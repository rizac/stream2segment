"""
Created on Feb 14, 2017

@author: riccardo
"""
import os
import re
from itertools import product
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import yaml
from pandas._testing import assert_frame_equal
from pandas.errors import EmptyDataError
from tables import HDF5ExtError

from stream2segment.process.db.models import Event, Segment
from stream2segment.process import SkipSegment
from stream2segment.resources import get_templates_fpath
from stream2segment.process.main import _run_and_write as process_main_run, process
from stream2segment.process.writers import SEGMENT_ID_COLNAME, BaseWriter


@pytest.fixture
def config_dict(pytestdir):
    """global fixture returning the dict from paramtable.yaml"""
    def func(**overridden_pars):
        with open(get_templates_fpath('paramtable.yaml')) as _:
            return {**yaml.safe_load(_), **overridden_pars}
    return func


def readcsv(filename, header=True):
    return pd.read_csv(filename, header=None) if not header else pd.read_csv(filename)


class patches:
    # paths container for class-level patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    get_session = 'stream2segment.process.main.get_session'
    close_session = 'stream2segment.process.main.close_session'
    run_process = 'stream2segment.process.main._run_and_write'
    # configlog4processing = 'stream2segment.process.main.configlog4processing'


class Test:

    # The class-level `init` fixture is marked with autouse=true which implies that
    # all test methods in the class will use this fixture without a need to state it
    # in the test function signature or with a class-level usefixtures decorator.
    # For info see:
    # https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, pytestdir, db4process):
        db4process.create(to_file=True)
        session = db4process.session
        # sets up the mocked functions: db session handling (using the already
        # created session) and log file handling:
        with patch(patches.get_session, return_value=session):
            with patch(patches.close_session,
                       side_effect=lambda *a, **v: None):
                yield

    # ======== ACTUAL TESTS: ================================

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @patch(patches.run_process, side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run,
                                            # fixtures:
                                            capsys, db4process, config_dict):
        """test a case where save inventory is True, and that we saved inventories"""
        from stream2segment.resources.templates import paramtable
        _ = process(dburl=db4process.dburl, pyfunc=paramtable.main,
                    segments_selection={'has_data': 'true'},
                    config=config_dict(snr_threshold=0),
                    verbose=True, logfile=False)

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]

        # assert the passed outputfile is None:
        assert args[4] is None
        # assert "Output file:  n/a" in result output:
        _ = capsys.readouterr()
        output, error = _.out, _.err
        assert not error
        assert re.search('Output file:\\s+n/a', output)

        # Note that apparently CliRunner() puts stderr and stdout together
        # (https://github.com/pallets/click/pull/868)
        # So we should test that we have these string twice:
        for subs in ["Processing function: ", "Config. file: "]:
            idx = output.find(subs)
            assert idx > -1

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("file_extension, options",
                             product(['.h5', '.csv'], [{},
                                                       {'chunksize': 1},
                                                       {'chunksize': 1, 'multi_process': True},
                                                       {'multi_process': True},
                                                       {'chunksize': 1, 'multi_process': 1},
                                                       ]))
    def test_simple_run_retDict_complex_select(self, file_extension, options,
                                               # fixtures:
                                               capsys, pytestdir, db4process, config_dict):
        """test a case where we have a more complex select involving joins"""
        # advanced_settings, cmdline_opts = options
        session = db4process.session
        # select the event times for the segments with data:
        etimes = sorted(_[1] for _ in session.query(Segment.id, Event.time).
                        join(Segment.event).filter(Segment.has_data))

        _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        from stream2segment.resources.templates import paramtable
        filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db4process.dburl, pyfunc=paramtable.main,
                    segments_selection={'has_data': 'true',
                                        'event.time': '<=%s' % (max(etimes).isoformat())},
                    config=config_dict(snr_threshold=0),
                    outfile=filename,
                    verbose=True, logfile=logfile, **options)

        # check file has been correctly written:
        if file_extension == '.csv':
            csv1 = readcsv(filename)
            assert len(csv1) == 1
            assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        else:
            dfr = pd.read_hdf(filename)
            assert len(dfr) == 1
            assert dfr.iloc[0][SEGMENT_ID_COLNAME] == expected_first_row_seg_id

        with open(logfile, 'r') as _:
            logcontent = _.read()
        segs = session.query(Segment.id).filter(Segment.has_data).all()
        assert "%d segment(s) found to process" % len(segs) in logcontent
        assert "1 of %d segment(s) successfully processed" % len(segs) in logcontent
        assert ("%d of %d segment(s) skipped with error message reported in the log " \
                "file") % (len(segs)-1, len(segs)) in logcontent

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    def test_simple_run_retDict_high_snr_threshold(self,
                                                   # fixtures:
                                                   capsys, pytestdir, db4process, config_dict):
        """same as `test_simple_run_retDict_saveinv` above
        but with a very high snr threshold => no rows processed"""
        session = db4process.session

        from stream2segment.resources.templates import paramtable
        options = {}
        file_extension = ".csv"
        filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db4process.dburl, pyfunc=paramtable.main,
                    segments_selection={'has_data': 'true'},
                    config=config_dict(snr_threshold=3),
                    outfile=filename,
                    verbose=True, logfile=logfile, **options)

        # no file written (see next comment for details). Check outfile is empty:
        with pytest.raises(EmptyDataError):
            csv1 = readcsv(filename)

        with open(logfile, 'r') as _:
            logcontent = _.read()
        segs = session.query(Segment.id).filter(Segment.has_data).all()
        assert ("""4 segment(s) found to process

segment (id=1): low snr 1.350154
segment (id=2): 4 traces (probably gaps/overlaps)
segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

0 of 4 segment(s) successfully processed
4 of 4 segment(s) skipped with error message reported in the log file""") in logcontent

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize('select_with_data, seg_chunk',
                             [(True, None), (True, 1), (False, None), (False, 1)])
    def test_simple_run_retDict_seg_select_empty_and_err_segments(self,
                                                                  select_with_data, seg_chunk,
                                                                  # fixtures:
                                                                  capsys,
                                                                  pytestdir,
                                                                  db4process, config_dict):
        """test a segment selection that takes only non-processable segments"""
        from stream2segment.resources.templates import paramtable
        options = {}
        if seg_chunk is not None:
            options['chunksize'] = seg_chunk
        seg_sel = {'station.latitude': '<10', 'station.longitude': '<10'}
        if select_with_data:
            seg_sel['has_data'] = 'true'
        file_extension = ".csv"
        filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db4process.dburl, pyfunc=paramtable.main,
                    segments_selection=seg_sel,
                    config=config_dict(snr_threshold=0),
                    outfile=filename,
                    verbose=True, logfile=logfile, **options)

        # check file has not been written (no data):
        with pytest.raises(EmptyDataError):
            csv1 = readcsv(filename)

        with open(logfile, 'r') as _:
            logcontent = _.read()

        if select_with_data:
            # selecting only with data means out of the three candidate segments, one
            # is discarded prior to processing:
            assert ("""2 segment(s) found to process

segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

0 of 2 segment(s) successfully processed
2 of 2 segment(s) skipped with error message reported in the log file""") in logcontent
        else:
            assert ("""3 segment(s) found to process

segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)
segment (id=6): MiniSeed error: no data

0 of 3 segment(s) successfully processed
3 of 3 segment(s) skipped with error message reported in the log file""") in logcontent

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    def test_simple_run_ret_list(self,
                                 # fixtures:
                                 capsys,
                                 pytestdir,
                                 db4process, config_dict):
        """test processing returning list, and also when we specify a different
        main function"""
        _seg = db4process.segments(with_inventory=True, with_data=True,
                                   with_gap=False).one()
        expected_first_row_seg_id = _seg.id

        from stream2segment.resources.templates import paramtable
        def main_return_list(segment, config):
            return list(paramtable.main(segment, config).values())
        options = {}
        seg_sel = {'has_data': 'true'}
        file_extension = ".csv"
        filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db4process.dburl, pyfunc=main_return_list,
                    segments_selection=seg_sel,
                    config=config_dict(snr_threshold=0),
                    outfile=filename,
                    verbose=True, logfile=logfile, **options)

        output, error = capsys.readouterr()
        assert not error
        # check file has been correctly written:
        csv1 = readcsv(filename)  # read first with header:
        # assert no rows:
        assert csv1.empty
        # now read without header:
        csv1 = readcsv(filename, header=False)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id

        with open(logfile, 'r') as _:
            logcontent = _.read()
        assert ("""4 segment(s) found to process

segment (id=2): 4 traces (probably gaps/overlaps)
segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

1 of 4 segment(s) successfully processed
3 of 4 segment(s) skipped with error message reported in the log file""") in logcontent
        # assert logfile exists:
        assert os.path.isfile(logfile)

    # Even though we are not interested here to check what is there on the created db,
    # because we test errors,
    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("err_type",
                             [ImportError,
                              AttributeError,
                              TypeError])
    def test_errors_process_not_run(self,
                                    err_type,
                                    # fixtures:
                                    capsys, pytestdir, db4process, config_dict):
        """test processing in case of severla 'critical' errors (which do not launch the process
          None means simply a bad argument (funcname missing)"""

        main = None
        if err_type == ImportError:
            def main(segment, config):
                import asdbasdabsdabsdasdbasdb

        elif err_type == AttributeError:
            def main(segment, config):
                return segment.___attribute_that_does_not_exist___()

        elif err_type == TypeError:
            def main(segment, config, wrong_argument):
                return {}

        options = {}
        seg_sel = {'has_data': 'true'}
        file_extension = ".csv"
        filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        with pytest.raises(Exception) as excinfo:
            _ = process(dburl=db4process.dburl, pyfunc=main,
                        segments_selection=seg_sel,
                        config=config_dict(snr_threshold=0),
                        outfile=filename,
                        verbose=True, logfile=logfile, **options)

        stdout, stderr = capsys.readouterr()
        # we did open the output file:
        assert os.path.isfile(filename)
        # and we never wrote on it:
        assert os.stat(filename).st_size == 0
        # check correct outputs, in both log and output:
        with open(logfile) as _:
            logfilecontent = _.read()
        outputs = [stdout, logfilecontent]
        for output in outputs:
            # Check that the err_type name is in the output traceback. But note that
            # ImportError is "ModuleNotFoundError" in recent versions of Python (3.9?),
            # so:
            err_names = [err_type.__name__]
            if err_type == ImportError:
                err_names.append('ModuleNotFoundError')
            # Try to loosely assert the messages is on standard output:
            assert any(e in output and 'Traceback' in output and ' line ' in output
                       for e in err_names)

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("err_type", [None, SkipSegment])
    def test_errors_process_completed(self, err_type,
                                      # fixtures:
                                      capsys, pytestdir, db4process, config_dict):
        """test processing in case of non 'critical' errors i.e., which do not prevent the process
          to be completed. None means we do not override SEG_SEL_STR which, with the current
          templates, causes no segment to be selected"""
        from stream2segment.resources.templates import paramtable
        if err_type == SkipSegment:
            seg_sel = {'has_data': 'true'}
            def main2(segment, config):
                raise SkipSegment(ValueError("invalid literal for .* with base 10: '4d'"))
        else:
            seg_sel = {'maxgap_numsamples': '[-0.5, 0.5]', 'has_data': 'true'}
            def main2(segment, config):
                return paramtable.main(segment, config)

        options = {}
        file_extension = ".csv"
        filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db4process.dburl, pyfunc=main2,
                    segments_selection=seg_sel,
                    config=config_dict(),
                    outfile=filename,
                    verbose=True, logfile=logfile, **options)

        output, error = capsys.readouterr()
        with open(logfile) as _:
            logcontent = _.read()

        assert not error
        # we did open the output file:
        assert os.path.isfile(filename)
        # and we never wrote on it:
        assert os.stat(filename).st_size == 0
        # check correct outputs, in both log and output:
        if err_type is None:  # no segments processed
            # we want to check that a particular string (str2check) is in the stdout
            # But consider that string changes according to py versions so use regex:
            str2check = \
                (r"0 segment\(s\) found to process\n"
                 r"\n+"
                 r"0 of 0 segment\(s\) successfully processed\n"
                 r"0 of 0 segment\(s\) skipped with error message reported in the log file")
            assert re.search(str2check, output)
            assert re.search(str2check, logcontent)
        else:
            # we want to check that a particular string (str2check) is in the stdout
            # But consider that string changes according to py versions so use regex:
            str2check = \
                (r'4 segment\(s\) found to process\n'
                 r'\n+'
                 r'0 of 4 segment\(s\) successfully processed\n'
                 r'4 of 4 segment\(s\) skipped with error message reported in the log file')
            assert re.search(str2check, output)

            str2check = \
                (r"4 segment\(s\) found to process\n"
                 r"\n+"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"\n+"
                 r"0 of 4 segment\(s\) successfully processed\n"
                 r"4 of 4 segment\(s\) skipped with error message reported in the log file")
            try:
                assert re.search(str2check, logcontent)
            except AssertionError:
                asd =9

    @patch(patches.run_process, side_effect=process_main_run)
    def test_save2file(self, mock_run,
                       # fixtures:
                       capsys, pytestdir, db4process, config_dict):
        """test the save2file python module, and also test a case when
        no output file is provided
        """
        # set values which will override the yaml config in templates folder:
        dir_ = pytestdir.makedir()

        from stream2segment.resources.templates import save2fs
        options = {}
        seg_sel = {'has_data': 'true'}
        file_extension = ".csv"
        # filename = pytestdir.newfile(file_extension)
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db4process.dburl, pyfunc=save2fs.main,
                    segments_selection=seg_sel,
                    config=config_dict(snr_threshold=0, root_dir=os.path.abspath(dir_)),
                    outfile=None,
                    verbose=True, logfile=logfile, **options)

        # output, error = capsys.readouterr()
        # with open(logfile) as _:
        #     logcontent = _.read()

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_only_written_segment = \
            db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        # get seiscomp path of OK segment before the session is closed:
        path = os.path.join(dir_, expected_only_written_segment.sds_path())

        output, error = capsys.readouterr()

        filez = os.listdir(os.path.dirname(path))
        assert len(filez) == 2
        from obspy import read
        stream1 = read(os.path.join(os.path.dirname(path), filez[0]), format='MSEED')
        stream2 = read(os.path.join(os.path.dirname(path), filez[1]), format='MSEED')
        assert len(stream1) == len(stream2) == 1
        assert not np.allclose(stream1[0].data, stream2[0].data)

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]

        # asssert passed outputfile is None:
        assert args[4] is None
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', output)

        # Note that apparently CliRunner() (see clirunner fixture) puts stderr and stdout
        # together (https://github.com/pallets/click/pull/868)
        # Reminder: previously, log erros where redirected to stderr
        # This is dangerous as we use a redirect to avoid external libraries to pritn to stderr
        # and logging to stderr might cause 'operation on closed file'.
        for subs in ["Processing function: ", "Config. file: "]:
            idx = output.find(subs)
            assert idx > -1

    # appending to file:

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize('hdf', [True, False])
    @pytest.mark.parametrize("output_file_empty", [False, True])
    @pytest.mark.parametrize('processing_py_return_list', [True, False])
    def test_append_on_badly_formatted_outfile(self, processing_py_return_list,
                                                     output_file_empty,
                                                     hdf,
                                                     # fixtures:
                                                     capsys, pytestdir, db4process,
                                                     config_dict):
        """test a case where we append on an badly formatted output file
        (no segment id column found)"""
        if processing_py_return_list and hdf:
            # hdf does not support returning lists
            return
        seg_sel = {'has_data': 'true'}
        config = config_dict(snr_threshold=0)
        options = {'append': True}

        from stream2segment.resources.templates import paramtable
        if processing_py_return_list:
            def main(segment, config):
                return list(paramtable.main(segment, config).values())
        else:
            main = paramtable.main

        outfilepath = pytestdir.newfile('.hdf' if hdf else '.csv', create=True)
        if not output_file_empty:
            if hdf:
                pd.DataFrame(columns=['-+-', '[[['], data=[[1, 'a']]).to_hdf(outfilepath,
                                                                             format='t',
                                                                             key='f')
            else:
                with open(outfilepath, 'wt') as _:
                    _.write('asdasd')

        logfile = pytestdir.newfile('.log')

        # this are the cases where the append is ok:
        should_be_ok = processing_py_return_list or \
                       (not hdf and output_file_empty)
        try:
            _ = process(dburl=db4process.dburl, pyfunc=main,
                        segments_selection=seg_sel,
                        config=config,
                        outfile=outfilepath,
                        verbose=True, logfile=logfile, **options)
            assert should_be_ok
        except (HDF5ExtError, BaseWriter._SEGID_NOTFOUND_ERR.__class__) as exc:
            if isinstance(exc, HDF5ExtError):
                assert hdf
            assert not should_be_ok

        output, error = capsys.readouterr()

        if not should_be_ok:
            # if hdf and output file is empty, the error is a HDF error
            # (because an emopty file cannot be opened as HDF, a CSV apparently
            # can)
            is_empty_hdf_file = hdf and output_file_empty
            if not is_empty_hdf_file:
                # otherwise, it's a s2s error where we could not find the
                # segment id column:
                assert ("TypeError: Cannot append to file, segment_id column " \
                        "name not found") in output
            return

        with open(logfile) as _:
            logtext = _.read()
        assert len(logtext) > 0
        assert "Appending results to existing file" in logtext

    from stream2segment.process.writers import _SEGMENT_ID_COLNAMES

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize('hdf', [True, False])
    @pytest.mark.parametrize("segment_id_colname", _SEGMENT_ID_COLNAMES)
    @pytest.mark.parametrize('processing_py_return_list', [True, False])
    # @patch('stream2segment.cli.click.confirm', return_value=True)
    def test_append(self,
                    processing_py_return_list, segment_id_colname, hdf,
                    # fixtures:
                    capsys, pytestdir, db4process, config_dict):
        """test a typical case where we supply the append option"""
        if processing_py_return_list and hdf:
            # hdf does not support returning lists
            pytest.skip("Python function cannot return lists when output is HDF")

        with patch('stream2segment.process.writers.SEGMENT_ID_COLNAME',
              segment_id_colname):
            options = {'append': True}
            config = config_dict(snr_threshold=0)

            _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
            expected_first_row_seg_id = _seg.id
            station_id_whose_inventory_is_saved = _seg.station.id

            session = db4process.session

            outfilepath = pytestdir.newfile('.hdf' if hdf else '.csv')
            logfile = pytestdir.newfile('.log')

            from stream2segment.resources.templates import paramtable
            main = paramtable.main
            if processing_py_return_list:
                def main(segment, config):
                    return list(paramtable.main(segment, config).values())

            _ = process(dburl=db4process.dburl, pyfunc=main,
                        segments_selection={'has_data': 'true'},
                        config=config,
                        outfile=outfilepath,
                        verbose=True, logfile=logfile, **options)

            processing_df1 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df1) == 1
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df1.loc[0, segid_column] == expected_first_row_seg_id
            with open(logfile) as _:
                logtext1 = _.read()
            assert "4 segment(s) found to process" in logtext1
            assert "Skipping 1 already processed segment(s)" not in logtext1
            assert "Ignoring `append` functionality: output file does not exist or not provided" \
                in logtext1
            assert "1 of 4 segment(s) successfully processed" in logtext1

            # now test a second call, the same as before:
            logfile = pytestdir.newfile('.log')
            _ = process(dburl=db4process.dburl, pyfunc=main,
                        segments_selection={'has_data': 'true'},
                        config=config,
                        outfile=outfilepath,
                        verbose=True, logfile=logfile, **options)
            # check file has been correctly written:
            processing_df2 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df2) == 1
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df2.loc[0, segid_column] == expected_first_row_seg_id
            with open(logfile) as _:
                logtext2 = _.read()
            assert "3 segment(s) found to process" in logtext2
            assert "Skipping 1 already processed segment(s)" in logtext2
            assert "Appending results to existing file" in logtext2
            assert "0 of 3 segment(s) successfully processed" in logtext2
            # assert two rows are equal:
            assert_frame_equal(processing_df1, processing_df2, check_dtype=True)

            # change the segment id of the written segment
            seg = session.query(Segment).filter(Segment.id == expected_first_row_seg_id).\
                first()
            new_seg_id = seg.id * 100
            seg.id = new_seg_id
            session.commit()

            # now test a second call, the same as before:
            logfile = pytestdir.newfile('.log')
            _ = process(dburl=db4process.dburl, pyfunc=main,
                        segments_selection={'has_data': 'true'},
                        config=config,
                        outfile=outfilepath,
                        verbose=True, logfile=logfile, **options)
            # check file has been correctly written:
            processing_df3 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df3) == 2
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df3.loc[0, segid_column] == expected_first_row_seg_id
            assert processing_df3.loc[1, segid_column] == new_seg_id
            with open(logfile) as _:
                logtext3 = _.read()
            assert "4 segment(s) found to process" in logtext3
            assert "Skipping 1 already processed segment(s)" in logtext3
            assert "Appending results to existing file" in logtext3
            assert "1 of 4 segment(s) successfully processed" in logtext3
            # assert two rows are equal:
            assert_frame_equal(processing_df1, processing_df3[:1], check_dtype=True)

            # last try: no append (also set no-prompt to test that we did not
            # prompt the user)
            logfile = pytestdir.newfile('.log')
            options_ = {**options, 'append': False}
            _ = process(dburl=db4process.dburl, pyfunc=main,
                        segments_selection={'has_data': 'true'},
                        config=config,
                        outfile=outfilepath,
                        verbose=True, logfile=logfile, **options_)
            # check file has been correctly written:
            processing_df4 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df4) == 1
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df4.loc[0, segid_column] == new_seg_id
            with open(logfile) as _:
                logtext4 = _.read()
            assert "4 segment(s) found to process" in logtext4
            assert "Skipping 1 already processed segment(s)" not in logtext4
            assert "Appending results to existing file" not in logtext4
            assert "1 of 4 segment(s) successfully processed" in logtext4
            assert 'Overwriting existing output file' in logtext4


def read_processing_output(filename, header=True):  # <- header only for csv
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.hdf':
        return pd.read_hdf(filename).reset_index(drop=True, inplace=False)
    elif ext == '.csv':
        return pd.read_csv(filename, header=None) if not header \
            else pd.read_csv(filename)
    else:
        raise ValueError('Unrecognized extension %s' % ext)