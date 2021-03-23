'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import object

import re
import os
import sys
from os.path import splitext

import mock
from mock import patch

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from stream2segment.cli import cli
from stream2segment.process.db import Segment
from stream2segment.utils.resources import get_templates_fpath
from stream2segment.process.main import run as process_main_run
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.process.writers import BaseWriter, _SEGMENT_ID_COLNAMES, \
    SEGMENT_ID_COLNAME

SEG_SEL_STR = 'segments_selection'


@pytest.fixture
def yamlfile(pytestdir):
    '''global fixture wrapping pytestdir.yamlfile'''
    def func(**overridden_pars):
        return pytestdir.yamlfile(get_templates_fpath('paramtable.yaml'), **overridden_pars)

    return func


def read_processing_output(filename, header=True):  # <- header only for csv
    ext = splitext(filename)[1].lower()
    if ext == '.hdf':
        return pd.read_hdf(filename).reset_index(drop=True, inplace=False)
    elif ext == '.csv':
        return pd.read_csv(filename, header=None) if not header \
            else pd.read_csv(filename)
    else:
        raise ValueError('Unrecognized extension %s' % ext)


class Test(object):

    pyfile = get_templates_fpath("paramtable.py")

    @classmethod
    def cp_pyfile(cls, dest_py_file, return_lists=False):
        """Return cls.pyfile but returning lists insted of dicts"""

        # modify python so taht 'main' returns a list by calling the default 'main'
        # and returning its keys:
        with open(cls.pyfile, 'r') as opn:
            content = opn.read()

        pyfile = dest_py_file  # pytestdir.newfile('.py')

        if return_lists:
            cont2 = content.replace("def main(segment, config):", """def main(segment, config):
    return list(main2(segment, config).values())
def main2(segment, config):""")

        with open(pyfile, 'wb') as _opn:
            _opn.write(cont2.encode('utf8'))

        return pyfile

    @property
    def logfilecontent(self):
        assert os.path.isfile(self._logfilename)
        with open(self._logfilename) as opn:
            return opn.read()

    # The class-level `init` fixture is marked with autouse=true which implies that all test
    # methods in the class will use this fixture without a need to state it in the test
    # function signature or with a class-level usefixtures decorator. For info see:
    # https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, pytestdir, db4process):
        db4process.create(to_file=True)
        session = db4process.session
        # sets up the mocked functions: db session handling (using the already created session)
        # and log file handling:
        with patch('stream2segment.utils.inputvalidation.valid_session', return_value=session):
            with patch('stream2segment.main.closesession',
                       side_effect=lambda *a, **v: None):
                with patch('stream2segment.main.configlog4processing') as mock2:

                    def clogd(logger, logfilebasepath, verbose):
                        # config logger as usual, but redirects to a temp file
                        # that will be deleted by pytest, instead of polluting the program
                        # package:
                        ret = o_configlog4processing(logger,
                                                     pytestdir.newfile('.log') \
                                                     if logfilebasepath else None,
                                                     verbose)

                        self._logfilename = ret[0].baseFilename
                        return ret

                    mock2.side_effect = clogd

                    yield

# ## ======== ACTUAL TESTS: ================================

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run,
                                            # fixtures:
                                            db4process, clirunner, yamlfile):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,
                            SEG_SEL_STR: {'has_data': 'true'}}
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                        '-p', self.pyfile, '-c', yamlfile(**config_overrides),
                                        '-a'])
        assert clirunner.ok(result)

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        # assert third argument (`ondone` callback) is None 'ondone' or is a BaseWriter (no-op)
        # class:
        assert args[2] is None or \
            type(args[2]) == BaseWriter  # pylint: disable=unidiomatic-typecheck
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', result.output)
        # assert "Output file:  n/a" in result output:
        assert re.search('Ignoring `append` functionality: output file does not exist '
                         'or not provided', result.output)

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, ['-a']),
                              ])
    @pytest.mark.parametrize('hdf', [True, False])
    @pytest.mark.parametrize("output_file_empty", [False, True])
    @pytest.mark.parametrize('processing_py_return_list', [True, False])
    def test_simple_run_append_on_badly_formatted_outfile(self, processing_py_return_list,
                                                          output_file_empty,
                                                          hdf, advanced_settings,
                                                          cmdline_opts,
                                                          # fixtures:
                                                          pytestdir, db4process,
                                                          clirunner, yamlfile):
        '''test a case where we append on an badly formatted output file
        (no segment id column found)'''
        if processing_py_return_list and hdf:
            # hdf does not support returning lists
            return
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,
                            SEG_SEL_STR: {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        _seg = db4process.segments(with_inventory=True, with_data=True,
                                   with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        pyfile = self.pyfile
        if processing_py_return_list:
            pyfile = self.cp_pyfile(pytestdir.newfile('.py'), return_lists=True)

        # output_file_empty=True
        # hdf = False
        # processing_py_return_list = False

        outfilepath = pytestdir.newfile('.hdf' if hdf else '.csv', create=True)
        if not output_file_empty:
            if hdf:
                pd.DataFrame(columns=['-+-', '[[['], data=[[1, 'a']]).to_hdf(outfilepath,
                                                                             format='t',
                                                                             key='f')
            else:
                with open(outfilepath, 'wt') as _:
                    _.write('asdasd')

        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                  '-p', pyfile, '-c', yamlfile(**config_overrides),
                                  outfilepath] + cmdline_opts)

        # this are the cases where the append is ok:
        should_be_ok = processing_py_return_list or \
                       (not hdf and output_file_empty)

        if not should_be_ok:
            try:
                assert not clirunner.ok(result)
            except AssertionError:
                asd = 9
            # if hdf and output file is empty, the error is a HDF error
            # (because an emopty file cannot be opened as HDF, a CSV apparently
            # can)
            is_empty_hdf_file = hdf and output_file_empty
            if not is_empty_hdf_file:
                # otherwise, it's a s2s error where we could not find the
                # segment id column:
                assert ("TypeError: Cannot append to file, segment_id column " \
                        "name not found") in result.output
            return

        try:
            assert clirunner.ok(result)
        except AssertionError:
            asd = 9
        # # SKIP check file has been correctly written (see test below):
        # def read_hdf(filename):
        #     return pd.read_hdf(filename).reset_index(drop=True, inplace=False)
        #
        # # check file has been correctly written:
        # processing_df1 = read_processing_output(outfilepath,
        #                                         header=not processing_py_return_list)
        # assert len(processing_df1) == 1
        # assert processing_df1.loc[0,
        #                           processing_df1.columns[0]] == expected_first_row_seg_id

        logtext = self.logfilecontent
        assert len(logtext) > 0
        assert "Appending results to existing file" in logtext

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize('hdf', [True, False])
    @pytest.mark.parametrize("segment_id_colname", _SEGMENT_ID_COLNAMES)
    @pytest.mark.parametrize('processing_py_return_list', [True, False])
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, ['-a']),
                              ({}, ['-a', '--multi-process']),
                              ])
    @mock.patch('stream2segment.cli.click.confirm', return_value=True)
    # @mock.patch('stream2segment.process.writers.SEGMENT_ID_COLNAME')
    def test_append(self, # mock_seg_id_colname,
                    mock_click_confirm, advanced_settings, cmdline_opts,
                    processing_py_return_list, segment_id_colname, hdf,
                    # fixtures:
                    pytestdir, db4process, clirunner, yamlfile):
        '''test a typical case where we supply the append option'''
        if processing_py_return_list and hdf:
            # hdf does not support returning lists
            pytest.skip("Python function cannot return lists when output is HDF")

        # also, these tests take a lot of time when multi process is on. In this
        # case, avoid testing with old segment id column name:
        if cmdline_opts == ['-a', '--multi-process'] and segment_id_colname != SEGMENT_ID_COLNAME:
            pytest.skip("Skipping time-consuming tests with old segment id column names "
                        "and multiprocess")

        with patch('stream2segment.process.writers.SEGMENT_ID_COLNAME',
              segment_id_colname):
            # assign the default segment id column name:
            # mock_seg_id_colname.return_value=segment_id_colname

            # set values which will override the yaml config in templates folder:
            config_overrides = {'snr_threshold': 0,
                                SEG_SEL_STR: {'has_data': 'true'}}
            if advanced_settings:
                config_overrides['advanced_settings'] = advanced_settings
            yaml_file = yamlfile(**config_overrides)

            _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
            expected_first_row_seg_id = _seg.id
            station_id_whose_inventory_is_saved = _seg.station.id

            session = db4process.session

            outfilepath = pytestdir.newfile('.hdf' if hdf else '.csv')

            pyfile = self.pyfile
            if processing_py_return_list:
                pyfile = self.cp_pyfile(pytestdir.newfile('.py'), return_lists=True)

            mock_click_confirm.reset_mock()
            result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                            '-p', pyfile, '-c', yaml_file,
                                            outfilepath]
                                      + cmdline_opts)
            assert clirunner.ok(result)

            processing_df1 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df1) == 1
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df1.loc[0, segid_column] == expected_first_row_seg_id
            logtext1 = self.logfilecontent
            assert "4 segment(s) found to process" in logtext1
            assert "Skipping 1 already processed segment(s)" not in logtext1
            assert "Ignoring `append` functionality: output file does not exist or not provided" \
                in logtext1
            assert "1 of 4 segment(s) successfully processed" in logtext1
            assert not mock_click_confirm.called

            # now test a second call, the same as before:
            mock_click_confirm.reset_mock()
            result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                            '-p', pyfile, '-c', yaml_file, outfilepath]
                                      + cmdline_opts)
            # check file has been correctly written:
            processing_df2 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df2) == 1
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df2.loc[0, segid_column] == expected_first_row_seg_id
            logtext2 = self.logfilecontent
            assert "3 segment(s) found to process" in logtext2
            assert "Skipping 1 already processed segment(s)" in logtext2
            assert "Appending results to existing file" in logtext2
            assert "0 of 3 segment(s) successfully processed" in logtext2
            assert not mock_click_confirm.called
            # assert two rows are equal:
            assert_frame_equal(processing_df1, processing_df2, check_dtype=True)

            # change the segment id of the written segment
            seg = session.query(Segment).filter(Segment.id == expected_first_row_seg_id).\
                first()
            new_seg_id = seg.id * 100
            seg.id = new_seg_id
            session.commit()

            if processing_py_return_list is False and hdf is False:
                asd = 9
            # now test a second call, the same as before:
            mock_click_confirm.reset_mock()
            result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl, '-p', pyfile,
                                            '-c', yaml_file, outfilepath] + cmdline_opts)
            # check file has been correctly written:
            processing_df3 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df3) == 2
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df3.loc[0, segid_column] == expected_first_row_seg_id
            assert processing_df3.loc[1, segid_column] == new_seg_id
            logtext3 = self.logfilecontent
            assert "4 segment(s) found to process" in logtext3
            assert "Skipping 1 already processed segment(s)" in logtext3
            assert "Appending results to existing file" in logtext3
            assert "1 of 4 segment(s) successfully processed" in logtext3
            assert not mock_click_confirm.called
            # assert two rows are equal:
            assert_frame_equal(processing_df1, processing_df3[:1], check_dtype=True)

            # last try: no append (also set no-prompt to test that we did not prompt the user)
            mock_click_confirm.reset_mock()
            result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl, '-p', pyfile,
                                            '-c', yaml_file, outfilepath] + cmdline_opts[1:])
            # check file has been correctly written:
            processing_df4 = read_processing_output(outfilepath,
                                                    header=not processing_py_return_list)
            assert len(processing_df4) == 1
            segid_column = segment_id_colname if hdf else processing_df1.columns[0]
            assert processing_df4.loc[0, segid_column] == new_seg_id
            logtext4 = self.logfilecontent
            assert "4 segment(s) found to process" in logtext4
            assert "Skipping 1 already processed segment(s)" not in logtext4
            assert "Appending results to existing file" not in logtext4
            assert "1 of 4 segment(s) successfully processed" in logtext4
            assert 'Overwriting existing output file' in logtext4
            assert mock_click_confirm.called

            # last try: prompt return False
            mock_click_confirm.reset_mock()
            mock_click_confirm.return_value = False
            result = clirunner.invoke(cli, ['process',  '--dburl', db4process.dburl, '-p', pyfile,
                                            '-c', yaml_file, outfilepath] + cmdline_opts[1:])
            assert result.exception
            assert type(result.exception) == SystemExit
            assert result.exception.code == 1
