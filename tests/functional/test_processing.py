'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from past.utils import old_div
import os
import sys
from datetime import datetime, timedelta
import mock
from mock import patch
import re

import pytest
import pandas as pd
from pandas.errors import EmptyDataError
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.utils.inputargs import yaml_load as orig_yaml_load
from stream2segment.utils.resources import get_templates_fpaths, get_templates_fpath
from stream2segment.process.db import get_inventory
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.process.main import run as process_main_run, query4process
from stream2segment.utils.url import URLError
from stream2segment.process.writers import BaseWriter
from stream2segment.io.utils import dumps_inv


@pytest.fixture
def yamlfile(pytestdir):
    '''global fixture wrapping pytestdir.yamlfile'''
    def func(**overridden_pars):
        return pytestdir.yamlfile(get_templates_fpath('paramtable.yaml'), **overridden_pars)

    return func


def readcsv(filename, header=True):
    return pd.read_csv(filename, header=None) if not header else pd.read_csv(filename)


class Test(object):

    pyfile = get_templates_fpath("paramtable.py")

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
        with patch('stream2segment.utils.inputargs.get_session', return_value=session):
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

    def inlogtext(self, string):
        '''Checks that `string` is in log text.
        The assertion `string in self.logfilecontent` fails in py3.5, although the differences
        between characters is the same position is zero. We did not find any better way than
        fixing it via this cumbersome function'''
        logtext = self.logfilecontent
        i = 0
        while len(logtext[i:i+len(string)]) == len(string):
            if (sum(ord(a)-ord(b) for a, b in zip(string, logtext[i:i+len(string)]))) == 0:
                return True
            i += 1
        return False

# ## ======== ACTUAL TESTS: ================================

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @patch('stream2segment.process.db.get_inventory', side_effect=get_inventory)
    def test_segwrapper(self, mock_getinv,
                        # fixtures:
                        db4process, data):
        session = db4process.session
        segids = query4process(session, {}).all()
        seg_with_inv = \
            db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        sta_with_inv_id = seg_with_inv.station.id
        invcache = {}
        prev_staid = None
        for (segid, staid) in segids:
            assert prev_staid is None or staid >= prev_staid
            staequal = prev_staid is not None and staid == prev_staid
            prev_staid = staid
            segment = session.query(Segment).filter(Segment.id == segid).first()
            sta = segment.station
            segment.station._inventory = invcache.get(sta.id, None)

            mock_getinv.reset_mock()
            if sta.id != sta_with_inv_id:
                with pytest.raises(Exception):  # all inventories are None
                    segment.inventory()
                assert mock_getinv.called
                # re-call it and assert we raise the previous Exception:
                ccc = mock_getinv.call_count
                with pytest.raises(Exception):  # all inventories are None
                    segment.inventory()
                assert mock_getinv.call_count == ccc
            else:
                invcache[sta.id] = segment.inventory()
                if staequal:
                    assert not mock_getinv.called
                else:
                    assert mock_getinv.called
                assert len(segment.station.inventory_xml) > 0
            segs = segment.siblings().all()
            # as channel's channel is either 'ok' or 'err' we should never have
            # other components
            assert len(segs) == 0

        # NOW TEST OTHER ORIENTATION PROPERLY. WE NEED TO ADD WELL FORMED SEGMENTS WITH CHANNELS
        # WHOSE ORIENTATION CAN BE DERIVED:
        staid = session.query(Station.id).first()[0]
        dcid = session.query(DataCenter.id).first()[0]
        eid = session.query(Event.id).first()[0]
        dwid = session.query(Download.id).first()[0]
        # add channels
        c_1 = Channel(station_id=staid, location='ok', channel="AB1", sample_rate=56.7)
        c_2 = Channel(station_id=staid, location='ok', channel="AB2", sample_rate=56.7)
        c_3 = Channel(station_id=staid, location='ok', channel="AB3", sample_rate=56.7)
        session.add_all([c_1, c_2, c_3])
        session.commit()
        # add segments. Create attributes (although not strictly necessary to have bytes data)
        atts = data.to_segment_dict('trace_GE.APE.mseed')
        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c_1.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        sg2 = Segment(channel_id=c_2.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        sg3 = Segment(channel_id=c_3.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        session.add_all([sg1, sg2, sg3])
        session.commit()
        # start testing:
        segids = query4process(session, {}).all()

        for (segid, staid) in segids:
            segment = session.query(Segment).filter(Segment.id == segid).first()
            segs = segment.siblings()
            if segs.all():
                assert segment.id in (sg1.id, sg2.id, sg3.id)
                assert len(segs.all()) == 2

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run,
                                            # fixtures:
                                            db4process, yamlfile):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        yaml_file = yamlfile(**config_overrides)

        runner = CliRunner()

        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl,
                               '-p', self.pyfile, '-c', yaml_file])

        assert not result.exception

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        # assert third argument (`ondone` callback) is None 'ondone' or is a BaseWriter (no-op)
        # class:
        assert args[2] is None or \
            type(args[2]) == BaseWriter  # pylint: disable=unidiomatic-typecheck
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', result.output)

        # Note that apparently CliRunner() puts stderr and stdout together
        # (https://github.com/pallets/click/pull/868)
        # So we should test that we have these string twice:
        for subs in ["Processing function: ", "Config. file: "]:
            idx = result.output.find(subs)
            assert idx > -1

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    def test_simple_run_retDict_complex_select(self,
                                               advanced_settings,
                                               cmdline_opts,
                                               # fixtures:
                                               pytestdir, db4process, yamlfile):
        '''test a case where we have a more complex select involving joins'''
        session = db4process.session
        # select the event times for the segments with data:
        etimes = sorted(_[1] for _ in session.query(Segment.id, Event.time).
                        join(Segment.event).filter(Segment.has_data))

        config_overrides = {'snr_threshold': 0,
                            'segment_select': {'has_data': 'true',
                                               'event.time': '<=%s' % (max(etimes).isoformat())}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        # the selection above should be the same as the previous test:
        # test_simple_run_retDict_saveinv,
        # as segment_select[event.time] includes all segments in segment_select['has_data'],
        # thus the code is left as it was in the method above
        yaml_file = yamlfile(**config_overrides)

        _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl,
                               '-p', self.pyfile, '-c', yaml_file, filename] + cmdline_opts)

        assert not result.exception
        # check file has been correctly written:
        csv1 = readcsv(filename)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id

        self.inlogtext("""3 segment(s) found to process

segment (id=3): 4 traces (probably gaps/overlaps)
segment (id=2): Station inventory (xml) error: no data

1 of 3 segment(s) successfully processed
2 of 3 segment(s) skipped with error message (check log or details)""")
        # assert logfile exists:
        assert os.path.isfile(self._logfilename)

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    def test_simple_run_retDict_high_snr_threshold(self,
                                                   # fixtures:
                                                   pytestdir, db4process, yamlfile):
        '''same as `test_simple_run_retDict_saveinv` above
        but with a very high snr threshold => no rows processed'''
        # setup inventories:
        session = db4process.session
        # set values which will override the yaml config in templates folder:
        config_overrides = {  # snr_threshold 3 is high enough to discard the only segment
                              # we would process otherwise:
                            'snr_threshold': 3,
                            'segment_select': {'has_data': 'true'}}
        yaml_file = yamlfile(**config_overrides)

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl,
                               '-p', self.pyfile, '-c', yaml_file, filename])

        assert not result.exception
        # no file written (see next comment for details). Check outfile is empty:
        with pytest.raises(EmptyDataError):
            csv1 = readcsv(filename)
        # check file has been correctly written: 2 segments have no data, thus they are skipped
        # and not logged
        # 2 segments have gaps/overlaps, thus they are skipped and logged
        # 1 segment has data but no inventory, thus skipped and logged
        # 1 segment with data and inventory, but snr is too low: skipped and logged
        assert self.inlogtext("""4 segment(s) found to process

segment (id=1): low snr 1.350154
segment (id=2): 4 traces (probably gaps/overlaps)
segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

0 of 4 segment(s) successfully processed
4 of 4 segment(s) skipped with error message (check log or details)""")

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize('select_with_data, seg_chunk',
                             [(True, None), (True, 1), (False, None), (False, 1)])
    def test_simple_run_retDict_seg_select_empty_and_err_segments(self,
                                                                  select_with_data, seg_chunk,
                                                                  # fixtures:
                                                                  pytestdir,
                                                                  db4process, yamlfile):
        '''test a segment selection that takes only non-processable segments'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,  # take all segments
                            # the following will select the station with no inventory.
                            # There are three segments associated with it:
                            # one with data and no gaps, one with data and gaps,
                            # the third with no data
                            'segment_select': {'station.latitude': '<10',
                                               'station.longitude': '<10'}}
        if select_with_data:
            config_overrides['segment_select']['has_data'] = 'true'
        if seg_chunk is not None:
            config_overrides['advanced_settings'] = {'segments_chunk': seg_chunk}

        yaml_file = yamlfile(**config_overrides)

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                     '-p', self.pyfile,
                                     '-c', yaml_file,
                                     filename])
        assert not result.exception
        # check file has not been written (no data):
        with pytest.raises(EmptyDataError):
            csv1 = readcsv(filename)

        # see comment aboive on segments_select
        if select_with_data:
            # selecting only with data means out of the three candidate segments, one
            # is discarded prior to processing:
            assert self.inlogtext("""2 segment(s) found to process

segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

0 of 2 segment(s) successfully processed
2 of 2 segment(s) skipped with error message (check log or details)""")
        else:
            assert self.inlogtext("""3 segment(s) found to process

segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)
segment (id=6): MiniSeed error: no data

0 of 3 segment(s) successfully processed
3 of 3 segment(s) skipped with error message (check log or details)""")

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    def test_simple_run_ret_list(self, advanced_settings, cmdline_opts,
                                 # fixtures:
                                 pytestdir,
                                 db4process, yamlfile):
        '''test processing returning list, and also when we specify a different main function'''

        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,  # take all segments
                            'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        yaml_file = yamlfile(**config_overrides)

        _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        pyfile = self.pyfile

        # Now wrtite pyfile into a named temp file, with the method:
        # def main_retlist(segment, config):
        #    return main(segment, config).keys()
        # the method returns a list (which is what we want to test
        # and this way, we do not need to keep synchronized any additional file
        filename = pytestdir.newfile('.csv')
        pyfile2 = pytestdir.newfile('.py')
        if not os.path.isfile(pyfile2):

            with open(pyfile, 'r') as opn:
                content = opn.read()

            cont2 = content.replace("def main(segment, config):", """def main_retlist(segment, config):
    return list(main(segment, config).values())
def main(segment, config):""")
            with open(pyfile2, 'wb') as _opn:
                _opn.write(cont2.encode('utf8'))

        runner = CliRunner()
        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                     '-p', pyfile2, '-f', "main_retlist",
                                     '-c', yaml_file,
                                     filename] + cmdline_opts)

        assert not result.exception
        # check file has been correctly written:
        csv1 = readcsv(filename)  # read first with header:
        # assert no rows:
        assert csv1.empty
        # now read without header:
        csv1 = readcsv(filename, header=False)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id

        assert self.inlogtext("""4 segment(s) found to process

segment (id=2): 4 traces (probably gaps/overlaps)
segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

1 of 4 segment(s) successfully processed
3 of 4 segment(s) skipped with error message (check log or details)""")
        # assert logfile exists:
        assert os.path.isfile(self._logfilename)

    # Even though we are not interested here to check what is there on the created db,
    # because we test errors,
    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("cmdline_opts",
                             [[], ['--multi-process'],
                              ['--multi-process', '--num-processes', '1']])
    @pytest.mark.parametrize("err_type, expects_log_2_be_configured",
                             [(None, False),
                              (ImportError, False),
                              (AttributeError, True),
                              (TypeError, True)])
    def test_errors_process_not_run(self,
                                    err_type, expects_log_2_be_configured, cmdline_opts,
                                    # fixtures:
                                    pytestdir, db4process, yamlfile):
        '''test processing in case of severla 'critical' errors (which do not launch the process
          None means simply a bad argument (funcname missing)'''
        pyfile = self.pyfile

        # REMEMBER THAT BY DEFAULT LEAVING THE segment_select IMPLEMENTED in conffile
        # WE WOULD HAVE NO SEGMENTS, as maxgap_numsamples is None for all segments of this test
        # Thus provide config overrides:
        yaml_file = yamlfile(segment_select={'has_data': 'true'})

        runner = CliRunner()
        # Now wrtite pyfile into a named temp file, BUT DO NOT SUPPLY EXTENSION
        # This seems to fail in python3 (FIXME: python2?)
        filename = pytestdir.newfile('.csv')
        pyfile2 = pytestdir.newfile('.py')

        with open(pyfile, 'r') as opn:
            content = opn.read()

        # here replace the stuff we need:
        if err_type == ImportError:
            # create the exception: implement a fake import
            content = content.replace("def main(", """import abcdefghijk_blablabla_456isjfger
def main2(""")
        elif err_type == AttributeError:
            # create the exception. Implement a bad signature whci hraises a TypeError
            content = content.replace("def main(", """def main2(segment, config):
    return "".attribute_that_does_not_exist_i_guess_blabla()

def main(""")
        elif err_type == TypeError:
            # create the exception. Implement a bad signature whci hraises a TypeError
            content = content.replace("def main(", """def main2(segment, config, wrong_argument):
    return int(None)

def main(""")
        else:  # err_type is None
            # this case does not do anything, but since we will call 'main2' as funcname
            # in `runner.invoke` (see below), we should raise a BadArgument
            pass

        with open(pyfile2, 'wb') as _opn:
            _opn.write(content.encode('utf8'))

        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl, '--no-prompt',
                                     '-p', pyfile2, '-f', "main2",
                                     '-c', yaml_file,
                                     filename] + cmdline_opts)

        assert result.exception
        assert result.exit_code != 0
        stdout = result.output
        if expects_log_2_be_configured:
            # these cases raise BEFORE running pyfile
            # assert log config has not been called: (see self.init):
            assert self._logfilename is not None
            # we did open the output file:
            assert os.path.isfile(filename)
            # and we never wrote on it:
            assert os.stat(filename).st_size == 0
            # check correct outputs, in both log and output:
            outputs = [stdout, self.logfilecontent]
            for output in outputs:
                # Try to assert the messages on standard output being compatible with PY2,
                # as the messages might change
                assert err_type.__name__ in output \
                    and 'Traceback' in output and ' line ' in output
        else:
            # these cases raise BEFORE running pyfile
            # assert log config has not been called: (see self.init):
            with pytest.raises(Exception):
                # basically, assert we do not have the log file
                _ = self.logfilecontent
            assert 'Invalid value for "pyfile": ' in stdout
            further_string = 'main2' if err_type is None else 'No module named'
            assert further_string in stdout
            # we did NOt open the output file:
            assert not os.path.isfile(filename)

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("err_type", [None, ValueError])
    def test_errors_process_completed(self, err_type,
                                      # fixtures:
                                      pytestdir, db4process, yamlfile):
        '''test processing in case of non 'critical' errors i.e., which do not prevent the process
          to be completed. None means we do not override segment_select which, with the current
          templates, causes no segment to be selected'''
        pyfile = self.pyfile

        # REMEMBER THAT BY DEFAULT LEAVING THE segment_select IMPLEMENTED in conffile
        # WE WOULD HAVE NO SEGMENTS, as maxgap_numsamples is None for all segments of this test
        # Thus provide config overrides:
        if err_type is not None:
            yaml_file = yamlfile(segment_select={'has_data': 'true'})
        else:
            yaml_file = yamlfile()

        runner = CliRunner()
        # Now wrtite pyfile into a named temp file, BUT DO NOT SUPPLY EXTENSION
        # This seems to fail in python3 (FIXME: python2?)
        filename = pytestdir.newfile('.csv')
        pyfile2 = pytestdir.newfile('.py')

        with open(pyfile, 'r') as opn:
            content = opn.read()

        if err_type == ValueError:
            # create the exception. Implement a bad signature whci hraises a TypeError
            content = content.replace("def main(", """def main2(segment, config):
    return int('4d')

def main(""")
        else:
            # rename main to main2, as we will call 'main2' as funcname in 'runner.invoke' below
            # REMEMBER THAT THIS CASE HAS ACTUALLY NO SEGMENTS TO BE PROCESSED, see
            # 'yamlfile' fixture above
            content = content.replace("def main(", """def main2(""")

        with open(pyfile2, 'wb') as _opn:
            _opn.write(content.encode('utf8'))

        result = runner.invoke(cli, ['process', '--dburl', db4process.dburl, '--no-prompt',
                                     '-p', pyfile2, '-f', "main2",
                                     '-c', yaml_file,
                                     filename])

        assert not result.exception
        assert result.exit_code == 0
        stdout = result.output
        # these cases raise BEFORE running pyfile
        # assert log config has not been called: (see self.init):
        assert self._logfilename is not None
        # we did open the output file:
        assert os.path.isfile(filename)
        # and we never wrote on it:
        assert os.stat(filename).st_size == 0
        # check correct outputs, in both log and output:
        logfilecontent = self.logfilecontent
        if err_type is None:  # no segments processed
            # we want to check that a particular string (str2check) is in the stdout
            # However, str2check newlines count is not constant through
            # libraries and python versions. It might be due to click progressbar not showing on
            # eclipse. Therefore, assert a regex, where we relax the condition on newlines (\n+)
            str2check = \
                (r"0 segment\(s\) found to process\n"
                 r"\n+"
                 r"0 of 0 segment\(s\) successfully processed\n"
                 r"0 of 0 segment\(s\) skipped with error message \(check log or details\)")
            assert re.search(str2check, stdout)
            assert re.search(str2check, logfilecontent)
        else:
            # we want to check that a particular string (str2check) is in the stdout
            # However, str2check newlines count is not constant through
            # libraries and python versions. It might be due to click progressbar not showing on
            # eclipse. Therefore, assert a regex, where we relax the condition on newlines (\n+)
            str2check = \
                (r'4 segment\(s\) found to process\n'
                 r'\n+'
                 r'0 of 4 segment\(s\) successfully processed\n'
                 r'4 of 4 segment\(s\) skipped with error message \(check log or details\)')
            assert re.search(str2check, stdout)

            # logfile has also the messages of what was wrong. Note that
            # py2 prints:
            # "invalid literal for long() with base 10: '4d'"
            # and PY3 prints:
            # ""invalid literal for int() with base 10: '4d'"
            # instead of writing:
            # if PY2:
            #     assert "invalid literal for long() with base 10: '4d'" in logfilecontent
            # else:
            #     assert "invalid literal for int() with base 10: '4d'" in logfilecontent
            # let's be more relaxed (use .*). Also, use a regexp for cross-versions
            # compatibility about newlines (see comments above)
            str2check = \
                (r"4 segment\(s\) found to process\n"
                 r"\n+"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
                 r"\n+"
                 r"0 of 4 segment\(s\) successfully processed\n"
                 r"4 of 4 segment\(s\) skipped with error message \(check log or details\)")
            assert re.search(str2check, logfilecontent)
