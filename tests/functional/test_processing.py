'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

# from builtins import str, object

from past.utils import old_div
import os, sys
from datetime import datetime, timedelta
import mock
from mock import patch
from future.backports.urllib.error import URLError
import pytest
import pandas as pd
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.utils.inputargs import yaml_load as orig_yaml_load
from stream2segment.utils.resources import get_templates_fpaths
from stream2segment.process.utils import get_inventory_url, save_inventory as original_saveinv
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.process.main import run as process_main_run, query4process
# from future import standard_library
from stream2segment.process.utils import enhancesegmentclass
import re
from stream2segment.process.writers import BaseWriter
from pandas.errors import EmptyDataError
from future.utils import PY2
# standard_library.install_aliases()


def yaml_load_side_effect(**overrides):
    """Side effect for the function reading the yaml config which enables the input
    of parameters to be overridden just after reading and before any other operation"""
    if overrides:
        def func(*a, **v):
            ret = orig_yaml_load(*a, **v)
            ret.update(overrides)  # note: this OVERRIDES nested dicts
            # whereas passing coverrides as second argument of orig_yaml_load MERGES their keys
            # with existing one
            return ret
        return func
    return orig_yaml_load


def readcsv(filename, header=True):
    return pd.read_csv(filename, header=None) if not header else pd.read_csv(filename)

class Test(object):

    pyfile, conffile = get_templates_fpaths("paramtable.py", "paramtable.yaml")

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data, pytestdir):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=True)

        # init db:
        session = db.session

        # setup a run_id:
        r = Download()
        session.add(r)
        session.commit()
        self.run = r

        ws = WebService(id=1, url='eventws')
        session.add(ws)
        session.commit()
        self.ws = ws
        # setup an event:
        e1 = Event(id=1, webservice_id=ws.id, event_id='abc1', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e2 = Event(id=2, webservice_id=ws.id, event_id='abc2', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e3 = Event(id=3, webservice_id=ws.id, event_id='abc3', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e4 = Event(id=4, webservice_id=ws.id, event_id='abc4', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e5 = Event(id=5, webservice_id=ws.id, event_id='abc5', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        session.add_all([e1, e2, e3, e4, e5])
        session.commit()
        self.evt1, self.evt2, self.evt3, self.evt4, self.evt5 = e1, e2, e3, e4, e5

        d = DataCenter(station_url='asd', dataselect_url='sdft')
        session.add(d)
        session.commit()
        self.dc = d

        # s_ok stations have lat and lon > 11, other stations do not
        s_ok = Station(datacenter_id=d.id, latitude=11, longitude=12, network='ok', station='ok',
                       start_time=datetime.utcnow())
        session.add(s_ok)
        session.commit()
        self.sta_ok = s_ok

        s_err = Station(datacenter_id=d.id, latitude=-21, longitude=5, network='err', station='err',
                        start_time=datetime.utcnow())
        session.add(s_err)
        session.commit()
        self.sta_err = s_err

        s_none = Station(datacenter_id=d.id, latitude=-31, longitude=-32, network='none',
                         station='none', start_time=datetime.utcnow())
        session.add(s_none)
        session.commit()
        self.sta_none = s_none

        c_ok = Channel(station_id=s_ok.id, location='ok', channel="ok", sample_rate=56.7)
        session.add(c_ok)
        session.commit()
        self.cha_ok = c_ok

        c_err = Channel(station_id=s_err.id, location='err', channel="err", sample_rate=56.7)
        session.add(c_err)
        session.commit()
        self.cha_err = c_err

        c_none = Channel(station_id=s_none.id, location='none', channel="none", sample_rate=56.7)
        session.add(c_none)
        session.commit()
        self.cha_none = c_none

        atts = data.to_segment_dict('trace_GE.APE.mseed')

        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e1.id, download_id=r.id,
                      event_distance_deg=35, **atts)

        # this segment should have inventory returning an exception (see url_read above)
        sg2 = Segment(channel_id=c_err.id, datacenter_id=d.id, event_id=e2.id, download_id=r.id,
                      event_distance_deg=45, **atts)
        # segment with gaps
        atts = data.to_segment_dict('IA.BAKI..BHZ.D.2016.004.head')
        sg3 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e3.id, download_id=r.id,
                      event_distance_deg=55, **atts)

        # build two segments without data:
        # empty segment
        atts['data'] = b''
        atts['request_start'] += timedelta(seconds=1)  # avoid unique constraint
        sg4 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e4.id, download_id=r.id,
                      event_distance_deg=45, **atts)

        # null segment
        atts['data'] = None
        atts['request_start'] += timedelta(seconds=2)  # avoid unique constraint
        sg5 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e5.id, download_id=r.id,
                      event_distance_deg=45, **atts)

        session.add_all([sg1, sg2, sg3, sg4, sg5])
        session.commit()
        self.seg1 = sg1
        self.seg2 = sg2
        self.seg_gaps = sg2
        self.seg_empty = sg3
        self.seg_none = sg4


        # mock get inventory:
        def url_read(*a, **v):
            '''mock urlread for inventories. Checks in the url (first arg if there is the 'err',
            'ok' or none' substring and returns appropriated data'''
            if "=err" in a[0]:
                raise URLError('error')
            elif "=none" in a[0]:
                return None, 500, 'Server error'
            else:
                return data.read("inventory_GE.APE.xml"), 200, 'Ok'

        with patch('stream2segment.process.utils.urlread', side_effect=url_read) as mock1:
            self.mock_url_read = mock1
            with patch('stream2segment.utils.inputargs.get_session', return_value=session):
                with patch('stream2segment.main.closesession',
                           side_effect=lambda *a, **v: None):

                    self._logfilename = None
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

    @property
    def logfilecontent(self):
        assert os.path.isfile(self._logfilename)
        with open(self._logfilename) as opn:
            return opn.read()

# ## ======== ACTUAL TESTS: ================================

    @mock.patch('stream2segment.process.utils.save_inventory', side_effect=original_saveinv)
    def test_segwrapper(self, mock_saveinv, db, data):

        segids = query4process(db.session, {}).all()
        prev_staid = None

        assert not hasattr(Segment, "_config")  # assert we are not in enhanced Segment "mode"
        with enhancesegmentclass():
            for saveinv in [True, False]:
                prev_staid = None
                assert hasattr(Segment, "_config")  # assert we are still in the with above
                # we could avoid the with below but we want to test overwrite:
                with enhancesegmentclass({'save_inventory': saveinv}, overwrite_config=True):
                    for (segid, staid) in segids:
                        # mock_get_inventory.reset_mock()
                        # staid = db.session.query(Segment).filter(Segment.id == segid).one().station.id
                        assert prev_staid is None or staid >= prev_staid
                        staequal = prev_staid is not None and staid == prev_staid
                        prev_staid = staid
                        segment = db.session.query(Segment).filter(Segment.id == segid).first()

                        mock_saveinv.reset_mock()
                        sta_url = get_inventory_url(segment.station)
                        if "=err" in sta_url or "=none" in sta_url:
                            with pytest.raises(Exception):  # all inventories are None
                                segment.inventory()
                            assert not mock_saveinv.called
                        else:
                            segment.inventory()
                            if staequal:
                                assert not mock_saveinv.called
                            else:
                                assert mock_saveinv.called == saveinv
                            assert len(segment.station.inventory_xml) > 0
                        segs = segment.siblings().all()
                        # as channel's channel is either 'ok' or 'err' we should never have
                        # other components
                        assert len(segs) == 0

        assert not hasattr(Segment, "_config")  # assert we are not in enhanced Segment "mode"

        # NOW TEST OTHER ORIENTATION PROPERLY. WE NEED TO ADD WELL FORMED SEGMENTS WITH CHANNELS
        # WHOSE ORIENTATION CAN BE DERIVED:
        staid = db.session.query(Station.id).first()[0]
        dcid = db.session.query(DataCenter.id).first()[0]
        eid = db.session.query(Event.id).first()[0]
        dwid = db.session.query(Download.id).first()[0]
        # add channels
        c1 = Channel(station_id=staid, location='ok', channel="AB1", sample_rate=56.7)
        c2 = Channel(station_id=staid, location='ok', channel="AB2", sample_rate=56.7)
        c3 = Channel(station_id=staid, location='ok', channel="AB3", sample_rate=56.7)
        db.session.add_all([c1, c2, c3])
        db.session.commit()
        # add segments. Create attributes (although not strictly necessary to have bytes data)
        atts = data.to_segment_dict('trace_GE.APE.mseed')
        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c1.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        sg2 = Segment(channel_id=c2.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        sg3 = Segment(channel_id=c3.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        db.session.add_all([sg1, sg2, sg3])
        db.session.commit()
        # start testing:
        segids = query4process(db.session, {}).all()

        with enhancesegmentclass():
            for (segid, staid) in segids:
                segment = db.session.query(Segment).filter(Segment.id == segid).first()
                segs = segment.siblings()
                if segs.all():
                    assert segment.id in (sg1.id, sg2.id, sg3.id)
                    assert len(segs.all()) == 2

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Save inventory and check it's saved
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run, mock_yaml_load, db):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.seg1.id)
        station_id_whose_inventory_is_saved = self.sta_ok.id

        runner = CliRunner()

        pyfile, conffile = self.pyfile, self.conffile
        result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                               '-p', pyfile, '-c', conffile])

        assert not result.exception

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        # assert third argument (`ondone` callback) is None 'ondone' or is a BaseWriter (no-op)
        # class:
        assert args[2] is None or type(args[2]) == BaseWriter
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', result.output)

        # Note that apparently CliRunner() puts stderr and stdout together
        # (https://github.com/pallets/click/pull/868)
        # So we should test that we have these string twice:
        for subs in ["Processing function: ", "Config. file: "]:
            idx = result.output.find(subs)
            assert idx > -1

        # these assertion are just copied from the test below and left here cause they
        # should still hold (db behaviour does not change of we provide output file or not):

        # save_downloaded_inventory True, test that we did save any:
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = db.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert db.session.query(Station).\
            filter(Station.id == station_id_whose_inventory_is_saved).first().inventory_xml

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Save inventory and check it's saved
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv(self, mock_yaml_load, advanced_settings, cmdline_opts,
                                        pytestdir, db):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': True,
                                 'snr_threshold': 0,
                                 'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = self.seg1.id
        station_id_whose_inventory_is_saved = self.sta_ok.id

        runner = CliRunner()
        # test with a temporary file, i.e. a file which is created BEFORE, and supply --no-prompt
        # test then that we print the message "overridden the file..." in log output
        outfile = pytestdir.newfile('output.csv', create=True)
        pyfile, conffile = self.pyfile, self.conffile
        result = runner.invoke(cli, ['process', '--dburl', db.dburl, '--no-prompt',
                               '-p', pyfile, '-c', conffile, outfile] + cmdline_opts)

        assert not result.exception
        # check file has been correctly written:
        csv1 = readcsv(outfile)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext = self.logfilecontent
        assert """Overwriting existing output file
3 segment(s) found to process

segment (id=3): 4 traces (probably gaps/overlaps)
segment (id=2): Station inventory (xml) error: <urlopen error error>

station inventories saved: 1
1 of 3 segment(s) successfully processed
2 of 3 segment(s) skipped with error message (check log or details)""" in logtext
        # assert logfile exists:
        assert os.path.isfile(self._logfilename)

        # save_downloaded_inventory True, test that we did save any:
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = db.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert db.session.query(Station).\
            filter(Station.id == station_id_whose_inventory_is_saved).first().inventory_xml

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Save inventory and check it's saved
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv_complex_select(self, mock_yaml_load,
                                                       advanced_settings,
                                                       cmdline_opts,
                                                       # fixtures:
                                                       pytestdir,
                                                       db):
        '''test a case where we have a more complex select involving joins'''
        # When we use our exprequery, we might join already joined tables.
        # previously, we had a
        # sqlalchemy warning in the log. But this is NOT ANYMORE THE CASE as now we
        # join only un-joined tables: it turned out that joining already joined tables
        # issued warnings in some cases, and errors in some other cases.
        # TEST HERE THAT WE DO NOT HAVE SUCH ERRORS
        # FIXME: we should investigate why. One possible explanation is that if the joined
        # table is in the query sql-alchemy issues a warning:
        # query(Segment,Station).join(Segment.station).join(Segment.station)
        # whereas if the joined table is not in the query, sql-alchemy issues an error:
        # query(Segment,Station).join(Segment.event).join(Segment.event)
        # an already added join is not added twice. as we realised that
        # joins added multiple times where OK

        # select the event times for the segments with data:
        etimes = sorted(_[1] for _ in db.session.query(Segment.id, Event.time).\
                            join(Segment.event).filter(Segment.has_data))

        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {'has_data': 'true',
                                               'event.time': '<=%s' % (max(etimes).isoformat())}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        # the selection above should be the same as the previous test:
        # test_simple_run_retDict_saveinv,
        # as segment_select[event.time] includes all segments in segment_select['has_data'],
        # thus the code is left as it was in the method above
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = self.seg1.id
        station_id_whose_inventory_is_saved = self.sta_ok.id

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        pyfile, conffile = self.pyfile, self.conffile
        result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                               '-p', pyfile, '-c', conffile, filename] + cmdline_opts)

        assert not result.exception
        # check file has been correctly written:
        csv1 = readcsv(filename)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext = self.logfilecontent
        assert """3 segment(s) found to process

segment (id=3): 4 traces (probably gaps/overlaps)
segment (id=2): Station inventory (xml) error: <urlopen error error>

station inventories saved: 1
1 of 3 segment(s) successfully processed
2 of 3 segment(s) skipped with error message (check log or details)""" in logtext

        # save_downloaded_inventory True, test that we did save any:
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = db.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert db.session.query(Station).\
            filter(Station.id == station_id_whose_inventory_is_saved).first().inventory_xml

    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv_high_snr_threshold(self, mock_yaml_load,
                                                           # fixtures:
                                                           pytestdir,
                                                           db):
        '''same as `test_simple_run_retDict_saveinv` above
        but with a very high snr threshold => no rows processed'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 3,  # 3 is high enough to discard the only segment we would process otherwise
                            'segment_select': {'has_data': 'true'}}
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = self.seg1.id
        station_id_whose_inventory_is_saved = self.sta_ok.id

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        pyfile, conffile = self.pyfile, self.conffile
        result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                               '-p', pyfile, '-c', conffile, filename])

        assert not result.exception
        # check file has been correctly written:
        with pytest.raises(EmptyDataError):
            csv1 = readcsv(filename)
        # assert len(csv1) == 0
        # assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext = self.logfilecontent
        assert """3 segment(s) found to process

segment (id=1): low snr 1.350154
segment (id=3): 4 traces (probably gaps/overlaps)
segment (id=2): Station inventory (xml) error: <urlopen error error>

station inventories saved: 1
0 of 3 segment(s) successfully processed
3 of 3 segment(s) skipped with error message (check log or details)""" in logtext

        # save_downloaded_inventory True, test that we did save any:
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = db.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert db.session.query(Station).\
            filter(Station.id == station_id_whose_inventory_is_saved).first().inventory_xml

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Don't save inventory and check it's
    # not saved
    @pytest.mark.parametrize("seg_chunk", [None, 1])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_dontsaveinv(self, mock_yaml_load, seg_chunk,
                                            # fixtures:
                                            pytestdir,
                                            db):
        '''same as `test_simple_run_retDict_saveinv` above
         but with a 0 snr threshold and do not save inventories'''

        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': False,
                            'snr_threshold': 0,  # don't skip any segment in processing
                            'segment_select': {'has_data': 'true'}}
        if seg_chunk is not None:
            config_overrides['advanced_settings'] = {'segments_chunk': seg_chunk}
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = self.seg1.id

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        pyfile, conffile = self.pyfile, self.conffile
        result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                               '-p', pyfile, '-c', conffile, filename])

        assert not result.exception
        # check file has been correctly written:
        csv1 = readcsv(filename)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext = self.logfilecontent
        # Note below: no 'station inventories saved' message:
        assert """3 segment(s) found to process

segment (id=3): 4 traces (probably gaps/overlaps)
segment (id=2): Station inventory (xml) error: <urlopen error error>

1 of 3 segment(s) successfully processed
2 of 3 segment(s) skipped with error message (check log or details)""" in logtext

        # save_downloaded_inventory False, test that we did not save any:
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) == 0

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing NO file. We implement a filter that excludes the only
    # processed file using associated stations lat and lon.
    @pytest.mark.parametrize('select_with_data', [True, False])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_seg_select_empty_and_err_segments(self, mock_yaml_load,
                                                                 select_with_data,
                                                                 # fixtures:
                                                                 pytestdir,
                                                                 db):
        '''test a segment selection that takes only non-processable segments'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': False,
                            'snr_threshold': 0,  # take all segments
                            'segment_select': {'station.latitude': '<10',
                                               'station.longitude': '<10'}}
        if select_with_data:
            config_overrides['segment_select']['has_data'] = 'true'
        # Note on segment_select above:
        # s_ok stations have lat and lon > 11, other stations do not
        # now we want to set a filter which gets us only the segments from stations not ok.
        # Note: has_data is not specified so we will get 3 segments (2 with data None, 1 with
        # data which raises errors for station inventory)
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = self.seg1.id

        runner = CliRunner()
        filename = pytestdir.newfile('.csv')
        pyfile, conffile = self.pyfile, self.conffile

        result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                                     '-p', pyfile,
                                     '-c', conffile,
                                     filename])

        assert not result.exception
        # check file has been correctly written:
        with pytest.raises(EmptyDataError):
            csv1 = readcsv(filename)
#             assert len(csv1) == 1
#             assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext = self.logfilecontent
        # Note below: no 'station inventories saved' message in any log:
        if select_with_data:
            assert """1 segment(s) found to process

segment (id=2): Station inventory (xml) error: <urlopen error error>

0 of 1 segment(s) successfully processed
1 of 1 segment(s) skipped with error message (check log or details)""" in logtext
        else:
            assert """3 segment(s) found to process

segment (id=2): Station inventory (xml) error: <urlopen error error>
segment (id=4): MiniSeed error: no data
segment (id=5): MiniSeed error: no data

0 of 3 segment(s) successfully processed
3 of 3 segment(s) skipped with error message (check log or details)""" in logtext

        # ===================================================================
        # NOW WE CAN CHECK IF THE URLREAD HAS BEEN CALLED ONCE.
        # Out of the three segments to process, two don't have data thus we do not reach
        # the inventory() method. It remains only the third one, whcih downloads
        # the inventory and calls mock_url_read:
        # ===================================================================
        assert self.mock_url_read.call_count == 1

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning list. Just check it works
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_ret_list(self, mock_yaml_load, advanced_settings, cmdline_opts,
                                 # fixtures:
                                 pytestdir,
                                 db):
        '''test processing returning list, and also when we specify a different main function'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': False,
                            'snr_threshold': 0,  # take all segments
                            'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        runner = CliRunner()
        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = self.seg1.id

        pyfile, conffile = self.pyfile, self.conffile

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

        result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                                     '-p', pyfile2, '-f', "main_retlist",
                                     '-c', conffile,
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
        logtext = self.logfilecontent
        assert """3 segment(s) found to process

segment (id=3): 4 traces (probably gaps/overlaps)
segment (id=2): Station inventory (xml) error: <urlopen error error>

1 of 3 segment(s) successfully processed
2 of 3 segment(s) skipped with error message (check log or details)""" in logtext
        # assert logfile exists:
        assert os.path.isfile(self._logfilename)

    @pytest.mark.parametrize("cmdline_opts",
                             [[], ['--multi-process'], ['--multi-process', '--num-processes', '1']])
    @pytest.mark.parametrize("err_type, expects_log_2_be_configured",
                             [(None, False),
                              (ImportError, False),
                              (AttributeError, True),
                              (TypeError, True)])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_errors_process_not_run(self, mock_yaml_load,
                                           err_type, expects_log_2_be_configured,
                                           cmdline_opts,
                                           # fixtures:
                                           pytestdir,
                                           db):
        '''test processing in case of severla 'critical' errors (which do not launch the process
          None means simply a bad argument (funcname missing)'''
        pyfile, conffile = self.pyfile, self.conffile

        # REMEMBER THAT BY DEFAULT LEAVING THE segment_select IMPLEMENTED in conffile
        # WE WOULD HAVE NO SEGMENTS, as maxgap_numsamples is None for all segments of this test
        # Thus provide config overrides:
        mock_yaml_load.side_effect = yaml_load_side_effect(segment_select={'has_data': 'true'})

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

        result = runner.invoke(cli, ['process', '--dburl', db.dburl, '--no-prompt',
                                     '-p', pyfile2, '-f', "main2",
                                     '-c', conffile,
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
            assert self._logfilename is None
            assert 'Invalid value for "pyfile": ' in stdout
            further_string = 'main2' if err_type is None else 'No module named'
            assert further_string in stdout
            # we did NOt open the output file:
            assert not os.path.isfile(filename)

    @pytest.mark.parametrize("err_type", [None, ValueError])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_errors_process_completed(self, mock_yaml_load, err_type,
                                      # fixtures:
                                      pytestdir,
                                      db):
        '''test processing in case of non 'critical' errors i.e., which do not prevent the process
          to be completed. None means we do not override segment_select which, with the current
          templates, causes no segment to be selected'''
        pyfile, conffile = self.pyfile, self.conffile

        # REMEMBER THAT BY DEFAULT LEAVING THE segment_select IMPLEMENTED in conffile
        # WE WOULD HAVE NO SEGMENTS, as maxgap_numsamples is None for all segments of this test
        # Thus provide config overrides:
        if err_type is not None:
            mock_yaml_load.side_effect = yaml_load_side_effect(segment_select={'has_data': 'true'})
        else:
            mock_yaml_load.side_effect = yaml_load_side_effect()

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
            # 'mock_yaml_load.side_effect' above
            content = content.replace("def main(", """def main2(""")

        with open(pyfile2, 'wb') as _opn:
            _opn.write(content.encode('utf8'))

        result = runner.invoke(cli, ['process', '--dburl', db.dburl, '--no-prompt',
                                     '-p', pyfile2, '-f', "main2",
                                     '-c', conffile,
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
            str2check = """0 segment(s) found to process


0 of 0 segment(s) successfully processed
0 of 0 segment(s) skipped with error message (check log or details)"""
            assert str2check in stdout
            assert str2check in logfilecontent
        else:
            str2check = """3 segment(s) found to process



0 of 3 segment(s) successfully processed
3 of 3 segment(s) skipped with error message (check log or details)"""
            assert str2check in stdout
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
            # let's be more relaxed:
            assert "invalid literal for " in logfilecontent
            assert "with base 10: '4d'" in logfilecontent
            # now also assert that any line of str2check not empty is in logfilecontent:
            assert all(l in logfilecontent for l in str2check.splitlines() if l.strip())
