'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

from tempfile import NamedTemporaryFile
from past.utils import old_div
import os, sys
from datetime import datetime, timedelta
import mock
from mock import patch
import tempfile
import csv
from future.backports.urllib.error import URLError
import pytest

from click.testing import CliRunner

from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
# from urllib.error import URLError
# import multiprocessing
from obspy.core.stream import read

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.utils.inputargs import yaml_load as load_proc_cfg
from stream2segment import process
from stream2segment.utils.resources import get_templates_fpaths
from stream2segment.process.utils import get_inventory_url, save_inventory as original_saveinv
from stream2segment.process.core import query4process

from stream2segment.process.core import run as process_core_run
# from future import standard_library
from stream2segment.process.utils import enhancesegmentclass
# standard_library.install_aliases()


class DB(object):
    def __init__(self):
        self.dburi = os.getenv("DB_URL", "sqlite:///:memory:")
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        self.engine = create_engine(self.dburi)
        # Base.metadata.drop_all(cls.engine)
        Base.metadata.create_all(self.engine)  # @UndefinedVariable
        # create a configured "Session" class

    def create(self):
        Session = sessionmaker(bind=self.engine)
        # create a Session
        self.session = Session()

        # setup a run_id:
        r = Download()
        self.session.add(r)
        self.session.commit()
        self.run = r

        ws = WebService(id=1, url='eventws')
        self.session.add(ws)
        self.session.commit()
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
        self.session.add_all([e1, e2, e3, e4, e5])
        self.session.commit()
        self.evt1, self.evt2, self.evt3, self.evt4, self.evt5 = e1, e2, e3, e4, e5

        d = DataCenter(station_url='asd', dataselect_url='sdft')
        self.session.add(d)
        self.session.commit()
        self.dc = d

        # s_ok stations have lat and lon > 11, other stations do not
        s_ok = Station(datacenter_id=d.id, latitude=11, longitude=12, network='ok', station='ok',
                       start_time=datetime.utcnow())
        self.session.add(s_ok)
        self.session.commit()
        self.sta_ok = s_ok

        s_err = Station(datacenter_id=d.id, latitude=-21, longitude=5, network='err', station='err',
                        start_time=datetime.utcnow())
        self.session.add(s_err)
        self.session.commit()
        self.sta_err = s_err

        s_none = Station(datacenter_id=d.id, latitude=-31, longitude=-32, network='none',
                         station='none', start_time=datetime.utcnow())
        self.session.add(s_none)
        self.session.commit()
        self.sta_none = s_none

        c_ok = Channel(station_id=s_ok.id, location='ok', channel="ok", sample_rate=56.7)
        self.session.add(c_ok)
        self.session.commit()
        self.cha_ok = c_ok

        c_err = Channel(station_id=s_err.id, location='err', channel="err", sample_rate=56.7)
        self.session.add(c_err)
        self.session.commit()
        self.cha_err = c_err

        c_none = Channel(station_id=s_none.id, location='none', channel="none", sample_rate=56.7)
        self.session.add(c_none)
        self.session.commit()
        self.cha_none = c_none

        data = Test.read_stream_raw('trace_GE.APE.mseed')

        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e1.id, download_id=r.id,
                      event_distance_deg=35, **data)

        # this segment should have inventory returning an exception (see url_read above)
        sg2 = Segment(channel_id=c_err.id, datacenter_id=d.id, event_id=e2.id, download_id=r.id,
                      event_distance_deg=45, **data)
        # segment with gaps
        data = Test.read_stream_raw('IA.BAKI..BHZ.D.2016.004.head')
        sg3 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e3.id, download_id=r.id,
                      event_distance_deg=55, **data)

        # build two segments without data:
        # empty segment
        data['data'] = b''
        data['request_start'] += timedelta(seconds=1)  # avoid unique constraint
        sg4 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e4.id, download_id=r.id,
                      event_distance_deg=45, **data)

        # null segment
        data['data'] = None
        data['request_start'] += timedelta(seconds=2)  # avoid unique constraint
        sg5 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e5.id, download_id=r.id,
                      event_distance_deg=45, **data)

        self.session.add_all([sg1, sg2, sg3, sg4, sg5])
        self.session.commit()
        self.seg1 = sg1
        self.seg2 = sg2
        self.seg_gaps = sg2
        self.seg_empty = sg3
        self.seg_none = sg4

    def close(self):
        if self.engine:
            if self.session:
                try:
                    self.session.rollback()
                    self.session.close()
                except:
                    pass
            try:
                Base.metadata.drop_all(self.engine)  # @UndefinedVariable
            except:
                pass
#        self.session.close()
#         self.patcher1.stop()
#         self.patcher2.stop()


class Test(unittest.TestCase):

    dburi = ""
    file = None

    @staticmethod
    def cleanup(self):
        db = getattr(self, 'db', None)
        if db:
            db.close()

        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @property
    def is_sqlite(self):
        return str(self.db.engine.url).startswith("sqlite:///")

    @property
    def is_postgres(self):
        return str(self.db.engine.url).startswith("postgresql://")

    def setUp(self):
        # add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self)

        # values to override the config, if specified:
        self.config_overrides = {}
        self.inventory = True

        self.db = DB()
        self.db.create()
        self.session = self.db.session
        self.dburi = self.db.dburi

        self.patchers = []
        # mock get inventory:
        self.patchers.append(patch('stream2segment.process.utils.urlread'))
        self.mock_url_read = self.patchers[-1].start()
        self.mock_url_read.side_effect = self.url_read

        self.patchers.append(patch('stream2segment.utils.inputargs.get_session'))
        self.mock_session = self.patchers[-1].start()
        self.mock_session.return_value = self.session

        self.patchers.append(patch('stream2segment.main.closesession'))
        self.mock_closing = self.patchers[-1].start()
        self.mock_closing.side_effect = lambda *a, **v: None

    @staticmethod
    def read_stream_raw(file_name):
        '''returns a dict to be passed as argument for creating new Segment(s), by reading
        an existing miniseed'''
        stream = read(Test.get_file(file_name))

        start_time = stream[0].stats.starttime
        end_time = stream[0].stats.endtime

        # set arrival time to one third duration
        return dict(data=Test.read_data_raw(file_name),
                    arrival_time=(start_time + old_div((end_time - start_time), 3)).datetime,
                    request_start=start_time.datetime,
                    request_end=end_time.datetime,
                    sample_rate=stream[0].stats.sampling_rate)

    @staticmethod
    def read_data_raw(file_name):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        with open(os.path.join(folder, file_name), 'rb') as opn:
            return opn.read()

    @staticmethod
    def read_and_remove(filepath):
        assert os.path.isfile(filepath)
        with open(filepath) as opn:
            sss = opn.read()
        os.remove(filepath)
        return sss

    @staticmethod
    def get_file(filename):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        path = os.path.abspath(os.path.join(folder, filename))
        assert os.path.isfile(path)
        return path

    @staticmethod
    def get_processing_files():
        pyfile, conffile = get_templates_fpaths("processing.py", "processing.yaml") #pylint: disable=unbalanced-tuple-unpacking
        return pyfile, conffile

    @staticmethod
    def url_read(*a, **v):
        '''mock urlread for inventories. Checks in the url (first arg if there is the 'err', 'ok'
        or none' substring and returns appropriated data'''
        if "=err" in a[0]:
            raise URLError('error')
        elif "=none" in a[0]:
            return None, 500, 'Server error'
        else:
            return Test.read_data_raw("inventory_GE.APE.xml"), 200, 'Ok'

    def load_proc_cfg(self, *a, **kw):
        """called by mocked read config: updates the parsed dict with the custom config"""
        cfg = load_proc_cfg(*a, **kw)
        cfg.update(self.config_overrides)
        return cfg

# ## ======== ACTUAL TESTS: ================================

    @mock.patch('stream2segment.process.utils.save_inventory', side_effect=original_saveinv)
    def test_segwrapper(self, mock_saveinv):

        for val in query4process(self.session, {}):
            pass

        segids = query4process(self.session, {}).all()
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
                        # staid = self.session.query(Segment).filter(Segment.id == segid).one().station.id
                        assert prev_staid is None or staid >= prev_staid
                        staequal = prev_staid is not None and staid == prev_staid
                        prev_staid = staid
                        segment = self.session.query(Segment).filter(Segment.id == segid).first()

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
        staid = self.session.query(Station.id).first()[0]
        dcid = self.session.query(DataCenter.id).first()[0]
        eid = self.session.query(Event.id).first()[0]
        dwid = self.session.query(Download.id).first()[0]
        # add channels
        c1 = Channel(station_id=staid, location='ok', channel="AB1", sample_rate=56.7)
        c2 = Channel(station_id=staid, location='ok', channel="AB2", sample_rate=56.7)
        c3 = Channel(station_id=staid, location='ok', channel="AB3", sample_rate=56.7)
        self.session.add_all([c1, c2, c3])
        self.session.commit()
        # add segments. Create attributes (although not strictly necessary to have bytes data)
        data = Test.read_stream_raw('trace_GE.APE.mseed')
        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c1.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **data)
        sg2 = Segment(channel_id=c2.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **data)
        sg3 = Segment(channel_id=c3.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **data)
        self.session.add_all([sg1, sg2, sg3])
        self.session.commit()
        # start testing:
        segids = query4process(self.session, {}).all()

        with enhancesegmentclass():
            for (segid, staid) in segids:
                segment = self.session.query(Segment).filter(Segment.id == segid).first()
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
    @mock.patch('stream2segment.process.main.process_core_run', side_effect=process_core_run)
    def test_simple_run_no_outfile_provided(self, mock_run, mock_load_cfg):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': True,
                                 'snr_threshold': 0,
                                 'segment_select': {'has_data': 'true'}}
        mock_load_cfg.side_effect = self.load_proc_cfg

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)
        station_id_whose_inventory_is_saved = self.db.sta_ok.id

        # need to reset this global variable: FIXME: better handling?
        process.main._inventories = {}
        runner = CliRunner()
        
        pyfile, conffile = self.get_processing_files()
        result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                               '-p', pyfile, '-c', conffile])

        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        assert args[2] is None  # assert third argument (`ondone` callback) is None 'ondone' 
        assert "Output file:" not in result.output

        # Note that apparently CliRunner() puts stderr and stdout together 
        # (https://github.com/pallets/click/pull/868)
        # So we should test that we have these string twice:
        for subs in ["Executing 'main' in ", "Config. file: "]:
            idx = result.output.find(subs)
            assert idx > -1
            assert result.output.find(subs, idx+1) > idx
        
        
        
        # these assertion are just copied from the test below and left here cause they
        # should still hold (db behaviour does not change of we provide output file or not):
        
        # save_downloaded_inventory True, test that we did save any:
        assert len(self.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = self.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert self.session.query(Station).\
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
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv(self, mock_load_cfg):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': True,
                                 'snr_threshold': 0,
                                 'segment_select': {'has_data': 'true'}}
        mock_load_cfg.side_effect = self.load_proc_cfg

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)
        station_id_whose_inventory_is_saved = self.db.sta_ok.id

        # need to reset this global variable: FIXME: better handling?
        process.main._inventories = {}
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                   '-p', pyfile, '-c', conffile, file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False
                return

            # check file has been correctly written:
            with open(file.name, 'r') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == expected_first_row_seg_id
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 2
                logtext = self.read_and_remove(file.name+".log")
                assert len(logtext) > 0

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)

        # save_downloaded_inventory True, test that we did save any:
        assert len(self.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = self.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert self.session.query(Station).\
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
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv_complex_select(self, mock_load_cfg):
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
        etimes = sorted(_[1] for _ in self.session.query(Segment.id, Event.time).\
                            join(Segment.event).filter(Segment.has_data))

        self.config_overrides = {'save_inventory': True,
                                 'snr_threshold': 0,
                                 'segment_select': {'has_data': 'true',
                                                    'event.time': '<=%s' % (max(etimes).isoformat())}}
        # the selection above should be the same as the previous test:
        # test_simple_run_retDict_saveinv,
        # as segment_select[event.time] includes all segments in segment_select['has_data'],
        # thus the code is left as it was in the method above
        mock_load_cfg.side_effect = self.load_proc_cfg

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)
        station_id_whose_inventory_is_saved = self.db.sta_ok.id

        # need to reset this global variable: FIXME: better handling?
        process.main._inventories = {}
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                   '-p', pyfile, '-c', conffile, file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False
                return

            # check file has been correctly written:
            with open(file.name, 'r') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == expected_first_row_seg_id
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 2
                logtext = self.read_and_remove(file.name+".log")
                assert len(logtext) > 0

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)

        # save_downloaded_inventory True, test that we did save any:
        assert len(self.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = self.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert self.session.query(Station).\
            filter(Station.id == station_id_whose_inventory_is_saved).first().inventory_xml

    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv_high_snr_threshold(self, mock_load_cfg):
        '''same as `test_simple_run_retDict_saveinv` above
        but with a very high snr threshold'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': True,
                                 'snr_threshold': 3,  # 3 is high enough to discard the only segment we would process otherwise
                                 'segment_select': {'has_data': 'true'}}
        mock_load_cfg.side_effect = self.load_proc_cfg

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)
        station_id_whose_inventory_is_saved = self.db.sta_ok.id

        # need to reset this global variable: FIXME: better handling?
        process.main._inventories = {}
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                   '-p', pyfile, '-c', conffile, file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False
                return

            # check file has been correctly written:
            with open(file.name, 'r') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == expected_first_row_seg_id
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 0
                logtext = self.read_and_remove(file.name+".log")
                assert "low snr" in logtext

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)

        # save_downloaded_inventory True, test that we did save any:
        assert len(self.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = self.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert self.session.query(Station).\
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
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_dontsaveinv(self, mock_load_cfg):
        '''same as `test_simple_run_retDict_saveinv` above
         but with a 0 snr threshold and do not save inventories'''
        
        # test also segment chunks:
        for seg_chunk in (None, 1):
            # set values which will override the yaml config in templates folder:
            self.config_overrides = {'save_inventory': False,
                                     'snr_threshold': 0,  # don't skip any segment in processing
                                     'segment_select': {'has_data': 'true'}}
            if seg_chunk is not None:
                self.config_overrides['advanced_settings'] = {'segments_chunk': seg_chunk}
            mock_load_cfg.side_effect = self.load_proc_cfg

            # query data for testing now as the program will expunge all data from the session
            # and thus we want to avoid DetachedInstanceError(s):
            expected_first_row_seg_id = str(self.db.seg1.id)

            # need to reset this global variable: FIXME: better handling?
            process.main._inventories = {}
            runner = CliRunner()
            with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
                pyfile, conffile = self.get_processing_files()
                result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                       '-p', pyfile, '-c', conffile, file.name])

                if result.exception:
                    import traceback
                    traceback.print_exception(*result.exc_info)
                    print(result.output)
                    assert False
                    return

                # check file has been correctly written:
                with open(file.name, 'r') as csvfile:
                    spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                    rowz = 0
                    for row in spamreader:
                        rowz += 1
                        if rowz == 2:
                            assert row[0] == expected_first_row_seg_id
    #                         assert row[1] == self.db.seg1.start_time.isoformat()
    #                         assert row[2] == self.db.seg1.end_time.isoformat()
                    assert rowz == 2
                    logtext = self.read_and_remove(file.name+".log")
                    assert len(logtext) > 0
    
            # save_downloaded_inventory False, test that we did not save any:
            assert len(self.session.query(Station).filter(Station.has_inventory).all()) == 0

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
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_seg_select_empty_and_err_segments(self, mock_load_cfg):
        '''test that segment selection works'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': False,
                                 'snr_threshold': 0,  # take all segments
                                 'segment_select': {'station.latitude': '<10',
                                                    'station.longitude': '<10'}}
        # Note on segment_select above:
        # s_ok stations have lat and lon > 11, other stations do not
        # now we want to set a filter which gets us only the segments from stations not ok.
        # Note: has_data is not specified so we will get 3 segments (2 with data None, 1 with
        # data which raises errors for station inventory)
        mock_load_cfg.side_effect = self.load_proc_cfg

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()

            result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                         '-p', pyfile,
                                         '-c', conffile,
                                         file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False
                return

            # check file has been correctly written, we should have written two files
            with open(file.name, 'r') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == expected_first_row_seg_id
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 0
                logtext = self.read_and_remove(file.name+".log")

                # THE TEST BELOW IS USELESS AS WE DO NOT CAPTURENWARNINGS ANYMORE
                # as we have joined twice segment with stations (one is done by default, the other
                # has been set in custom_config['segment_select'] above), we should have a
                # sqlalchemy warning in the log. But this is NOT ANYMORE THE CASE as now we
                # join only un-joined tables: it turned out that joining already joined tables
                # issued warnings in some cases, and errors in some other cases.
                # FIXME: we should investigate why. One possible explanation is that if the joined
                # table is in the query sql-alchemy issues a warning:
                # query(Segment,Station).join(Segment.station).join(Segment.station)
                # whereas if the joined table is not in the query, sql-alchemy issues an error:
                # query(Segment,Station).join(Segment.event).join(Segment.event)
                # an already added join is not added twice. as we realised that
                # joins added multiple times where OK
                assert "SAWarning: Pathed join target" not in logtext

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
    # Here a simple test for a processing NO file. We implement a filter that excludes the only
    # processed file using associated stations lat and lon. 
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_seg_select_only_one_err_segment(self, mock_load_cfg):
        '''test that segment selection works (2)'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': False,
                                 'snr_threshold': 0,  # take all segments
                                 'segment_select': {'has_data': 'true',
                                                    'station.latitude': '<10',
                                                    'station.longitude': '<10'}}
        # Note on segment_select above: s_ok stations have lat and lon > 11, other stations do not
        # now we want to set a filter which gets us only the segments from stations not ok.
        # Note: withdata is True so we will get 1 segment (1 with data which raises
        # errors for station inventory)
        mock_load_cfg.side_effect = self.load_proc_cfg

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)

        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()

            result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                         '-p', pyfile,
                                         '-c', conffile,
                                         file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False
                return

            # check file has been correctly written, we should have written two files
            with open(file.name, 'r') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == expected_first_row_seg_id
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 0
                logtext = self.read_and_remove(file.name+".log")
                assert len(logtext) > 0
                # ===================================================================
                # NOW WE CAN CHECK IF THE URLREAD HAS BEEN CALLED ONCE AND NOT MORE:
                # out of three segmens, we called urlread
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
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_ret_list(self, mock_load_cfg):
        '''test processing returning list, and also when we specify a different main function'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': False,
                                 'snr_threshold': 0,  # take all segments
                                 'segment_select': {'has_data': 'true'}}
        mock_load_cfg.side_effect = self.load_proc_cfg

        runner = CliRunner()
        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.db.seg1.id)
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()

            # Now wrtite pyfile into a named temp file, with the method:
            # def main_retlist(segment, config):
            #    return main(segment, config).keys()
            # the method returns a list (which is what we want to test
            # and this way, we do not need to keep synchronized any additional file
            with tempfile.NamedTemporaryFile(suffix='.py') as pyfile2:  # @ReservedAssignment

                with open(pyfile, 'r') as opn:
                    content = opn.read()

                cont2 = content.replace("def main(segment, config):", """def main_retlist(segment, config):
    return main(segment, config).keys()
def main(segment, config):""")
                pyfile2.write(cont2.encode('utf8'))
                pyfile2.seek(0)

                result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                             '-p', pyfile2.name, '-f', "main_retlist",
                                             '-c', conffile,
                                             file.name])

                if result.exception:
                    import traceback
                    traceback.print_exception(*result.exc_info)
                    print(result.output)
                    assert False
                    return

                # check file has been correctly written:
                with open(file.name, 'r') as csvfile:
                    spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                    rowz = 0
                    for row in spamreader:
                        rowz += 1
                        if rowz == 1:
                            assert row[0] == expected_first_row_seg_id
    #                         assert row[1] == self.db.seg1.start_time.isoformat()
    #                         assert row[2] == self.db.seg1.end_time.isoformat()
                    assert rowz == 1
                    logtext = self.read_and_remove(file.name+".log")
                    assert len(logtext) > 0

    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_wrong_pyfile(self, mock_load_cfg):
        '''test processing when supplying a wrong python file (not py extension, this seem
        to raise when importing it in python3)'''
        # set values which will override the yaml config in templates folder:

        mock_load_cfg.side_effect = self.load_proc_cfg

        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()

            # Now wrtite pyfile into a named temp file, BUT DO NOT SUPPLY EXTENSION
            # This seems to fail in python3 (FIXME: python2?)
            with tempfile.NamedTemporaryFile() as pyfile2:  # @ReservedAssignment

                with open(pyfile, 'r') as opn:
                    content = opn.read()

                pyfile2.write(content.encode('utf8'))
                pyfile2.seek(0)

                result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                             '-p', pyfile2.name, '-f', "main_retlist",
                                             '-c', conffile,
                                             file.name])

                # we did not raise but printed to stdout. However, click apparently
                # makes result.exception being truthy
                assert result.exception
                assert 'Invalid value for "pyfile": ' in result.output
                assert result.exit_code != 0

                # check file has NOT be written:
                # Note that file should in principle not exist, but we opened here 
                # as temporary file
                with open(file.name, 'r') as csvfile:
                    spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                    rowz = sum(1 for _ in spamreader)
                    assert rowz == 0
                # however, log file has not been created:
                assert not os.path.isfile(file.name+".log")


    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_codeerror(self, mock_load_cfg):
        '''test processing type error(wrong argumens)'''
        # set values which will override the yaml config in templates folder:
        self.config_overrides = {'save_inventory': False,
                                 'snr_threshold': 0,  # take all segments
                                 'segment_select': {'has_data': 'true'}}
        mock_load_cfg.side_effect = self.load_proc_cfg

        runner = CliRunner()

        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            # pyfile = self.get_file("processing.py")  # custom one

            with NamedTemporaryFile(suffix='.py') as tmpfile:

                with open(pyfile) as opn:
                    content = opn.read()

                content = content.replace("def main(", """def main_typeerr(segment, config, wrong_argument):
    return [6]

def main(""")
                tmpfile.write(content.encode('utf8'))
                tmpfile.seek(0)

                result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                             '-p', tmpfile.name, '-f', "main_typeerr",
                                             '-c', conffile,
                                             file.name])

                # the file above are bad implementation (old one)
                # we should not write anything
                logtext = Test.read_and_remove(file.name+".log")
                # messages very from python 2 to 3. If python4 changes again, write it here below
                # the case py3 is something like: TypeError: main_typeerr() missing 1 required
                # positional argument...
                # py2 is something like: TypeError: main_typeerr() takes...
                # so build a general string:
                string2check = "TypeError: main_typeerr() "
                assert string2check in logtext

    def test_simple_run_codeerror_nosegs(self):
        '''test processing type error(wrong argumens), but test that
        since we do not have segments to process, the type error is not reached
        '''

        runner = CliRunner()

        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            # pyfile = self.get_file("processing.py")  # custom one
            with NamedTemporaryFile(suffix='.py') as tmpfile:

                with open(pyfile) as opn:
                    content = opn.read()

                content = content.replace("def main(", """def main_typeerr(segment, config, wrong_argument):
    return [6]

def main(""")
                tmpfile.write(content.encode('utf8'))
                tmpfile.seek(0)

                result = runner.invoke(cli, ['process', '--dburl', self.dburi,
                                             '-p', tmpfile.name, '-f', "main_typeerr",
                                             '-c', conffile,
                                             file.name])

                # the file above are bad implementation (old one)
                # we should not write anything
                logtext = Test.read_and_remove(file.name+".log")
                string2check = "0 segments"
                assert string2check in logtext
