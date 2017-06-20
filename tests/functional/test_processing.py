'''
Created on Feb 14, 2017

@author: riccardo
'''
import unittest
import os
import mock, os, sys
from datetime import datetime, timedelta
from mock import patch
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from stream2segment.io.db import models
from urllib2 import URLError
from click.testing import CliRunner
from stream2segment.main import main, closing
import tempfile
from stream2segment.io.db.models import Base, Event, Class, Station, WebService
import csv
from itertools import cycle
from future.backports.urllib.error import URLError
import pytest
import multiprocessing
from stream2segment.process.main import load_proc_cfg
from stream2segment import process
from obspy.core.stream import read
import StringIO
from stream2segment.utils.resources import get_templates_fpaths

class DB():
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
        r = models.Run()
        self.session.add(r)
        self.session.commit()
        self.run = r

        ws = WebService(id=1, url='eventws')
        self.session.add(ws)
        self.session.commit()
        self.ws = ws
        # setup an event:
        e = models.Event(id=1, webservice_id=ws.id, eventid='abc', latitude=8, longitude=9, magnitude=5, depth_km=4,
                         time=datetime.utcnow())
        self.session.add(e)
        self.session.commit()
        self.evt = e

        d = models.DataCenter(station_url='asd', dataselect_url='sdft')
        self.session.add(d)
        self.session.commit()
        self.dc = d

        # s_ok stations have lat and lon > 11, other stations do not
        s_ok = models.Station(datacenter_id=d.id, latitude=11, longitude=12, network='ok', station='ok',
                              start_time=datetime.utcnow())
        self.session.add(s_ok)
        self.session.commit()
        self.sta_ok = s_ok

        s_err = models.Station(datacenter_id=d.id, latitude=-21, longitude=5, network='err', station='err',
                              start_time=datetime.utcnow())
        self.session.add(s_err)
        self.session.commit()
        self.sta_err = s_err

        s_none = models.Station(datacenter_id=d.id, latitude=-31, longitude=-32, network='none', station='none',
                              start_time=datetime.utcnow())
        self.session.add(s_none)
        self.session.commit()
        self.sta_none = s_none

        c_ok = models.Channel(station_id=s_ok.id, location='ok', channel="ok", sample_rate=56.7)
        self.session.add(c_ok)
        self.session.commit()
        self.cha_ok = c_ok

        c_err = models.Channel(station_id=s_err.id, location='err', channel="err", sample_rate=56.7)
        self.session.add(c_err)
        self.session.commit()
        self.cha_err = c_err

        c_none = models.Channel(station_id=s_none.id, location='none', channel="none", sample_rate=56.7)
        self.session.add(c_none)
        self.session.commit()
        self.cha_none = c_none

        data = Test.read_stream_raw('trace_GE.APE.mseed')

        # build three segments with data:
        # "normal" segment
        sg1 = models.Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=35, **data)

        # this segment should have inventory returning an exception (see url_read above)
        sg2 = models.Segment(channel_id=c_err.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)
        # segment with gaps
        data = Test.read_stream_raw('IA.BAKI..BHZ.D.2016.004.head')
        sg3 = models.Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=55, **data)

        # build two segments without data:
        # empty segment
        data['data'] = b''
        data['start_time'] += timedelta(seconds=1)  # avoid unique constraint
        sg4 = models.Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)

        # null segment
        data['data'] = None
        data['start_time'] += timedelta(seconds=2)  # avoid unique constraint
        sg5 = models.Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
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

        patchers = [getattr(self, 'patcher', None),
                    getattr(self, 'patcher1', None),
                    getattr(self, 'patcher2', None),]
        for patcher in patchers:
            if patcher:
                patcher.stop()

    @property
    def is_sqlite(self):
        return str(self.db.engine.url).startswith("sqlite:///")
    
    @property
    def is_postgres(self):
        return str(self.db.engine.url).startswith("postgresql://")

    def setUp(self):
        
        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self)

        # remove file if not removed:
        #Test.cleanup(None, self.file)

        self.custom_config= {'save_downloaded_inventory': False, 'inventory': True}
        self.inventory=True

        self.db = DB()
        self.db.create()
        self.session = self.db.session
        self.dburi = self.db.dburi

        # mock get inventory:
        self.patcher = patch('stream2segment.download.utils.urlread')
        self.mock_url_read = self.patcher.start()
        self.mock_url_read.side_effect = self.url_read

        

        self.patcher1 = patch('stream2segment.main.get_session')
        self.mock_session = self.patcher1.start()
        self.mock_session.return_value = self.session

        self.patcher2 = patch('stream2segment.main.closing')
        self.mock_closing = self.patcher2.start()
        self.mock_closing.side_effect = lambda dburl: closing(dburl, close_session=False)
        
        

        

#     @staticmethod
#     def _get_seg_times():
#         start_time = datetime.utcnow()
#         end_time = start_time+timedelta(seconds=5)
#         a_time = start_time + timedelta(seconds=2)
#         return start_time, a_time, end_time

    @staticmethod
    def read_stream_raw(file_name):
        stream=read(Test.get_file(file_name))

        start_time = stream[0].stats.starttime
        end_time = stream[0].stats.endtime

        # set arrival time to one third duration
        return dict(
        data = Test.read_data_raw(file_name),
        arrival_time = (start_time + (end_time - start_time)/3).datetime,
        start_time = start_time.datetime,
        end_time = end_time.datetime,
        )

    @staticmethod
    def read_data_raw(file_name):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        with open(os.path.join(folder, file_name)) as opn:
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
    def get_processing_files(get_template_returning_list=False):
        pyfile, conffile = get_templates_fpaths("processing.py", "processing.yaml")
        if get_template_returning_list:
            pyfile = os.path.join(os.path.dirname(pyfile),
                                  os.path.basename(pyfile).replace(".py", ".returning_list.py"))
        return pyfile, conffile

    # DEPRECATED: NOT USED ANYMORE (WE DONT USE PYTHON MULTIPROCESSING ANYMORE):
    # IMPLEMENTED BECAUSE
    # self.mock_url read.call_count seems not to work woth
    # multiprocessing: mock multiprocessing
    def mocked_ascompleted(self, iterable):
        """mocks run_async cause we want to test also without multiprocessing (see below)"""
        class mockfuture(object):
            def __init__(self, res, cnc):
                self.res = res
                self.cnc = cnc

            def cancelled(self):
                return self.cnc

            def exception(self, *a, **v):
                if isinstance(self.res, Exception):
                    return self.res
                return None

            def result(self, *a, **v):
                if self.exception() is not None:
                    raise self.exception()
                return self.res

        for obj in iterable:
            try:
                res = obj[0](*obj[1], **obj[2])
            except Exception as exc:
                res = exc
            yield mockfuture(res, False)

    # DEPRECATED: NOT USED ANYMORE (WE DONT USE PYTHON MULTIPROCESSING ANYMORE):
    # IMPLEMENTED BECAUSE
    # self.mock_url read.call_count seems not to work woth
    # multiprocessing: mock multiprocessing
    def mocked_processpoolexecutor(self):

        class mppe(object):
            self.elements = []
            def __enter__(self, *a, **v):
                return self

            def __exit__(self, *a, **v):
                pass

            def submit(self, func, *a, **kw):
                return (func, a, kw)

        return mppe()

    @staticmethod
    def url_read(*a, **v):
        if "=err" in a[0]:
            raise URLError('error')
        elif "=none" in a[0]:
            return None, 500, 'Server error'
        else:
            return Test.read_data_raw("inventory_GE.APE.xml"), 200, 'Ok'

    def load_proc_cfg(self, *a, **kw):
        """called by mocked read config: updates the parsed dict with the custom config"""
        cfg = load_proc_cfg(*a, **kw)
        cfg.update(self.custom_config)
        return cfg

### ======== ACTUAL TESTS: ================================

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Save inventory and check it's saved
    @mock.patch('stream2segment.process.main.load_proc_cfg')
    def test_simple_run_retDict_saveinv(self, mock_load_cfg):
        self.custom_config['inventory']=True
        self.custom_config['save_downloaded_inventory']=True
        mock_load_cfg.side_effect = self.load_proc_cfg

        # need to reset this global variable: FIXME: better handling?
        process.main._inventories = {}
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   pyfile, conffile, file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print result.output
                assert False
                return

            # check file has been correctly written:
            with open(file.name, 'rb') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == str(self.db.seg1.id)
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 2
                assert len(self.read_and_remove(file.name+".log")) > 0

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)


        # save_downloaded_inventory True, test that we did save any:
        assert len(self.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = self.session.query(models.Station).all()
        assert any(s.inventory_xml for s in stas)
        assert self.session.query(models.Station).filter(models.Station.id == self.db.sta_ok.id).first().inventory_xml


    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Don't save inventory and check it's not saved
    @mock.patch('stream2segment.process.main.load_proc_cfg')
    def test_simple_run_retDict_dontsaveinv(self, mock_load_cfg):
        self.custom_config['inventory']=True
        self.custom_config['save_downloaded_inventory']=False
        mock_load_cfg.side_effect = self.load_proc_cfg

        # need to reset this global variable: FIXME: better handling?
        process.main._inventories={}
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   pyfile, conffile, file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print result.output
                assert False
                return

            # check file has been correctly written:
            with open(file.name, 'rb') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == str(self.db.seg1.id)
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 2
                assert len(self.read_and_remove(file.name+".log")) > 0

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)


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
    # Here a simple test for a processing NO file. We implement a filter that excludes the only processed file
    # using associated stations lat and lon. 
    @mock.patch('stream2segment.process.main.load_proc_cfg')
    def test_simple_run_retDict_seg_select_empty_and_err_segments(self, mock_load_cfg):
        # s_ok stations have lat and lon > 11, other stations do not
        # now we want to set a filter which gets us only the segments from stations not ok.
        # Note: withdata is not specified so we will get 3 segments (2 with data None, 1 with data which raises
        # errors for station inventory)
        self.custom_config['segment_select'] = {'station.latitude': '<10', 'station.longitude': '<10'}
        self.custom_config['inventory'] = True
        mock_load_cfg.side_effect = self.load_proc_cfg
        
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   pyfile,
                                   conffile,
                                   file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print result.output
                assert False
                return

            # check file has been correctly written, we should have written two files
            with open(file.name, 'rb') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == str(self.db.seg1.id)
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 0
                sss = self.read_and_remove(file.name+".log")
                # as we have joined twice segment with stations (one is done by default, the other
                # has been set in custom_config['segment_select'] above, we should have this sqlalchemy msg
                # in the log:
                assert "SAWarning: Pathed join target" in sss
                
                # ===================================================================
                # NOW WE CAN CHECK IF THE URLREAD HAS BEEN CALLED TWICE AND NOT MORE:
                # once for the none segments, once for the err segment
                # ===================================================================
                assert self.mock_url_read.call_count == 2

# Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing NO file. We implement a filter that excludes the only processed file
    # using associated stations lat and lon. 
    @mock.patch('stream2segment.process.main.load_proc_cfg')
    def test_simple_run_retDict_seg_select_only_one_err_segment(self, mock_load_cfg):
        # s_ok stations have lat and lon > 11, other stations do not
        # now we want to set a filter which gets us only the segments from stations not ok.
        # Note: withdata is True so we will get 1 segment (1 with data which raises
        # errors for station inventory)
        self.custom_config['segment_select'] = {'has_data': 'true', 'station.latitude': '<10', 'station.longitude': '<10'}
        self.custom_config['inventory'] = True
        mock_load_cfg.side_effect = self.load_proc_cfg
        
        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   pyfile,
                                   conffile,
                                   file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print result.output
                assert False
                return

            # check file has been correctly written, we should have written two files
            with open(file.name, 'rb') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 2:
                        assert row[0] == str(self.db.seg1.id)
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 0
                assert len(self.read_and_remove(file.name+".log")) > 0
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
    def test_simple_run_ret_list(self):
        runner = CliRunner()

        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files(True)
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   pyfile,
                                   conffile,
                                   file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print result.output
                assert False
                return

            # check file has been correctly written:
            with open(file.name, 'rb') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 1:
                        assert row[0] == str(self.db.seg1.id)
#                         assert row[1] == self.db.seg1.start_time.isoformat()
#                         assert row[2] == self.db.seg1.end_time.isoformat()
                assert rowz == 1
                assert len(self.read_and_remove(file.name+".log")) > 0


    def test_simple_run_codeerror(self):

        runner = CliRunner()

        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            # pyfile, conffile = Test.get_processing_files()
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   Test.get_file("processing_test_freqs2csv_dict.py"),
                                   Test.get_file('processing.config.yaml'),
                                   file.name])

            # the file above are bad implementation (old one)
            # we should not write anything
            sss = Test.read_and_remove(file.name+".log")

            assert "TypeError: main() takes exactly 2 arguments (4 given)" in sss


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()