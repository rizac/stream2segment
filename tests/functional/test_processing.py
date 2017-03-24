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
from stream2segment.main import main
import tempfile
from stream2segment.io.db.models import Base, Event, Class
import csv
from itertools import cycle
from future.backports.urllib.error import URLError
import pytest
import multiprocessing
from stream2segment.process.wrapper import load_proc_cfg
from stream2segment import process
from stream2segment.utils import resources
from obspy.core.stream import read

class Test(unittest.TestCase):

    dburi = ""
    file = None

    @staticmethod
    def cleanup(session, file_, *patchers):
        if session:
            session.close()
        if file_ and os.path.isfile(file_):
            os.remove(file_)
        for patcher in patchers:
            patcher.stop()

    @classmethod
    def setUpClass(cls):
        file_ = os.path.dirname(__file__)
        filedata = os.path.join(file_, "..", "data")
        url = os.path.join(filedata, "_test.sqlite")
        cls.dburi = 'sqlite:///' + url
        cls.file = url

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.file):
            os.remove(cls.file)

    def setUp(self):
        # remove file if not removed:
        Test.cleanup(None, self.file)

        self.save_inventory=False

        # mock get inventory:
        self.patcher = patch('stream2segment.download.utils.urlread')
        self.mock_url_read = self.patcher.start()

        self.mock_url_read.side_effect = self.url_read

        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        self.engine = create_engine(self.dburi)
        # Base.metadata.drop_all(cls.engine)
        Base.metadata.create_all(self.engine)  # @UndefinedVariable
        # create a configured "Session" class
        Session = sessionmaker(bind=self.engine)
        # create a Session
        self.session = Session()

        # setup a run_id:
        r = models.Run()
        self.session.add(r)
        self.session.commit()
        self.run = r

        # setup an event:
        e = models.Event(id='abc', latitude=8, longitude=9, magnitude=5, depth_km=4,
                         time=datetime.utcnow())
        self.session.add(e)
        self.session.commit()
        self.evt = e

        d = models.DataCenter(station_query_url='asd', dataselect_query_url='sdft')
        self.session.add(d)
        self.session.commit()
        self.dc = d

        s_ok = models.Station(datacenter_id=d.id, latitude=81, longitude=56, network='ok', station='ok')
        self.session.add(s_ok)
        self.session.commit()
        self.sta_ok = s_ok

        s_err = models.Station(datacenter_id=d.id, latitude=81, longitude=56, network='err', station='err')
        self.session.add(s_err)
        self.session.commit()
        self.sta_err = s_err

        s_none = models.Station(datacenter_id=d.id, latitude=81, longitude=56, network='none', station='none')
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

        c_none = models.Channel(station_id=s_err.id, location='none', channel="none", sample_rate=56.7)
        self.session.add(c_none)
        self.session.commit()
        self.cha_none = c_none

        data = self.read_stream_raw('trace_GE.APE.mseed')
        
        # "normal" segment
        sg1 = models.Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)

        # this segment should have inventory returning an exception (see url_read above)
        sg2 = models.Segment(channel_id=c_err.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)
        # segment with gaps
        data = self.read_stream_raw('IA.BAKI..BHZ.D.2016.004.head')
        sg3 = models.Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)

        # empty segment
        data['data'] = b''
        data['start_time'] += timedelta(seconds=1)  # avoid unique constraint
        sg4 = models.Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)

        # null segment
        data['data'] = None
        data['start_time'] += timedelta(seconds=2)  # avoid unique constraint
        sg5 = models.Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e.id, run_id=r.id,
                             event_distance_deg=45, **data)


        self.session.add_all([sg1, sg2, sg3, sg4, sg5])
        self.session.commit()
        self.seg1 = sg1
        self.seg2 = sg2
        self.seg_gaps = sg2
        self.seg_empty = sg3
        self.seg_none = sg4

        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self.session, self.file, self.patcher)


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

    def get_processing_files(self):
        return resources.get_proc_template_files()

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

    # FIRST TEST. to workaround the fact that self.mock_url read.call_count seems not to work woth
    # multiprocessing: mock multiprocessing
    @mock.patch('stream2segment.process.wrapper.ProcessPoolExecutor')
    @mock.patch('stream2segment.process.wrapper.as_completed')
    def test_simple_run(self, mock_as_completed, mock_ppe):
        mock_as_completed.side_effect = lambda iterable: self.mocked_ascompleted(iterable)
        mock_ppe.return_value = self.mocked_processpoolexecutor()
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

            # check file has been correctly written:
            with open(file.name, 'rb') as csvfile:
                spamreader = csv.reader(csvfile)  # , delimiter=' ', quotechar='|')
                rowz = 0
                for row in spamreader:
                    rowz += 1
                    if rowz == 1:
                        assert row[0] == self.seg1.channel.id
                        assert row[1] == self.seg1.start_time.isoformat()
                        assert row[2] == self.seg1.end_time.isoformat()
                assert rowz == 1
                assert len(self.read_and_remove(file.name+".log")) > 0
                # ===================================================================
                # NOW WE CAN CHECK IF THE URLREAD HAS BEEN CALLED TWICE AND NOT MORE:
                # ===================================================================
                assert self.mock_url_read.call_count == 2

    def url_read(self, *a, **v):
        if "=err" in a[0]:
            raise URLError('error')
        elif "=none" in a[0]:
            return None
        else:
            return self.read_data_raw("inventory_GE.APE.xml")


    def load_proc_cfg(self, *a, **kw):
        cfg = load_proc_cfg(*a, **kw)
        cfg['inventory'] = self.save_inventory
        return cfg

    @mock.patch('stream2segment.process.wrapper.urlread')
    @mock.patch('stream2segment.process.wrapper.load_proc_cfg')
    def test_simple_run_list_nosave(self, mock_load_cfg, mock_url_read):
        mock_url_read.side_effect = self.url_read
        self.save_inventory=False
        mock_load_cfg.side_effect = self.load_proc_cfg

        runner = CliRunner()
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   pyfile, conffile,
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
                        assert row[0] == self.seg1.channel.id
                        assert row[1] == self.seg1.start_time.isoformat()
                        assert row[2] == self.seg1.end_time.isoformat()
                assert rowz == 1
                assert len(self.read_and_remove(file.name+".log")) > 0

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)

        # test we did not save any inventory:
        stas = self.session.query(models.Station).all()
        assert not any(s.inventory_xml for s in stas)


    @mock.patch('stream2segment.process.wrapper.urlread')
    @mock.patch('stream2segment.process.wrapper.load_proc_cfg')
    def test_simple_run_list_save(self, mock_load_cfg, mock_url_read):
        mock_url_read.side_effect = self.url_read
        self.save_inventory=True
        mock_load_cfg.side_effect = self.load_proc_cfg

        # need to reset this global variable: FIXME: better handling?
        process.wrapper._inventories={}
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
                    if rowz == 1:
                        assert row[0] == self.seg1.channel.id
                        assert row[1] == self.seg1.start_time.isoformat()
                        assert row[2] == self.seg1.end_time.isoformat()
                assert rowz == 1
                assert len(self.read_and_remove(file.name+".log")) > 0

                # REMEMBER, THIS DOES NOT WORK:
                # assert mock_url_read.call_count == 2
                # that's why we tested above by mocking multiprocessing
                # (there must be some issue with multiprocessing)

        # test we did not save any inventory:
        stas = self.session.query(models.Station).all()
        assert any(s.inventory_xml for s in stas)
        assert self.session.query(models.Station).filter(models.Station.id == self.sta_ok.id).first().inventory_xml

    @mock.patch('stream2segment.process.wrapper.urlread')
    def test_simple_run_dict(self, mock_url_read):
        mock_url_read.side_effect = self.url_read
        runner = CliRunner()

        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            pyfile, conffile = self.get_processing_files()
            result = runner.invoke(main, ['p', '--dburl', self.dburi,
                                   self.get_file("processing_test_freqs2csv_dict.py"),
                                   self.get_file('processing.config.yaml'),
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
                    if rowz == 2:
                        assert row[0] == self.seg1.channel.id
                        assert row[1] == self.seg1.start_time.isoformat()
                        assert row[2] == self.seg1.end_time.isoformat()
                assert rowz == 2
                assert len(self.read_and_remove(file.name+".log")) > 0


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()