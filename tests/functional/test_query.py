#@PydevCodeAnalysisIgnore
'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')

import numpy as np
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta
from StringIO import StringIO

import unittest, os
from sqlalchemy.engine import create_engine
from stream2segment.io.db.models import Base, Event
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError
from stream2segment.main import main
from click.testing import CliRunner
from stream2segment.io.db import models
# from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_col_names
import pandas as pd
from stream2segment.download.query import get_events, get_datacenters,\
    make_ev2sta, get_segments_df, download_segments
from obspy.core.stream import Stream
from stream2segment.io.db.models import DataCenter
from itertools import cycle, repeat, count
from urllib2 import URLError
import socket
from obspy.taup.helper_classes import TauModelError

# import logging
# from logging import StreamHandler
import sys
# from stream2segment.main import logger as main_logger
from sqlalchemy.sql.expression import func

class Test(unittest.TestCase):
    
    dburi = ""
    file = None

    @staticmethod
    def cleanup(session, file, *patchers):
        if session:
            session.close()
        if file and os.path.isfile(file):
            os.remove(file)
        for patcher in patchers:
            patcher.stop()

    @classmethod
    def setUpClass(cls):
        file = os.path.dirname(__file__)
        filedata = os.path.join(file,"..","data")
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
        
        self.patcher = patch('stream2segment.utils.url.urllib2.urlopen')
        self.mock_urlopen = self.patcher.start()
        
        self.patcher2 = patch('stream2segment.download.query.get_arrival_time')
        self.mock_arrival_time = self.patcher2.start()
        
        self.patcher3 = patch('stream2segment.main.logger')
        self.mock_main_logger = self.patcher3.start()
        
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

        # side effects:
        
        self._evt_urlread_sideeffect =  ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", ""]
        self._dc_urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
    (indentation is bad! :)    http://geofon.gfz-potsdam.de/fdsnws/station/1/query""", ""]

        self._sta_urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
--- ERROR --- MALFORMED|12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", "", """#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
HT|AGG||HHE|--- ERROR --- NONNUMERIC |22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HHE|95.6|22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
HT|LKD2||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", "",  URLError('wat'), socket.timeout()]
        # self._sta_urlread_sideeffect = cycle([partial_valid, '', invalid, '', '', URLError('wat'), socket.timeout()])

        self._atime_sideeffect = [datetime.utcnow(), TauModelError('wat?'), ValueError('wat?')]
        self._seg_urlread_sideeffect = [b'data','', '', URLError('wat'), socket.timeout()]

        #add cleanup (in case tearDown is not 
        self.addCleanup(Test.cleanup, self.session, self.file, self.patcher, self.patcher2,
                        self.patcher3)

        
    def tearDown(self):
        # see cleanup static method
        pass
    
    def setup_urlopen(self, urlread_side_effect):
        """setup urlopen return value. 
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.read, that will be converted
        to an itertools.cycle(side_effect) REMEMBER that any element of urlread_side_effect which is a nonempty
        string must be followed by an EMPTY
        STRINGS TO STOP reading otherwise we fall into an infinite loop if the argument
        blocksize of url read is not negative !"""
        self.mock_urlopen.reset_mock()
        a = Mock()
        a.read.side_effect =  cycle(urlread_side_effect)
        self.mock_urlread = a.read
        self.mock_urlopen.return_value = a


    def get_events(self, url_read_side_effect=None, *a, **kw):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_events( *a, **kw)


    @patch('stream2segment.download.query.get_query', return_value='a')
    def test_get_events(self, mock_query):
        data = self.get_events(None, self.session,
                               "eventws", "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
        # assert only three events where succesfully saved to db
        # (one is
        assert len(self.session.query(Event).all()) == len(data) == 3
        # assert mock_urlread.call_args[0] == (mock_query.return_value, )

# =================================================================================================

    def get_datacenters(self, url_read_side_effect=None, *a, **kw):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_datacenters(*a, **kw)


    @patch('stream2segment.download.query.get_query', return_value='a')
    def test_get_dcs(self, mock_query):
        data = self.get_datacenters(session=self.session)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        assert len(self.session.query(models.DataCenter).all()) == 2

# =================================================================================================
    
    def make_ev2sta(self, url_read_side_effect=None, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return make_ev2sta(*a, **kw)
    
    def test_make_ev2sta(self):
        events = self.get_events(None, self.session,
                               "eventws", "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
        datacenters = self.get_datacenters(session=self.session)
        evt2stations, stats = self.make_ev2sta(None,  # use self._seg_urlread_side_effect
                                               **dict(session=self.session,
                                                      events = events,
                                                      datacenters = datacenters,
                                                      sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )
        
        # we should have called mock_urlopen_in_async datacenters * events:
        numcalls_urlopen_in_async = len(events) * len(datacenters)
        assert self.mock_urlopen.call_count == numcalls_urlopen_in_async

        # we expect each datacenter has at least one error (_stations_raw return value)
        expected_datacenters_with_errors = len(datacenters)
        datacenters_with_errors = 0
        for k in stats.keys():
            substat = stats[k]
            for msg in substat.keys():
                if msg.lower() != 'Ok':
                    datacenters_with_errors +=1
                    break
        assert datacenters_with_errors == expected_datacenters_with_errors

        # we should have only 3 channels saved
        assert len(self.session.query(models.Channel).all()) == 5
        # and 2 stations saved ():
        assert len(self.session.query(models.Station).all()) == 3

# =================================================================================================

    def get_segments_df(self, atime_side_effect=None, *a, **kw) : # , ):
        self.mock_arrival_time.reset_mock()
        self.mock_arrival_time.side_effect = self._atime_sideeffect if atime_side_effect is None else atime_side_effect
        # self.setup_mock_arrival_time(mock_arr_time)
        return get_segments_df(*a, **kw)

    def test_get_segments_df(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        events = self.get_events(None, self.session,
                               "eventws", "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
        datacenters = self.get_datacenters(session=self.session)
        evt2stations, stats = self.make_ev2sta(None,  # use self._seg_urlread_side_effect
                                               **dict(session=self.session,
                                                      events = events,
                                                      datacenters = datacenters,
                                                      sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )
        
        segments, skipped_already_d =  self.get_segments_df(None,  # atime_side_effect
                                                        self.session, events,
                                                        datacenters, evt2stations, [1,2], ['P', 'Q'],
                                                        'ak135', False)
        
        # as database is empty by default, skipped already downloaded should have zeros
        for val in skipped_already_d.itervalues():
            assert val == 0
        
        # as get_arrival_time raises errors, some segments are not written. Assert that the
        # total number of segments is lower than the total number of channels
        numchannels = sum(len(d) for d in evt2stations.itervalues())
        numsegments = len(segments)
        
        assert numchannels > numsegments
        
        h = 9
    
# =================================================================================================

    def download_segments(self, url_read_side_effect=None, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return download_segments(*a, **kw)
 
 
    def test_download_segments(self):
        events = self.get_events(None, self.session,
                               "eventws", "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
        datacenters = self.get_datacenters(session=self.session)
        evt2stations, stats = self.make_ev2sta(None,  # use self._seg_urlread_side_effect
                                               **dict(session=self.session,
                                                      events = events,
                                                      datacenters = datacenters,
                                                      sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )
        segments_df, skipped_already_d =  self.get_segments_df(None,  # None: self.atime_side_effect is default 
                                                           self.session, events,
                                                           datacenters, evt2stations, [1,2], ['P', 'Q'],
                                                           'ak135', False)
        
        stats = self.download_segments(None, # None: self._seg_urlread_sideeffect is default
                                       self.session, segments_df, self.run.id,
                                       max_error_count=5, max_thread_workers=5,
                                       timeout=40, download_blocksize=-1)
        
        # stats = self.download_segments(dc2segments) 
   
        # we expect not all segments are written (mock side_effect set it self.download_segments
        # returns errors and emtpy)
        numsegmentssaved = len(self.session.query(models.Segment).all())
        assert numsegmentssaved == 2
        assert numsegmentssaved <= len(segments_df)
        assert 4 == sum(len(stats[d]) for d in stats)
        

    def test_download_segments_max_err(self):
        events = self.get_events(None, self.session,
                               "eventws", "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
        datacenters = self.get_datacenters(session=self.session)
        evt2stations, stats = self.make_ev2sta(None,  # use self._seg_urlread_side_effect
                                               **dict(session=self.session,
                                                      events = events,
                                                      datacenters = datacenters,
                                                      sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )
        # keep only first event. This prevents to have duplicated stations and we get some value
        # in dc2segments below (how is that possible we should check...)
        evt2stations = {evt2stations.keys()[0] : evt2stations.values()[0]}
        
        segments_df, skipped_already_d =  self.get_segments_df(None,  # None: atime_side_effect is default 
                                                           self.session, events,
                                                           datacenters, evt2stations, [1,2], ['P', 'Q'],
                                                           'ak135', False)
        
        stats = self.download_segments([URLError('w')], # raise always exception is url read
                                                    self.session, segments_df, self.run.id,
                                                    max_error_count=1, max_thread_workers=5,
                                                    timeout=40, download_blocksize=-1)
   
        urlerror_found=0
        discarded_after_1_trial=0
        for k, v in stats.iteritems():
            if v and not  (urlerror_found and discarded_after_1_trial):
                urlerror_found = 0
                discarded_after_1_trial = 0
                for a, b in v.iteritems():
                    if a.startswith('URLError:'):
                        urlerror_found+=1
                    elif " after 1 previous errors" in a:
                        discarded_after_1_trial+=1
                    if urlerror_found and discarded_after_1_trial:
                        break
                    
        assert urlerror_found and discarded_after_1_trial
        

    @patch('stream2segment.download.query.get_events')
    @patch('stream2segment.download.query.get_datacenters')
    @patch('stream2segment.download.query.make_ev2sta')
    @patch('stream2segment.download.query.get_segments_df')
    @patch('stream2segment.download.query.download_segments')
    @patch('stream2segment.main.get_session')
    def test_cmdline(self, mock_get_sess, mock_download_segments, mock_get_seg_df, mock_make_ev2sta,
                     mock_get_datacenter, mock_get_events):
        
        ddd = datetime.utcnow()
        # setup arrival time side effect with a constant date (by default is datetime.utcnow)
        # we might pass it as custom argument below actually
        self._atime_sideeffect[0] = ddd
        
        mock_get_sess.return_value = self.session

        def dsegs(*a, **v):
            return self.download_segments(None, *a, **v)
        mock_download_segments.side_effect = dsegs
        
        def segdf(*a, **v):
            return self.get_segments_df(None, *a, **v)
        mock_get_seg_df.side_effect = segdf

        def ev2sta(*a, **v):
            return self.make_ev2sta(None, *a, **v)
        mock_make_ev2sta.side_effect = ev2sta

        def getdc(*a, **v):
            return self.get_datacenters(None, *a, **v)
        mock_get_datacenter.side_effect = getdc

        def getev(*a, **v):
            return self.get_events(None, *a, **v)
        mock_get_events.side_effect = getev
        
        
        
        # prevlen = len(self.session.query(models.Segment).all())
    
        runner = CliRunner()
        result = runner.invoke(main , ['--dburl', self.dburi, '-a', 'd',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print result.output
            assert False
            return
            
        segments = self.session.query(models.Segment).all()
        assert len(segments) == 2
        assert segments[0].data == b'data'
        assert not segments[1].data
        emptySegId = segments[1].id
        
        # re-launch with the same setups.
        # what we want to test is the addition of an already downloaded segment which was empty
        # before and now is not. So:
        newdata = b'dataABC'
        self._seg_urlread_sideeffect = [newdata, '']  # empty STRING at end otherwise urlread has infinite loop!
        
        # relaunch with 'd' (empty segments no retry):
        runner = CliRunner()
        result = runner.invoke(main , ['--dburl', self.dburi, '-a', 'd',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print result.output
            assert False
            return
        # test that nothing happened:
        segments = self.session.query(models.Segment).all()
        assert segments[1].id == emptySegId and not segments[1].data
        
        # relaunch with 'D' (retry empty or Null segments data)
        runner = CliRunner()
        result = runner.invoke(main , ['--dburl', self.dburi, '-a', 'D',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print result.output
            assert False
            return
        # test that now empty segment has new data:
        segments = self.session.query(models.Segment).all()
        assert segments[1].id == emptySegId and segments[1].data == newdata
        


    @patch('stream2segment.download.query.get_events')
    @patch('stream2segment.download.query.get_datacenters')
    @patch('stream2segment.download.query.make_ev2sta')
    @patch('stream2segment.download.query.get_segments_df')
    @patch('stream2segment.download.query.download_segments')
    @patch('stream2segment.main.get_session')
    def test_cmdline_singleevent_singledatacenter(self, mock_get_sess, mock_download_segments,
                                                  mock_get_seg_df, mock_make_ev2sta,
                     mock_get_datacenter, mock_get_events):
        
        mock_get_sess.return_value = self.session

        def dsegs(*a, **v):
            return self.download_segments(None, *a, **v)
        mock_download_segments.side_effect = dsegs
        
        def segdf(*a, **v):
            return self.get_segments_df(None, *a, **v)
        mock_get_seg_df.side_effect = segdf

        def ev2sta(*a, **v):
            return self.make_ev2sta(None, *a, **v)
        mock_make_ev2sta.side_effect = ev2sta

        def getdc(*a, **v):
            # return only one datacenter
            return self.get_datacenters(["""http://ws.resif.fr/fdsnws/station/1/query""", ""], *a, **v)
        mock_get_datacenter.side_effect = getdc

        def getev(*a, **v):
            # return only one event
            url_read_side_effect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN)""", ""]
            _ =  self.get_events(url_read_side_effect, *a, **v)
            return _
        mock_get_events.side_effect = getev

        ddd = datetime.utcnow()
        # setup arrival time side effect
        self._atime_sideeffect = cycle([ddd, TauModelError('wat?'), ValueError('wat?')])
        
        # prevlen = len(self.session.query(models.Segment).all())
    
        runner = CliRunner()
        result = runner.invoke(main , ['--dburl', self.dburi, '-a', 'd',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print result.output
            assert False
            return
            
        segments = self.session.query(models.Segment).all()
        assert len(segments) == 0
        