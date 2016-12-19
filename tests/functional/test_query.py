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
from stream2segment.download.query import get_datacenters as gdc_orig, get_events, get_datacenters,\
    make_ev2sta, make_dc2seg, download_segments
from obspy.core.stream import Stream
from stream2segment.io.db.models import DataCenter
from itertools import cycle, repeat, count
from urllib2 import URLError
import socket
from obspy.taup.helper_classes import TauModelError

import logging
from logging import StreamHandler
import sys
from stream2segment.main import logger as main_logger


class Test(unittest.TestCase):
    
    engine = None
    dburi = ""


#     @classmethod
#     def setUpClass(cls):
#         file = os.path.dirname(__file__)
#         filedata = os.path.join(file,"..","data")
#         url = os.path.join(filedata, "_test.sqlite")
#         Test.dburi = 'sqlite:///' + url
#         # an Engine, which the Session will use for connection
#         # resources
#         # some_engine = create_engine('postgresql://scott:tiger@localhost/')
#         Test.engine = create_engine(Test.dburi)
#         # Base.metadata.drop_all(cls.engine)
#         Base.metadata.create_all(cls.engine)
# 
#     @classmethod
#     def tearDownClass(cls):
#         Base.metadata.drop_all(cls.engine)

    def setUp(self):
        file = os.path.dirname(__file__)
        filedata = os.path.join(file,"..","data")
        url = os.path.join(filedata, "_test.sqlite")
        self.dburi = 'sqlite:///' + url
        self.file = url
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
        
        # logging.getLogger("stream2segment").addHandler(StreamHandler(stream=sys.stdout))
        

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)  # @UndefinedVariable
        os.remove(self.file)
        for hand in main_logger.handlers:
            # IMPOERTANT TO REMOVE LOGGER CAUSE MAIN COFNIGURES LOGGERS AND
            # USES THE DB, EVERYTHING IS MESSED UP THEN AND IT's NOT BECAUSE OF ERRORS WE SHOULD
            # CATCH
            main_logger.removeHandler(hand)

#     @staticmethod
#     def checkstats(stats, check_hasok=True, check_haserrors=True, check_hasempty=True):
#         numok=numerrors=numemtpy=0
#         for k in stats.keys():
#             substat = stats[k]
#             for msg in substat.keys():
#                 if msg.lower() == 'ok':
#                     numok+=1
#                 elif "empty" == msg.lower():
#                     numemtpy += 1
#                 else:
#                     numerrors += 1
#         
#         if check_hasok and not numok:
#             raise ValueError('stats has no "ok" key')
#         elif check_hasempty and not numemtpy:
#             raise ValueError('stats has no "empty" key')
#         elif check_haserrors and not numerrors:
#             raise ValueError('stats has no errors')
        
    @property
    def _events_raw(self):
        return """"1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""

    def geteventsargs(self):
        return ["eventws", "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso"], {}


    def get_events(self, mock_url_read):
        mock_url_read.reset_mock()
        mock_url_read.return_value = self._events_raw
        args = self.geteventsargs()
        return get_events(self.session, *args[0], **args[1])


    @patch('stream2segment.download.query.get_query', return_value='a')
    @patch('stream2segment.download.query.url_read')
    def test_get_events(self, mock_urlread, mock_query):
        data = self.get_events(mock_urlread)

        # addert only three events where succesfully saved to db
        # (one is
        assert len(self.session.query(Event).all()) == len(data) == 3
#         argz = {k:k for k in args}
#         argz.update({'format': 'text'})
#         argz.pop(args[0])
#         mock_query.assert_called_with(args[0], minmagnitude=args[1], minlat=args[2], maxlat=args[3],
#                           minlon=args[4], maxlon=args[5], start=args[6], end=args[7], format='text')
        assert mock_urlread.call_args[0] == (mock_query.return_value, )

# =================================================================================================

    @property
    def _datacenters_raw(self):
        return """http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
    (indentation is bad! :)    http://geofon.gfz-potsdam.de/fdsnws/station/1/query"""

    def getdatacentersargs(self):
        return [], {}


    def get_datacenters(self, mock_url_read):
        mock_url_read.reset_mock()
        mock_url_read.return_value = self._datacenters_raw
        args = self.getdatacentersargs()
        return get_datacenters(self.session, *args[0], **args[1])


    @patch('stream2segment.download.query.get_query', return_value='a')
    @patch('stream2segment.download.query.url_read')
    def test_get_dcs(self, mock_urlread, mock_query):
        data = self.get_datacenters(mock_urlread)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        mock_urlread.assert_called_once()  # we might be more fine grained, see code
        assert len(self.session.query(models.DataCenter).all()) == 2

# =================================================================================================

    @property
    def _stations_raw(self):
        invalid = """#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
--- ERROR --- MALFORMED|12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
"""
        partial_valid = """#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
HT|AGG||HHE|--- ERROR --- NONNUMERIC |22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HHE|95.6|22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
HT|LKD2||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
"""     
        return cycle([partial_valid, '', invalid, '', '', URLError('wat'), socket.timeout()])

    
    def setup_mock_urlopen_in_async_for_stations(self, mock_urlopen_in_async):
        mock_urlopen_in_async.reset_mock()
        a = Mock()
        a.read.side_effect =  self._stations_raw
        mock_urlopen_in_async.return_value = a
       
    
    def make_ev2sta(self, events, datacenters, mock_urlopen_in_async):
        kw = dict(
            session=self.session,
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
        self.setup_mock_urlopen_in_async_for_stations(mock_urlopen_in_async)
         
        return make_ev2sta(**kw)

    @patch('stream2segment.download.query.url_read')
    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_make_ev2sta(self, mock_urlopen_in_async, mock_url_read):
        events = self.get_events(mock_url_read)
        datacenters = self.get_datacenters(mock_url_read)
        evt2stations, stats = self.make_ev2sta(events, datacenters, mock_urlopen_in_async)
        
        # we should have called mock_urlopen_in_async datacenters * events:
        numcalls_urlopen_in_async = len(events) * len(datacenters)
        assert mock_urlopen_in_async.call_count == numcalls_urlopen_in_async

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
    @staticmethod
    def setup_mock_arrival_time(mock_arr_time):
        mock_arr_time.reset_mock()
        mock_arr_time.side_effect = cycle([datetime.utcnow(), TauModelError('wat?'), ValueError('wat?')])

    def make_dc2seg(self, events, datacenters, evt2stations, mock_arr_time):
        self.setup_mock_arrival_time(mock_arr_time)
        return make_dc2seg(self.session, events, datacenters, evt2stations, [1,2], ['P', 'Q'])

    @patch('stream2segment.download.query.get_arrival_time')
    @patch('stream2segment.download.query.url_read')
    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_make_dc2seg(self, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        events = self.get_events(mock_url_read)
        datacenters = self.get_datacenters(mock_url_read)
        evt2stations, stats = self.make_ev2sta(events, datacenters, mock_urlopen_in_async)
        segments, skipped_already_d =  self.make_dc2seg(events, datacenters, evt2stations, mock_arr_time)
        
        # as database is empty by default, skipped already downloaded should have zeros
        for val in skipped_already_d.itervalues():
            assert val == 0
        
        # as get_arrival_time raises errors, some segments are not written. Assert that the
        # total number of segments is lower than the total number of channels
        numchannels = sum(len(d) for d in evt2stations.itervalues())
        numsegments = sum(len(d) for d in segments.itervalues())
        
        assert numchannels > numsegments
        
        h = 9
    
# =================================================================================================
    
    @staticmethod
    def setup_mock_urlopen_in_async_for_segments(mock_urlopen_in_async):
        mock_urlopen_in_async.reset_mock()
        a = Mock()
        a.read.side_effect =  cycle([b'data','', '', URLError('wat'), socket.timeout()])
        mock_urlopen_in_async.return_value = a


    def download_segments(self, dc2segments, mock_urlopen_in_async):
        self.setup_mock_urlopen_in_async_for_segments(mock_urlopen_in_async)

        r = models.Run()
        self.session.add(r)
        self.session.commit()
        
        stats = {}
        for dcen_id, segments_df in dc2segments.iteritems():
            stats_ = download_segments(self.session, segments_df, r.id,  # arguments below are just to fill the call
                                      5,
                                      5,
                                      40,
                                      -1)
            stats[dcen_id] = stats_
        
        return stats
        
    @patch('stream2segment.download.query.get_arrival_time')
    @patch('stream2segment.download.query.url_read')
    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_download_segments(self, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        events = self.get_events(mock_url_read)
        datacenters = self.get_datacenters(mock_url_read)
        evt2stations, stats = self.make_ev2sta(events, datacenters, mock_urlopen_in_async)
        dc2segments, skipped_already_d =  self.make_dc2seg(events, datacenters, evt2stations, mock_arr_time)
        stats = self.download_segments(dc2segments, mock_urlopen_in_async) 
   
        # we expect not all segments are written (mock side_effect set it self.download_segments
        # returns errors and emtpy)
        numsegmentssaved = len(self.session.query(models.Segment).all())
        assert numsegmentssaved == 0
        assert numsegmentssaved <= sum(len(d) for d in dc2segments.itervalues())
        assert 0 == sum(len(stats[d]) for d in stats)
        
        # we expect each datacenter has at least one error (mock side_effect set it self.download_segments)
#         expected_datacenters_with_errors = len(datacenters)
#         datacenters_with_errors = 0
#         for k in stats.keys():
#             substat = stats[k]
#             for msg in substat.keys():
#                 if msg.lower() != 'Ok':
#                     datacenters_with_errors +=1
#                     break
#         assert datacenters_with_errors == expected_datacenters_with_errors

    
    @patch('stream2segment.download.query.get_arrival_time')
    @patch('stream2segment.download.query.url_read')
    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_cmdline(self, mock_urlopen_in_async, mock_url_read, mock_arr_time):
         
        self.setup_mock_arrival_time(mock_arr_time)
        # setup urlasync for stations. This means that when reading segments a station raw data
        # is returned, but we should not care (it will be stored as binary)
        self.setup_mock_urlopen_in_async_for_stations(mock_urlopen_in_async)
        
        # setup urlread for events first and then for datacenters:
        mock_url_read.reset_mock()
        mock_url_read.side_effect = [self._events_raw, self._datacenters_raw]
        
        prevlen = len(self.session.query(models.Segment).all())
    
        runner = CliRunner()
        result = runner.invoke(main , ['--dburl', self.dburi, '-a', 'd',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'], catch_exceptions=False)
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print result.output
            
        assert len(self.session.query(models.Segment).all()) == 1
          
                 
#         # common query for 1 event found and all datacenters (takes more or less 10 to 20 minutes):
#         # stream2segment -f 2016-05-08T22:45:00 -t 2016-05-08T23:00:00
#         