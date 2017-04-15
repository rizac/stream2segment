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
from stream2segment.io.db.models import Base, Event, Class
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError
from stream2segment.main import main, closing
from click.testing import CliRunner
# from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_col_names
import pandas as pd
from stream2segment.download.query import add_classes, get_events_df, get_datacenters_df
# ,\
#     get_fdsn_channels_df, save_stations_and_channels, get_dists_and_times, set_saved_dist_and_times,\
#     download_segments, drop_already_downloaded, set_download_urls, save_segments
from obspy.core.stream import Stream
from stream2segment.io.db.models import DataCenter, Segment, Run, Station
from itertools import cycle, repeat, count
from urllib2 import URLError
import socket
from obspy.taup.helper_classes import TauModelError

# import logging
# from logging import StreamHandler
import sys
# from stream2segment.main import logger as main_logger
from sqlalchemy.sql.expression import func
from stream2segment.utils import get_session
from stream2segment.io.db.pd_sql_utils import withdata

class Test(unittest.TestCase):

    @staticmethod
    def cleanup(session, *patchers):
        if session:
            try:
                session.flush()
                session.commit()
            except SQLAlchemyError as _:
                pass
                # self.session.rollback()
            session.close()
            session.bind.dispose()
        
        for patcher in patchers:
            patcher.stop()

    def _get_sess(self, *a, **v):
        return self.session

    def setUp(self):

        from sqlalchemy import create_engine
        self.dburi = 'sqlite:///:memory:'
        engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(engine)
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        # create a Session
        self.session = Session()
        
        
        self.patcher = patch('stream2segment.utils.url.urllib2.urlopen')
        self.mock_urlopen = self.patcher.start()
        
        # this mocks get_session to return self.session:
        self.patcher1 = patch('stream2segment.main.get_session')
        self.mock_get_session = self.patcher1.start()
        self.mock_get_session.side_effect = self._get_sess
        
        # this mocks closing to actually NOT close the session (we will do it here):
        self.patcher2 = patch('stream2segment.main.closing')
        self.mock_closing = self.patcher2.start()
        def clsing(*a, **v):
            if len(a) >= 4:
                a[3] = False
            else:
                v['close_session'] = False
            return closing(*a, **v)
        self.mock_closing.side_effect = clsing
        
        
        self.patcher3 = patch('stream2segment.download.utils.get_arrival_time')
        self.mock_arrival_time = self.patcher3.start()
        
        self.patchers = [self.patcher, self.patcher1, self.patcher2, self.patcher3]
        #self.patcher3 = patch('stream2segment.main.logger')
        #self.mock_main_logger = self.patcher3.start()
        
        # setup a run_id:
        r = Run()
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

        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self.session, *self.patchers)
                        #self.patcher3)

    
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

    def test_add_classes(self):
        cls = {'a' : 'bla', 'b' :'c'}
        add_classes(self.session, cls)
        assert len(self.session.query(Class).all()) == 2
        add_classes(self.session, cls)
        assert len(self.session.query(Class).all()) == 2
    
# ===========================

    def get_events_df(self, url_read_side_effect, eventws="eventws", **args):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_events_df(self.session, eventws, *args)


    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_events(self, mock_query):
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", ""]
        
        data = self.get_events_df(urlread_sideeffect)
        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == len(pd.unique(data['id'])) == 3
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        assert len(data) == 3
        # assert mock_urlread.call_args[0] == (mock_query.return_value, )
        
        

    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_events_toomany_requests_FIXME_TOBEIMPLEMENTED(self, mock_query): # FIXME: implement it!
        ## FIXMEEEEE TO BE IMPLEMENTED!!!!
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", ""]
        
#         data = self.get_events_df(urlread_sideeffect, self.session,
#                                "eventws")  # , "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
        


# =================================================================================================

    def get_datacenters_df(self, url_read_side_effect, **args):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_datacenters_df(self.session, **args)
    

    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_dcs(self, mock_query):
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
    (indentation is bad! :)    http://geofon.gfz-potsdam.de/fdsnws/station/1/query""", ""]
        data, reg = self.get_datacenters_df(urlread_sideeffect)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        assert len(self.session.query(DataCenter).all()) == 2
        
    @patch('stream2segment.download.query.urljoin', return_value='a')
    def tst_get_dcs2(self, mock_query):
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
    (indentation is bad! :)    http://geofon.gfz-potsdam.de/fdsnws/station/1/query""", ""]
        data = self.get_datacenters_df(urlread_sideeffect)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        assert len(self.session.query(DataCenter).all()) == 2

# =================================================================================================
    
    def get_fdsn_channels_df(self, url_read_side_effect, events_df, datacenters_df, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_fdsn_channels_df(self.session, events_df, datacenters_df, *a, **kw)
     
    def tst_get_fdsn_channels_df(self):
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", ""]
        events_df = self.get_events_df(urlread_sideeffect)

        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
    (indentation is bad! :)    http://geofon.gfz-potsdam.de/fdsnws/station/1/query""", ""]
        datacenters_df = self.get_datacenters_df(urlread_sideeffect)
        fdsn_sta_df, stats = self.get_fdsn_channels_df(None,  # use self._seg_urlread_side_effect
                                                       events_df,
                                                       datacenters_df,
                                               **dict(sradius_minmag=5, sradius_maxmag=6, 
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
        numcalls_urlopen_in_async = len(events_df) * len(datacenters_df)
        assert self.mock_urlopen.call_count == numcalls_urlopen_in_async
 
        # we expect each datacenter has at least one error (_stations_raw return value)
        expected_datacenters_with_errors = len(datacenters_df)
        datacenters_with_errors = 0
        for k in stats.keys():
            substat = stats[k]
            for msg in substat.keys():
                if msg.lower() != 'Ok':
                    datacenters_with_errors +=1
                    break
        assert datacenters_with_errors == expected_datacenters_with_errors


    def tst_save_stations_and_channels(self):
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", ""]
        events_df = self.get_events_df(urlread_sideeffect)

        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
    (indentation is bad! :)    http://geofon.gfz-potsdam.de/fdsnws/station/1/query""", ""]
        datacenters_df = self.get_datacenters_df(urlread_sideeffect)
        fdsn_sta_df, stats = self.get_fdsn_channels_df(None,
                                                       events_df,
                                                       datacenters_df,
                                               **dict(sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )
    
        channels_df = save_stations_and_channels(fdsn_sta_df, self.session)
        # we should have only 3 channels saved
        assert len(self.session.query(Channel).all()) == 5
        # and 2 stations saved ():
        assert len(self.session.query(Station).all()) == 3
 
# # =================================================================================================
# 
    def getset_dists_and_times(self, atime_side_effect, events_df, segments_df, *a, **kw) : # , ):
        segments_df = set_saved_dist_and_times(self.session, segments_df)
        self.mock_arrival_time.reset_mock()
        self.mock_arrival_time.side_effect = self._atime_sideeffect if atime_side_effect is None else atime_side_effect
        # self.setup_mock_arrival_time(mock_arr_time)
        return get_dists_and_times(events_df, segments_df, *a, **kw)
 
    def tst_getset_dists_and_times(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        events_df = self.get_events_df(None)

        datacenters_df = self.get_datacenters_df(None)
        _sta_urlread_side_effect = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
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
        
        fdsn_sta_df, stats = self.get_fdsn_channels_df(_sta_urlread_side_effect,  events_df, datacenters_df,
                                               **dict(sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )

        
        channels_df = save_stations_and_channels(fdsn_sta_df, self.session)

        #
        # IMPORTANT: WE NEED A DETERMINISTIC WAY TO HANDLE ARRIVAL TIME CALCULATION, 
        # AS APPARENTLY BEING IN A PROCESSPOOLEXECUTOR LEADS TO UNEXPECTED ORDERS
        # (CF EG BY RUNNING IN ECLIPSE OR IN TERMINAL). SO:
        #
        def deterministic_atime_sideeffect(*a, **kw):
            evt_depth_km = a[1]
            if evt_depth_km == 60:
                return datetime.utcnow()
            raise TauModelError('wat?')
        
        segments_df =  self.getset_dists_and_times(deterministic_atime_sideeffect, events_df, channels_df,
                                                   [1,2], ['P', 'Q'],
                                                        'ak135')

        # If u want to know how many channels we expect to be good after arrival time calculation:
        evtids = events_df[events_df['depth_km'] == 60]['id'].values
        expected_good = len(channels_df[channels_df['event_id'].isin(evtids)])

        # as get_arrival_time raises errors, some segments are not written. Assert that the
        # total number of segments is lower than the total number of channels
        assert expected_good == len(segments_df)
         
        h = 9

# # =================================================================================================
# 
    def download_segments(self, url_read_side_effect, segments_df, *a, **kw) : # , ):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return download_segments(segments_df, *a, **kw)

    def tst_download_segments(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        events_df = self.get_events_df(None)

        datacenters_df = self.get_datacenters_df(None)
        _sta_urlread_side_effect = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
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
        
        fdsn_sta_df, stats = self.get_fdsn_channels_df(_sta_urlread_side_effect,  events_df, datacenters_df,
                                               **dict(sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )

        
        channels_df = save_stations_and_channels(fdsn_sta_df, self.session)

        #
        # IMPORTANT: WE NEED A DETERMINISTIC WAY TO HANDLE ARRIVAL TIME CALCULATION, 
        # AS APPARENTLY BEING IN A PROCESSPOOLEXECUTOR LEADS TO UNEXPECTED ORDERS
        # (CF EG BY RUNNING IN ECLIPSE OR IN TERMINAL). SO:
        #
        def deterministic_atime_sideeffect(*a, **kw):
            evt_depth_km = a[1]
            if evt_depth_km == 60:
                return datetime.utcnow()
            raise TauModelError('wat?')
        
        segments_df =  self.getset_dists_and_times(deterministic_atime_sideeffect, events_df, channels_df,
                                                   [1,2], ['P', 'Q'],
                                                        'ak135')

        # If u want to know how many channels we expect to be good after arrival time calculation:
        evtids = events_df[events_df['depth_km'] == 60]['id'].values
        expected_good = len(channels_df[channels_df['event_id'].isin(evtids)])

        # as get_arrival_time raises errors, some segments are not written. Assert that the
        # total number of segments is lower than the total number of channels
        segments_df = drop_already_downloaded(self.session, segments_df, True)
        # we do not have any already downloaded segment (we initialized the db for this function):
        assert len(segments_df) == expected_good
        segments_df = set_download_urls(segments_df, datacenters_df)
        _seg_urlread_sideeffect = [b'data','', '', URLError('wat'), socket.timeout()]
        segments_ok = int(np.true_divide(len(segments_df), len(_seg_urlread_sideeffect)))
        d_stats = self.download_segments(_seg_urlread_sideeffect, segments_df,max_error_count=5, max_thread_workers=5,
                                         timeout=40, download_blocksize=-1)
        
        assert np.array_equal(pd.isnull(segments_df['data']).values, [False, False, False, True, True])
        assert np.array_equal(((segments_df['data']=='') |
                               pd.isnull(segments_df['data'])).values, [False, True, True, True, True])



    def tst_save_segments(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        events_df = self.get_events_df(None)

        datacenters_df = self.get_datacenters_df(None)
        _sta_urlread_side_effect = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
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
        
        fdsn_sta_df, stats = self.get_fdsn_channels_df(_sta_urlread_side_effect,  events_df, datacenters_df,
                                               **dict(sradius_minmag=5, sradius_maxmag=6, 
                                                      sradius_minradius=7, 
                                                      sradius_maxradius=8,
                                                      station_timespan = [1,3],
                                                      channels= ['a', 'b', 'c'],
                                                      min_sample_rate = 100,
                                                      max_thread_workers=5,
                                                      timeout=10,
                                                      blocksize=5)
                                               )

        
        channels_df = save_stations_and_channels(fdsn_sta_df, self.session)

        #
        # IMPORTANT: WE NEED A DETERMINISTIC WAY TO HANDLE ARRIVAL TIME CALCULATION, 
        # AS APPARENTLY BEING IN A PROCESSPOOLEXECUTOR LEADS TO UNEXPECTED ORDERS
        # (CF EG BY RUNNING IN ECLIPSE OR IN TERMINAL). SO:
        #
        def deterministic_atime_sideeffect(*a, **kw):
            evt_depth_km = a[1]
            if evt_depth_km == 60:
                return datetime.utcnow()
            raise TauModelError('wat?')
        
        segments_df =  self.getset_dists_and_times(deterministic_atime_sideeffect, events_df, channels_df,
                                                   [1,2], ['P', 'Q'],
                                                        'ak135')

        # If u want to know how many channels we expect to be good after arrival time calculation:
        # not used here, but left in case it will be used
        evtids = events_df[events_df['depth_km'] == 60]['id'].values
        expected_good = len(channels_df[channels_df['event_id'].isin(evtids)])


        # as get_arrival_time raises errors, some segments are not written. Assert that the
        # total number of segments is lower than the total number of channels
        segments_df = drop_already_downloaded(self.session, segments_df, True)
        segments_df = set_download_urls(segments_df, datacenters_df)
        _seg_urlread_sideeffect = [b'data','', '', URLError('wat'), socket.timeout()]
        segments_ok = int(np.true_divide(len(segments_df), len(_seg_urlread_sideeffect)))
        d_stats = self.download_segments(_seg_urlread_sideeffect, segments_df,max_error_count=5, max_thread_workers=5,
                                         timeout=40, download_blocksize=-1)
        
        save_segments(self.session, segments_df, self.run.id, sync_session_on_update=False)
        
        # we expect not all segments are written (mock side_effect set it self.download_segments
        # returns errors and emtpy)
        numsegmentssaved = len(self.session.query(Segment).all())
        assert numsegmentssaved == 5
        num_segmentssaved_withdata = len(self.session.query(Segment).filter(withdata(Segment.data)).all())
        assert num_segmentssaved_withdata == 1
        assert numsegmentssaved == len(self.session.query(Segment).filter(Segment.run_id == self.run.id).all())
        
        # try to see what happens now. We should have unique constraint failure, so nothing is
        # saved. 
        save_segments(self.session, segments_df, self.run.id, sync_session_on_update=False)
        # we expect the same things as above
        numsegmentssaved2 = len(self.session.query(Segment).all())
        assert numsegmentssaved == numsegmentssaved2
        num_segmentssaved_withdata2 = len(self.session.query(Segment).filter(withdata(Segment.data)).all())
        assert num_segmentssaved_withdata2 == num_segmentssaved_withdata
        assert numsegmentssaved == len(self.session.query(Segment).filter(Segment.run_id == self.run.id).all())
        
        # change one thing: the data on the second try to see what happens now:
        # first get the id of the None case:
        seg = self.session.query(Segment).filter(Segment.data == None).first()
        segments_df.loc[pd.isnull(segments_df['data']), 'id'] = seg.id
        segments_df.loc[pd.isnull(segments_df['data']), 'data'] = 'wat'
        save_segments(self.session, segments_df, self.run.id, sync_session_on_update=False)
        # we expect the same segments on db
        numsegmentssaved2 = len(self.session.query(Segment).all())
        assert numsegmentssaved2 == numsegmentssaved
        # BUT we expect one segment more has been saved:
        num_segmentssaved_withdata2 = len(self.session.query(Segment).filter(withdata(Segment.data)).all())
        assert num_segmentssaved_withdata2 == num_segmentssaved_withdata + 1

#  DO NOT TEST MAX ERR: WE WILL REMOVE IT IN THE FUTURE         
# 
#     def test_download_segments_max_err(self):
#         events = self.get_events(None, self.session,
#                                "eventws", )  # "minmag", "minlat", "maxlat", "minlon", "maxlon", "startiso", "endiso")
#         datacenters = self.get_datacenters(session=self.session)
#         evt2stations, stats = self.make_ev2sta(None,  # use self._seg_urlread_side_effect
#                                                **dict(session=self.session,
#                                                       events = events,
#                                                       datacenters = datacenters,
#                                                       sradius_minmag=5, sradius_maxmag=6, 
#                                                       sradius_minradius=7, 
#                                                       sradius_maxradius=8,
#                                                       station_timespan = [1,3],
#                                                       channels= ['a', 'b', 'c'],
#                                                       min_sample_rate = 100,
#                                                       max_thread_workers=5,
#                                                       timeout=10,
#                                                       blocksize=5)
#                                                )
#         # keep only first event. This prevents to have duplicated stations and we get some value
#         # in dc2segments below (how is that possible we should check...)
#         evt2stations = {evt2stations.keys()[0] : evt2stations.values()[0]}
#         
#         segments_df, skipped_already_d =  self.get_segments_df(None,  # None: atime_side_effect is default 
#                                                            self.session, events,
#                                                            datacenters, evt2stations, [1,2], ['P', 'Q'],
#                                                            'ak135', False)
#         
#         stats = self.download_segments([URLError('w')], # raise always exception is url read
#                                                     self.session, segments_df, self.run.id,
#                                                     max_error_count=1, max_thread_workers=5,
#                                                     timeout=40, download_blocksize=-1)
#    
#         urlerror_found=0
#         discarded_after_1_trial=0
#         for k, v in stats.iteritems():
#             if v and not  (urlerror_found and discarded_after_1_trial):
#                 urlerror_found = 0
#                 discarded_after_1_trial = 0
#                 for a, b in v.iteritems():
#                     if a.startswith('URLError:'):
#                         urlerror_found+=1
#                     elif " after 1 previous errors" in a:
#                         discarded_after_1_trial+=1
#                     if urlerror_found and discarded_after_1_trial:
#                         break
#                     
#         assert urlerror_found and discarded_after_1_trial
#         
# 
#     @patch('stream2segment.download.query.get_events')
#     @patch('stream2segment.download.query.get_datacenters')
#     @patch('stream2segment.download.query.make_ev2sta')
#     @patch('stream2segment.download.query.get_segments_df')
#     @patch('stream2segment.download.query.download_segments')
#     def test_cmdline(self, mock_download_segments, mock_get_seg_df, mock_make_ev2sta,
#                      mock_get_datacenter, mock_get_events):
#         
#         ddd = datetime.utcnow()
#         # setup arrival time side effect with a constant date (by default is datetime.utcnow)
#         # we might pass it as custom argument below actually
#         self._atime_sideeffect[0] = ddd
# 
#         def dsegs(*a, **v):
#             return self.download_segments(None, *a, **v)
#         mock_download_segments.side_effect = dsegs
#         
#         def segdf(*a, **v):
#             return self.get_segments_df(None, *a, **v)
#         mock_get_seg_df.side_effect = segdf
# 
#         def ev2sta(*a, **v):
#             return self.make_ev2sta(None, *a, **v)
#         mock_make_ev2sta.side_effect = ev2sta
# 
#         def getdc(*a, **v):
#             return self.get_datacenters(None, *a, **v)
#         mock_get_datacenter.side_effect = getdc
# 
#         def getev(*a, **v):
#             return self.get_events(None, *a, **v)
#         mock_get_events.side_effect = getev
#         
#         
#         
#         # prevlen = len(self.session.query(Segment).all())
#     
#         runner = CliRunner()
#         result = runner.invoke(main , ['d', '--dburl', self.dburi,
#                                        '--start', '2016-05-08T00:00:00',
#                                        '--end', '2016-05-08T9:00:00'])
#         if result.exception:
#             import traceback
#             traceback.print_exception(*result.exc_info)
#             print result.output
#             assert False
#             return
#             
#         segments = self.session.query(Segment).all()
#         assert len(segments) == 2
#         assert segments[0].data == b'data'
#         assert not segments[1].data
#         emptySegId = segments[1].id
#         
#         # re-launch with the same setups.
#         # what we want to test is the addition of an already downloaded segment which was empty
#         # before and now is not. So:
#         newdata = b'dataABC'
#         self._seg_urlread_sideeffect = [newdata, '']  # empty STRING at end otherwise urlread has infinite loop!
#         
#         # relaunch with 'd' (empty segments no retry):
#         runner = CliRunner()
#         result = runner.invoke(main , ['d', '--dburl', self.dburi,
#                                        '--start', '2016-05-08T00:00:00',
#                                        '--end', '2016-05-08T9:00:00'])
#         if result.exception:
#             import traceback
#             traceback.print_exception(*result.exc_info)
#             print result.output
#             assert False
#             return
#         # test that nothing happened:
#         segments = self.session.query(Segment).all()
#         assert segments[1].id == emptySegId and not segments[1].data
#         run_id = segments[1].run_id
#         
#         # relaunch (retry empty or Null segments data)
#         runner = CliRunner()
#         result = runner.invoke(main , ['d', '--dburl', self.dburi, '--retry',
#                                        '--start', '2016-05-08T00:00:00',
#                                        '--end', '2016-05-08T9:00:00'])
#         if result.exception:
#             import traceback
#             traceback.print_exception(*result.exc_info)
#             print result.output
#             assert False
#             return
#         # test that now empty segment has new data:
#         segments = self.session.query(Segment).all()
#         assert segments[1].id == emptySegId and segments[1].data == newdata
#         # assert run_id changed:
#         assert segments[1].run_id != run_id
# 
# 
#     @patch('stream2segment.download.query.get_events')
#     @patch('stream2segment.download.query.get_datacenters')
#     @patch('stream2segment.download.query.make_ev2sta')
#     @patch('stream2segment.download.query.get_segments_df')
#     @patch('stream2segment.download.query.download_segments')
#     def test_cmdline_singleevent_singledatacenter(self, mock_download_segments,
#                                                   mock_get_seg_df, mock_make_ev2sta,
#                      mock_get_datacenter, mock_get_events):
#         
#         def dsegs(*a, **v):
#             return self.download_segments(None, *a, **v)
#         mock_download_segments.side_effect = dsegs
#         
#         def segdf(*a, **v):
#             return self.get_segments_df(None, *a, **v)
#         mock_get_seg_df.side_effect = segdf
# 
#         def ev2sta(*a, **v):
#             return self.make_ev2sta(None, *a, **v)
#         mock_make_ev2sta.side_effect = ev2sta
# 
#         def getdc(*a, **v):
#             # return only one datacenter
#             return self.get_datacenters(["""http://ws.resif.fr/fdsnws/station/1/query""", ""], *a, **v)
#         mock_get_datacenter.side_effect = getdc
# 
#         def getev(*a, **v):
#             # return only one event
#             url_read_side_effect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
# 20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN)""", ""]
#             _ =  self.get_events(url_read_side_effect, *a, **v)
#             return _
#         mock_get_events.side_effect = getev
# 
#         ddd = datetime.utcnow()
#         # setup arrival time side effect
#         self._atime_sideeffect = cycle([ddd, TauModelError('wat?'), ValueError('wat?')])
#         
#         # prevlen = len(self.session.query(Segment).all())
#     
#         runner = CliRunner()
#         result = runner.invoke(main , ['d', '--dburl', self.dburi,
#                                        '--start', '2016-05-08T00:00:00',
#                                        '--end', '2016-05-08T9:00:00'])
#         if result.exception:
#             import traceback
#             traceback.print_exception(*result.exc_info)
#             print result.output
#             assert False
#             return
#             
#         segments = self.session.query(Segment).all()
#         assert len(segments) == 0
# 
# 
# 
#     @patch('stream2segment.download.query.make_ev2sta')
#     @patch('stream2segment.download.query.get_events')
#     @patch('stream2segment.download.query.get_datacenters')
#     def test_cmdline_datacenter_query_error(self, mock_get_datacenter, mock_get_events,
#                                             mock_make_ev2sta):
#         
#         def getev(*a, **v):
#             # return only one event (assure event does not raise errors, we want to test the datacenters)
#             url_read_side_effect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
# 20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN)""", ""]
#             _ =  self.get_events(url_read_side_effect, *a, **v)
#             return _
#         mock_get_events.side_effect = getev
#         
#         
#         def getdc(*a, **v):
#             return self.get_datacenters([URLError('oops')], *a, **v)
#         mock_get_datacenter.side_effect = getdc
# 
#         def ev2sta(*a, **v):
#             # whatever is ok, as we will test to NOT have called this function!
#             return self.make_ev2sta(None, *a, **v)
#         mock_make_ev2sta.side_effect = ev2sta
#         # prevlen = len(self.session.query(Segment).all())
#     
#         runner = CliRunner()
#         result = runner.invoke(main , ['d', '--dburl', self.dburi,
#                                        '--start', '2016-05-08T00:00:00',
#                                        '--end', '2016-05-08T9:00:00'])
#         if result.exception:
#             import traceback
#             traceback.print_exception(*result.exc_info)
#             print result.output
#             assert False
#             return
#         
#         assert not mock_make_ev2sta.called
#         segments = self.session.query(Segment).all()
#         assert len(segments) == 0
#     
#     @patch('stream2segment.download.query.make_ev2sta')
#     @patch('stream2segment.download.query.get_events')
#     @patch('stream2segment.download.query.get_datacenters')
#     def test_cmdline_events_query_error(self, mock_get_datacenter, mock_get_events,
#                                             mock_make_ev2sta):
#         
#         def getev(*a, **v):
#             # return only one event (assure event does not raise errors, we want to test the datacenters)
#             url_read_side_effect = [URLError('oop')]
#             _ =  self.get_events(url_read_side_effect, *a, **v)
#             return _
#         mock_get_events.side_effect = getev
#         
#         
#         def getdc(*a, **v):
#             return self.get_datacenters([URLError('oops')], *a, **v)
#         mock_get_datacenter.side_effect = getdc
# 
#         def ev2sta(*a, **v):
#             # whatever is ok, as we will test to NOT have called this function!
#             return self.make_ev2sta(None, *a, **v)
#         mock_make_ev2sta.side_effect = ev2sta
#         # prevlen = len(self.session.query(Segment).all())
#     
#         runner = CliRunner()
#         result = runner.invoke(main , ['d', '--dburl', self.dburi,
#                                        '--start', '2016-05-08T00:00:00',
#                                        '--end', '2016-05-08T9:00:00'])
#         if result.exception:
#             import traceback
#             traceback.print_exception(*result.exc_info)
#             print result.output
#             assert False
#             return
#         
#         
#         # FIXME: write something here, like we warned that we had some problems to the screen or whatever

