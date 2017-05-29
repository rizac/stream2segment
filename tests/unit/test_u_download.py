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
from stream2segment.io.db.models import Base, Event, Class, WebService
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from stream2segment.main import main, closing
from click.testing import CliRunner
# from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_col_names
import pandas as pd
from stream2segment.download.main import add_classes, get_events_df, get_datacenters_df, logger as query_logger, \
get_channels_df, merge_events_stations, set_saved_arrivaltimes, get_arrivaltimes,\
    prepare_for_download, download_save_segments, _strcat, get_eventws_url,\
    QuitDownload
# ,\
#     get_fdsn_channels_df, save_stations_and_channels, get_dists_and_times, set_saved_dist_and_times,\
#     download_segments, drop_already_downloaded, set_download_urls, save_segments
from obspy.core.stream import Stream, read
from stream2segment.io.db.models import DataCenter, Segment, Run, Station, Channel, WebService
from itertools import cycle, repeat, count, product, izip
from urllib2 import URLError
import socket
from obspy.taup.helper_classes import TauModelError
# import logging
# from logging import StreamHandler
import sys
# from stream2segment.main import logger as main_logger
from sqlalchemy.sql.expression import func
from stream2segment.utils import get_session, mseedlite3, yaml_load
from stream2segment.io.db.pd_sql_utils import withdata, dbquery2df, insertdf_napkeys, updatedf
from logging import StreamHandler
import logging
from _io import BytesIO
import urllib2
from stream2segment.download.utils import get_url_mseed_errorcodes
from test.test_userdict import d1
from stream2segment.utils.mseedlite3 import MSeedError, unpack
import threading
from stream2segment.utils.url import read_async
from stream2segment.utils.resources import get_templates_fpath


# when debugging, I want the full dataframe with to_string(), not truncated
pd.set_option('display.max_colwidth', -1)

# hard-coding the responses messages here:
responses = {
    100: ('Continue', 'Request received, please continue'),
    101: ('Switching Protocols',
          'Switching to new protocol; obey Upgrade header'),

    200: ('OK', 'Request fulfilled, document follows'),
    201: ('Created', 'Document created, URL follows'),
    202: ('Accepted',
          'Request accepted, processing continues off-line'),
    203: ('Non-Authoritative Information', 'Request fulfilled from cache'),
    204: ('No Content', 'Request fulfilled, nothing follows'),
    205: ('Reset Content', 'Clear input form for further input.'),
    206: ('Partial Content', 'Partial content follows.'),

    300: ('Multiple Choices',
          'Object has several resources -- see URI list'),
    301: ('Moved Permanently', 'Object moved permanently -- see URI list'),
    302: ('Found', 'Object moved temporarily -- see URI list'),
    303: ('See Other', 'Object moved -- see Method and URL list'),
    304: ('Not Modified',
          'Document has not changed since given time'),
    305: ('Use Proxy',
          'You must use proxy specified in Location to access this '
          'resource.'),
    307: ('Temporary Redirect',
          'Object moved temporarily -- see URI list'),

    400: ('Bad Request',
          'Bad request syntax or unsupported method'),
    401: ('Unauthorized',
          'No permission -- see authorization schemes'),
    402: ('Payment Required',
          'No payment -- see charging schemes'),
    403: ('Forbidden',
          'Request forbidden -- authorization will not help'),
    404: ('Not Found', 'Nothing matches the given URI'),
    405: ('Method Not Allowed',
          'Specified method is invalid for this server.'),
    406: ('Not Acceptable', 'URI not available in preferred format.'),
    407: ('Proxy Authentication Required', 'You must authenticate with '
          'this proxy before proceeding.'),
    408: ('Request Timeout', 'Request timed out; try again later.'),
    409: ('Conflict', 'Request conflict.'),
    410: ('Gone',
          'URI no longer exists and has been permanently removed.'),
    411: ('Length Required', 'Client must specify Content-Length.'),
    412: ('Precondition Failed', 'Precondition in headers is false.'),
    413: ('Request Entity Too Large', 'Entity is too large.'),
    414: ('Request-URI Too Long', 'URI is too long.'),
    415: ('Unsupported Media Type', 'Entity body in unsupported format.'),
    416: ('Requested Range Not Satisfiable',
          'Cannot satisfy request range.'),
    417: ('Expectation Failed',
          'Expect condition could not be satisfied.'),

    500: ('Internal Server Error', 'Server got itself in trouble'),
    501: ('Not Implemented',
          'Server does not support this operation'),
    502: ('Bad Gateway', 'Invalid responses from another server/proxy.'),
    503: ('Service Unavailable',
          'The server cannot process the request due to a high load'),
    504: ('Gateway Timeout',
          'The gateway server did not receive a timely response'),
    505: ('HTTP Version Not Supported', 'Cannot fulfill request.'),
    }

class Test(unittest.TestCase):

    @staticmethod
    def cleanup(session, handler, *patchers):
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
        
        hndls = query_logger.handlers[:]
        handler.close()
        for h in hndls:
            if h is handler:
                query_logger.removeHandler(h)

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
        
        
        # mock threadpoolexecutor to run one instance at a time, so we get deterministic results:
        self.patcher23 = patch('stream2segment.download.main.read_async')
        self.mock_read_async = self.patcher23.start()
#         def readasync(iterable, ondone, *a, **v):
#             ret = list(iterable)
#             ondones = [None] * len(ret)
#             def _ondone(*a_):
#                 ondones[ret.index(a_[0])] = a_
#             
#             read_async(ret, _ondone, *a, **v)
#             
#             for k in ondones:
#                 ondone(*k)
        def readasync(iterable, *a, **v):
            # make readasync deterministic by returning the order of iterable
            ret = list(iterable)
            ondones = [None] * len(ret)
            
            for a_ in read_async(ret, *a, **v):
                ondones[ret.index(a_[0])] = a_

            for k in ondones:
                yield k

        self.mock_read_async.side_effect = readasync
        
        
        self.logout = StringIO()
        self.handler = StreamHandler(stream=self.logout)
        # THIS IS A HACK:
        query_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset (to level.INFO)
        # otherwise it stays what we set two lines above. Problems might arise if closing
        # sets a different level, but for the moment who cares
        
        query_logger.addHandler(self.handler)
        
        # MOCK ARRIVAL_TIME. REMEMBER: WITH PROCESSPOOLEXECUTOR DO NOT MOCK DIRECTLY THE FUNCTION PASSED
        # AS AS_COMPLETED, BUT A SUB FUNCTION. THIS IS PROBABLY DUE TO THE FACT THAT MOCKS ARE
        # NOT PICKABLE (SUB FUNCTIONS APPARENTLY DO NOT SUFFER NON PICKABILITY)
        
        self.patcher3 = patch('stream2segment.download.utils.get_min_travel_time')
        self.mock_min_travel_time = self.patcher3.start()
        
        self.patchers = [self.patcher, self.patcher1, self.patcher2, self.patcher3, self.patcher23]
        #self.patcher3 = patch('stream2segment.main.logger')
        #self.mock_main_logger = self.patcher3.start()
        
        # setup a run_id:
        r = Run()
        self.session.add(r)
        self.session.commit()
        self.run = r

        # side effects:
        
        self._evt_urlread_sideeffect =  """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|90|90|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        self._dc_urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

"""

# Note: by default we set sta_urlsideeffect to return such a channels which result in 12
# segments (see lat and lon of channels vs lat and lon of events above)
        self._sta_urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
GE|FLT1||HHE|1|1|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-01-01T00:00:00|
GE|FLT1||HHN|1|1|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-01-01T00:00:00|
GE|FLT1||HHZ|1|1|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-01-01T00:00:00|
n1|s||c1|90|90|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
n1|s||c2|90|90|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
n1|s||c3|90|90|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
IA|BAKI||BHE|1|1|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-01-01T00:00:00|
IA|BAKI||BHN|1|1|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-01-01T00:00:00|
IA|BAKI||BHZ|1|1|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-01-01T00:00:00|
n2|s||c1|90|90|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
n2|s||c2|90|90|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
n2|s||c3|90|90|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
"""]
        # self._sta_urlread_sideeffect = cycle([partial_valid, '', invalid, '', '', URLError('wat'), socket.timeout()])

        self._mintraveltime_sideeffect = cycle([1])

        _file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GE.FLT1..HH?.mseed")
        with open(_file, "rb") as _opn:
            self._seg_data = _opn.read()
        
        _file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "IA.BAKI..BHZ.D.2016.004.head")
        with open(_file, "rb") as _opn:
            self._seg_data_gaps = _opn.read()
            
        self._seg_data_empty = b''
            
        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_gaps, 413, 500, self._seg_data[:2],
                                        self._seg_data_empty,  413, URLError("++urlerror++"),
                                        socket.timeout()]

        self.service = ''  # so get_datacenters_df accepts any row by default

        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self.session, self.handler, *self.patchers)
                        #self.patcher3)
                        
        self.db_buf_size = 1
        
        self.routing_service = yaml_load(get_templates_fpath("download.yaml"))['advanced_settings']['routing_service_url']

    def log_msg(self):
        return self.logout.getvalue()
    
    def setup_urlopen(self, urlread_side_effect):
        """setup urlopen return value. 
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.read, that will be converted
        to an itertools.cycle(side_effect) REMEMBER that any element of urlread_side_effect which is a nonempty
        string must be followed by an EMPTY
        STRINGS TO STOP reading otherwise we fall into an infinite loop if the argument
        blocksize of url read is not negative !"""
#         self.mock_urlopen.reset_mock()
#         a = Mock()
#         # convert returned values to the given urlread return value (tuple data, code, msg)
#         # if k is an int, convert to an HTTPError
#         retvals = []
#         for k in urlread_side_effect:
#             if type(k) == int:
#                 retvals = (None, k, responses(k))
#             elif type(k) == str:
#                 retvals = (k, 200, 'OK')
#             else:
#                 retvals = k
#                 
#         a.read.side_effect =  cycle(retvals)
#         self.mock_urlread = a.read
#         self.mock_urlopen.return_value = a
#         
        self.mock_urlopen.reset_mock()
        # convert returned values to the given urlread return value (tuple data, code, msg)
        # if k is an int, convert to an HTTPError
        retvals = []
        if type(urlread_side_effect) == str or not hasattr(urlread_side_effect, "__iter__"):
            urlread_side_effect = [urlread_side_effect]

            
        for k in urlread_side_effect:
            a = Mock()
            if type(k) == int:
                a.read.side_effect = urllib2.HTTPError('url', int(k),  responses[k][0], None, None)
            elif type(k) == str:
                def func(k):
                    b = BytesIO(k)
                    def rse(*a, **v):
                        rewind = not a and not v
                        if not rewind:
                            currpos = b.tell()
                        ret = b.read(*a, **v)
                        # hacky workaround to support cycle below: if reached the end, go back to start
                        if not rewind:
                            cp = b.tell()
                            rewind = cp == currpos
                        if rewind:
                            b.seek(0, 0)
                        return ret
                    return rse
                a.read.side_effect = func(k)
                a.code = 200
                a.msg = responses[a.code][0]
            else:
                a.read.side_effect = k
            retvals.append(a)
#         
        self.mock_urlopen.side_effect = cycle(retvals)
#        self.mock_urlopen.side_effect = Cycler(urlread_side_effect)
        

    def test_add_classes(self):
        cls = {'a' : 'bla', 'b' :'c'}
        add_classes(self.session, cls, db_bufsize=self.db_buf_size)
        assert len(self.session.query(Class).all()) == 2
        logmsg = self.log_msg()
        assert "Writing to database table 'classes': 2 of 2 new items saved" in logmsg
        add_classes(self.session, cls, db_bufsize=self.db_buf_size)
        # assert we did not write any new class
        assert len(self.session.query(Class).all()) == 2
        # and also that nothing is printed to log (we print only new and discarded)
        assert logmsg == self.log_msg()
        
# ===========================

    
    @patch('stream2segment.download.main.yaml_load', return_value={'service1': {'event': 'http:event1'}, 'service2': {'event': 'http:event2', 'station': 'http:station2'}})
    def test_get_eventws_url(self, mock_yaml_load):
        with pytest.raises(ValueError):
            url = get_eventws_url(self.session, "eida")
        
        url = get_eventws_url(self.session, "service1")
        assert url == 'http:event1'
        url = get_eventws_url(self.session, "service2")
        assert url == 'http:event2'

    def get_events_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
#         if not eventws_url:
#             ptch = patch('stream2segment.download.main.yaml_load', return_value={'': {'event': 'http:event1'}})
#             eventws_url = get_eventws_url(self.session, "")
#             ptch.stop()
        return get_events_df(*a, **v)
        

    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_events(self, mock_query):
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""]
        
        
        data = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == len(pd.unique(data['id'])) == 3
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        assert len(data) == 3
        # assert mock_urlread.call_args[0] == (mock_query.return_value, )
        
        # now download again, with an url error:
        
        urlread_sideeffect = [413, """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", URLError('blabla23___')]
        
        data = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        # assert nothing new has added:
        assert len(self.session.query(Event).all()) == len(pd.unique(data['id'])) == 3
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        assert len(data) == 3
        # assert mock_urlread.call_args[0] == (mock_query.return_value, )
        
        assert "blabla23___" in self.log_msg()
        

    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_events_toomany_requests_raises(self, mock_query): # FIXME: implement it!
        
        urlread_sideeffect = [413, """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""]
        # as urlread returns alternatively a 413 and a good string, also sub-queries
        # will return that, so that we will end up having a 413 when the string is not
        # further splittable:
        with pytest.raises(QuitDownload):
            data = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", self.db_buf_size,
                                      start=datetime(2010,1,1).isoformat(),
                                      end=datetime(2011,1,1).isoformat())
        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == 0
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        with pytest.raises(NameError):
            assert len(data) == 3

        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == 0
        

#     @patch('stream2segment.download.main.urljoin', return_value='a')
#     @patch('stream2segment.download.main.dbsync')
#     def test_get_events_eventws_not_saved(self, mock_dbsync, mock_query): # FIXME: implement it!
#         urlread_sideeffect = [413]  # this is useless, we test stuff which raises before it
#         
#         # now we want to return all times 413, and see that we raise a ValueError:
#         
#         mock_dbsync.reset_mock()
#         mock_dbsync.side_effect = lambda *a, **v: dbsync(*a, **v)
    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_events_eventws_not_saved(self, mock_query): # FIXME: implement it!
        urlread_sideeffect = [413]  # this is useless, we test stuff which raises before it

        # we want to return all times 413, and see that we raise a ValueError:
        with pytest.raises(QuitDownload):
            # now it should raise because of a 413:
            data = self.get_events_df(urlread_sideeffect, self.session, "abcd", self.db_buf_size,
                                      start=datetime(2010,1,1).isoformat(),
                                         end=datetime(2011,1,1).isoformat())
            
        # assert we wrote the url
        assert len(self.session.query(WebService.url).filter(WebService.url=='abcd').all()) == 1
        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == 0
        # we cannot assert anything has been written to logger cause the exception are caucht
        # if we raun from main. This should be checked in functional tests where we test the whole
        # chain
        # assert "request entity too large" in self.log_msg()

# =================================================================================================

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_datacenters_df(*a, **v)
    

    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs_malformed(self, mock_query):
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query

ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session,
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        # geofon has actually a post line since 'indentation is bad..' is splittable)
        assert post_data_list[0] is None and post_data_list[1] is not None and \
            '\n' in post_data_list[1]
        
    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs2(self, mock_query):
        urlread_sideeffect = ["""http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query

ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
"""]
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service=None,
                                                       channels=['BH?'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        assert post_data_list[0] is not None and post_data_list[1] is None
        
        # now download with a channel matching:
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service=None,
                                                       channels=['H??'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        assert mock_query.call_count == 2  # we might be more fine grained, see code
        assert post_data_list[0] is not None and post_data_list[1] is not None
        # assert we have only one line for each post request:
        assert all('\n' not in r for r in post_data_list)


    @patch('stream2segment.download.main.get_dc_filterfunc')
    def test_get_dcs_service(self, mock_dc_filter):
        
        def func(service):
            if service == 'geofon':
                return lambda x: "geofon" in x
            elif service == 'resif':
                return lambda x: "resif" in x
            elif service == 'iris':
                return lambda x: "iris.edu" in x
            elif service == 'eida':
                return lambda x: "iris.edu" not in x
            return lambda x: True
        
        urlread_sideeffect = ["""http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25

http://ws.resif.fr/fdsnws/dataselect/1/query

ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

http://ws.iris.edu.org/fdsnws/dataselect/1/query

A * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
B * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
C * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
"""]
        mock_dc_filter.side_effect = func

        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service='', channels=['BH?'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 3
        assert post_data_list[0] is not None and all(post_data_list[i] is None for i in [1,2])
        
        self.session.query(DataCenter).delete()
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service="geofon", channels=None, db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 1
        assert post_data_list[0] is not None
        
        self.session.query(DataCenter).delete()
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service="resif", channels=None, db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 1
        assert post_data_list[0] is not None
        
        self.session.query(DataCenter).delete()
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service="iris", channels=None, db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 1
        assert post_data_list[0] is not None
        
        self.session.query(DataCenter).delete()
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service="eida", channels=None, db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        assert len(post_data_list) == 2 and all(p is not None for p in post_data_list) and len(post_data_list[0].split("\n")) ==2
        
        self.session.query(DataCenter).delete()
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                       service="eida", channels=['BH?'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        assert len(post_data_list) == 2
        assert len(post_data_list[0].split("\n")) == 2
        assert post_data_list[1] is None
        
        # test that if the routing service changes the station
        
        
    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs3(self, mock_query):
        urlread_sideeffect = [500, """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

""", 501]
        
        with pytest.raises(QuitDownload):
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect[0], self.session, self.routing_service,
                                                       service=None, channels=['BH?'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == 0
        mock_query.assert_called_once()  # we might be more fine grained, see code
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[1], self.session, self.routing_service,
                                                       service=None, channels=['BH?'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.call_count == 2  # we might be more fine grained, see code
        assert post_data_list[0] is not None and post_data_list[1] is None
        
        # this raises again a server error, but we have datacenters in cahce:
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[2], self.session, self.routing_service,
                                                       service=None, channels=['BH?'], db_bufsize=self.db_buf_size)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.call_count == 3  # we might be more fine grained, see code
        assert post_data_list is None
        
    
    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs_postdata_all_nones(self, mock_query):
        urlread_sideeffect = ["""http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
http://ws.resif.fr/fdsnws/dataselect/1/query
"""]
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[0], self.session, self.routing_service,
                                                       service=None, channels=['BH?'], db_bufsize=self.db_buf_size)
        assert all(x is None for x in post_data_list)
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[0], self.session, self.routing_service,
                                                       service=None, channels=None, db_bufsize=self.db_buf_size)
        assert all(x is None for x in post_data_list)

# =================================================================================================



    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_channels_df(*a, **kw)
     
    def test_get_channels_df(self):
        urlread_sideeffect = """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)

        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

"""
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                           service=None,
                                                           channels=channels, db_bufsize=self.db_buf_size)
        
        # url read for channels: Note: first response data raises, second has an error and
        #that error is skipped (other channels are added), and last two channels are from two
        # stations (BLA|BLA|...) with only different start time (thus stations should both be added)
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
--- ERROR --- MALFORMED|12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", 
# NOTE THAT THE CHANNELS ABOVE WILL BE OVERRIDDEN BY THE ONES BELOW (MULTIPLE NAMES< WE SHOULD NOT HAVE
# THIS CASE WITH THE EDIAWS ROUTING SERVICE BUT WE TEST HERE THE CASE)
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
HT|AGG||HHE|--- ERROR --- NONNUMERIC |22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|70.0|2008-02-12T00:00:00|
HT|AGG||HHE|95.6|22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
HT|LKD2||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|2019-01-01T00:00:00
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
"""]
                                      
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1, self.db_buf_size
                                               )
         
        # we should have called mock_urlopen_in_async times the datacenters
        assert self.mock_urlopen.call_count == len(datacenters_df)
 
        assert len(self.session.query(Station.id).all()) == 4
        assert len(self.session.query(Channel.id).all()) == 6
        
        # assert all good channels and stations have the id of the second datacenter
        id = datacenters_df.iloc[1].id
        assert all(sid[0] == id for sid in self.session.query(Station.datacenter_id).all())
        # assert channels are referring to those stations:
        sta_ids = [x[0] for x in self.session.query(Station.id).all()]
        assert all(c_staid[0] in sta_ids for c_staid in self.session.query(Channel.station_id).all())
        
        # now mock a datacenter null postdata (error in eida routing service)
        # and test that by querying the database we get the same data (the one we just saved)
        cha_df2 = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       None,
                                                       channels,
                                                       100, 'a', 'b', -1, self.db_buf_size
                                               )

        assert len(cha_df2) == len(cha_df)
        assert sorted(list(cha_df.columns)) == sorted(list(cha_df2.columns))
        
        # now mock a datacenter null postdata in the second item
        # (<==> no data in eida routing service under a specific datacenter)
        # and test that by querying the database we get the data we just saved.
        # NB: the line below raises cause the first datacenter has no channels to use
        # (response data malformed), therefore,
        # since the second datacenter is discarded, we won't have any data due to 
        # client server error. This is a download error that must raise
        # (in the main download program flow, it will be caught inside download.main.run.py)
        with pytest.raises(QuitDownload):
            cha_df2 = self.get_channels_df(urlread_sideeffect, self.session,
                                                           datacenters_df,
                                                           ['x', None],
                                                           channels,
                                                           100, 'a', 'b', -1, self.db_buf_size
                                                   )
        assert 'discarding response data' in self.log_msg()

        # now test the opposite, but note that urlreadside effect should return now an urlerror and a socket error:
        with pytest.raises(QuitDownload):
            cha_df2 = self.get_channels_df(URLError('urlerror_wat'), self.session,
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels,
                                                           100, 'a', 'b', -1, self.db_buf_size
                                                   )
        assert 'urlerror_wat' in self.log_msg()

        # now test again, we should ahve a socket timeout
        with pytest.raises(QuitDownload):
            cha_df2 = self.get_channels_df(socket.timeout(), self.session,
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels,
                                                           100, 'a', 'b', -1, self.db_buf_size
                                                   )
        assert 'timeout' in self.log_msg()
        
        # now the case where the none post request is for the "good" repsonse data:
        # Basically, replace ['x', None] with [None, 'x'] and swap urlread side effects
        cha_df2 = self.get_channels_df([urlread_sideeffect[1], urlread_sideeffect[0]],
                                       self.session,
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels,
                                                           100, 'a', 'b', -1, self.db_buf_size
                                                   )
        
        assert len(cha_df2) == 6
        # now change min sampling rate and see what happens (we do not have one channel more
        # cause we have overlapping names, and the 50 Hz channel is overridden by the second
        # query) 
        cha_df3 = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1, self.db_buf_size
                                               )
        assert len(cha_df3) == len(cha_df)
        
        # now change this:
        
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|B||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
E|F||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  URLError('wat'), socket.timeout()]
                                      
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1, self.db_buf_size
                                               )

        assert len(cha_df) == 1
        assert "sample rate <" in self.log_msg()
        
        # now decrease the sampling rate, we should have two channels (all):
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1, self.db_buf_size
                                               )

        assert len(cha_df) == 2
        
        # now change channels=['B??'], we should have no effect as the arg has effect
        # when postdata is None (=query to the db)
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       ['B??'],
                                                       10, 'a', 'b', -1, self.db_buf_size
                                               )

        assert len(cha_df) == 2
        
        # now change channels=['B??'], we should have an effect (QuitDownload) as the arg has effect
        # because postdata is now None (=query to the db)
        # QuitDownload is an exception raised in download module to say we cannot continue
        with pytest.raises(QuitDownload):
            cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                               datacenters_df,
                                                               None,
                                                               ['B??'],
                                                               10, 'a', 'b', -1, self.db_buf_size
                                                       )
        assert "Getting already-saved stations and channels from db" in self.log_msg()
        
        # same as above (no channels found) but use a very high sample rate:
        with pytest.raises(QuitDownload):
            cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                               datacenters_df,
                                                               postdata,
                                                               ['B??'],
                                                               1000000000, 'a', 'b', -1, self.db_buf_size
                                                       )

        # assert cha_df.empty
        assert "discarding %d channels (sample rate < 1000000000 Hz)" % len(cha_df) in self.log_msg()
        
        # this on the other hand must raise cause we get no data from the server
        with pytest.raises(QuitDownload):
            cha_df = self.get_channels_df("", self.session,
                                                               datacenters_df,
                                                               postdata,
                                                               ['B??'],
                                                               10, 'a', 'b', -1, self.db_buf_size
                                                       )

        
        

# FIXME: text save inventories!!!!

    def test_merge_event_stations(self):
        # get events with lat lon (1,1), (2,2,) ... (n, n)
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|2|2|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)

        # this urlread_sideeffect is actually to be considered for deciding which datacenters to store,
        # their post data is not specified as it
        # would be ineffective as it is overridden by the urlread_sideeffect
        # specified below for the channels
        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
http://ws.resif.fr/fdsnws/dataselect/1/query
"""
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(None, self.session, self.routing_service,
                                                           service=None, channels=channels, db_bufsize=self.db_buf_size)

        # url read for channels: Note: first response data raises, second has an error and
        #that error is skipped (other channels are added), and last two channels are from two
        # stations (BLA|BLA|...) with only different start time (thus stations should both be added)
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|a||HHZ|1|1|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
A|b||HHE|2|2|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|c||HHZ|3|3|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
BLA|e||HHZ|7|7|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|2019-01-01T00:00:00
BLA|e||HHZ|8|8|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  URLError('wat'), socket.timeout()]
                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1, self.db_buf_size
                                               )
        
        assert len(channels_df) == 5

    # events_df
#    id  magnitude  latitude  longitude  depth_km                    time
# 0  1   3.0        1.0       1.0        60.0     2016-05-08 05:17:11.500
# 1  2   4.0        90.0      90.0       2.0      2016-05-08 01:45:30.300

    # channels_df:
#     id station_id  latitude  longitude  datacenter_id start_time   end_time
# 0   1           1       1.0        1.0              1 2008-02-12        NaT
# 1   2           2       2.0        2.0              1 2009-01-01        NaT
# 2   3           3       3.0        3.0              2 2008-02-12        NaT
# 3   4           4       7.0        7.0              2 2009-01-01 2019-01-01
# 4   5           5       8.0        8.0              2 2019-01-01        NaT

        # for magnitude <10, max_radius is 0. For magnitude >10, max_radius is 200
        # we have only magnitudes <10, we have two events exactly on a station (=> dist=0)
        # which will be taken (the others dropped out)
        df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=0, maxmag_radius=200)
        
        assert len(df) == 2
        
        # for magnitude <1, max_radius is 100. For magnitude >1, max_radius is 200
        # we have only magnitudes <10, we have all event-stations closer than 100 deg
        # So we might have ALL channels taken BUT: one station start time is in 2019, thus
        # it will not fall into the case above!
        df = merge_events_stations(events_df, channels_df, minmag=1, maxmag=1,
                                   minmag_radius=100, maxmag_radius=2000)
        
        assert len(df) == (len(channels_df)-1) *len(events_df)
        # assert channel outside time bounds was in:
        assert not channels_df[channels_df[Segment.start_time.key] == datetime(2019,1,1)].empty
        # we need to get the channel id from channels_df cause in df we removed unnecessary columns (including start end time)
        ch_id = channels_df[channels_df[Segment.start_time.key] == datetime(2019,1,1)][Channel.id.key].iloc[0]
        # old Channel.id.key is Segment.channel_id.key in df:
        assert df[df[Segment.channel_id.key] == ch_id].empty
        
        # this is a more complex case, we want to drop the first event by setting a very low
        # threshold (sraidus_minradius=1) for magnitudes <=3 (the first event magnitude)
        # and maxradius very high for the other event (magnitude=4)
        df = merge_events_stations(events_df, channels_df, minmag=3, maxmag=4,
                                   minmag_radius=1, maxmag_radius=40)
        
        # assert we have only the second event except the first channel which is from the 1st event.
        # The first event is retrievable by its latitude (2)
        # FIXME: more fine grained tests based on distance?
        evid = events_df[events_df[Event.latitude.key]==2][Event.id.key].iloc[0]
        assert np.array_equal((df[Segment.event_id.key] == evid),
                              [False, True, True, True, True])
        





# # =================================================================================================
# 
    def get_arrivaltimes(self, mintraveltime_side_effect, *a, **kw) : # , ):
        
        # REMEMBER: WITH PROCESSPOOLEXECUTOR DO NOT MOCK DIRECTLY THE FUNCTION PASSED
        # AS AS_COMPLETED, BUT A SUB FUNCTION. THIS IS PROBABLY DUE TO THE FACT THAT MOCKS ARE
        # NOT PICKABLE (SUB FUNCTIONS APPARENTLY DO NOT SUFFER NON PICKABILITY)
        
        self.mock_min_travel_time.reset_mock()
        self.mock_min_travel_time.side_effect = self._mintraveltime_sideeffect if mintraveltime_side_effect is None else mintraveltime_side_effect
        # self.setup_mock_arrival_time(mock_arr_time)
        return get_arrivaltimes(*a, **kw)
 

    def test_getset_arrivaltimes(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                           service=None,
                                                           channels=channels, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1, self.db_buf_size
                                               )
        assert len(channels_df) == 12  # just to be sure. If failing, we might have changed the class default
    # events_df
#    id  magnitude  latitude  longitude  depth_km                    time
# 0  1   3.0        1.0       1.0        60.0     2016-05-08 05:17:11.500
# 1  2   4.0        90.0      90.0       2.0      2016-05-08 01:45:30.300

    # channels_df:
#    id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# 0  1   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHE   
# 1  2   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHN   
# 2  3   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHZ   
# 3  4   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c1    
# 4  5   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c2    
# 5  6   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c3    
# 6   7   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHE   
# 7   8   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHN   
# 8   9   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHZ   
# 9   10  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c1    
# 10  11  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c2    
# 11  12  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c3    

        # take all segments:
        df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=100, maxmag_radius=200)
        
        h = 9

        # NOW TEST GET ARRIVAL TIMES
        # FIRST WE HAVE DB EMPTY THUS NO UPADTE SHOULD BE MADE
        assert Segment.arrival_time.key not in df.columns
        evts_stations_df = set_saved_arrivaltimes(self.session, df)
        
# evts_stations_df:
#    channel_id  station_id  datacenter_id network station location channel  event_distance_deg  event_id  depth_km                    time               arrival_time
# 0  1           1           1              GE      FLT1             HHE     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 1  2           1           1              GE      FLT1             HHN     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 2  3           1           1              GE      FLT1             HHZ     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 3  4           2           1              n1      s                c1      89.000              1         60.0     2016-05-08 05:17:11.500 NaT                       
# 4  5           2           1              n1      s                c2      89.000              1         60.0     2016-05-08 05:17:11.500 NaT                       
# 5  6           2           1              n1      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 6  7           3           2              IA      BAKI             BHE     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT         
# 7  8           3           2              IA      BAKI             BHN     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT         
# 8  9           3           2              IA      BAKI             BHZ     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT         
# 9  10          4           2              n2      s                c1      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 10  11          4           2              n2      s                c2      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 11  12          4           2              n2      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 12  1           1           1              GE      FLT1             HHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 13  2           1           1              GE      FLT1             HHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 14  3           1           1              GE      FLT1             HHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 15  4           2           1              n1      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 16  5           2           1              n1      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 17  6           2           1              n1      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 18  7           3           2              IA      BAKI             BHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 19  8           3           2              IA      BAKI             BHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 20  9           3           2              IA      BAKI             BHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 21  10          4           2              n2      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 22  11          4           2              n2      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 23  12          4           2              n2      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         



        assert Segment.arrival_time.key in evts_stations_df.columns and \
            all(pd.isnull(evts_stations_df[Segment.arrival_time.key]))
        
        # now put a segment on the db and see that one arrival time is not null
        atime = datetime.utcnow()
        stime = atime - timedelta(minutes=1)
        etime = atime + timedelta(minutes=3)
        evdist = 500.555
        # we will set values on atime and evdist according to
        # Channel.station-id.key ('station_id')
        # Segment.event_id.key ('event_id')
        SID = 1
        EVID = 1
        self.session.add(Segment(id = 1, # , default=seg_pkey_default, 
                         event_id = EVID,
                         channel_id = 1,  # the channel here must have station id = SID (see above)
                         datacenter_id = 1,
                         # seed_identifier = 'abc',
                         event_distance_deg = evdist,
                         data = None, # lazy load only upon access
                         download_status_code = None,
                         start_time = stime,
                         arrival_time = atime,
                         end_time = etime,
                         sample_rate = 100,
                         run_id = self.run.id))
        self.session.commit()
        
        evts_stations_df = set_saved_arrivaltimes(self.session, df)
        filter = (evts_stations_df[Channel.station_id.key] == SID) & \
            (evts_stations_df[Segment.event_id.key] == EVID)
        # we changed only ONE item in df:
        existing_items = 3
        assert len(evts_stations_df[filter]) == existing_items
        assert all(evts_stations_df[filter][Segment.arrival_time.key] == atime)
        assert all(evts_stations_df[filter][Segment.event_distance_deg.key] == evdist)
        
        # NOW mock get_dist_and_times

        # test first with a function that returns a TypeError for any segment:
        def deterministic_mintraveltime_sideeffect(*a, **kw):
            return datetime.utcnow()  # we should return the TOTAL seconds, NOT a datetime. Thus we should have all errors

        # make a copy of evts_stations_df cause we will modify in place the data frame
        segments_df =  self.get_arrivaltimes(deterministic_mintraveltime_sideeffect, evts_stations_df.copy(),
                                                   [1,2], ['P', 'Q'],
                                                        'ak135', mp_max_workers=1)
        
        # all failed, except the one we just set by mocking the db:
        assert len(segments_df) == existing_items  # these were not recalculated
        
        # Test now with a function which raises in some cases
        # IMPORTANT: WE NEED A DETERMINISTIC WAY TO HANDLE ARRIVAL TIME CALCULATION, 
        # AS APPARENTLY BEING IN A PROCESSPOOLEXECUTOR LEADS TO UNEXPECTED ORDERS
        # (CF EG BY RUNNING IN ECLIPSE OR IN TERMINAL). SO:
        #
        # WE NEED A DETERMINISTIC WAY TO MOCK THE FUNCTION AS THIS BELOW MIGHT BE EXECUTED IN
        # UNEXPECTED ORDER (DUE TO MULTIPROCESSING)
        
        def deterministic_mintraveltime_sideeffect(*a, **kw):
            evt_depth_km = a[0]
            if evt_depth_km == 60:
                return 1  # seconds
            raise TauModelError('wat?')
        
        expected_length = len(evts_stations_df[(evts_stations_df[Event.depth_km.key]==60) |
                                               (~pd.isnull(evts_stations_df[Segment.arrival_time.key]))])
        
        assert Segment.start_time.key not in evts_stations_df.columns
        assert Segment.end_time.key not in evts_stations_df.columns
        assert Event.time.key in evts_stations_df.columns
        assert Event.depth_km.key in evts_stations_df.columns
        
        # make a copy of evts_stations_df cause we will modify in place the data frame
        segments_df =  self.get_arrivaltimes(deterministic_mintraveltime_sideeffect, evts_stations_df.copy(),
                                                   [1,2], ['P', 'Q'],
                                                        'ak135', mp_max_workers=1)
        
        # all failed, except the one we just set by mocking the db:
        assert len(segments_df) == expected_length
        assert Segment.start_time.key in segments_df.columns
        assert Segment.end_time.key in segments_df.columns

        # FIXME: we should assert times are correctly calculated with respect to event time


    def test_prepare_for_download(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                           service=None,
                                                           channels=channels, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1, self.db_buf_size
                                               )
        assert len(channels_df) == 12  # just to be sure. If failing, we might have changed the class default
    # events_df
#    id  magnitude  latitude  longitude  depth_km                    time
# 0  1   3.0        1.0       1.0        60.0     2016-05-08 05:17:11.500
# 1  2   4.0        90.0      90.0       2.0      2016-05-08 01:45:30.300

    # channels_df:
#    id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# 0  1   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHE   
# 1  2   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHN   
# 2  3   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHZ   
# 3  4   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c1    
# 4  5   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c2    
# 5  6   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c3    
# 6   7   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHE   
# 7   8   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHN   
# 8   9   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHZ   
# 9   10  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c1    
# 10  11  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c2    
# 11  12  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c3    

        # take all segments:
        df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=100, maxmag_radius=200)
        evts_stations_df = set_saved_arrivaltimes(self.session, df)
        
# evts_stations_df:
#    channel_id  station_id  datacenter_id network station location channel  event_distance_deg  event_id  depth_km                    time               arrival_time
# 0  1           1           1              GE      FLT1             HHE     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 1  2           1           1              GE      FLT1             HHN     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 2  3           1           1              GE      FLT1             HHZ     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 3  4           2           1              n1      s                c1      89.000              1         60.0     2016-05-08 05:17:11.500 NaT                       
# 4  5           2           1              n1      s                c2      89.000              1         60.0     2016-05-08 05:17:11.500 NaT                       
# 5  6           2           1              n1      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 6  7           3           2              IA      BAKI             BHE     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT         
# 7  8           3           2              IA      BAKI             BHN     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT         
# 8  9           3           2              IA      BAKI             BHZ     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT         
# 9  10          4           2              n2      s                c1      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 10  11          4           2              n2      s                c2      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 11  12          4           2              n2      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT         
# 12  1           1           1              GE      FLT1             HHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 13  2           1           1              GE      FLT1             HHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 14  3           1           1              GE      FLT1             HHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 15  4           2           1              n1      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 16  5           2           1              n1      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 17  6           2           1              n1      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 18  7           3           2              IA      BAKI             BHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 19  8           3           2              IA      BAKI             BHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 20  9           3           2              IA      BAKI             BHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT         
# 21  10          4           2              n2      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 22  11          4           2              n2      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         
# 23  12          4           2              n2      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT         

        
        
        
        # make a copy of evts_stations_df cause we will modify in place the data frame
        segments_df =  self.get_arrivaltimes(urlread_sideeffect, evts_stations_df.copy(),
                                                   [1,2], ['P', 'Q'],
                                                        'ak135', mp_max_workers=1)
        
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        
        assert not Segment.id.key in segments_df.columns
        assert not Segment.run_id.key in segments_df.columns
        
        segments_df = prepare_for_download(self.session, segments_df,
                                           self.run.id,
                                           retry_no_code=True,
                                           retry_url_errors=True,
                                           retry_mseed_errors=True,
                                           retry_4xx=True,
                                           retry_5xx=True)
# segments_df:
#    channel_id  datacenter_id network station location channel  event_distance_deg  event_id            arrival_time          start_time            end_time
# 0  1           1              GE      FLT1             HHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 1  2           1              GE      FLT1             HHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 2  3           1              GE      FLT1             HHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 3  4           1              n1      s                c1      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 4  5           1              n1      s                c2      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 5  6           1              n1      s                c3      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 6  7           2              IA      BAKI             BHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 7  8           2              IA      BAKI             BHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 8  9           2              IA      BAKI             BHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 9  10          2              n2      s                c1      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 10  11          2              n2      s                c2      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 11  12          2              n2      s                c3      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 12  1           1              GE      FLT1             HHE     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 13  2           1              GE      FLT1             HHN     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 14  3           1              GE      FLT1             HHZ     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 15  4           1              n1      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 16  5           1              n1      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 17  6           1              n1      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 18  7           2              IA      BAKI             BHE     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 19  8           2              IA      BAKI             BHN     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 20  9           2              IA      BAKI             BHZ     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 21  10          2              n2      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 22  11          2              n2      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 23  12          2              n2      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31


        assert Segment.id.key in segments_df.columns
        assert Segment.run_id.key in segments_df.columns
        assert len(segments_df) == expected
        # assert len(self.session.query(Segment.id).all()) == len(segments_df)
        
        assert all(x[0] is None for x in self.session.query(Segment.download_status_code).all())
        assert all(x[0] is None for x in self.session.query(Segment.data).all())

        # mock an already downloaded segment.
        # Set the first five to have a particular download status code
        urlerr, mseederr = get_url_mseed_errorcodes()
        downloadstatuscodes = [None, urlerr, mseederr, 413, 505]
        for i, download_status_code in enumerate(downloadstatuscodes):
            dic = segments_df.iloc[i].to_dict()
            dic['download_status_code'] = download_status_code
            dic['run_id'] = self.run.id
            # hack for deleting unused columns:
            for col in [Station.network.key, Station.station.key,
                        Channel.location.key, Channel.channel.key]:
                del dic[col]
            self.session.add(Segment(**dic))
        self.session.commit()
        
        assert len(self.session.query(Segment.id).all()) == len(downloadstatuscodes)
        
        # Now we have an instance of all possible errors on the db (5 in total) and a new
        # instance (not on the db). Assure all work:
        # set the num of instances to download anyway. Their number is the not saved ones, i.e.:
        to_download_anyway = len(segments_df) - len(downloadstatuscodes)
        for c in product([True, False], [True, False], [True, False], [True, False], [True, False]):
            s_df = prepare_for_download(self.session, segments_df,
                                        self.run.id,
                                           retry_no_code=c[0],
                                           retry_url_errors=c[1],
                                           retry_mseed_errors=c[2],
                                           retry_4xx=c[3],
                                           retry_5xx=c[4])
            to_download_in_this_case = sum(c)  # count the True's (bool sum works in python) 
            assert len(s_df) == to_download_anyway +  to_download_in_this_case


    def download_save_segments(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return download_save_segments(*a, **kw)
    
    @patch("stream2segment.download.main.mseedunpack")
    @patch("stream2segment.download.main.insertdf_napkeys")
    @patch("stream2segment.download.main.updatedf")
    def test_download_save_segments(self, mock_updatedf, mock_insertdf_napkeys, mseed_unpack):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        mseed_unpack.side_effect = lambda *a, **v: mseedlite3.unpack(*a, **v)
        mock_insertdf_napkeys.side_effect = lambda *a, **v: insertdf_napkeys(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, self.routing_service,
                                                           self.service,
                                                           channels, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1, self.db_buf_size
                                               )
        assert len(channels_df) == 12  # just to be sure. If failing, we might have changed the class default
    # events_df
#                  id  magnitude  latitude  longitude  depth_km  time  
# 0  20160508_0000129        3.0       1.0        1.0      60.0  2016-05-08 05:17:11.500
# 1  20160508_0000004        4.0       2.0        2.0       2.0  2016-05-08 01:45:30.300 

# channels_df (index not shown):
# columns:
# id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# data (not aligned with columns):
# 1   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHE
# 2   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHN
# 3   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHZ
# 4   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c1 
# 5   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c2 
# 6   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c3 
# 7   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHE
# 8   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHN
# 9   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHZ
# 10  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c1 
# 11  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c2 
# 12  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c3

        # take all segments:
        # use minmag and maxmag
        df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=10, maxmag_radius=10)
        
        h = 9
        
# df (index not shown):
# cid sid did n   s    l  c    ed   event_id          depth_km                time  <- LAST TWO ARE Event related columns that will be removed after arrival_time calculations
# 1   1   1   GE  FLT1    HHE  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 2   1   1   GE  FLT1    HHN  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 3   1   1   GE  FLT1    HHZ  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 7   3   2   IA  BAKI    BHE  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 8   3   2   IA  BAKI    BHN  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 9   3   2   IA  BAKI    BHZ  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 4   2   1   n1  s       c1   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 5   2   1   n1  s       c2   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 6   2   1   n1  s       c3   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 10  4   2   n2  s       c1   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 11  4   2   n2  s       c2   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 12  4   2   n2  s       c3   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300

# LEGEND:
# cid = channel_id
# sid = station_id
# scid = datacenter_id
# n, s, l, c = network, station, location, channel
# ed = event_distance_deg

        evts_stations_df = set_saved_arrivaltimes(self.session, df)
                # make a copy of evts_stations_df cause we will modify in place the data frame
        segments_df =  self.get_arrivaltimes(urlread_sideeffect, evts_stations_df.copy(),
                                                   [1,2], ['P', 'Q'],
                                                        'ak135', mp_max_workers=1)
        
        expected = len(segments_df)  # no segment on db, we should have all segments to download

        segments_df = prepare_for_download(self.session, segments_df,
                                           self.run.id,
                                           retry_no_code=True,
                                           retry_url_errors=True,
                                           retry_mseed_errors=True,
                                           retry_4xx=True,
                                           retry_5xx=True)
        
# segments_df
# COLUMNS:
# channel_id  datacenter_id network station location channel event_distance_deg event_id arrival_time start_time end_time id download_status_code run_id
# DATA (not aligned with columns):
#               channel_id  datacenter_id network station location channel  event_distance_deg  event_id            arrival_time          start_time            end_time    id download_status_code  run_id
# GE.FLT1..HHE  1           1              GE      FLT1             HHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1     
# GE.FLT1..HHN  2           1              GE      FLT1             HHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1     
# GE.FLT1..HHZ  3           1              GE      FLT1             HHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1     
# IA.BAKI..BHE  7           2              IA      BAKI             BHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1     
# IA.BAKI..BHN  8           2              IA      BAKI             BHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1     
# IA.BAKI..BHZ  9           2              IA      BAKI             BHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1     
# n1.s..c1      4           1              n1      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1     
# n1.s..c2      5           1              n1      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1     
# n1.s..c3      6           1              n1      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1     
# n2.s..c1      10          2              n2      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1     
# n2.s..c2      11          2              n2      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1     
# n2.s..c3      12          2              n2      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1     
        
        # self._segdata is the folder file of a "valid" 3-channel miniseed
        # The channels are: 
        # Thus, no match will be found and all segments will be written with a None
        # download status code

        
        # setup urlread: first three rows: ok
        # rows[3:6]: 413, retry them
        # rows[6:9]: malformed_data
        # rows[9:12] 413, retry them
        # then retry:
        # rows[3]: empty_data
        # rows[4]: data_with_gaps (but seed_id should notmatch)
        # rows[5]: data_with_gaps (seed_id should notmatch)
        # rows[9]: URLError
        # rows[10]: Http 500 error
        # rows[11]: 413
        
        # NOTE THAT THIS RELIES ON THE FACT THAT THREADS ARE EXECUTED IN THE ORDER OF THE DATAFRAME
        # WHICH SEEMS TO BE THE CASE AS THERE IS ONE SINGLE PROCESS
        # self._seg_data[:2] is a way to mock data corrupted
        urlread_sideeffect = [self._seg_data, 413, self._seg_data[:2], 413,
                              '', self._seg_data_gaps, self._seg_data_gaps, URLError("++urlerror++"), 500, 413]
        # Let's go:
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df, datacenters_df, 1,2,3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.id, Segment.channel_id, Segment.datacenter_id,
                Segment.download_status_code, Segment.max_gap_ovlap_ratio, \
                Segment.sample_rate, Segment.seed_identifier, Segment.data, Segment.run_id, Segment.start_time, Segment.end_time,
                ]
        db_segments_df = dbquery2df(self.session.query(*cols))
        
        # change data column otherwise we cannot display db_segments_df. When there is data just print "data"
        db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) & (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key] = 'data' 

        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

# db_segments_df:
#    id  channel_id  datacenter_id  download_status_code  max_gap_ovlap_ratio  sample_rate seed_identifier  data  run_id          start_time            end_time
# 0  1   1           1              200.0                 0.0001               100.0        GE.FLT1..HHE    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 1  2   2           1              200.0                 0.0001               100.0        GE.FLT1..HHN    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 2  3   3           1              200.0                 0.0001               100.0        GE.FLT1..HHZ    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 6  7   7           2              200.0                 NaN                  NaN          None                  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 7  8   8           2              NaN                   NaN                  NaN          None            None  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 8  9   9           2              200.0                 20.0                 20.0         IA.BAKI..BHZ    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 3  4   4           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 4  5   5           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 5  6   6           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 9  10  10          2              -1.0                  NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 10 11  11          2              500.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 11 12  12          2              413.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31


 
        assert len(ztatz) == len(datacenters_df)
        assert len(db_segments_df) == len(segments_df)
        assert mock_updatedf.call_count == 0
        # as we have 12 segments and a buf size of self.db_buf_size(=1, but it might change in the
        # future), this below is two
        # it might change if we changed the buf size in the future
        
        # test that we correctly called mock_insertdf_napkeys. Note that we assume that the
        # latter is called ONLY inside download.main.DbManager. To test that, as the number of stuff
        # to be added (length of the dataframes) varies, we need to implement a counter here:
        mock_insertdf_napkeys_call_count = 0
        _bufzise = 0
        for c in mock_insertdf_napkeys.call_args_list:
            c_args =  c[0]
            df_ = c_args[0]
            _bufzise += len(df_)
            if _bufzise >= self.db_buf_size:
                mock_insertdf_napkeys_call_count += 1
                _bufzise = 0
        
        assert mock_insertdf_napkeys.call_count == mock_insertdf_napkeys_call_count
        
        # assert data is consistent
        COL = Segment.data.key
        assert (db_segments_df.iloc[:3][COL] == 'data').all()
        assert (db_segments_df.iloc[3:4][COL] == '').all()
        assert pd.isnull(db_segments_df.iloc[4:5][COL]).all()
        assert (db_segments_df.iloc[5:6][COL] == 'data').all()
        assert pd.isnull(db_segments_df.iloc[6:][COL]).all()
        
        # assert downdload status code is consistent
        URLERR_CODE, MSEEDERR_CODE = get_url_mseed_errorcodes()
        # also this asserts that we grouped for dc starttime endtime
        COL = Segment.download_status_code.key
        assert (db_segments_df.iloc[:4][COL] == 200).all()
        assert pd.isnull(db_segments_df.iloc[4:5][COL]).all()
        assert (db_segments_df.iloc[5:6][COL] == 200).all()
        assert (db_segments_df.iloc[6:9][COL] == MSEEDERR_CODE).all()
        assert (db_segments_df.iloc[9][COL] == URLERR_CODE).all()
        assert (db_segments_df.iloc[10][COL] == 500).all()
        assert (db_segments_df.iloc[11][COL] == 413).all()
        
        # assert gaps are only in the given position
        URLERR_CODE, MSEEDERR_CODE = get_url_mseed_errorcodes()
        COL = Segment.max_gap_ovlap_ratio.key
        assert (db_segments_df.iloc[:3][COL] < 0.01).all()
        assert pd.isnull(db_segments_df.iloc[3:5][COL]).all()
        assert (db_segments_df.iloc[5][COL] > 1).all()
        assert pd.isnull(db_segments_df.iloc[6:][COL]).all()
        
        
        # now mock retry:
        segments_df = prepare_for_download(self.session, segments_df,
                                           self.run.id,
                                           retry_no_code=True,
                                           retry_url_errors=True,
                                           retry_mseed_errors=True,
                                           retry_4xx=True,
                                           retry_5xx=True)
        
        COL = Segment.download_status_code.key
        mask = (db_segments_df[COL] >= 400) | pd.isnull(db_segments_df[COL]) \
            | (db_segments_df[COL].isin([URLERR_CODE, MSEEDERR_CODE]))
        assert len(segments_df) == len(db_segments_df[mask])
        
        urlread_sideeffect = [413]
        mock_updatedf.reset_mock()
        mock_insertdf_napkeys.reset_mock()
        # Let's go:
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df, datacenters_df, 1,2,3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.download_status_code, Segment.channel_id]
        db_segments_df = dbquery2df(self.session.query(*cols))
        
        # change data column otherwise we cannot display db_segments_df. When there is data just print "data"
        # db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) & (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key] = 'data' 

        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

        assert (db_segments_df[COL] == 413).all()
        assert len(ztatz) == len(datacenters_df)
        assert len(db_segments_df) == len(segments_df)
        
        # same as above: but with updatedf: test that we correctly called mock_insertdf_napkeys. Note that we assume that the
        # latter is called ONLY inside download.main.DbManager. To test that, as the number of stuff
        # to be added (length of the dataframes) varies, we need to implement a counter here:
        mock_updatedf_call_count = 0
        _bufzise = 0
        for c in mock_updatedf.call_args_list:
            c_args =  c[0]
            df_ = c_args[0]
            _bufzise += len(df_)
            if _bufzise >= self.db_buf_size:
                mock_updatedf_call_count += 1
                _bufzise = 0
        
        assert mock_updatedf.call_count == mock_updatedf_call_count
        
        
        assert mock_insertdf_napkeys.call_count == 0
