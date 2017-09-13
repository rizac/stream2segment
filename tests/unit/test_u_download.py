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
from stream2segment.io.db.models import Base, Event, Class, WebService, DataCenter, fdsn_urls
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from stream2segment.main import main, closing
from click.testing import CliRunner
# from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_col_names
import pandas as pd
from stream2segment.download.main import get_events_df, get_datacenters_df, \
    logger as query_logger, get_channels_df, merge_events_stations, \
    prepare_for_download, download_save_segments, \
    QuitDownload, chaid2mseedid_dict
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
from stream2segment.utils import get_session, mseedlite3
from stream2segment.io.db.pd_sql_utils import dbquery2df, insertdf_napkeys, updatedf
from logging import StreamHandler
import logging
from io import BytesIO
import urllib2
from stream2segment.download.utils import get_url_mseed_errorcodes
from test.test_userdict import d1
from stream2segment.utils.mseedlite3 import MSeedError, unpack
import threading
from stream2segment.utils.url import read_async
from stream2segment.utils.resources import get_templates_fpath, yaml_load, get_ttable_fpath
from stream2segment.download.traveltimes.ttloader import TTTable
from stream2segment.download.utils import urljoin as original_urljoin

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
    def cleanup(me):
        engine, session, handler, patchers = me.engine, me.session, me.handler, me.patchers
        if me.engine:
            if me.session:
                try:
                    me.session.rollback()
                    me.session.close()
                except:
                    pass
            try:
                Base.metadata.drop_all(me.engine)
            except:
                pass
        
        for patcher in patchers:
            patcher.stop()
        
        hndls = query_logger.handlers[:]
        handler.close()
        for h in hndls:
            if h is handler:
                query_logger.removeHandler(h)

    def _get_sess(self, *a, **v):
        return self.session

    @property
    def is_sqlite(self):
        return str(self.engine.url).startswith("sqlite:///")
    
    @property
    def is_postgres(self):
        return str(self.engine.url).startswith("postgresql://")

    def setUp(self):
        url = os.getenv("DB_URL", "sqlite:///:memory:")

        from sqlalchemy import create_engine
        self.dburi = url
        engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(engine)
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        # create a Session
        self.session = Session()
        self.engine = engine
        
        
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
        

        self.patchers = [self.patcher, self.patcher1, self.patcher2, self.patcher23]
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
        self.addCleanup(Test.cleanup, self)
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
        

#     def test_add_classes(self):
#         cls = {'a' : 'bla', 'b' :'c'}
#         add_classes(self.session, cls, db_bufsize=self.db_buf_size)
#         assert len(self.session.query(Class).all()) == 2
#         logmsg = self.log_msg()
#         assert "Db table 'classes': 2 new items inserted (no sql errors)" in logmsg
#         #assert "Db table 'classes': no new item to insert" not in logmsg
#         add_classes(self.session, cls, db_bufsize=self.db_buf_size)
#         # assert we did not write any new class
#         assert len(self.session.query(Class).all()) == 2
#         # and also that nothing is printed to log (we print only new and discarded)
#         assert "Db table 'classes': no new item to insert" in self.log_msg()
        
# ===========================

    
#     @patch('stream2segment.download.main.yaml_load', return_value={'service1': {'event': 'http:event1'}, 'service2': {'event': 'http:event2', 'station': 'http:station2'}})
#     def test_get_eventws_url(self, mock_yaml_load):
#         with pytest.raises(ValueError):
#             url = get_eventws_url(self.session, "eida")
#         
#         url = get_eventws_url(self.session, "service1")
#         assert url == 'http:event1'
#         url = get_eventws_url(self.session, "service2")
#         assert url == 'http:event2'

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

    def test_models_fdsn_url(self):
        for url in ["https://mock/fdsnws/station/1/query", "https://mock/fdsnws/station/1/query?",
                    "https://mock/fdsnws/station/1/", "https://mock/fdsnws/station/1"]:
            res = fdsn_urls(url)
            assert res[0] == "https://mock/fdsnws/station/1/query"
            assert res[1] == "https://mock/fdsnws/dataselect/1/query"
        
        url = "http://www.google.com"
        assert fdsn_urls(url) is None
        
        url = "https://mock/fdsnws/station/1/whatever/query"
        res = fdsn_urls(url)
        assert res[0] == "https://mock/fdsnws/station/1/whatever/query"
        assert res[1] == "https://mock/fdsnws/dataselect/1/whatever/query"
        
# =================================================================================================

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_datacenters_df(*a, **v)
    
#     (session, service, routing_service_url, 
#     channels, starttime=None, endtime=None, 
#     db_bufsize=None)
    
    @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs_general(self, mock_urljoin):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query

ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        
        # no fdsn service ("http://myservice")
        with pytest.raises(QuitDownload):
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "http://myservice",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called # is called only when supplying eida
        
        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session,
                                                       "https://mock/fdsnws/station/1/query",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called # is called only when supplying eida
        assert len(self.session.query(DataCenter).all()) == len(data) == 1
        assert self.session.query(DataCenter).first().node_organization_name == None

        # iris:
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "iris",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called # is called only when supplying eida
        assert len(self.session.query(DataCenter).all()) == 2  # we had one already (added above)
        assert len(data) == 1
        assert len(self.session.query(DataCenter).filter(DataCenter.node_organization_name == 'iris').all()) == 1
        
        # eida:
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert mock_urljoin.called # is called only when supplying eida
        assert len(self.session.query(DataCenter).all()) == 3 # we had two already written, 1 written now
        assert len(data) == 1
        assert len(self.session.query(DataCenter).filter(DataCenter.node_organization_name == 'eida').all()) == 1
        # assert we wrote just resif (the first one, the other one are malformed):
        assert self.session.query(DataCenter).filter(DataCenter.node_organization_name == 'eida').first().station_url == \
            "http://ws.resif.fr/fdsnws/station/1/query"
    
        # now re-launch and assert we did not write anything to the db cause we already did:
        dcslen = len(self.session.query(DataCenter).all())
        self.get_datacenters_df(urlread_sideeffect, self.session,
                                                       "https://mock/fdsnws/station/1/query",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert dcslen == len(self.session.query(DataCenter).all())
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "iris",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert dcslen == len(self.session.query(DataCenter).all())
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                                       self.routing_service, None, None,
                                                       db_bufsize=self.db_buf_size)
        assert dcslen == len(self.session.query(DataCenter).all())
        

    # @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs_postdata(self):  # , mock_urljoin):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        
        d0 = datetime.utcnow()
        d1 = d0 + timedelta(minutes=1.1)

        for channels, starttime, endtime in product([['HH?'], ['HH?', 'BH?'], None], [None, d0],
                                                    [None, d1]):
            # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session,
                                                           "https://mock/fdsnws/station/1/query",
                                                           self.routing_service,
                                                           channels, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            assert len(post_data_list) == 1
            assert post_data_list[-1] ==  '* * * %s %s %s' % \
                (",".join(channels) if channels else "*", starttime.isoformat() if starttime else "*",
                 endtime.isoformat() if endtime else "*")
            
            # iris:
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "iris",
                                                       self.routing_service,
                                                       channels, starttime, endtime,
                                                       db_bufsize=self.db_buf_size)
            assert len(post_data_list) == 1
            assert post_data_list[-1] ==  '* * * %s %s %s' % \
                (",".join(channels) if channels else "*", starttime.isoformat() if starttime else "*",
                 endtime.isoformat() if endtime else "*")
            
            # eida:
            
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                                   self.routing_service,
                                                   channels, starttime, endtime,
                                                   db_bufsize=self.db_buf_size)
            assert len(post_data_list) == 2
            # post data returned are always the same. See string above
            assert post_data_list[0] == 'http://geofon.gfz-potsdam.de/fdsnws/station/1/query'
            assert post_data_list[1] == 'ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00\nUP ARJ * BHW 2013-08-01T00:00:00 2017-04-25'
            
            #assert postdata == '* * * *'
    
        # @patch('stream2segment.download.main.urljoin', return_value='a')
    def test_get_dcs_routingerror(self):  # , mock_urljoin):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = [URLError('wat?')]
        
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1.1)
        channels = ['HH?', 'BH?']
        
        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        # test that is ok because routing errors are not concerned here
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session,
                                                       "https://mock/fdsnws/station/1/query",
                                                       self.routing_service,
                                                       channels, starttime, endtime,
                                                       db_bufsize=self.db_buf_size)
        assert not self.mock_urlopen.called

        # iris:
        # test that is ok because routing errors are not concerned here
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "iris",
                                                   self.routing_service,
                                                   channels, starttime, endtime,
                                                   db_bufsize=self.db_buf_size)
        assert not self.mock_urlopen.called
        
        # eida:
        with pytest.raises(QuitDownload):
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                                   self.routing_service,
                                                   channels, starttime, endtime,
                                                   db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        
        # now let's write something to db
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        df, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                                   self.routing_service,
                                                   channels, starttime, endtime,
                                                   db_bufsize=self.db_buf_size)
        
        dcs = sorted([_[0] for _ in self.session.query(DataCenter.id).filter(DataCenter.node_organization_name == 'eida').all()])
        assert self.mock_urlopen.called
        assert len(dcs) == len(df)
        
        # and it should not raise anymore, but return the db stuff: pass None as service to check  that it
        # defaults to 'eida'
        urlread_sideeffect = [URLError('wat?')]
        df, post_data_list = self.get_datacenters_df(urlread_sideeffect, self.session, None,  # == "eida",
                                                   self.routing_service,
                                                   channels, starttime, endtime,
                                                   db_bufsize=self.db_buf_size)
        assert all(not post for post in post_data_list)
        dcs2 = sorted([_[0] for _ in self.session.query(DataCenter.id).filter(DataCenter.node_organization_name == 'eida').all()])
        assert dcs == dcs2
        assert self.mock_urlopen.called
        assert sorted(df[DataCenter.id.key].tolist()) == dcs

# =================================================================================================



    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_channels_df(*a, **kw)
    # get_channels_df(session, datacenters_df, post_data, channels, starttime, endtime, min_sample_rate, 
#     max_thread_workers, timeout, blocksize, db_bufsize, 
#     show_progress=False):
     
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
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, None, self.routing_service,
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
                                                       channels, None, None,
                                                       100, None, None, -1, self.db_buf_size
                                               )
         
        # we should have called mock_urlopen_in_async times the datacenters
        assert self.mock_urlopen.call_count == len(datacenters_df)
 
        assert len(self.session.query(Station.id).all()) == 4
        assert len(self.session.query(Channel.id).all()) == 6
        
        # we should have built the request object according to url dise effect defined BEFORE
        # self.get_datacenters_df:
        # """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
        # ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
        # UP ARJ * * 2013-08-01T00:00:00 2017-04-25
        # 
        # http://ws.resif.fr/fdsnws/dataselect/1/query
        # ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
        # """
        assert self.mock_urlopen.call_args_list[0][0][0].data == """format=text
level=channel
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25"""
        assert self.mock_urlopen.call_args_list[1][0][0].data == """format=text
level=channel
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999"""
        assert self.mock_urlopen.call_args_list[0][0][0].get_full_url() == "http://geofon.gfz-potsdam.de/fdsnws/station/1/query"
        assert self.mock_urlopen.call_args_list[1][0][0].get_full_url() == "http://ws.resif.fr/fdsnws/station/1/query"
        
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
                                                       [None] * len(postdata),
                                                       channels, None, None,
                                                       100, None, None, -1, self.db_buf_size
                                               )

        assert len(cha_df2) == len(cha_df)
        assert sorted(list(cha_df.columns)) == sorted(list(cha_df2.columns))
        
        # now mock a datacenter null (or empty) postdata in the second datacenter
        # (<==> no data in eida routing service under a specific datacenter)
        # and test that by querying the database we get the data we just saved.
        # as when postdata is falsy we query the database and
        # we have only channels of the 2nd datacenter
        cha_df3 = self.get_channels_df(urlread_sideeffect, self.session,
                                                           datacenters_df,
                                                           ['x', None],
                                                           channels, None, None,
                                                           100, None, None, -1, self.db_buf_size
                                                   )
        assert cha_df3.equals(cha_df2)
        assert 'discarding response data' in self.log_msg()

        # now test the opposite, but note that urlread side effect should return now an urlerror and a socket error:
        with pytest.raises(QuitDownload):
            cha_df2 = self.get_channels_df(URLError('urlerror_wat'), self.session,
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels, None, None,
                                                           100, None, None, -1, self.db_buf_size
                                                   )
        assert 'urlerror_wat' in self.log_msg()
        assert 'WARNING: unable to fetch stations from 1 data center(s)' in self.log_msg()

        # now test again, we should ahve a socket timeout
        with pytest.raises(QuitDownload):
            cha_df2 = self.get_channels_df(socket.timeout(), self.session,
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels, None, None,
                                                           100, None, None, -1, self.db_buf_size
                                                   )
        assert 'timeout' in self.log_msg()
        
        # now the case where the none post request is for the "good" repsonse data:
        # Basically, replace ['x', None] with [None, 'x'] and swap urlread side effects
        cha_df2 = self.get_channels_df([urlread_sideeffect[1], urlread_sideeffect[0]],
                                       self.session,
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels, None, None,
                                                           100, None, None, -1, self.db_buf_size
                                                   )
        
        assert len(cha_df2) == 6
        # now change min sampling rate and see what happens (we do not have one channel more
        # cause we have overlapping names, and the 50 Hz channel is overridden by the second
        # query) 
        cha_df3 = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels, None, None,
                                                       10, None, None, -1, self.db_buf_size
                                               )
        assert len(cha_df3) == len(cha_df)
        
        # now change this:
        
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|B||HBE|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|2010-02-12T00:00:00
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
E|F||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  URLError('wat'), socket.timeout()]
        
        # test high sample rate, we should get only one channel:
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels, None, None,
                                                       100, None, None, -1, self.db_buf_size
                                               )

        assert len(cha_df) == 1
        assert "sample rate <" in self.log_msg()
        
        # now decrease the sampling rate, we should have two channels (all):
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels, None, None,
                                                       10, None, None, -1, self.db_buf_size
                                               )

        assert len(cha_df) == 2
        
        # now change channels=['B??'], we should have the same result as before as
        # the `channel` argument has effect when postdata is None (=query to the db)
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       ['B??'], None, None,
                                                       10, None, None, -1, self.db_buf_size
                                               )

        assert len(cha_df) == 2
        
        # same as above but use a very high sample rate:
        with pytest.raises(QuitDownload):
            cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                               datacenters_df,
                                                               postdata,
                                                               ['B??'], None, None,
                                                               1000000000, None, None, -1, self.db_buf_size
                                                       )

        assert "discarding %d channels (sample rate < 1000000000 Hz)" % len(cha_df) in self.log_msg()
        
        

        # still channels=['B??'], but postdata is None, which means get_channels_df
        # queries the db. Since there
        # are no 'B??' channels saved (see urlread_sideeffect return value above)
        # which means a QuitDownload is raised
        with pytest.raises(QuitDownload):
            cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                               datacenters_df,
                                                               [None] * len(datacenters_df),
                                                               ['B??'], None, None,
                                                               10, None, None, -1, self.db_buf_size
                                                       )
        assert ("Fetching stations and channels from "
                "db for %d data-center(s)") % len(datacenters_df) in self.log_msg()
        
           
        # Test the case where all responses are empty: raise quit download
        with pytest.raises(QuitDownload) as qerr:
            cha_df = self.get_channels_df("", self.session,
                                                               datacenters_df,
                                                               postdata,
                                                               ['B??'], None, None,
                                                               10, None, None, -1, self.db_buf_size
                                                       )

        assert ("No channel found (Request error or empty response in all station queries, "
                "no data in cache (database). Check config and log for details)") == str(qerr.value)
                
        # same as above if all responses are errors
        with pytest.raises(QuitDownload) as qerr:
            cha_df = self.get_channels_df([500], self.session,
                                                               datacenters_df,
                                                               postdata,
                                                               ['B??'], None, None,
                                                               10, None, None, -1, self.db_buf_size
                                                       )
        
        # test channels and startime + entimes provided when querying the db (postdata None)
        # by iussuing the command:
        # dbquery2df(self.session.query(Channel.id, Channel.channel, Station.start_time,
        #                               Station.end_time, Channel.sample_rate).join(Station))
        # we found the following channels on the db:
        #
        # id  channel start_time  end_time             sample_rate
        # 1   HHE     2008-02-12  None                 100.0      
        # 2   HHZ     2008-02-12  None                 100.0      
        # 3   HHE     2009-01-01  None                 100.0      
        # 4   HHZ     2009-01-01  None                 100.0      
        # 5   HHZ     2009-01-01  2019-01-01 00:00:00  100.0      
        # 6   HHZ     2019-01-01  None                 100.0      
        # 7   HHZ     2019-01-01  None                 100.0      
        # 8   HBE     2008-02-12  2010-02-12 00:00:00  50.0     
        
        # Now according to the table above set a list of arguments:
        # Each key is: channel, starttime, edntime, mapped to the expected channel count
        args = {(('?B?',), None, None): 1,
                (('HB?',), datetime(2009,1,1), None): 1,
                #(('HB?',), datetime(2019,1,1), None): 0,
                #(('HB?',), None, datetime(2001,1,1)): 0,
                (None, datetime(2008,1,1), None): 8,
                (None, datetime(2012,1,1), None): 7,
                # (None, None, datetime(2001,1,1)): 0,
                (None, None, datetime(2010,1,1)): 6,
                (None, datetime(2010,1,1), datetime(2010,2,1)): 6,
                (None, datetime(2010,1,1), datetime(2030,2,1)): 8,
                }
        # test when we query the database (postdata is None) with different channels and datetime(s)
        for (cha, stime, etime), expected_length in args.iteritems():
            cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                               datacenters_df,
                                                               [None] * len(datacenters_df),
                                                               cha, stime, etime,
                                                               10, None, None, -1, self.db_buf_size
                                                       )
            if len(cha_df) != expected_length:  # this if just is for debug purposes in eclipse
                sdf = 9
            assert len(cha_df) == expected_length
        # now test the cases where we have zero channels. These are treated separately from the case
        # above cause 0 channels raises QuitDownload:
        args = {
                (('HB?',), datetime(2019,1,1), None): 0,
                (('HB?',), None, datetime(2001,1,1)): 0,
                (None, None, datetime(2001,1,1)): 0, 
                }
        # test when we query the database (postdata is None) with different channels and datetime(s)
        for (cha, stime, etime), expected_length in args.iteritems():
            with pytest.raises(QuitDownload):
                cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                               datacenters_df,
                                                               [None] * len(datacenters_df),
                                                               cha, stime, etime,
                                                               10, None, None, -1, self.db_buf_size
                                                       )
            

# FIXME: test save inventories!!!!

    def ttable(self, modelname=None):
        '''convenience function that loads a ttable from the data folder'''
        if modelname is None:
            modelname = 'ak135_tts+_5.npz'
        fle = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', modelname)
        if not os.path.splitext(fle)[1]:
            fle += '.npz'
        return TTTable(fle)

    def test_merge_event_stations(self):
        # get events with lat lon (1,1), (2,2,) ... (n, n)
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|2|2|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)


        channels = None
        datacenters_df, postdata = self.get_datacenters_df(None, self.session, None, self.routing_service,
                                                           channels=channels, db_bufsize=self.db_buf_size)

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
                                                       channels, None, None,
                                                       10, None, None, -1, self.db_buf_size
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

        tt_table = self.ttable()
        # for magnitude <10, max_radius is 0. For magnitude >10, max_radius is 200
        # we have only magnitudes <10, we have two events exactly on a station (=> dist=0)
        # which will be taken (the others dropped out)
        df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=0, maxmag_radius=200, tttable=tt_table)
        
        assert len(df) == 2
        
        # for magnitude <1, max_radius is 100. For magnitude >1, max_radius is 200
        # we have only magnitudes <10, we have all event-stations closer than 100 deg
        # So we might have ALL channels taken BUT: one station start time is in 2019, thus
        # it will not fall into the case above!
        df = merge_events_stations(events_df, channels_df, minmag=1, maxmag=1,
                                   minmag_radius=100, maxmag_radius=2000, tttable=tt_table)
        
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
                                   minmag_radius=1, maxmag_radius=40, tttable=tt_table)
        
        # assert we have only the second event except the first channel which is from the 1st event.
        # The first event is retrievable by its latitude (2)
        # FIXME: more fine grained tests based on distance?
        evid = events_df[events_df[Event.latitude.key]==2][Event.id.key].iloc[0]
        assert np.array_equal((df[Segment.event_id.key] == evid),
                              [False, True, True, True, True])
        
        
        # test arrival times are properly set: Set all event locations to [0,0] as well
        # as stations locations. This should result in all arrival times equal to event time
        #
        _events_df = events_df
        _channels_df = channels_df
        events_df = events_df.copy()
        events_df.loc[:, Event.latitude.key] = 0
        events_df.loc[:, Event.longitude.key] = 0
        event_ids = pd.unique(events_df[Event.id.key])
        # We have two events, set the depth of the first one to zero the other to 60 
        evtid1, evtid2 = event_ids[0], event_ids[1]
        evttime1 = events_df[events_df[Event.id.key] == evtid1][Event.time.key].iloc[0]
        evttime2 = events_df[events_df[Event.id.key] == evtid2][Event.time.key].iloc[0]
        events_df.loc[events_df[Event.id.key] == evtid1, Event.depth_km.key] = 0
        events_df.loc[events_df[Event.id.key] == evtid2, Event.depth_km.key] = 60
         
        channels_df = channels_df.copy()
        channels_df.loc[:, Station.latitude.key] = 0
        channels_df.loc[:, Station.longitude.key] = 0
        df = merge_events_stations(events_df, channels_df, minmag=3, maxmag=4,
                                   minmag_radius=1, maxmag_radius=40, tttable=tt_table)
        # assert for events of depth 0 arrival times are queal to event times
        assert (df[df[Segment.event_id.key] == evtid1][Segment.arrival_time.key] == evttime1).all()
        # assert for events of depth > 0 arrival times are GREATER than event times
        assert (df[df[Segment.event_id.key] == evtid2][Segment.arrival_time.key] > evttime2).all()
         
        # now set the first event time out-of bounds:
        events_df.loc[events_df[Event.id.key] == evtid1, Event.depth_km.key] = 600000
        df = merge_events_stations(events_df, channels_df, minmag=3, maxmag=4,
                                   minmag_radius=1, maxmag_radius=40, tttable=tt_table)
        # assert for events of depth 0 arrival times are queal to event times
        # as nans are dropped from the returned dataframe, assert we do not have segments with
        # event_id == evtid1:
        assert df[df[Segment.event_id.key] == evtid1][Segment.arrival_time.key].empty
        # still assert for events of depth > 0 arrival times are GREATER than event times
        assert (df[df[Segment.event_id.key] == evtid2][Segment.arrival_time.key] > evttime2).all()
        


# # =================================================================================================


    def test_prepare_for_download(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, None, self.routing_service,
                                                           channels=channels, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels, None, None,
                                                       100, None, None, -1, self.db_buf_size
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
        segments_df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=100, maxmag_radius=200, tttable=self.ttable())

        
# segments_df:
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
#         segments_df =  self.get_arrivaltimes(urlread_sideeffect, evts_stations_df.copy(),
#                                                    [1,2], ['P', 'Q'],
#                                                         'ak135', mp_max_workers=1)
        
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        wtimespan = [1,2]
        assert not Segment.id.key in segments_df.columns
        assert not Segment.run_id.key in segments_df.columns
        orig_seg_df = segments_df.copy()
        segments_df = prepare_for_download(self.session, orig_seg_df, wtimespan,
                                           retry_no_code=True,
                                           retry_url_errors=True,
                                           retry_mseed_errors=True,
                                           retry_4xx=True,
                                           retry_5xx=True)
# segments_df: (not really the real dataframe, some columns are removed but relevant data is ok):
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
        assert Segment.run_id.key not in segments_df.columns
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
                if col in dic:
                    del dic[col]
            self.session.add(Segment(**dic))
        self.session.commit()
        
        assert len(self.session.query(Segment.id).all()) == len(downloadstatuscodes)
        
        # Now we have an instance of all possible errors on the db (5 in total) and a new
        # instance (not on the db). Assure all work:
        # set the num of instances to download anyway. Their number is the not saved ones, i.e.:
        to_download_anyway = len(segments_df) - len(downloadstatuscodes)
        for c in product([True, False], [True, False], [True, False], [True, False], [True, False]):
            s_df = prepare_for_download(self.session, orig_seg_df, wtimespan,
                                           retry_no_code=c[0],
                                           retry_url_errors=c[1],
                                           retry_mseed_errors=c[2],
                                           retry_4xx=c[3],
                                           retry_5xx=c[4])
            to_download_in_this_case = sum(c)  # count the True's (bool sum works in python) 
            assert len(s_df) == to_download_anyway +  to_download_in_this_case

        # now change the window time span and see that everything is to be downloaded again:
        # do it for any retry combinations, as it should ALWAYS return "everything has to be re-downloaded"
        wtimespan[1] += 5
        for c in product([True, False], [True, False], [True, False], [True, False], [True, False]):
            s_df = prepare_for_download(self.session, orig_seg_df, wtimespan,
                                           retry_no_code=c[0],
                                           retry_url_errors=c[1],
                                           retry_mseed_errors=c[2],
                                           retry_4xx=c[3],
                                           retry_5xx=c[4])
            assert len(s_df) == len(orig_seg_df)
        # this hol

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
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, self.session, self.service,
                                                           self.routing_service,
                                                           channels, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       postdata,
                                                       channels, None, None,
                                                       10, None, None, -1, self.db_buf_size
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

        assert all(_ in channels_df.columns for _ in [Station.network.key, Station.station.key,
                                                      Channel.location.key, Channel.channel.key])
        chaid2mseedid = chaid2mseedid_dict(channels_df)
        # check that we removed the columns:
        assert not any(_ in channels_df.columns for _ in [Station.network.key, Station.station.key,
                                                      Channel.location.key, Channel.channel.key])
        
        # take all segments:
        # use minmag and maxmag
        ttable = self.ttable()
        segments_df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=10, maxmag_radius=10, tttable=ttable)
        
        assert len(pd.unique(segments_df['arrival_time'])) == 2
        
        h = 9
    
    
        
# segments_df (index not shown). Note that 
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


        wtimespan = [1,2]
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        orig_segments_df = segments_df.copy()
        segments_df = prepare_for_download(self.session, orig_segments_df, wtimespan,
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
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df, datacenters_df, 
                                            chaid2mseedid,
                                            self.run.id, 1,2,3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.id, Segment.channel_id, Segment.datacenter_id,
                Segment.download_status_code, Segment.max_gap_ovlap_ratio, \
                Segment.sample_rate, Segment.seed_identifier, Segment.data, Segment.run_id, Segment.start_time, Segment.end_time,
                ]
        db_segments_df = dbquery2df(self.session.query(*cols))
        assert Segment.run_id.key in db_segments_df.columns
        
        
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
        segments_df = prepare_for_download(self.session, orig_segments_df, wtimespan,
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
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df, datacenters_df,
                                            chaid2mseedid,
                                            self.run.id, 1,2,3, db_bufsize=self.db_buf_size)
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
