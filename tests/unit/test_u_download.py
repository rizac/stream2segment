#@PydevCodeAnalysisIgnore
# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')

from future import standard_library
standard_library.install_aliases()

from builtins import str
import numpy as np
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta

import random

import sys
# this can apparently not be avoided neither with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
try:
    from cStringIO import StringIO  # python2.x
except ImportError:
    from io import StringIO

import unittest, os
from sqlalchemy.engine import create_engine
from stream2segment.io.db.models import Base, Event, Class, WebService, DataCenter, fdsn_urls
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from click.testing import CliRunner

import pandas as pd
from stream2segment.download.modules.events import get_events_df
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import prepare_for_download, download_save_segments,\
    chaid2mseedid_dict
from stream2segment.download.utils import QuitDownload
# ,\
#     get_fdsn_channels_df, save_stations_and_channels, get_dists_and_times, set_saved_dist_and_times,\
#     download_segments, drop_already_downloaded, set_download_urls, save_segments
from obspy.core.stream import Stream, read
from stream2segment.io.db.models import DataCenter, Segment, Download, Station, Channel, WebService
from itertools import cycle, repeat, count, product
from urllib.error import URLError
import socket
from obspy.taup.helper_classes import TauModelError
# import logging
# from logging import StreamHandler

# from stream2segment.main import logger as main_logger
from sqlalchemy.sql.expression import func
from stream2segment.io.db.pdsql import dbquery2df, insertdf, updatedf
from logging import StreamHandler
import logging
from io import BytesIO
import urllib.request, urllib.error, urllib.parse
from stream2segment.download.utils import custom_download_codes 
from stream2segment.download.modules.mseedlite import MSeedError, unpack
import threading
from stream2segment.utils.url import read_async
from stream2segment.utils.resources import get_templates_fpath, yaml_load, get_ttable_fpath
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.download.utils import dblog
from stream2segment.utils import urljoin as original_urljoin



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

import logging

query_logger = logger = logging.getLogger("stream2segment")

class Test(unittest.TestCase):

    @staticmethod
    def cleanup(me):
        if hasattr(me, "_pd_display_maxcolwidth"):
            pd.set_option('display.max_colwidth', me._pd_display_maxcolwidth)

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
        # when debugging, I want the full dataframe with to_string(), not truncated
        # NOTE: this messes up right alignment of numbers in DownloadStats (see utils.py)
        # FIRST, remember current settings and restore them in cleanup:
        self._pd_display_maxcolwidth = pd.get_option('display.max_colwidth')
        pd.set_option('display.max_colwidth', -1)
        
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
        
        self.patchers = []
        
        self.patchers.append(patch('stream2segment.utils.url.urllib.request.urlopen'))
        self.mock_urlopen = self.patchers[-1].start()
        
        # mock ThreadPool (tp) to run one instance at a time, so we get deterministic results:
        class MockThreadPool(object):
            
            def __init__(self, *a, **kw):
                pass
                
            def imap(self, func, iterable, *args):
                # make imap deterministic: same as standard python map:
                # everything is executed in a single thread the right input order
                return map(func, iterable)
            
            def imap_unordered(self, func, iterable, *args):
                # make imap_unordered deterministic: same as standard python map:
                # everything is executed in a single thread in the right input order
                return map(func, iterable)
            
            def close(self, *a, **kw):
                pass
        # assign patches and mocks:
        self.patchers.append(patch('stream2segment.utils.url.ThreadPool'))
        self.mock_tpool = self.patchers[-1].start()
        self.mock_tpool.side_effect = MockThreadPool
        
        
        self.logout = StringIO()
        self.handler = StreamHandler(stream=self.logout)
        self._logout_cache = ""

        # THIS IS A HACK:
        query_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset (to level.INFO)
        # otherwise it stays what we set two lines above. Problems might arise if closing
        # sets a different level, but for the moment who cares
        
        query_logger.addHandler(self.handler)
        
        # setup a run_id:
        r = Download()
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
        idx = len(self._logout_cache)
        self._logout_cache = self.logout.getvalue()
        if len(self._logout_cache) == idx:
            idx = None # do not slice
        return self._logout_cache[idx:]
    
    def setup_urlopen(self, urlread_side_effect):
        """setup urlopen return value. 
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.read, that will be converted
        to an itertools.cycle(side_effect) REMEMBER that any element of urlread_side_effect which is a nonempty
        string must be followed by an EMPTY
        STRINGS TO STOP reading otherwise we fall into an infinite loop if the argument
        blocksize of url read is not negative !"""

        self.mock_urlopen.reset_mock()
        # convert returned values to the given urlread return value (tuple data, code, msg)
        # if k is an int, convert to an HTTPError
        retvals = []
        # Check if we have an iterable (where strings are considered not iterables):
        if not hasattr(urlread_side_effect, "__iter__") or isinstance(urlread_side_effect, (bytes, str)):
            # it's not an iterable (wheere str/bytes/unicode are considered NOT iterable in both py2 and 3)
            urlread_side_effect = [urlread_side_effect]
            
        for k in urlread_side_effect:
            a = Mock()
            if type(k) == int:
                a.read.side_effect = urllib.error.HTTPError('url', int(k),  responses[k][0], None, None)
            elif type(k) in (bytes, str):
                def func(k):
                    b = BytesIO(k.encode('utf8') if type(k) == str else k)  # py2to3 compatible
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
        
        
    def test_dblog(self):
        dblog(Station, 0, 0, 0, 0)
        s = self.log_msg()
        assert "Db table 'stations': no new row to insert, no row to update" in s
        dblog(Station, 0, 0, 0, 1)
        s = self.log_msg()
        assert "Db table 'stations': 0 rows updated, 1 discarded (sql errors)" in s
        dblog(Station, 0, 1, 0, 1)
        s = self.log_msg()
        assert """Db table 'stations': 0 new rows inserted, 1 discarded (sql errors)
Db table 'stations': 0 rows updated, 1 discarded (sql errors)""" in s
        dblog(Station, 0, 1, 0, 0)
        s = self.log_msg()
        assert "Db table 'stations': 0 new rows inserted, 1 discarded (sql errors)" in s
        dblog(Station, 1, 5, 4, 1)
        s = self.log_msg()
        assert """Db table 'stations': 1 new row inserted, 5 discarded (sql errors)
Db table 'stations': 4 rows updated, 1 discarded (sql errors)""" in s
        dblog(Station, 3, 0, 4, 1)
        s = self.log_msg()
        assert """Db table 'stations': 3 new rows inserted (no sql error)
Db table 'stations': 4 rows updated, 1 discarded (sql errors)""" in s
        dblog(Station, 3, 5, 1, 0)
        s = self.log_msg()
        assert """Db table 'stations': 3 new rows inserted, 5 discarded (sql errors)
Db table 'stations': 1 row updated (no sql error)""" in s
        dblog(Station, 3, 0, 4, 0)
        s = self.log_msg()
        assert """Db table 'stations': 3 new rows inserted (no sql error)
Db table 'stations': 4 rows updated (no sql error)""" in s
        h = 9

    def get_events_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_events_df(*a, **v)
        

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    def test_get_events(self, mock_query):
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""]
        
        
        data = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        # assert only first two events events were successfully saved 
        assert len(self.session.query(Event).all()) == len(pd.unique(data['id'])) == 2
        # AND data to save has length 2:
        assert len(data) == 2

        # now download again, with an url error:        
        urlread_sideeffect = [413, """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", URLError('blabla23___')]
        
        data = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        # assert we got the same result as above:
        assert len(self.session.query(Event).all()) == len(pd.unique(data['id'])) == 2
        assert len(data) == 2
        # and since first response is 413, that having split the request into two, the
        # second response is our URLError (we could test it better, anyway):
        assert "blabla23___" in self.log_msg()
        

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
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
        

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
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
        # get_datacenters_df(session, service, routing_service_url, 
        #                    net, sta, loc, cha, starttime=None, endtime=None, 
        #                    db_bufsize=None)
    
    @patch('stream2segment.download.modules.datacenters.urljoin', return_value='a')
    def test_get_dcs_general(self, mock_urljoin):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query

ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        
        # provide defaults for arguments not tested here:
        net, sta, loc, cha, start, end = [], [], [], [], None, None
        
        # no fdsn service ("http://myservice")
        with pytest.raises(QuitDownload):
            data, _ = self.get_datacenters_df(urlread_sideeffect, self.session,
                                              "http://myservice", self.routing_service,
                                              net, sta, loc, cha, start, end,
                                              db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called # is called only when supplying eida
        
        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session,
                                    "https://mock/fdsnws/station/1/query",self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called # is called only when supplying eida
        assert len(self.session.query(DataCenter).all()) == len(data) == 1
        assert self.session.query(DataCenter).first().organization_name == None
        assert eidavalidator is None # no eida

        # iris:
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session,
                                    "iris", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called # is called only when supplying eida
        assert len(self.session.query(DataCenter).all()) == 2  # we had one already (added above)
        assert len(data) == 1
        assert len(self.session.query(DataCenter).filter(DataCenter.organization_name == 'iris').all()) == 1
        assert eidavalidator is None # no eida

        # eida:
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session,
                                    "eida", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        assert mock_urljoin.called # is called only when supplying eida
        assert len(self.session.query(DataCenter).all()) == 3 # we had two already written, 1 written now
        assert len(data) == 1
        assert len(self.session.query(DataCenter).filter(DataCenter.organization_name == 'eida').all()) == 1
        # assert we wrote just resif (the first one, the other one are malformed):
        assert self.session.query(DataCenter).filter(DataCenter.organization_name == 'eida').first().station_url == \
            "http://ws.resif.fr/fdsnws/station/1/query"
        assert eidavalidator is not None # no eida
    
        # now re-launch and assert we did not write anything to the db cause we already did:
        dcslen = len(self.session.query(DataCenter).all())
        self.get_datacenters_df(urlread_sideeffect, self.session,
                                "https://mock/fdsnws/station/1/query", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(self.session.query(DataCenter).all())
        self.get_datacenters_df(urlread_sideeffect, self.session,
                                "iris", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(self.session.query(DataCenter).all())
        
        self.get_datacenters_df(urlread_sideeffect, self.session,
                                "eida", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(self.session.query(DataCenter).all())
        

    @patch('stream2segment.download.modules.datacenters.urljoin', side_effect = lambda *a, **v: original_urljoin(*a, **v))
    def test_get_dcs_postdata(self, mock_urljoin):  # , mock_urljoin):
        '''test fetching datacenters eida, iris, custom url and test that postdata is what we
        expected (which is eida/iris/whatever independent)'''
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        
        d0 = datetime.utcnow()
        d1 = d0 + timedelta(minutes=1.1)
        
        nsl = [['ABC'], []]
        chans = [['HH?'], ['HH?', 'BH?'], []]
        
        for net, sta, loc, cha, starttime, endtime in product(nsl, nsl, nsl, chans,
                                                                   [None, d0],
                                                                   [None, d1]):
            mock_urljoin.reset_mock()
            # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, self.session,
                                                           "https://mock/fdsnws/station/1/query",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            assert eida_validator is None
            assert not mock_urljoin.called
            
            # iris:
            mock_urljoin.reset_mock()
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, self.session, "iris",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            assert eida_validator is None
            assert not mock_urljoin.called
            
            # eida:
            mock_urljoin.reset_mock()
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            
            geofon_id = data[data[DataCenter.station_url.key] == 'http://geofon.gfz-potsdam.de/fdsnws/station/1/query'].iloc[0].id
            resif_id = data[data[DataCenter.station_url.key] == 'http://ws.resif.fr/fdsnws/station/1/query'].iloc[0].id
            
            j= mock_urljoin.call_args_list
            assert len(j) == 1
            call_ = j[0]
            args = call_[0]
            kwargs = call_[1]
            assert len(args) == 1
            assert args[0] == 'http://rz-vm258.gfz-potsdam.de/eidaws/routing/1/query'
            assert kwargs['service'] == 'dataselect'
            assert kwargs['format'] == 'post'
            if net:
                assert kwargs['net'] == ','.join(net)
            else:
                assert 'net' not in kwargs
            if sta:
                assert kwargs['sta'] == ','.join(sta)
            else:
                assert 'sta' not in kwargs
            if loc:
                assert kwargs['loc'] == ','.join(loc)
            else:
                assert 'loc' not in kwargs
            if cha:
                assert kwargs['cha'] == ','.join(cha)
            else:
                assert 'cha' not in kwargs
            if starttime:
                assert kwargs['start'] == starttime.isoformat()
            else:
                assert 'start' not in kwargs
            if endtime:
                assert kwargs['end'] == endtime.isoformat()
            else:
                assert 'end' not in kwargs
            
    
    def test_get_dcs_routingerror(self):  # , mock_urljoin):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = [URLError('wat?')]
        
        # we might set the following params as defaults because not used, let's provide anyway
        # something meaningful:
        net, sta, loc, cha = ['*'], [], [],['HH?', 'BH?']
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1.1)

        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        # test that everything is ok because we do not call urlread
        data, _ = self.get_datacenters_df(urlread_sideeffect, self.session,
                                          "https://mock/fdsnws/station/1/query",
                                          self.routing_service,
                                          net, sta, loc, cha, starttime, endtime,
                                          db_bufsize=self.db_buf_size)
        assert not self.mock_urlopen.called

        # iris:
        # test that everything is ok because we do not call urlread
        data, _ = self.get_datacenters_df(urlread_sideeffect, self.session, "iris",
                                          self.routing_service,
                                          net, sta, loc, cha, starttime, endtime,
                                          db_bufsize=self.db_buf_size)
        assert not self.mock_urlopen.called
        
        # eida:
        # test that everything is not ok because urlread raises and we do not have data on the db
        with pytest.raises(QuitDownload) as qdown:
            data, _ = self.get_datacenters_df(urlread_sideeffect, self.session, "eida",
                                              self.routing_service,
                                              net, sta, loc, cha, starttime, endtime,
                                              db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert "Eida routing service error, no eida data-center saved in database" \
            in str(qdown.value)
        
        # now let's write something to db
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        df, eidavalidator = self.get_datacenters_df(urlread_sideeffect, self.session,
                                                    "eida",
                                                    self.routing_service,
                                                    net, sta, loc, cha, starttime, endtime,
                                                    db_bufsize=self.db_buf_size)
        # check that all datacenters are eida
        dcs = sorted([_[0] for _ in self.session.query(DataCenter.id).filter(DataCenter.organization_name == 'eida').all()])
        assert self.mock_urlopen.called
        assert len(dcs) == len(df)
        assert eidavalidator is not None
       
# =================================================================================================



    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_channels_df(*a, **kw)
    # def get_channels_df(session, datacenters_df, eidavalidator, # <- can be none
    #                     net, sta, loc, cha, starttime, endtime, 
    #                     min_sample_rate, update, 
    #                     max_thread_workers, timeout, blocksize, db_bufsize, 
    #                     show_progress=False):
     
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
        # we tried to add two events with the same id, check we printed out the msg:
        assert "Duplicated instances violate db constraint" in self.log_msg()
        
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)
        
        # first we mock url errors in all queries. We still did not write anything in the db
        # so we should quit:
        with pytest.raises(QuitDownload) as qd:
            _ = self.get_channels_df(URLError('urlerror_wat'), self.session,
                                                           datacenters_df,
                                                           eidavalidator,
                                                           net, sta, loc, cha, None, None, 100,
                                                           False, None, None, -1, self.db_buf_size)
        assert 'urlerror_wat' in self.log_msg()
        assert "Unable to fetch stations" in self.log_msg()
        assert "Fetching stations from database for 2 (of 2) data-center(s)" in self.log_msg()
        # Test that the exception message is correct
        # note that this message is in the log if we run the method from the main
        # function (which is not the case here):
        assert ("Unable to fetch stations from all data-centers, "
                                      "no data to fetch from the database. "
                                      "Check config and log for details") in str(qd.value)
        

        # now use filters on hannels start-time and end-time arguments, by assuring post data
        # is well formed. Include negations (!) to check that they are not included in the request
        # post data
        st = [datetime(2001,1,1), None]
        et = [datetime(2001,3,1), None]
        nslc = [[], ['HH?'], ['HHL', 'HH?'], ['!A'], ['!AZ', 'BH?']]
        # now we might want to use 
        # for net, sta, loc, cha, s, e in product(nslc, nslc, nslc, nslc, [None, st], [None, et]):
        # which runs 2500 iterations. Way too much. We pick 1000 tests by randomly choicing 
        # in those elements
        for i in range(1000):
            _net, _sta, _loc, _cha = random.choice(nslc), random.choice(nslc),\
                random.choice(nslc), random.choice(nslc)
            s, e = random.choice(st), random.choice(et)
            with pytest.raises(QuitDownload) as qd:
                _ = self.get_channels_df(URLError('urlerror_wat'), self.session,
                                                           datacenters_df,
                                                           eidavalidator,
                                                           _net, _sta, _loc, _cha, s, e, 100,
                                                           False, None, None, -1,
                                                           self.db_buf_size)
                
                n_ = (",".join(_ for _ in _net if _[0:1] != '!') or '*').encode()
                s_ = (",".join(_ for _ in _sta if _[0:1] != '!') or '*').encode()
                l_ = (",".join(_ for _ in _loc if _[0:1] != '!') or '*').encode()
                c_ = (",".join(_ for _ in _cha if _[0:1] != '!') or '*').encode()
                s_ = '*' if st is None else st.isoformat().encode()
                e_ = '*' if et is None else et.isoformat().encode()
                expected == b"""format=text
level=channel
%s %s %s %s %s %s""" % (n_, s_, l_, c_, s_, e_)
                assert self.mock_urlopen.call_args_list[0][0][0].data == expected
                assert self.mock_urlopen.call_args_list[1][0][0].data == expected
        
        

        # now get channels with a mocked custom urlread_sideeffect below:
        # IMPORTANT: url read for channels: Note: first response data raises, second has an error and
        # that error is skipped (the other channels are added)
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
--- ERROR --- MALFORMED|12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", 
# NOTE THAT THE CHANNELS ABOVE WILL BE OVERRIDDEN BY THE ONES BELOW (MULTIPLE NAMES< WE SHOULD NOT HAVE
# THIS CASE WITH THE EDIAWS ROUTING SERVICE BUT WE TEST HERE THE CASE)
# NOTE THE USE OF HTß as SensorDescription (to check non-asci characters do not raise)
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
HT|AGG||HHE|--- ERROR --- NONNUMERIC |22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|70.0|2008-02-12T00:00:00|
HT|AGG||HLE|95.6|22.336|622.0|0.0|90.0|0.0|GFZ:HTß1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HLZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|90.0|2009-01-01T00:00:00|
HT|LKD2||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|90.0|2009-01-01T00:00:00|
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|2019-01-01T00:00:00
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2018-01-01T00:00:00|
"""]                     

        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 90,
                                                       False, None, None, -1, self.db_buf_size
                                               )
        # assert we have a message for discarding the response data
        # (first arg of urlread):
        assert "Discarding response data" in self.log_msg()
        # we should have called mock_urlopen_in_async times the datacenters
        assert self.mock_urlopen.call_count == len(datacenters_df)
        assert len(self.session.query(Station.id).all()) == 4
        # the last two channels of the second item of `urlread_sideeffect` are from two
        # stations (BLA|BLA|...) with only different start time. Thus they should both be added:
        assert len(self.session.query(Channel.id).all()) == 6
        # as net, sta, loc, cha are all empty lists and start = end = None (all default=>no filter),
        # this is the post data passed to urlread for the 1st datacenter:
        assert self.mock_urlopen.call_args_list[0][0][0].data == b"""format=text
level=channel
* * * * * *"""
        # as net, sta, loc, cha are all empty lists and start = end = None (all default=>no filter),
        # this is the post data passed to urlread for the 2nd datacenter:
        assert self.mock_urlopen.call_args_list[1][0][0].data == b"""format=text
level=channel
* * * * * *"""
        assert self.mock_urlopen.call_args_list[0][0][0].get_full_url() == \
            "http://geofon.gfz-potsdam.de/fdsnws/station/1/query"
        assert self.mock_urlopen.call_args_list[1][0][0].get_full_url() == \
            "http://ws.resif.fr/fdsnws/station/1/query"
        # assert all downloaded stations have datacenter_id of the second datacenter:
        dcid = datacenters_df.iloc[1].id
        assert all(sid[0] ==dcid for sid in self.session.query(Station.datacenter_id).all())
        # assert all downloaded channels have station_id in the set of downloaded stations only:
        sta_ids = [x[0] for x in self.session.query(Station.id).all()]
        assert all(c_staid[0] in sta_ids for c_staid in self.session.query(Channel.station_id).all())

        # now mock again url errors in all queries. As we wrote something in the db
        # so we should NOT quit
        cha_df2 = self.get_channels_df(URLError('urlerror_wat'), self.session,
                                       datacenters_df,
                                       eidavalidator,
                                       net, sta, loc, cha, datetime(2020, 1, 1), None, 100,
                                       False, None, None, -1, self.db_buf_size)

        # Note above that min sample rate = 100 and a starttime proivded should return 3 channels: 
        assert len(cha_df2) == 3
        assert "Fetching stations from database for 2 (of 2) data-center(s)" in self.log_msg()

        # now test again with a socket timeout
        cha_df2 = self.get_channels_df(socket.timeout(), self.session,
                                       datacenters_df,
                                       eidavalidator,
                                       net, sta, loc, cha, None, None, 100,
                                       False, None, None, -1, self.db_buf_size)
        assert 'timeout' in self.log_msg()
        assert "Fetching stations from database for 2 (of 2) data-center(s)" in self.log_msg()

        # now mixed case: 
        
        # now change min sampling rate and see that we should get one channel less
        cha_df3 = self.get_channels_df(urlread_sideeffect, self.session,
                                       datacenters_df,
                                       eidavalidator,
                                       net, sta, loc, cha, None, None,  100,
                                       False, None, None, -1, self.db_buf_size)
        assert len(cha_df3) == len(cha_df)-2
        assert "2 channel(s) discarded (sample rate < 100 Hz)" in self.log_msg()
        
        # now change this:
        
        urlread_sideeffect  = [URLError('wat'), 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|B|10|HBE|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-02-12T00:00:00|2010-02-12T00:00:00
E|F|11|HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  URLError('wat'), socket.timeout()]
        
        
        # now change channels=['B??']. In the urlread_sideeffect above, for the 1st, 3rd and 4th
        # case we fallback to a db query, but we do not have such a channel, so nothing is returned
        # The dataframe currently saved on db is:
        #    id channel start_time   end_time  sample_rate  datacenter_id
        # 0  1   HLE    2008-02-12 NaT         100.0        2            
        # 1  2   HLZ    2008-02-12 NaT         100.0        2            
        # 2  3   HHE    2009-01-01 NaT         90.0         2            
        # 3  4   HHZ    2009-01-01 NaT         90.0         2            
        # 4  5   HHZ    2009-01-01 2019-01-01  100.0        2            
        # 5  6   HHZ    2018-01-01 NaT         100.0        2
        # for the second case, the mocked response returns two channels and in this case
        # we might put whatever filter here below. Assert that the number of channels returned is 2
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                      datacenters_df,
                                      eidavalidator,
                                      net, sta, loc, ['B??'], None, None, 10,
                                      False, None, None, -1, self.db_buf_size)
        assert len(cha_df) == 2
        
        # test channels and startime + entimes provided when querying the db (postdata None)
        # by iussuing the command:
        # dbquery2df(self.session.query(Channel.id, Station.network, Station.station,
        #  Channel.location, Channel.channel, Station.start_time,Station.end_time,
        #  Channel.sample_rate, Station.datacenter_id).join(Channel.station))
        # This is the actual state of the db:
        # ----------------------------------------------
        # channel_id network station location channel start_time    end_time   sample_rate  datacenter_id
        #          1      HT     AGG              HLE    2008-02-12 NaT              100.0              2               
        #          2      HT     AGG              HLZ    2008-02-12 NaT              100.0              2               
        #          3      HT     LKD2             HHE    2009-01-01 NaT              90.0               2               
        #          4      HT     LKD2             HHZ    2009-01-01 NaT              90.0               2               
        #          5      BLA    BLA              HHZ    2009-01-01 2019-01-01       100.0              2               
        #          6      BLA    BLA              HHZ    2018-01-01 NaT              100.0              2               
        #          7      A      B         10     HBE    2003-02-12 2010-02-12       100.0              2               
        #          8      E      F         11     HHZ    2019-01-01 NaT              100.0              2  
        # ----------------------------------------------
        # Now according to the table above set a list of arguments:
        # Each key is: the argument, each value IS A LIST OF BOOLEAN MAPPED TO EACH ROW OF THE
        # DATAFRAME ABOVE, telling if the row matches according to the argument:
        nets = {('*',): [1, 1, 1, 1, 1, 1, 1, 1],
                # ('HT', 'BLA'): [1, 1, 1, 1, 1, 1, 0, 0],
                ('*A*',): [0, 0, 0, 0, 1, 1, 1, 0]
                }
        stas = {('B*',): [0, 0, 0, 0, 1, 1, 1, 0],
                ('B??',): [0, 0, 0, 0, 1, 1, 0, 0]}
        locs = {('',): [1, 1, 1, 1, 1, 1, 0, 0],  # note that we do NOT assume '--' can be given, as this should be the parsed output of `nslc_lists`
                ('1?',): [0, 0, 0, 0, 0, 0, 1, 1],}
        chans = {('?B?',): [0, 0, 0, 0, 0, 0, 1, 0],
                  ('HL?', '?B?'): [1, 1, 0, 0, 0, 0, 1, 0],
                  ('HHZ',): [0, 0, 0, 1, 1, 1, 0, 1],
                 }
        stimes={None: [1, 1, 1, 1, 1, 1, 1, 1],
                    datetime(2002, 1, 1): [1, 1, 1, 1, 1, 1, 1, 1],
                    # datetime(2011, 1, 1): [1, 1, 1, 1, 1, 1, 0, 1],
                    datetime(2099, 1, 1): [1, 1, 1, 1, 0, 1, 0, 1],
                    }
        etimes={None: [1, 1, 1, 1, 1, 1, 1, 1],
                    datetime(2002, 1, 1): [0, 0, 0, 0, 0, 0, 0, 0],
                    datetime(2011, 1, 1): [1, 1, 1, 1, 1, 0, 1, 0],
                    datetime(2099, 1, 1): [1, 1, 1, 1, 1, 1, 1, 1],
                    }
        minsr = {90: [1, 1, 1, 1, 1, 1, 1, 1],
                 # 95: [1, 1, 0, 0, 1, 1, 1, 1],
                 100: [1, 1, 0, 0, 1, 1, 1, 1],
                 105: [0, 0, 0, 0, 0, 0, 0, 0]}
        # no url read: set socket.tiomeout as urlread side effect. This will force
        # querying the database to test that the filtering works as expected:
        for n, s, l, c, st, e, m in product(nets, stas, locs, chans, stimes, etimes, minsr):
            matches = np.array(nets[n]) * np.array(stas[s]) * np.array(locs[l]) * \
                np.array(chans[c]) * np.array(stimes[st]) * np.array(etimes[e]) * np.array(minsr[m])
            expected_length = matches.sum()
            # Now: if expected length is zero, it means we do not have data matches on the db
            # This raises a quitdownload (avoiding pytest.raises cause in this
            # case it's easier like done below):
            try:
                cha_df = self.get_channels_df(socket.timeout(), self.session,
                                              datacenters_df.loc[datacenters_df[DataCenter.id.key] == 2],
                                              eidavalidator,
                                              n, s, l, c, st, e, m,
                                              False, None, None, -1, self.db_buf_size)
                assert len(cha_df) == expected_length
            except QuitDownload as qd:
                assert expected_length == 0
                assert "Unable to fetch stations from all data-centers" in str(qd)       

        # Same test as above, but test negative assertions with "!". Reminder: data on db is:
        # ----------------------------------------------
        # channel_id network station location channel start_time    end_time   sample_rate  datacenter_id
        #          1      HT     AGG              HLE    2008-02-12 NaT              100.0              2               
        #          2      HT     AGG              HLZ    2008-02-12 NaT              100.0              2               
        #          3      HT     LKD2             HHE    2009-01-01 NaT              90.0               2               
        #          4      HT     LKD2             HHZ    2009-01-01 NaT              90.0               2               
        #          5      BLA    BLA              HHZ    2009-01-01 2019-01-01       100.0              2               
        #          6      BLA    BLA              HHZ    2018-01-01 NaT              100.0              2               
        #          7      A      B         10     HBE    2003-02-12 2010-02-12       100.0              2               
        #          8      E      F         11     HHZ    2019-01-01 NaT              100.0              2  
        # ----------------------------------------------
        # Now according to the table above set a list of arguments:
        # Each key is: the argument, each value IS A LIST OF BOOLEAN MAPPED TO EACH ROW OF THE
        # DATAFRAME ABOVE, telling if the row matches according to the argument:
        nets = {('!*A*', 'A'): [1, 1, 1, 1, 0, 0, 1, 1],
                ('E', 'A'): [0, 0, 0, 0, 0, 0, 1, 1]
                }
        stas = {('!*B*', 'B'): [1, 1, 1, 1, 0, 0, 1, 1],
                ('!???2',): [1, 1, 0, 0, 1, 1, 1, 1]}
        locs = {('',): [1, 1, 1, 1, 1, 1, 0, 0],  # note that we do NOT assume '--' can be given, as this should be the parsed output of `nslc_lists`
                ('!',): [0, 0, 0, 0, 0, 0, 1, 1],}
        chans = { ('HHZ', '!*E'): [0, 1, 0, 1, 1, 1, 0, 1],
                 ('!?H?',): [1, 1, 0, 0, 0, 0, 1, 0]
                 }
        stimes={None: [1, 1, 1, 1, 1, 1, 1, 1]}
        etimes={None: [1, 1, 1, 1, 1, 1, 1, 1]}
        minsr = {-1: [1, 1, 1, 1, 1, 1, 1, 1]}
        # no url read: set socket.tiomeout as urlread side effect. This will force
        # querying the database to test that the filtering works as expected:
        for n, s, l, c, st, e, m in product(nets, stas, locs, chans, stimes, etimes, minsr):
            matches = np.array(nets[n]) * np.array(stas[s]) * np.array(locs[l]) * \
                np.array(chans[c]) * np.array(stimes[st]) * np.array(etimes[e]) * np.array(minsr[m])
            expected_length = matches.sum()
            # Now: if expected length is zero, it means we do not have data matches on the db
            # This raises a quitdownload (avoiding pytest.raises cause in this
            # case it's easier like done below):
            try:
                cha_df = self.get_channels_df(socket.timeout(), self.session,
                                              datacenters_df.loc[datacenters_df[DataCenter.id.key] == 2],
                                              eidavalidator,
                                              n, s, l, c, st, e, m,
                                              False, None, None, -1, self.db_buf_size)
                
                assert len(cha_df) == expected_length
            except QuitDownload as qd:
                assert expected_length == 0
                assert "Unable to fetch stations from all data-centers" in str(qd)       

  
        # now make the second url_side_effect raise => force query from db, and the first good
        # => fetch from the web
        # We want to test the mixed case: some fetched from db, some from the web
        # ---------------------------------------------------
        # first we query the db to check what we have:
        cha_df = dbquery2df(self.session.query(Channel.id, Station.datacenter_id,
                                               Station.network).join(Station))
        # build a new network:
        newnetwork = 'U'
        while newnetwork in cha_df[Station.network.key]:
            newnetwork += 'U'
        urlread_sideeffect2  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
%s|W||HBE|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|2010-02-12T00:00:00
""" % newnetwork,  socket.timeout()]
        # now note: the first url read raised, now it does not: write the channel above with
        # network = newnetwork (surely non existing to the db)
        #  The second url read did not raise, now it does (socket.timeout): fetch from the db
        # we issue a ['???'] as 'channel' argument in order to fetch everything from the db
        # (we would have got the same by passing None as 'channel' argument)
        # The three [] before ['???'] are net, sta, loc and mean: no filter on those params
        cha_df_ = self.get_channels_df(urlread_sideeffect2, self.session,
                                       datacenters_df, eidavalidator,
                                       [], [], [], ['???'], None, None, 10,
                                       False, None, None, -1, self.db_buf_size)
        
        # we should have the channel with network 'U' to the first datacenter
        dcid = datacenters_df.iloc[0][DataCenter.id.key]
        assert len(cha_df_[cha_df_[Station.datacenter_id.key] == dcid]) == 1
        assert cha_df_[cha_df_[Station.datacenter_id.key] == dcid][Station.network.key][0] == \
            newnetwork
        # but we did not query other channels for datacenter id = dcid, as the web response
        # was successful, we rely on that. Conversely, for the other datacenter we should have all
        # channels fetched from db
        dcid = datacenters_df.iloc[1][DataCenter.id.key]
        chaids_of_dcid = cha_df_[cha_df_[Station.datacenter_id.key] == dcid][Channel.id.key].tolist()
        db_chaids_of_dcid = cha_df[cha_df[Station.datacenter_id.key] == dcid][Channel.id.key].tolist()
        assert chaids_of_dcid == db_chaids_of_dcid


    def test_get_channels_df_eidavalidator_station_and_channel_duplicates(self):
        urlread_sideeffect = """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)

        # urlread for datacenters: will be called only if we have eida (the case here)
        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
A1 * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
A2 a2 * * 2013-08-01T00:00:00 2017-04-25
XX xx * * 2013-08-01T00:00:00 2017-04-25
YY yy * HH? 2013-08-01T00:00:00 2017-04-25

http://ws.resif.fr/fdsnws/dataselect/1/query
B1 * * HH? 2002-09-01T00:00:00 2005-10-20T00:00:00
XX xx * * 2013-08-01T00:00:00 2017-04-25
YY yy * DE? 2013-08-01T00:00:00 2017-04-25
"""
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)
        
        # MOCK THE CASE OF DUPLICATED STATIONS (I.E., RETURNED BY MORE THAN ONE DATACENTER)
        # The behaviour we want to test is the following: Given a downloaded station,
        # 1) if the station if it's unique (only one datacenter returned it), just fetch
        # the station ID from the db
        # (note that this does not touch the station datacenter so we might end up
        # downloading station's segment from a different datacenter than the station's
        # datacenter saved on the database, which is what we want).
        # 2) if the station is NOT unique, then query the eida routing service
        # (or the database if the eda-rs is down):
        # the FIRST station among our duplicates (sorted by datacenter id) which has a match
        # on the eida routing service or db is taken, the other(s) discarded.
        # If no station match (the station is not supposed to be
        # returned by ANY datacenter in the eida-rs, or we do not have that station on the db)
        # the station, and all its channels, are discarded
        #
        # To test what we just said, we write here the datacenters station query responses.
        # LEGEND. In the channel:
        # First letter: D=has dupes, N=no dupes. If N, the channel is always added
        # Second letter: E=expected (is in the eida routing service of this datacenter), N: not expected
        # Third letter: No meaning, (left free to be able play with regexps)
        # Note that in Sensor description we writeif the channel should be saved
        # ("OK: [explanation, if any]") or not ("NO: [explanation, if any]"). Channels starting with
        # 'N', has said, do not need explanation as they have no dupes => written
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A1|aa||DEZ|3|4|6|0|0|0|OK:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A1|aa||DEL|3|4|6|0|0|0|OK:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|ww||NNL|3|4|6|0|0|0|OK:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|xx||DNL|3|4|6|0|0|0|NO: we cannot guess                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
XX|xx||DEL|3|4|6|0|0|0|OK: it's the first                                 |8|0.1|M/S|50.0|2008-02-12T00:00:00|
YY|yy||DEL|3|4|6|0|0|0|NO: channel check done cause it's dupe: no match   |8|0.1|M/S|50.0|2008-02-12T00:00:00|
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
B1|bb||NEZ|3|4|6|0|0|0|OK: channel check not doneo taken                  |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A1|aa||DNZ|3|4|6|0|0|0|NO:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A1|aa||NNZ|3|4|6|0|0|0|OK: starttime changed -> new station               |8|0.1|M/S|50.0|2018-02-12T00:00:00|
A2|xx||DNL|3|4|6|0|0|0|NO: we cannot guess                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|a2||NNL|3|4|6|0|0|0|OK:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
XX|xx||DEL|3|4|6|0|0|0|NO: it's not the first                             |8|0.1|M/S|50.0|2008-02-12T00:00:00|
YY|yy||DEL|3|4|6|0|0|0|OK: channel check done cause it's dupe: matches    |8|0.1|M/S|50.0|2008-02-12T00:00:00|
"""]

        # get channels with the above implemented urlread_sideeffect:
        cha_df = self.get_channels_df(urlread_sideeffect, self.session,
                                      datacenters_df, eidavalidator,
                                      net, sta, loc, cha, None, None, 10,
                                      False, None, None, -1, self.db_buf_size)
        # if u want to check what has been taken, issue in the debugger:
        # str(dbquery2df(self.session.query(Channel.id, Station.network, Station.station, Channel.channel, Channel.station_id, Station.datacenter_id).join(Station)))
        csd = dbquery2df(self.session.query(Channel.sensor_description))
        assert len(csd) == 8
        # assert that the OK string is in the sensor description
        assert all("OK: " for c in csd[Channel.sensor_description.key])
        
        # what happens if we need to query the db? We should get the same isn't it?
        # then set eidavalidator = None
        cha_df2 = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       None,
                                                       net, sta, loc, cha, None, None, 10,
                                                       False, None, None, -1, self.db_buf_size)
        
        # assert that we get the same result as when eidavalidator is None:
        assert cha_df2.equals(cha_df)
    
    
        # now test when the response is different
        urlread_sideeffect[0], urlread_sideeffect[1]  = urlread_sideeffect[1], urlread_sideeffect[0]
        # get channels with the above implemented urlread_sideeffect:
        cha_df3 = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 10,
                                                       False, None, None, -1, self.db_buf_size)
        # we tested visually that everything is ok visually by issuing a 
        # str(dbquery2df(self.session.query(Channel.id, Station.network, Station.station, Channel.channel, Channel.station_id, Station.datacenter_id).join(Station)))
        #, we might add some more specific assert here
        assert len(cha_df3) == 7

        # get channels with the above implemented urlread_sideeffect:
        cha_df4 = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       None,
                                                       net, sta, loc, cha, None, None, 10,
                                                       False, None, None, -1, self.db_buf_size)
        
        assert not cha_df3.equals(cha_df4)  # WHY?? cause they are not sorted, we might have
        # updated some portion of dataframe and inserted the rest on dbmanager.close, or the other way around
        # sort them by id, reset dataframes indices... and they should be the same:
        assert cha_df3.sort_values(by=['id']).reset_index(drop=True).equals(cha_df4.sort_values(by=['id']).reset_index(drop=True))
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


        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(None, self.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)

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
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 10,
                                                       False, None, None, -1, self.db_buf_size
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
        assert not channels_df[channels_df[Station.start_time.key] == datetime(2019,1,1)].empty
        # we need to get the channel id from channels_df cause in df we removed unnecessary columns (including start end time)
        ch_id = channels_df[channels_df[Station.start_time.key] == datetime(2019,1,1)][Channel.id.key].iloc[0]
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
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 100,
                                                       False, None, None, -1, self.db_buf_size
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
        assert not Segment.download_id.key in segments_df.columns
        orig_seg_df = segments_df.copy()
        segments_df, request_timebounds_need_update = \
            prepare_for_download(self.session, orig_seg_df, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True, 
                                 retry_timespan_warn=True)
        assert request_timebounds_need_update is False

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
        assert Segment.download_id.key not in segments_df.columns
        assert len(segments_df) == expected
        # assert len(self.session.query(Segment.id).all()) == len(segments_df)
        
        assert all(x[0] is None for x in self.session.query(Segment.download_code).all())
        assert all(x[0] is None for x in self.session.query(Segment.data).all())

        # mock an already downloaded segment.
        # Set the first 7 to have a particular download status code
        urlerr, mseederr, outtime_err, outtime_warn = custom_download_codes()
        downloadstatuscodes = [None, urlerr, mseederr, 413, 505, outtime_err, outtime_warn]
        for i, download_code in enumerate(downloadstatuscodes):
            dic = segments_df.iloc[i].to_dict()
            dic['download_code'] = download_code
            dic['download_id'] = self.run.id
            # hack for deleting unused columns:
            for col in [Station.network.key, Station.station.key,
                        Channel.location.key, Channel.channel.key]:
                if col in dic:
                    del dic[col]
            # convet numpy values to python scalars:
            # pandas 20+ seems to keep numpy types in to_dict
            # https://github.com/pandas-dev/pandas/issues/13258
            # this was not the case in pandas 0.19.2
            # sql alchemy does not like that        
            # (Curiosly, our pd2sql methods still work fine (we should check why)
            #So, quick and dirty:
            for k in dic.keys():
                if hasattr(dic[k], "item"):
                    dic[k] = dic[k].item()
            # now we can safely add it:
            self.session.add(Segment(**dic))

        self.session.commit()
        
        assert len(self.session.query(Segment.id).all()) == len(downloadstatuscodes)
        
        # Now we have an instance of all possible errors on the db (5 in total) and a new
        # instance (not on the db). Assure all work:
        # set the num of instances to download anyway. Their number is the not saved ones, i.e.:
        to_download_anyway = len(segments_df) - len(downloadstatuscodes)
        for c in product([True, False], [True, False], [True, False], [True, False], [True, False],
                         [True, False], [True, False]):
            s_df, request_timebounds_need_update = \
                prepare_for_download(self.session, orig_seg_df, wtimespan,
                                     retry_seg_not_found=c[0],
                                     retry_url_err=c[1],
                                     retry_mseed_err=c[2],
                                     retry_client_err=c[3],
                                     retry_server_err=c[4],
                                     retry_timespan_err=c[5],
                                     retry_timespan_warn=c[6])
            to_download_in_this_case = sum(c)  # count the True's (bool sum works in python) 
            assert len(s_df) == to_download_anyway +  to_download_in_this_case
            assert request_timebounds_need_update is False

        # now change the window time span and see that everything is to be downloaded again:
        # do it for any retry combinations, as it should ALWAYS return "everything has to be re-downloaded"
        wtimespan[1] += 5
        for c in product([True, False], [True, False], [True, False], [True, False], [True, False],
                         [True, False], [True, False]):
            s_df, request_timebounds_need_update = \
                prepare_for_download(self.session, orig_seg_df, wtimespan,
                                     retry_seg_not_found=c[0],
                                     retry_url_err=c[1],
                                     retry_mseed_err=c[2],
                                     retry_client_err=c[3],
                                     retry_server_err=c[4],
                                     retry_timespan_err=c[5],
                                     retry_timespan_warn=c[6])
            assert len(s_df) == len(orig_seg_df)
            assert request_timebounds_need_update is True  # because we changed wtimespan
        # this hol


    def test_prepare_for_download_sametimespans(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare. event ws returns two events very close by
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|2.01|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 05:17:12.300000|1.001|1.001|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        net, sta, loc, cha = [], [], [], []
        urlread_sideeffect = None
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 100,
                                                       False, None, None, -1, self.db_buf_size
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
        assert not Segment.download_id.key in segments_df.columns
        orig_seg_df = segments_df.copy()
        segments_df, request_timebounds_need_update = \
            prepare_for_download(self.session, orig_seg_df, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True, 
                                 retry_timespan_warn=True)
        
        assert request_timebounds_need_update is False

        logmsg = self.log_msg()
        # the dupes should be the number of segments divided by the events set (2) which are
        # very close
        expected_dupes = len(segments_df) / len(events_df)
        assert ("%d suspiciously duplicated segments found" % expected_dupes) in logmsg

    
    def download_save_segments(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return download_save_segments(*a, **kw)
    
    @patch("stream2segment.download.modules.segments.mseedunpack")
    @patch("stream2segment.io.db.pdsql.insertdf")
    @patch("stream2segment.io.db.pdsql.updatedf")
    def test_download_save_segments(self, mock_updatedf, mock_insertdf, mseed_unpack):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        # mseed unpack takes no starttime and endtime arguments, so that 
        # we do not discard any correct chunk
        mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session, self.service,
                                    self.routing_service, net, sta, loc, cha,
                                    db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 10,
                                                       False, None, None, -1, self.db_buf_size
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
        segments_df, request_timebounds_need_update = \
            prepare_for_download(self.session, orig_segments_df, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True, 
                                 retry_timespan_warn=True)
        
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
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df,
                                            datacenters_df, 
                                            chaid2mseedid,
                                            self.run.id, request_timebounds_need_update,
                                            1,2,3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.id, Segment.channel_id, Segment.datacenter_id,
                Segment.download_code, Segment.maxgap_numsamples, \
                Segment.sample_rate, Segment.data_seed_id, Segment.data, Segment.download_id,
                Segment.request_start, Segment.request_end, Segment.start_time, Segment.end_time
                ]
        db_segments_df = dbquery2df(self.session.query(*cols))
        assert Segment.download_id.key in db_segments_df.columns
        
        
        # change data column otherwise we cannot display db_segments_df. When there is data just print "data"
        db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
                           (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key] = b'data' 

        # assert we have 4 segments with "data" properly set:
        assert len(db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
                           (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key]) == 4

        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

# db_segments_df:
#    id  channel_id  datacenter_id  download_status_code  max_gap_ovlap_ratio  sample_rate data_seed_id     data  run_id          start_time            end_time
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
        
        dsc = db_segments_df[Segment.download_code.key]
        exp_dsc = np.array( [200,200,200, 200, np.nan, 200, -2,-2,-2, -1, 500, 413])
        assert ((dsc == exp_dsc) | (np.isnan(dsc) & np.isnan(exp_dsc))).all()
        # as we have 12 segments and a buf size of self.db_buf_size(=1, but it might change in the
        # future), this below is two
        # it might change if we changed the buf size in the future
        
        # test that we correctly called mock_insertdf. Note that we assume that the
        # latter is called ONLY inside DbManager. To test that, as the number of stuff
        # to be added (length of the dataframes) varies, we need to implement a counter here:
        mock_insertdf_call_count = 0
        _bufzise = 0
        for c in mock_insertdf.call_args_list:
            c_args =  c[0]
            df_ = c_args[0]
            _bufzise += len(df_)
            if _bufzise >= self.db_buf_size:
                mock_insertdf_call_count += 1
                _bufzise = 0
        
        assert mock_insertdf.call_count == mock_insertdf_call_count
        
        # assert data is consistent
        COL = Segment.data.key
        assert (db_segments_df.iloc[:3][COL] == b'data').all()
        assert (db_segments_df.iloc[3:4][COL] == b'').all()
        assert pd.isnull(db_segments_df.iloc[4:5][COL]).all()
        assert (db_segments_df.iloc[5:6][COL] == b'data').all()
        assert pd.isnull(db_segments_df.iloc[6:][COL]).all()
        
        # assert downdload status code is consistent
        URLERR_CODE, MSEEDERR_CODE, OUTTIME_ERR, OUTTIME_WARN = custom_download_codes()

        # also this asserts that we grouped for dc starttime endtime
        COL = Segment.download_code.key
        assert (db_segments_df.iloc[:4][COL] == 200).all()
        assert pd.isnull(db_segments_df.iloc[4:5][COL]).all()
        assert (db_segments_df.iloc[5:6][COL] == 200).all()
        assert (db_segments_df.iloc[6:9][COL] == MSEEDERR_CODE).all()
        assert (db_segments_df.iloc[9][COL] == URLERR_CODE).all()
        assert (db_segments_df.iloc[10][COL] == 500).all()
        assert (db_segments_df.iloc[11][COL] == 413).all()
        
        # assert gaps are only in the given position
        COL = Segment.maxgap_numsamples.key
        assert (db_segments_df.iloc[:3][COL] < 0.01).all()
        assert pd.isnull(db_segments_df.iloc[3:5][COL]).all()
        assert (db_segments_df.iloc[5][COL] == 20).all()
        assert pd.isnull(db_segments_df.iloc[6:][COL]).all()
        
        
        # now mock retry:
        segments_df, request_timebounds_need_update = \
            prepare_for_download(self.session, orig_segments_df, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True, 
                                 retry_timespan_warn=True)
        
        assert request_timebounds_need_update is False

        COL = Segment.download_code.key
        mask = (db_segments_df[COL] >= 400) | pd.isnull(db_segments_df[COL]) \
            | (db_segments_df[COL].isin([URLERR_CODE, MSEEDERR_CODE]))
        assert len(segments_df) == len(db_segments_df[mask])
        
        urlread_sideeffect = [413]
        mock_updatedf.reset_mock()
        mock_insertdf.reset_mock()
        # Let's go:
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df, datacenters_df,
                                            chaid2mseedid,
                                            self.run.id, request_timebounds_need_update,
                                            1,2,3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.download_code, Segment.channel_id]
        db_segments_df = dbquery2df(self.session.query(*cols))
        
        # change data column otherwise we cannot display db_segments_df. When there is data just print "data"
        # db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) & (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key] = b'data' 

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
        
        
        assert mock_insertdf.call_count == 0
        
        
    @patch("stream2segment.download.modules.segments.mseedunpack")
    @patch("stream2segment.io.db.pdsql.insertdf")
    @patch("stream2segment.io.db.pdsql.updatedf")
    def test_download_save_segments_timebounds(self, mock_updatedf, mock_insertdf, mseed_unpack):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        # mseed unpack takes no starttime and endtime arguments, so that 
        mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        
        # mock event response: it's the same as self._evt_urlread_sideeffect but modify the dates
        # as NOW. This means, any segment downloaded later will
        # be out-of-bound
        utcnow = datetime.utcnow()
        utcnow_iso = utcnow.isoformat().replace("T", " ")
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|%s|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|%s|90|90|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
""" % (utcnow_iso, utcnow_iso)
        events_df = self.get_events_df(urlread_sideeffect, self.session, "http://eventws", db_bufsize=self.db_buf_size)
        # restore urlread_side_effect:
        urlread_sideeffect = None
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, self.session, self.service,
                                    self.routing_service, net, sta, loc, cha,
                                    db_bufsize=self.db_buf_size)                                      
        channels_df = self.get_channels_df(urlread_sideeffect, self.session,
                                                       datacenters_df,
                                                       eidavalidator,
                                                       net, sta, loc, cha, None, None, 10,
                                                       False, None, None, -1, self.db_buf_size
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


        wtimespan = [1,2]  # in minutes
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        orig_segments_df = segments_df.copy()
        segments_df, request_timebounds_need_update = \
            prepare_for_download(self.session, orig_segments_df, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True, 
                                 retry_timespan_warn=True)
        
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
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, segments_df,
                                            datacenters_df, 
                                            chaid2mseedid,
                                            self.run.id, request_timebounds_need_update,
                                            1,2,3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.id, Segment.channel_id, Segment.datacenter_id,
                Segment.download_code, Segment.maxgap_numsamples, \
                Segment.sample_rate, Segment.data_seed_id, Segment.data, Segment.download_id,
                Segment.request_start, Segment.request_end, Segment.start_time, Segment.end_time
                ]
        db_segments_df = dbquery2df(self.session.query(*cols))
        assert Segment.download_id.key in db_segments_df.columns
        
        URLERR_CODE, MSEEDERR_CODE, OUTTIME_ERR, OUTTIME_WARN = custom_download_codes()
        # assert no segment has data (time out of bounds):
        assert len(db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
                           (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key]) == 0
        # assert the number of "correctly" downloaded segments, i.e. with data (4) has now
        # code = TIMEBOUND_ERR
        assert len(db_segments_df[db_segments_df[Segment.download_code.key] == OUTTIME_ERR]) == 4
        
        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

# db_segments_df:
#    id  channel_id  datacenter_id  download_status_code  max_gap_ovlap_ratio  sample_rate data_seed_id     data  run_id          start_time            end_time
# 0  1   1           1              -3                    0.0001               100.0        GE.FLT1..HHE    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 1  2   2           1              -3                    0.0001               100.0        GE.FLT1..HHN    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 2  3   3           1              -3                    0.0001               100.0        GE.FLT1..HHZ    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 6  7   7           2              200.0                 NaN                  NaN          None                  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 7  8   8           2              NaN                   NaN                  NaN          None            None  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 8  9   9           2              -3                 20.0                 20.0         IA.BAKI..BHZ    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 3  4   4           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 4  5   5           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 5  6   6           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 9  10  10          2              -1.0                  NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 10 11  11          2              500.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 11 12  12          2              413.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31


 
        # now modify the first row time bounds:
        # first we need to assign the database id to our segments_df, to prevent
        # db contraint error when writing to db:
        # `download_save_segments` below needs toi UPDATE the segments and it does it by checking if an 
        # id is present
        # check that the channel_ids align:
        assert (segments_df[Segment.channel_id.key].values == db_segments_df[Segment.channel_id.key].values).all()
        # so that we can simply do this:
        segments_df[Segment.id.key] = db_segments_df[Segment.id.key]
        
        # first read the miniseed:
        stream = read(BytesIO(self._seg_data))
        tstart = stream[0].stats.starttime.datetime
        tend = stream[0].stats.endtime.datetime
        segments_df.loc[segments_df[Segment.channel_id.key]==1, Segment.request_start.key] = tstart
        segments_df.loc[segments_df[Segment.channel_id.key]==1, Segment.request_end.key] = tstart + (tend-tstart)/2
        
        segments_df.loc[segments_df[Segment.channel_id.key]==2, Segment.request_start.key] = tstart
        segments_df.loc[segments_df[Segment.channel_id.key]==2, Segment.request_end.key] = tend
        
        # build a segments_df of the three segments belonging to the same channel:
        new_segments_df = segments_df.loc[segments_df[Segment.channel_id.key].isin([1,2,3]), :]
        # change urlread_side_effect to provide, for the first three segments, the same
        # sequence of bytes. The sequence actually is OK, but in the first case it will be PARTIALLY saved
        # in the second case TOTALLY, and in the thrid case NOT AT ALL:
        urlread_sideeffect = [self._seg_data, self._seg_data, self._seg_data]
        ztatz = self.download_save_segments(urlread_sideeffect, self.session, new_segments_df,
                                            datacenters_df, 
                                            chaid2mseedid,
                                            self.run.id, request_timebounds_need_update,
                                            1,2,3, db_bufsize=self.db_buf_size)
        db_segments_df = dbquery2df(self.session.query(*cols))
        # re-sort db_segments_df to match the segments_df:
        ret = [db_segments_df[db_segments_df[Segment.channel_id.key] == cha]
               for cha in segments_df[Segment.channel_id.key]]
        db_segments_df = pd.concat(ret, axis=0)
        
        # assert the 1st segment whose time range has been modified has data, BUT download_status_code still TIMEBOUNDS_ERROR
        df__ = db_segments_df.loc[db_segments_df[Segment.channel_id.key]==1, :]
        assert len(df__) == 1
        row__ = df__.iloc[0]
        assert row__[Segment.download_code.key] == OUTTIME_WARN
        assert len(row__[Segment.data.key]) > 0
        
        # assert the 2nd segment whose time range has been modified has data, AND download_status_code 200 (ok)
        df__ = db_segments_df.loc[db_segments_df[Segment.channel_id.key]==2, :]
        assert len(df__) == 1
        row__ = df__.iloc[0]
        assert row__[Segment.download_code.key] == 200
        assert len(row__[Segment.data.key]) > 0
        
         # assert the 3rd segment whose time range has NOT been modified has no data, AND download_status_code is still TIMEBOUNDS_ERROR
        df__ = db_segments_df.loc[db_segments_df[Segment.channel_id.key]==3, :]
        assert len(df__) == 1
        row__ = df__.iloc[0]
        assert row__[Segment.download_code.key] == OUTTIME_ERR
        assert len(row__[Segment.data.key]) == 0
