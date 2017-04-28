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
from stream2segment.download.query import add_classes, get_events_df, get_datacenters_df, logger as query_logger, \
get_channels_df, merge_events_stations, set_saved_arrivaltimes, get_arrivaltimes,\
    prepare_for_download
# ,\
#     get_fdsn_channels_df, save_stations_and_channels, get_dists_and_times, set_saved_dist_and_times,\
#     download_segments, drop_already_downloaded, set_download_urls, save_segments
from obspy.core.stream import Stream
from stream2segment.io.db.models import DataCenter, Segment, Run, Station, Channel
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
from logging import StreamHandler
import logging
from _io import BytesIO
import urllib2

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
        
        self.patchers = [self.patcher, self.patcher1, self.patcher2, self.patcher3]
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
20160508_0000004|2016-05-08 01:45:30.300000|2|2|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        self._dc_urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

"""

        self._sta_urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|a||HHZ|1|1|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
A|b||HHE|2|2|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|c||HHZ|3|3|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
BLA|e||HHZ|7|7|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|2019-01-01T00:00:00
BLA|e||HHZ|8|8|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
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
            
        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_empty, self._seg_data_gaps,
                                        URLError('url_error_segment'), 500]

        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self.session, self.handler, *self.patchers)
                        #self.patcher3)

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

        def scoped_func(string):
            bio = BytesIO(string)
            def func(*a, **v):
                return bio.read(*a, **v)
            return func

        bytezios = {}  # needed to use proper inner scope ref (in py3 there is 'let')
        for k in urlread_side_effect:
            a = Mock()
            if type(k) == int:
                a.read.side_effect = urllib2.HTTPError('url', int(k),  responses[k][0], None, None)
            elif type(k) == str:
                func = scoped_func(k)
                a.read.side_effect = func
                a.code = 200
                a.msg = responses[a.code][0]
            else:
                a.read.side_effect = k
            retvals.append(a)
        
        self.mock_urlopen.side_effect = cycle(retvals)
        
        

    def test_add_classes(self):
        cls = {'a' : 'bla', 'b' :'c'}
        add_classes(self.session, cls)
        assert len(self.session.query(Class).all()) == 2
        logmsg = self.log_msg()
        assert "2 new item(s) saved" in logmsg
        add_classes(self.session, cls)
        # assert we did not write any new class
        assert len(self.session.query(Class).all()) == 2
        # and also that nothing is printed to log (we print only new and discarded)
        assert logmsg == self.log_msg()
        
# ===========================

    def get_events_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_events_df(self.session, *a, **v)


    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_events(self, mock_query):
        urlread_sideeffect = ["""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""]
        
        data = self.get_events_df(urlread_sideeffect, "http://eventws")
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
        
        data = self.get_events_df(urlread_sideeffect, "http://eventws")
        # assert nothing new has added:
        assert len(self.session.query(Event).all()) == len(pd.unique(data['id'])) == 3
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        assert len(data) == 3
        # assert mock_urlread.call_args[0] == (mock_query.return_value, )
        
        assert "blabla23___" in self.log_msg()
        

    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_events_toomany_requests_raises(self, mock_query): # FIXME: implement it!
        ## FIXMEEEEE TO BE IMPLEMENTED!!!!
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
        with pytest.raises(ValueError):
            data = self.get_events_df(urlread_sideeffect, "http://eventws", start=datetime(2010,1,1).isoformat(),
                                      end=datetime(2011,1,1).isoformat())
        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == 0
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        with pytest.raises(NameError):
            assert len(data) == 3
        

    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_events_toomany_requests_doesntraise(self, mock_query): # FIXME: implement it!
        ## FIXMEEEEE TO BE IMPLEMENTED!!!!
        urlread_sideeffect = [413, """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", ""]
        # as urlread returns alternatively a 413 and a good string, also sub-queries
        # will return that, so that we will end up having a 413 when the string is not
        # further splittable:
        data = self.get_events_df(urlread_sideeffect, "http://eventws", start=datetime(2010,1,1).isoformat(),
                                      end=datetime(2011,1,1).isoformat())
        # assert only three events were successfully saved to db (two have same id) 
        assert len(self.session.query(Event).all()) == 3
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        assert len(data) == 3

        # assert logger has been written with the first 413 error:
        assert "request entity too large" in self.log_msg()

# =================================================================================================

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_datacenters_df(self.session, *a, **v)
    

    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_dcs_malformed(self, mock_query):
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query

ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, None)
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        # geofon has actually a post line since 'indentation is bad..' is splittable)
        assert post_data_list[0] is None and post_data_list[1] is not None and \
            '\n' in post_data_list[1]
        
    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_dcs2(self, mock_query):
        urlread_sideeffect = ["""http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query

ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
"""]
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, ['BH?'])
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.assert_called_once()  # we might be more fine grained, see code
        assert post_data_list[0] is not None and post_data_list[1] is None
        
        # now download with a channel matching:
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect, ['H??'])
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        assert mock_query.call_count == 2  # we might be more fine grained, see code
        assert post_data_list[0] is not None and post_data_list[1] is not None
        # assert we have only one line for each post request:
        assert all('\n' not in r for r in post_data_list)
        
    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_dcs3(self, mock_query):
        urlread_sideeffect = [500, """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

""", 501]
        
        with pytest.raises(ValueError):
            data, post_data_list = self.get_datacenters_df(urlread_sideeffect[0], ['BH?'])
        assert len(self.session.query(DataCenter).all()) == 0
        mock_query.assert_called_once()  # we might be more fine grained, see code
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[1], ['BH?'])
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.call_count == 2  # we might be more fine grained, see code
        assert post_data_list[0] is not None and post_data_list[1] is None
        
        # this raises again a server error, but we have datacenters in cahce:
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[2], ['BH?'])
        assert len(self.session.query(DataCenter).all()) == len(data) == 2
        mock_query.call_count == 3  # we might be more fine grained, see code
        assert post_data_list is None
        
    
    @patch('stream2segment.download.query.urljoin', return_value='a')
    def test_get_dcs_postdata_all_nones(self, mock_query):
        urlread_sideeffect = ["""http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
http://ws.resif.fr/fdsnws/dataselect/1/query
"""]
        
        data, post_data_list = self.get_datacenters_df(urlread_sideeffect[0], ['BH?'])
        assert all(x is None for x in post_data_list)

# =================================================================================================



    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_channels_df(self.session, *a, **kw)
     
    def test_get_channels_df(self):
        urlread_sideeffect = """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""
        events_df = self.get_events_df(urlread_sideeffect, "http://eventws")

        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25
http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

"""
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, channels)
        
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
                                      
        cha_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1
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
        cha_df2 = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       None,
                                                       channels,
                                                       100, 'a', 'b', -1
                                               )

        assert len(cha_df2) == len(cha_df)
        assert sorted(list(cha_df.columns)) == sorted(list(cha_df2.columns))
        
        # now mock a datacenter null postdata in the second item (no data in eida routing service under a specific datacenter)
        # and test that by querying the database we get the same data (the one we just saved)
        # this raises cause the first datacenter has no channels (malformed)
        with pytest.raises(ValueError):
            cha_df2 = self.get_channels_df(urlread_sideeffect,
                                                           datacenters_df,
                                                           ['x', None],
                                                           channels,
                                                           100, 'a', 'b', -1
                                                   )
        assert 'channels: discarding response data' in self.log_msg()

        # now test the opposite, but note that urlreadside effect should return now an urlerror and a socket error:
        with pytest.raises(ValueError):
            cha_df2 = self.get_channels_df(URLError('urlerror_wat'),
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels,
                                                           100, 'a', 'b', -1
                                                   )
        assert 'urlerror_wat' in self.log_msg()

        # now test again, we should ahve a socket timeout
        with pytest.raises(ValueError):
            cha_df2 = self.get_channels_df(socket.timeout(),
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels,
                                                           100, 'a', 'b', -1
                                                   )
        assert 'timeout' in self.log_msg()
        
        # now the case where the none post request is for the "good" repsonse data:
        # Basically, replace ['x', None] with [None, 'x'] and swap urlread side effects
        cha_df2 = self.get_channels_df([urlread_sideeffect[1], urlread_sideeffect[0]],
                                                           datacenters_df,
                                                           [None, 'x'],
                                                           channels,
                                                           100, 'a', 'b', -1
                                                   )
        
        assert len(cha_df2) == 6
        # now change min sampling rate and see what happens (we do not have one channel more
        # cause we have overlapping names, and the 50 Hz channel is overridden by the second
        # query) 
        cha_df3 = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1
                                               )
        assert len(cha_df3) == len(cha_df)
        
        # now change this:
        
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|B||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
""", 
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
E|F||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  URLError('wat'), socket.timeout()]
                                      
        cha_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1
                                               )

        assert len(cha_df) == 1
        assert "sample rate <" in self.log_msg()
        
        # now decrease the sampling rate, we should have two channels (all):
        cha_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1
                                               )

        assert len(cha_df) == 2
        
        # now change channels=['B??'], we should have no effect as the channels takes effect
        # when postdata is None (=query to the db)
        cha_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       ['B??'],
                                                       10, 'a', 'b', -1
                                               )

        assert len(cha_df) == 2
        
        # let's see if now channels=['BH?'] has effect:
        # now change channels=['B??'], we should have no effect as the channels takes effect
        # when postdata is None (=query to the db)
        # remember that empty data raises ValueError
        with pytest.raises(ValueError):
            cha_df = self.get_channels_df(urlread_sideeffect,
                                                           datacenters_df,
                                                           None,
                                                           ['B??'],
                                                           10, 'a', 'b', -1
                                                   )

        

# FIXME: text save inventories!!!!

    def test_merge_event_stations(self):
        # get events with lat lon (1,1), (2,2,) ... (n, n)
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|2|2|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        events_df = self.get_events_df(urlread_sideeffect, "http://eventws")

        # this urlread_sideeffect is actually to be considered for deciding which datacenters to store,
        # their post data is not specified as it
        # would be ineffective as it is overridden by the urlread_sideeffect
        # specified below for the channels
        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
http://ws.resif.fr/fdsnws/dataselect/1/query
"""
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(None, channels)

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
                                      
        channels_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       10, 'a', 'b', -1
                                               )
        
        assert len(channels_df) == 5

    # events_df
#                  id  magnitude  latitude  longitude  depth_km  time  
# 0  20160508_0000129        3.0       1.0        1.0      60.0  2016-05-08 05:17:11.500
# 1  20160508_0000004        4.0       2.0        2.0       2.0  2016-05-08 01:45:30.300 

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
        df = merge_events_stations(events_df, channels_df, sradius_minmag=10, sradius_maxmag=10,
                                   sradius_minradius=0, sradius_maxradius=200)
        
        assert len(df) == 2
        
        # for magnitude <1, max_radius is 100. For magnitude >1, max_radius is 200
        # we have only magnitudes <10, we have all event-stations closer than 100 deg
        # So we might have ALL channels taken BUT: one station start time is in 2019, thus
        # it will not fall into the case above!
        df = merge_events_stations(events_df, channels_df, sradius_minmag=1, sradius_maxmag=1,
                                   sradius_minradius=100, sradius_maxradius=2000)
        
        assert len(df) == (len(channels_df)-1) *len(events_df)
        # assert channel outside time bounds was in:
        assert not channels_df[channels_df[Segment.start_time.key] == datetime(2019,1,1)].empty
        # we need to get the channel id from channels_df cause in df we removed unnecessary columns (including start end time)
        ch_id = channels_df[channels_df[Segment.start_time.key] == datetime(2019,1,1)][Channel.id.key].iloc[0]
        # old Channel.id.key is Segment.channel_id.key in df:
        assert df[df[Segment.channel_id.key] == ch_id].empty
        
        # this is a more complex case, we want to drop the first event by setting a very low
        # threshold (sraidus_minradius=1) for magnitudes <=3 (the first event magnitude)
        # and sradius_maxradius very high for the other event (magnitude=4)
        df = merge_events_stations(events_df, channels_df, sradius_minmag=3, sradius_maxmag=4,
                                   sradius_minradius=1, sradius_maxradius=40)
        
        # assert we have only the second event except the first channel which is from the 1st event.
        # FIXME: more fine grained tests based on distance?
        assert np.array_equal((df[Segment.event_id.key] == '20160508_0000004'),
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
 

    def tst_getset_arrivaltimes(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, "http://eventws")
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, channels)                                      
        channels_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1
                                               )
        assert len(channels_df) == 4  # just to be sure. If failing, we might have changed the class default
    # events_df
#                  id  magnitude  latitude  longitude  depth_km  time  
# 0  20160508_0000129        3.0       1.0        1.0      60.0  2016-05-08 05:17:11.500
# 1  20160508_0000004        4.0       2.0        2.0       2.0  2016-05-08 01:45:30.300 

    # channels_df:
#     id  station_id  latitude  longitude  datacenter_id start_time   end_time
#  0   1           1       3.0        3.0              2 2008-02-12        NaT
#  1   2           2       7.0        7.0              2 2009-01-01 2019-01-01
#  2   3           3       8.0        8.0              2 2019-01-01        NaT
#  4   4           4       2.0        2.0              1 2009-01-01        NaT

        # take all segments:
        df = merge_events_stations(events_df, channels_df, sradius_minmag=10, sradius_maxmag=10,
                                   sradius_minradius=100, sradius_maxradius=200)
        
        h = 9
        
# df:
#    channel_id  station_id  latitude  longitude  datacenter_id event_id         event_distance_deg depth_km time  
# 0  1           1           2.0       2.0        1             20160508_0000129 1.413962           60.0     2016-05-08 05:17:11.500  
# 1  2           2           3.0       3.0        2             20160508_0000129 2.827494           60.0     2016-05-08 05:17:11.500  
# 2  3           3           7.0       7.0        2             20160508_0000129 8.472983           60.0     2016-05-08 05:17:11.500  
# 3  1           1           2.0       2.0        1             20160508_0000004 0.000000            2.0     2016-05-08 01:45:30.300  
# 4  2           2           3.0       3.0        2             20160508_0000004 1.413532            2.0     2016-05-08 01:45:30.300  
# 5  3           3           7.0       7.0        2             20160508_0000004 7.059033            2.0     2016-05-08 01:45:30.300


        # NOW TEST GET ARRIVAL TIMES
        # FIRST WE HAVE DB EMPTY THUS NO UPADTE SHOULD BE MADE
        assert Segment.arrival_time.key not in df.columns
        evts_stations_df = set_saved_arrivaltimes(self.session, df)
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
        EVID = '20160508_0000129'
        self.session.add(Segment(id = 1, # , default=seg_pkey_default, 
                         event_id = EVID,
                         channel_id = 1,  # the channel here must have station id = SID (see above)
                         datacenter_id = 1,
                         seed_identifier = 'abc',
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
        assert len(evts_stations_df[filter]) == 1
        assert all(evts_stations_df[filter][Segment.arrival_time.key] == atime)
        assert all(evts_stations_df[filter][Segment.event_distance_deg.key] == evdist)
        
        # NOW mock get_dist_and_times

        # test first with a function that returns a TypeError for any segment:
        def deterministic_mintraveltime_sideeffect(*a, **kw):
            return datetime.utcnow()  # we should return the TOTAL seconds, NOT a datetime. Thus we should have all errors

        expected_length = 1
        # make a copy of evts_stations_df cause we will modify in place the data frame
        segments_df =  self.get_arrivaltimes(deterministic_mintraveltime_sideeffect, evts_stations_df.copy(),
                                                   [1,2], ['P', 'Q'],
                                                        'ak135')
        
        # all failed, except the one we just set by mocking the db:
        assert len(segments_df) == expected_length  # cause 1 was just added to the db and it's not recalculated
        
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
                                                        'ak135')
        
        # all failed, except the one we just set by mocking the db:
        assert len(segments_df) == expected_length
        assert Segment.start_time.key in segments_df.columns
        assert Segment.end_time.key in segments_df.columns

        # FIXME: we should assert times are correctly calculated with respect to event time


        
    def test_download_save_segments(self):  #, mock_urlopen_in_async, mock_url_read, mock_arr_time):
        # prepare:
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, "http://eventws")
        channels = None
        datacenters_df, postdata = self.get_datacenters_df(urlread_sideeffect, channels)                                      
        channels_df = self.get_channels_df(urlread_sideeffect,
                                                       datacenters_df,
                                                       postdata,
                                                       channels,
                                                       100, 'a', 'b', -1
                                               )
        assert len(channels_df) == 4  # just to be sure. If failing, we might have changed the class default
    # events_df
#                  id  magnitude  latitude  longitude  depth_km  time  
# 0  20160508_0000129        3.0       1.0        1.0      60.0  2016-05-08 05:17:11.500
# 1  20160508_0000004        4.0       2.0        2.0       2.0  2016-05-08 01:45:30.300 

    # channels_df:
#     id  station_id  latitude  longitude  datacenter_id start_time   end_time
#  0   1           1       3.0        3.0              2 2008-02-12        NaT
#  1   2           2       7.0        7.0              2 2009-01-01 2019-01-01
#  2   3           3       8.0        8.0              2 2019-01-01        NaT
#  4   4           4       2.0        2.0              1 2009-01-01        NaT

        # take all segments:
        df = merge_events_stations(events_df, channels_df, sradius_minmag=10, sradius_maxmag=10,
                                   sradius_minradius=100, sradius_maxradius=200)
        
        h = 9
        
# df:
#    channel_id  station_id  latitude  longitude  datacenter_id event_id         event_distance_deg depth_km time  
# 0  1           1           2.0       2.0        1             20160508_0000129 1.413962           60.0     2016-05-08 05:17:11.500  
# 1  2           2           3.0       3.0        2             20160508_0000129 2.827494           60.0     2016-05-08 05:17:11.500  
# 2  3           3           7.0       7.0        2             20160508_0000129 8.472983           60.0     2016-05-08 05:17:11.500  
# 3  1           1           2.0       2.0        1             20160508_0000004 0.000000            2.0     2016-05-08 01:45:30.300  
# 4  2           2           3.0       3.0        2             20160508_0000004 1.413532            2.0     2016-05-08 01:45:30.300  
# 5  3           3           7.0       7.0        2             20160508_0000004 7.059033            2.0     2016-05-08 01:45:30.300


        evts_stations_df = set_saved_arrivaltimes(self.session, df)
                # make a copy of evts_stations_df cause we will modify in place the data frame
        segments_df =  self.get_arrivaltimes(urlread_sideeffect, evts_stations_df.copy(),
                                                   [1,2], ['P', 'Q'],
                                                        'ak135')
        
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        segments_df = prepare_for_download(self.session, segments_df,
                                           retry_no_code=True,
                                           retry_url_errors=True,
                                           retry_mseed_errors=True,
                                           retry_4xx=True,
                                           retry_5xx=True)
        
        assert len(segments_df) == expected
        assert len(session.query(Segment.id).all()) == len(segments_df)
        
        # segments_df:
# channel_id  station_id  datacenter_id event_id         event_distance_deg arrival_time            start_time          end_time
# 1           1           1             20160508_0000129 1.413962           2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 2           2           2             20160508_0000129 2.827494           2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 3           3           2             20160508_0000129 8.472983           2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 1           1           1             20160508_0000004 0.000000           2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 2           2           2             20160508_0000004 1.413532           2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 3           3           2             20160508_0000004 7.059033           2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31












































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
        def deterministic_mintraveltime_sideeffect(*a, **kw):
            evt_depth_km = a[1]
            if evt_depth_km == 60:
                return datetime.utcnow()
            raise TauModelError('wat?')
        
        segments_df =  self.getset_dists_and_times(deterministic_mintraveltime_sideeffect, events_df, channels_df,
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
        def deterministic_mintraveltime_sideeffect(*a, **kw):
            evt_depth_km = a[1]
            if evt_depth_km == 60:
                return datetime.utcnow()
            raise TauModelError('wat?')
        
        segments_df =  self.getset_dists_and_times(deterministic_mintraveltime_sideeffect, events_df, channels_df,
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

