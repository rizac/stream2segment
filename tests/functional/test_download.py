#@PydevCodeAnalysisIgnore
'''
Created on Feb 4, 2016

@author: riccardo
'''
from __future__ import print_function
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')

from future import standard_library
standard_library.install_aliases()
from builtins import str
import re
import numpy as np
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta
import sys

# this can apparently not be avoided neither with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
try:
    from cStringIO import StringIO  # python2.x
except ImportError:
    from io import StringIO

import unittest, os
from sqlalchemy.engine import create_engine
from stream2segment.io.db.models import Base, Event, Class, WebService
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from stream2segment.main import main, closing
from click.testing import CliRunner
# from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_col_names
import pandas as pd
from stream2segment.download.main import get_events_df, get_datacenters_df, \
get_channels_df, merge_events_stations, \
    prepare_for_download, download_save_segments, save_inventories
# ,\
#     get_fdsn_channels_df, save_stations_and_channels, get_dists_and_times, set_saved_dist_and_times,\
#     download_segments, drop_already_downloaded, set_download_urls, save_segments
from obspy.core.stream import Stream, read
from stream2segment.io.db.models import DataCenter, Segment, Run, Station, Channel, WebService,\
    withdata
from itertools import cycle, repeat, count, product
from urllib.error import URLError
import socket
from obspy.taup.helper_classes import TauModelError
# import logging
# from logging import StreamHandler

# from stream2segment.main import logger as main_logger
from sqlalchemy.sql.expression import func
from stream2segment.utils import get_session, mseedlite3
from stream2segment.io.db.pd_sql_utils import dbquery2df, insertdf_napkeys, updatedf
from logging import StreamHandler
import logging
from io import BytesIO
import urllib.request, urllib.error, urllib.parse
from stream2segment.download.utils import get_url_mseed_errorcodes
from test.test_userdict import d1
from stream2segment.utils.mseedlite3 import MSeedError, unpack
import threading
from stream2segment.utils.url import read_async
from stream2segment.utils.resources import get_templates_fpath
from stream2segment.utils.log import configlog4download
from stream2segment.io.db.pd_sql_utils import _get_max as _get_db_autoinc_col_max

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
        
#         hndls = s2s_download_logger.handlers[:]
#         handler.close()
#         for h in hndls:
#             if h is handler:
#                 s2s_download_logger.removeHandler(h)

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
        engine = create_engine(self.dburi, echo=False)
        Base.metadata.create_all(engine)
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        # create a Session
        self.session = Session()
        self.engine = engine
        
        self.patcher = patch('stream2segment.utils.url.urllib.request.urlopen')
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
        self.patcher23 = patch('stream2segment.download.main.original_read_async')
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
        # s2s_download_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset (to level.INFO)
        # otherwise it stays what we set two lines above. Problems might arise if closing
        # sets a different level, but for the moment who cares
        # s2s_download_logger.addHandler(self.handler)
        
        self.patcher29 = patch('stream2segment.main.configlog4download')
        self.mock_config4download = self.patcher29.start()
        def c4d(logger, *a, **v):
            ret = configlog4download(logger, *a, **v)
            logger.addHandler(self.handler)
            return ret
        self.mock_config4download.side_effect = c4d
             
        
        self.patchers = [self.patcher, self.patcher1, self.patcher2, self.patcher23,
                         self.patcher29]
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
        
        self.configfile = get_templates_fpath("download.yaml")
        
    def log_msg(self):
        return self.logout.getvalue()
    
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
#        self.mock_urlopen.side_effect = Cycler(urlread_side_effect)
        

    
    def get_events_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_events_df(*a, **v)
        


    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_datacenters_df(*a, **v)
    

    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return get_channels_df(*a, **kw)

# # ================================================================================================= 

    def download_save_segments(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None else url_read_side_effect)
        return download_save_segments(*a, **kw)
    
    def save_inventories(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._get_inv() if url_read_side_effect is None else url_read_side_effect)
        save_inventories(*a, **v)

    
    def _get_inv(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "inventory_GE.APE.xml")
        with open(path, 'rb') as opn_:
            return opn_.read()


    @patch('stream2segment.io.db.pd_sql_utils._get_max')
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.main.mseedunpack')
    @patch('stream2segment.download.main.insertdf_napkeys')
    @patch('stream2segment.download.main.updatedf')
    def test_cmdline_dberr(self, mock_updatedf, mock_insertdf_napkeys, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                    mock_get_datacenters_df, mock_get_events_df, mock_autoinc_db):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf_napkeys.side_effect = lambda *a, **v: insertdf_napkeys(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(self.session.query(Segment).all())
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(self.session.query(Run.id).all())


        # first thing to check is that, when we have a db error, channels and stations are correctly
        # written.
        # this calls the original function whihc returns the autoincrement col BUT
        # with -1 we should actually mess up things so elements are not written
        # maybe except the first one which is zero
        # BUT ONLY FOR SEGMENTS!
        def _mapkey(session, column):
            ret = _get_db_autoinc_col_max(session, column)
            if column.class_ == Segment:
                ret = 0  # this should make the first bulk of segments writtable
                # but from the next one on failing. Not really a deterministic way to know
                # how many we wrote, but it's already something
            return ret
                
        mock_autoinc_db.side_effect = _mapkey
        
        runner = CliRunner()
        result = runner.invoke(main , ['d',
                                       '-c', self.configfile,
                                        '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            print("EXCEPTION")
            print("=========")
            print("")
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            print("")
            print("=========")
            assert False
            return
        
        assert self.session.query(Station).count() == 4
        
        # assert log msg printed
        assert """duplicate key value violates unique constraint "segments_pkey"
DETAIL:  Key (id)=(1) already exists""" if self.is_postgres else \
"(UNIQUE constraint failed: segments.id)" in self.log_msg() 
    
        
        # get the excpeted segment we should have downloaded:
        segments_df = mock_download_save_segments.call_args_list[0][0][1]
        assert self.session.query(Segment).count() < len(segments_df)
        
        # get the first group written to the db. Note that as we mocked read_async (see above)
        # the first dataframe given to urlread should be also the first to be written to db,
        # and so on for the second, third. If the line below fails, check that maybe it's not
        # the case and we should be less strict. Actually, we will be less strict, 
        # turns out the check is undeterministic Comment out:
#         first_segments_df = segments_df.groupby(['datacenter_id', 'start_time', 'end_time'], sort=False).first()
        assert self.session.query(Segment).count()  == 3  # len(first_segments_df)
        # assert 
        assert self.session.query(Channel).count() == 12
        assert self.session.query(Event).count() == 2
        
        # assert run log has been written correctly, i.e. that the db error on the segments
        # has not affected further writing operations. To do this quickly, assert that
        # all run.log have something written in (not null, not empty)
        assert self.session.query(withdata(Run.log)).count() == self.session.query(Run).count()
        
             
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.main.mseedunpack')
    @patch('stream2segment.download.main.insertdf_napkeys')
    @patch('stream2segment.download.main.updatedf')
    def test_cmdline(self, mock_updatedf, mock_insertdf_napkeys, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                    mock_get_datacenters_df, mock_get_events_df):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf_napkeys.side_effect = lambda *a, **v: insertdf_napkeys(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(self.session.query(Segment).all())
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(self.session.query(Run.id).all())



        runner = CliRunner()
        result = runner.invoke(main , ['d',
                                       '-c', self.configfile,
                                        '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            print("EXCEPTION")
            print("=========")
            print("")
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            print("")
            print("=========")
            assert False
            return
        
        assert len(self.session.query(Run.id).all()) == runs + 1
        runs += 1
        segments = self.session.query(Segment).all()
        assert len(segments) == 12
        segments = self.session.query(Segment).filter(Segment.has_data).all()
        assert len(segments) == 4
        
        assert len(self.session.query(Station).filter(Station.has_inventory).all()) == 0
        
        assert not mock_updatedf.called
        assert mock_insertdf_napkeys.called
        
        dfres1 = dbquery2df(self.session.query(Segment.id, Segment.channel_id, Segment.datacenter_id,
                                               Segment.event_id,
                                         Segment.download_status_code, Segment.data,
                                         Segment.max_gap_ovlap_ratio, Segment.run_id,
                                         Segment.sample_rate, Segment.seed_identifier))
        dfres1.sort_values(by=Segment.id.key, inplace=True)  # for easier visual compare
        dfres1.reset_index(drop=True, inplace=True)  # we need to normalize indices for comparison later
        # just change the value of the bytes so that we can better 
        # visually inspect dataframe under clipse, in case:
        dfres1.loc[(~pd.isnull(dfres1[Segment.data.key])) & (dfres1[Segment.data.key].str.len()>0),
                  Segment.data.key] = b'data'




        # re-launch with the same setups.
        # what we want to test is the 413 error on every segment. This should split downloads and
        # retry until there is just one segment and this should write 413 for all segments to retry
        # WARNING: THIS TEST COULD FAIL IF WE CHANGE THE DEFAULTS. CHANGE `mask` IN CASE
        mock_download_save_segments.reset_mock()
        mock_updatedf.reset_mock()
        mock_insertdf_napkeys.reset_mock()
        self._seg_urlread_sideeffect = [413]
        idx = len(self.log_msg())
        runner = CliRunner()
        result = runner.invoke(main , ['d',
                                       '-c', self.configfile,
                                        '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        
        dfres2 = dbquery2df(self.session.query(Segment.id, Segment.channel_id, Segment.datacenter_id,
                                               Segment.event_id,
                                         Segment.download_status_code, Segment.data,
                                         Segment.max_gap_ovlap_ratio, Segment.run_id,
                                         Segment.sample_rate, Segment.seed_identifier))
        dfres2.sort_values(by=Segment.id.key, inplace=True)  # for easier visual compare
        dfres2.reset_index(drop=True, inplace=True)  # we need to normalize indices for comparison later
        # just change the value of the bytes so that we can better 
        # visually inspect dataframe under clipse, in case:
        dfres2.loc[(~pd.isnull(dfres2[Segment.data.key])) & (dfres2[Segment.data.key].str.len()>0),
                  Segment.data.key] = b'data'
        
        assert mock_updatedf.called
        assert not mock_insertdf_napkeys.called
        
        URLERROR, MSEEDERROR = get_url_mseed_errorcodes()
        
        assert len(dfres2) == len(dfres1)
        assert len(self.session.query(Run.id).all()) == runs + 1
        runs += 1
        # asssert we changed the download status code for segments which should be retried
        # WARNING: THIS TEST COULD FAIL IF WE CHANGE THE DEFAULTS. CHANGE `mask` IN CASE
        mask = dfres1[Segment.download_status_code.key].between(500, 599.999, inclusive=True) | \
                      (dfres1[Segment.download_status_code.key] == URLERROR) | \
                      pd.isnull(dfres1[Segment.download_status_code.key])
        retried = dfres2.loc[mask, :]
        assert (retried[Segment.download_status_code.key] == 413).all()
        # asssert we changed the run_id for segments which should be retried
        # WARNING: THIS TEST COULD FAIL IF WE CHANGE THE DEFAULTS. CHANGE THE `mask` IN CASE
        assert (retried[Segment.run_id.key] > dfres1.loc[retried.index, Segment.run_id.key]).all()
        
        assert mock_download_save_segments.called



        # Ok, now with the current config 413 is not retried: 
        # check that now we should skip all segments
        mock_download_save_segments.reset_mock()
        runner = CliRunner()
        result = runner.invoke(main , ['d', 
                                       '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        assert not mock_download_save_segments.called
        
        
        # test some edge cases, if run from eclipse, a debugger and inspection of self.log_msg()
        # might be needed to check that everything is printed right. IF WE CHANGE THE MESSAGES
        # TO BE DISPLAYED, THEN CHANGE THE STRING BELOW:
        str_err = "routing service error, trying to work with 2 already saved data-centers"
        assert str_err not in self.log_msg()
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(500, *a, **v) 
        mock_download_save_segments.reset_mock()
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        assert not mock_download_save_segments.called
        assert str_err in self.log_msg()
        str_1 = "Fetching stations and channels from db for %d data-center(s)" % self.session.query(DataCenter).count()
        assert str_1 in self.log_msg()
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        

        # test some edge cases, if run from eclipse, a debugger and inspection of self.log_msg()
        # might be needed to check that everything is printed right. IF WE CHANGE THE MESSAGES
        # TO BE DISPLAYED, THEN CHANGE THE STRING BELOW:
        str_err = "No channel found with sample rate"
        assert str_err not in self.log_msg()
        # assert str_err not in self.log_msg()
        def mgdf(*a, **v):
            aa = list(a)
            aa[6] = 100000  # change min sample rate to a huge number
            return self.get_channels_df(None, *aa, **v) 
        mock_get_channels_df.side_effect = mgdf
        
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        assert str_err in self.log_msg()
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v) 
        
        
        # now we mock urlread for stations: it always raises 500 error 500
        # thus the program should query the database as station urlread raises
        # and we cannot get any station from the web
        str_2 = "Nothing to download: all segments already downloaded according to the current configuration"
        idx = self.log_msg().find(str_2)
        assert idx > -1
        rem = self._sta_urlread_sideeffect
        self._sta_urlread_sideeffect = 500
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        # assert we wrote again str_2:
        assert self.log_msg().rfind(str_2) > idx
        
        # reset to default:
        self._sta_urlread_sideeffect = rem
        
        
        # test with loading station inventories:
        
        # we should not have inventories saved:
        stainvs = self.session.query(Station).filter(Station.has_inventory).all()
        assert len(stainvs) == 0
        # calculate the expected stations:
        expected_invs_to_download_ids = [x[0] for x in self.session.query(Station.id).filter((~Station.has_inventory) &
                   (Station.segments.any(Segment.has_data))).all()]
        # test that we have data, but also errors
        num_expected_inventories_to_download = len(expected_invs_to_download_ids)
        assert num_expected_inventories_to_download == 2  # just in order to set the value below
        # and be more safe about the fact that we will have only ONE station inventory saved
        inv_urlread_ret_val = [self._get_inv(), URLError('a')]
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(inv_urlread_ret_val, *a, **v)
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        stainvs = self.session.query(Station).filter(Station.has_inventory).all()
        assert len(stainvs) == 1
        assert "Unable to save inventory" in self.log_msg()
        ix = self.session.query(Station.id, Station.inventory_xml).filter(Station.has_inventory).all()
        num_downloaded_inventories_first_try = len(ix)
        assert len(ix) == num_downloaded_inventories_first_try
        staid, invdata = ix[0][0], ix[0][1]
        expected_invs_to_download_ids.remove(staid)  # remove the saved inventory
        assert not invdata.startswith(b'<?xml ') # assert we compressed data  
        assert mock_save_inventories.called                                                    
        mock_save_inventories.reset_mock


        # Now mock empty data:
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories([b""], *a, **v)
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        stainvs = self.session.query(Station).filter(Station.has_inventory).all()
        # assert we still have one station (the one we saved before):
        assert len(stainvs) == num_downloaded_inventories_first_try
        mock_save_inventories.reset_mock

        
        # now mock url returning always data (the default: it returns self._get_inv():
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        ix = self.session.query(Station.id, Station.inventory_xml).filter(Station.has_inventory).all()
        assert len(ix) == num_expected_inventories_to_download
        
        
        # check now that none is downloaded
        mock_save_inventories.reset_mock()
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        stainvs2 = self.session.query(Station).filter(Station.has_inventory).all()
        assert len(stainvs2) == num_expected_inventories_to_download
        assert not mock_save_inventories.called  
                                                                                    
        
        # now test that if a station chanbges datacenter "owner", then the new datacenter
        # is used. Test also that if we remove a single miniseed component of a download that
        # miniseed only is downloaded again
        dfz = dbquery2df(self.session.query(Segment.id, Segment.seed_identifier,
                                            Segment.datacenter_id, Channel.station_id).
                         join(Segment.station, Segment.channel).filter(Segment.has_data))
        
        # dfz:
    #     id  seed_identifier datacenter_id  Station.datacenter_id
    #  0  1   GE.FLT1..HHE    1              1            
    #  1  2   GE.FLT1..HHN    1              1            
    #  2  3   GE.FLT1..HHZ    1              1            
    #  3  6   IA.BAKI..BHZ    2              2 
        
        # remove the first one:
        deleted_seg_id = 1
        seed_to_redownload = dfz[dfz[Segment.id.key] == deleted_seg_id].iloc[0]
        # deleted_seed_id = dfz[dfz[Segment.id.key] == deleted_seg_id].iloc[0][Segment.seed_identifier.key]
        self.session.query(Segment).filter(Segment.id == deleted_seg_id).delete()
        # be sure we deleted it:
        assert len(self.session.query(Segment.id).filter(Segment.has_data).all()) == len(dfz) - 1
        
        oldst_se = self._sta_urlread_sideeffect  # keep last side effect to restore it later
        self._sta_urlread_sideeffect = oldst_se[::-1]  # swap station return values from urlread
    
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
 
        # try to get
        dfz2 = dbquery2df(self.session.query(Segment.id, Segment.seed_identifier,
                                             Segment.datacenter_id, Channel.station_id,
                                             Station.network, Station.station, Channel.location, Channel.channel).
                         join(Segment.station, Segment.channel))
        
        # build manually the seed identifier id:
        
        
        dfz2[Segment.seed_identifier.key] = dfz2[Station.network.key].str.cat(dfz2[Station.station.key].str.cat(dfz2[Channel.location.key].str.cat(dfz2[Channel.channel.key], "."),"."), ".")
        seed_redownloaded = dfz2[dfz2[Segment.seed_identifier.key] == seed_to_redownload[Segment.seed_identifier.key]]
        assert len(seed_redownloaded) == 1
        seed_redownloaded = seed_redownloaded.iloc[0]
        
        # assert the seed_to_redownload and seed_redownloaded have still the same station_id:
        assert seed_redownloaded[Channel.station_id.key] == seed_to_redownload[Channel.station_id.key]
        # but different datacenters:
        assert seed_redownloaded[Segment.datacenter_id.key] != seed_to_redownload[Segment.datacenter_id.key]

        # restore default:
        self._sta_urlread_sideeffect =  oldst_se




        
        # test a type error in the url_segment_side effect
        self.session.query(Segment).delete()
        assert len(self.session.query(Segment).all()) == 0
        errmsg_py2 = '_sre.SRE_Pattern object is not an iterator'  # python2
        errmsg_py3 = "'_sre.SRE_Pattern' object is not an iterator"  # python3
        assert errmsg_py2 not in self.log_msg()
        assert errmsg_py3 not in self.log_msg()
        suse = self._seg_urlread_sideeffect  # remainder (reset later)
        self._seg_urlread_sideeffect = re.compile(".*")  # just return something not number nor string
        runner = CliRunner()
        result = runner.invoke(main , ['d', '-c', self.configfile,
                                       '--dburl', self.dburi,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            assert result.exc_info[0] == TypeError
            assert errmsg_py2 in self.log_msg() or errmsg_py3 in self.log_msg()
        else:
            print("DID NOT RAISE!!")
            assert False
        self._seg_urlread_sideeffect = suse  # restore default
        