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
from stream2segment.cli import cli
from click.testing import CliRunner
import pandas as pd

from stream2segment.download.main import get_events_df, get_datacenters_df, \
get_channels_df, merge_events_stations, \
    prepare_for_download, download_save_segments, save_inventories
# ,\
#     get_fdsn_channels_df, save_stations_and_channels, get_dists_and_times, set_saved_dist_and_times,\
#     download_segments, drop_already_downloaded, set_download_urls, save_segments
from obspy.core.stream import Stream, read
from stream2segment.io.db.models import DataCenter, Segment, Download, Station, Channel, WebService,\
    withdata
from itertools import cycle, repeat, count, product
from urllib.error import URLError
import socket
from obspy.taup.helper_classes import TauModelError
# import logging
# from logging import StreamHandler

# from stream2segment.main import logger as main_logger
from sqlalchemy.sql.expression import func
from stream2segment.io.db.pdsql import dbquery2df, insertdf, updatedf,  _get_max as _get_db_autoinc_col_max
from logging import StreamHandler
import logging
from io import BytesIO
import urllib.request, urllib.error, urllib.parse
from stream2segment.download.utils import custom_download_codes
from stream2segment.download.modules.mseedlite import MSeedError, unpack
import threading
from stream2segment.utils.url import read_async
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from stream2segment.utils.log import configlog4download


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

        self.patchers = []
        self.patchers.append(patch('stream2segment.utils.url.urllib.request.urlopen'))
        self.mock_urlopen = self.patchers[-1].start()
        
        # this mocks get_session to return self.session:
        self.patchers.append(patch('stream2segment.utils.inputargs.get_session'))
        self.mock_get_session = self.patchers[-1].start()
        self.mock_get_session.side_effect = self._get_sess
        
        # this mocks closing to actually NOT close the session (we will do it here):
        self.patchers.append(patch('stream2segment.main.closesession'))
        self.mock_closing = self.patchers[-1].start()
        self.mock_closing.side_effect = lambda *a, **v: None
        
        # this mocks yaml_load and sets inventory to False, as tests rely on that.
        # Moreover, we set the 
        self.patchers.append(patch('stream2segment.utils.inputargs.yaml_load'))
        self.mock_yaml_load = self.patchers[-1].start()
        def yload(*a, **v):
            dic = yaml_load(*a, **v)
            if 'inventory' not in v:
                dic['inventory'] = False
            else:
                sdf = 0
            # also set timespan
            dic['timespan'] = [1, 3]
            return dic
        self.mock_yaml_load.side_effect = yload
        
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

        
        # setup a run_id:
        r = Download()
        self.session.add(r)
        self.session.commit()
        self.run = r

        # side effects:
        # THESE ARE COPIED VIA DEBUGGING FROM A CASE WHERE 
        # WE HAD TIME BOUNDS ERRORS
        
        self._evt_urlread_sideeffect =  """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID |  MagType | Magnitude | MagAuthor | EventLocationName
20160605_0000085|2016-06-05T21:06:04.7Z|45.51|25.91|49.0|BUC|EMSC-RTS|BUC|510656|ml|3.9|BUC|ROMANIA
"""
        eidafile = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eida_routing_service_response.txt")
        with open(eidafile, 'r') as opn:
            self._dc_urlread_sideeffect = opn.read()


# Note: by default we set sta_urlsideeffect to return such a channels which result in 12
# segments (see lat and lon of channels vs lat and lon of events above)
        self._sta_urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
BS|BLKB||BHE|43.6227|22.675|650.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|BLKB||BHN|43.6227|22.675|650.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|BLKB||BHZ|43.6227|22.675|650.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|BLKB||HHE|43.6227|22.675|650.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|BLKB||HHN|43.6227|22.675|650.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|BLKB||HHZ|43.6227|22.675|650.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|BLKB||HNE|43.6227|22.675|650.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|BLKB||HNN|43.6227|22.675|650.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|BLKB||HNZ|43.6227|22.675|650.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|DOBAM||HNE|43.5811|27.831|246.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|DOBAM||HNN|43.5811|27.831|246.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|DOBAM||HNZ|43.5811|27.831|246.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|ELND||BHE|42.9287|25.8751|334.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|ELND||BHN|42.9287|25.8751|334.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|ELND||BHZ|42.9287|25.8751|334.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|ELND||HHE|42.9287|25.8751|334.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|ELND||HHN|42.9287|25.8751|334.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|ELND||HHZ|42.9287|25.8751|334.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|ELND||HNE|42.9287|25.8751|334.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|ELND||HNN|42.9287|25.8751|334.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|ELND||HNZ|42.9287|25.8751|334.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|JMB||BHE|42.466999|26.583|216.0|0.0|270.0|0.0|CMG40|120720000.0|0.02|M/S|20.0|2004-04-01T00:00:00|
BS|JMB||BHN|42.466999|26.583|216.0|0.0|180.0|0.0|CMG40|120720000.0|0.02|M/S|20.0|2004-04-01T00:00:00|
BS|JMB||BHZ|42.466999|26.583|216.0|0.0|180.0|90.0|CMG40|120720000.0|0.02|M/S|20.0|2004-04-01T00:00:00|
BS|KALB||BHE|43.4059|28.4162|121.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KALB||BHN|43.4059|28.4162|121.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KALB||BHZ|43.4059|28.4162|121.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KALB||HHE|43.4059|28.4162|121.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KALB||HHN|43.4059|28.4162|121.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KALB||HHZ|43.4059|28.4162|121.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KALB||HNE|43.4059|28.4162|121.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KALB||HNN|43.4059|28.4162|121.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KALB||HNZ|43.4059|28.4162|121.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KOZAM||HNE|43.8274|23.2364|760.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KOZAM||HNN|43.8274|23.2364|760.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KOZAM||HNZ|43.8274|23.2364|760.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KUBB||BHE|43.8024|26.4941|261.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KUBB||BHN|43.8024|26.4941|261.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KUBB||BHZ|43.8024|26.4941|261.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KUBB||HHE|43.8024|26.4941|261.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KUBB||HHN|43.8024|26.4941|261.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KUBB||HHZ|43.8024|26.4941|261.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KUBB||HNE|43.8024|26.4941|261.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KUBB||HNN|43.8024|26.4941|261.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KUBB||HNZ|43.8024|26.4941|261.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|LOZB||BHE|43.3701|26.593|342.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|LOZB||BHN|43.3701|26.593|342.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|LOZB||BHZ|43.3701|26.593|342.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|LOZB||HHE|43.3701|26.593|342.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|LOZB||HHN|43.3701|26.593|342.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|LOZB||HHZ|43.3701|26.593|342.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|LOZB||HNE|43.3701|26.593|342.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|LOZB||HNN|43.3701|26.593|342.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|LOZB||HNZ|43.3701|26.593|342.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|MALO||HNE|43.3559|23.7402|370.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2013-01-11T00:00:00|
BS|MALO||HNN|43.3559|23.7402|370.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2013-01-11T00:00:00|
BS|MALO||HNZ|43.3559|23.7402|370.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2013-01-11T00:00:00|
BS|MNNAM||HNE|43.4111|23.2262|0.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|MNNAM||HNN|43.4111|23.2262|0.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|MNNAM||HNZ|43.4111|23.2262|0.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVAM||HNE|43.9523|22.8502|880.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVAM||HNN|43.9523|22.8502|880.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVAM||HNZ|43.9523|22.8502|880.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVB||BHE|43.387|24.6207|199.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|PLVB||BHN|43.387|24.6207|199.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|PLVB||BHZ|43.387|24.6207|199.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|PLVB||HHE|43.387|24.6207|199.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|PLVB||HHN|43.387|24.6207|199.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|PLVB||HHZ|43.387|24.6207|199.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|PLVB||HNE|43.387|24.6207|199.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVB||HNN|43.387|24.6207|199.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVB||HNZ|43.387|24.6207|199.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZAM||HNE|43.6451|25.1243|820.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZAM||HNN|43.6451|25.1243|820.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZAM||HNZ|43.6451|25.1243|820.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZG||BHE|43.5662|26.5079|383.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|RAZG||BHN|43.5662|26.5079|383.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|RAZG||BHZ|43.5662|26.5079|383.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|RAZG||HHE|43.5662|26.5079|383.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|RAZG||HHN|43.5662|26.5079|383.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|RAZG||HHZ|43.5662|26.5079|383.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|RAZG||HNE|43.5662|26.5079|383.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZG||HNN|43.5662|26.5079|383.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZG||HNZ|43.5662|26.5079|383.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RUSAM||HNE|43.8462|25.9612|0.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RUSAM||HNN|43.8462|25.9612|0.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RUSAM||HNZ|43.8462|25.9612|0.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SHAB||BHE|43.5389|28.6057|430.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|SHAB||BHN|43.5389|28.6057|430.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|SHAB||BHZ|43.5389|28.6057|430.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|SHAB||HHE|43.5389|28.6057|430.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|SHAB||HHN|43.5389|28.6057|430.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|SHAB||HHZ|43.5389|28.6057|430.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|SHAB||HNE|43.5389|28.6057|430.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SHAB||HNN|43.5389|28.6057|430.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SHAB||HNZ|43.5389|28.6057|430.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SILAM||HNE|44.1046|27.2665|840.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SILAM||HNN|44.1046|27.2665|840.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SILAM||HNZ|44.1046|27.2665|840.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|VETAM||HNE|43.0805|25.6367|224.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|VETAM||HNN|43.0805|25.6367|224.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|VETAM||HNZ|43.0805|25.6367|224.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|"""]
        # self._sta_urlread_sideeffect = cycle([partial_valid, '', invalid, '', '', URLError('wat'), socket.timeout()])

        # the segments downloads returns ALWAYS the same miniseed, which is the BS network
        # in a specified 
        _file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "BS.*.*.*.2016-06-05.21:05-09.47.mseed")
        with open(_file, "rb") as _opn:
            self._seg_data = _opn.read()
            
        self._seg_urlread_sideeffect = [self._seg_data]


        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self)
                        #self.patcher3)
        
        self.configfile = get_templates_fpath("download.yaml")
        # self._logout_cache = ""
    
    def log_msg(self):
        return self.logout.getvalue()
#         idx = len(self._logout_cache)
#         self._logout_cache = self.logout.getvalue()
#         if len(self._logout_cache) == idx:
#             idx = None # do not slice
#         return self._logout_cache[idx:]

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
        return save_inventories(*a, **v)

    
    def _get_inv(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "inventory_GE.APE.xml")
        with open(path, 'rb') as opn_:
            return opn_.read()


    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    def test_cmdline_outofbounds(self, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                                 mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                                 mock_get_datacenters_df, mock_get_events_df):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(self.session.query(Segment).all())
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(self.session.query(Download.id).all())



        runner = CliRunner()
        result = runner.invoke(cli , ['download',
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
        
        assert len(self.session.query(Download.id).all()) == runs + 1
        runs += 1
        
        url_err, mseed_err, timespan_err, timespan_warn = custom_download_codes()
        
        # assert when we have a timespan error we do not have data:
        assert self.session.query(Segment).filter(Segment.has_data & (Segment.download_code==timespan_err)).count() == 0 
        
        # assert when we have a timespan warn or 200 response we have data:
        seg_with_data = self.session.query(Segment).filter(Segment.has_data).count()
        assert self.session.query(Segment).filter(Segment.has_data &
                                                  ((Segment.download_code==200) |
                                                   (Segment.download_code==timespan_warn))).count() == seg_with_data 
        
        # pickup some examples and test them (the examples where found by debugging download_segments):
        data = {timespan_warn: ['BS.BLKB..HHE', 'BS.BLKB..HHN', 'BS.BLKB..HHZ', 'BS.BLKB..HNZ', 'BS.DOBAM..HNE',
                                'BS.DOBAM..HNN', 'BS.DOBAM..HNZ'],
                timespan_err: ['BS.KALB..HHE', 'BS.KALB..HHN', 'BS.KALB..HHZ'],
                200: ['BS.BLKB..HNE', 'BS.BLKB..HNN']}
    
        for downloadcode, mseedids in data.items():
            for mseedid in mseedids:
                seg = self.session.query(Segment).filter(Segment.seed_id == mseedid).first()
                assert seg.download_code == downloadcode
                if (downloadcode == timespan_err):
                    assert not seg.has_data
                else:
                    assert seg.has_data
        




    