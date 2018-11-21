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
import random
import yaml
import stream2segment
from stream2segment.download.modules.segments import DcDataselectManager
standard_library.install_aliases()
from builtins import str, map
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

from itertools import product, combinations

import unittest, os
from sqlalchemy.engine import create_engine
from stream2segment.io.db.models import Base, Event, Class, WebService, Fdsnws
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from stream2segment.cli import cli
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
# import urllib.request, urllib.error, urllib.parse
from stream2segment.download.utils import custom_download_codes
from stream2segment.download.modules.mseedlite import MSeedError, unpack
import threading
# from urllib.error import URLError
from stream2segment.utils.url import read_async, URLError, HTTPError, get_opener
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from stream2segment.utils.log import configlog4download

from future.utils import PY2
if PY2:
    from BaseHTTPServer import BaseHTTPRequestHandler
    responses = BaseHTTPRequestHandler.responses
else:
    from http.client import responses

# when debugging, I want the full dataframe with to_string(), not truncated
pd.set_option('display.max_colwidth', -1)


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data, pytestdir):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)
        
        
        self.logout = StringIO()
        self.handler = StreamHandler(stream=self.logout)
        # THIS IS A HACK:
        # s2s_download_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset (to level.INFO)
        # otherwise it stays what we set two lines above. Problems might arise if closing
        # sets a different level, but for the moment who cares
        # s2s_download_logger.addHandler(self.handler)
             

        # setup a run_id:
        r = Download()
        db.session.add(r)
        db.session.commit()
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
        
        self._seg_data = data.read("GE.FLT1..HH?.mseed")
        self._seg_data_gaps = data.read("IA.BAKI..BHZ.D.2016.004.head")
        self._seg_data_empty = b''
            
        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_gaps, 413, 500, self._seg_data[:2],
                                        self._seg_data_empty,  413, URLError("++urlerror++"),
                                        socket.timeout()]


        self._inv_data = data.read("inventory_GE.APE.xml")

        self.service = ''  # so get_datacenters_df accepts any row by default

        self.configfile = get_templates_fpath("download.yaml")
        self.config_overrides = {}
        
        # store DcDataselectManager method here:
        self.dc_get_data_open = DcDataselectManager._get_data_open
        self.dc_get_data_from_userpass = DcDataselectManager._get_data_from_userpass
        # get data from token accepts a custom urlread side effect:
        _get_data_from_token = DcDataselectManager._get_data_from_token
        def dc_get_data_from_token_func(url_read_side_effect=None, *a, **kw):
            if url_read_side_effect is not None:
                self.setup_urlopen(url_read_side_effect)
            return _get_data_from_token(*a, **kw)
        self.dc_get_data_from_token = dc_get_data_from_token_func
        
        # class-level patchers:
        with patch('stream2segment.utils.url.urlopen') as mock_urlopen:
            self.mock_urlopen = mock_urlopen
            with patch('stream2segment.utils.inputargs.get_session', return_value=db.session):
                # this mocks yaml_load and sets inventory to False, as tests rely on that
                with patch('stream2segment.main.closesession'):  # no-op (do not close session)

                    def yload(*a, **v):
                        dic = yaml_load(*a, **v)
                        dic.update(self.config_overrides)
                        return dic

                    with patch('stream2segment.utils.inputargs.yaml_load',
                               side_effect=yload) as mock_yaml_load:
                        self.mock_yaml_load = mock_yaml_load

                        # mock ThreadPool (tp) to run one instance at a time, so we
                        # get deterministic results:
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
                        with patch('stream2segment.utils.url.ThreadPool',
                                   side_effect=MockThreadPool) as mock_thread_pool:
                            
                            def c4d(logger, logfilebasepath, verbose):
                                # config logger as usual, but redirects to a temp file
                                # that will be deleted by pytest, instead of polluting the program
                                # package:
                                ret = configlog4download(logger, pytestdir.newfile('.log'),
                                                         verbose)
                                logger.addHandler(self.handler)
                                return ret
                            with patch('stream2segment.main.configlog4download',
                                       side_effect=c4d) as mock_config4download:
                                self.mock_config4download = mock_config4download

                                yield
    
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
                a.read.side_effect = HTTPError('url', int(k),  responses[k], None, None)
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
                a.msg = responses[a.code]
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
        self.setup_urlopen(self._inv_data if url_read_side_effect is None else url_read_side_effect)
        return save_inventories(*a, **v)

    # only last 4 patches are actually needed, the other are there
    # simply because this module was copied-pasted from other tests. too lazy to check/remove
    # them, and we might need those patches in the future
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_open')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_from_userpass')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_from_token')
    def test_cmdline(self, mock_get_data_from_token, mock_get_data_from_userpass,
                     mock_get_data_open, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                     mock_get_datacenters_df, mock_get_events_df,
                     # fixtures:
                     db, clirunner, pytestdir):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        # mseed unpack is mocked by accepting only first arg (so that time bounds are not considered)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())

        # patching class methods while preserving the original call requires storing once
        # the original methods (as class attributes). Sets the side effect of the mocked method
        # to those class attributes as to preserve the original functionality
        # and be able to assert mock_* functions are called and so on
        # For info see:
        # https://stackoverflow.com/a/29563665
        mock_get_data_open.side_effect = self.dc_get_data_open
        mock_get_data_from_userpass.side_effect = self.dc_get_data_from_userpass
        mock_get_data_from_token.side_effect = self.dc_get_data_from_token
        
        # TEST 1: NORMAL CASE (NO AUTH):
        # mock yaml_load to override restricted_data:
        self.config_overrides = {'restricted_data': ''}
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())
        result = clirunner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        assert clirunner.ok(result)
        assert 'restricted_data: disabled, download open data only' in result.output
        assert mock_get_data_open.called
        assert not mock_get_data_from_token.called
        assert not mock_get_data_from_userpass.called
        # some assertions to check data properly written
        assert len(db.session.query(Download.id).all()) == runs + 1
        runs += 1
        segments = db.session.query(Segment).all()
        assert len(segments) == 12
        segments = db.session.query(Segment).filter(Segment.has_data).all()
        assert len(segments) == 4        
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) == 2
        assert mock_updatedf.called  # called while saving inventories
        assert mock_insertdf.called
        
        # TEST 1: USERPASS AND EIDA (PROBLEM):
        # test that we provide userpass and eida: error:
        # mock yaml_load to override restricted_data:
        mock_get_data_open.reset_mock()
        mock_get_data_from_token.reset_mock()
        mock_get_data_from_userpass.reset_mock()
        self.config_overrides = {'restricted_data': ['user', 'password'], 'dataws': 'eida'}
        result = clirunner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        assert not clirunner.ok(result)
        assert ('Error: Invalid value for "restricted_data": '
                'downloading from EIDA requires a token') in result.output
                
        # TEST 2: TOKEN FILE NOT EXISTING
        mock_get_data_open.reset_mock()
        mock_get_data_from_token.reset_mock()
        mock_get_data_from_userpass.reset_mock()
        self.config_overrides = {'restricted_data': 'abcdg465du97_Sdr4fvssgflero',
                                 'dataws': 'eida'}
        result = clirunner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        assert not clirunner.ok(result)
        assert ('Invalid token. If you passed a file path, '
                'check also that the file exists') in result.output

        # TEST 2: TOKEN FILE EXISTS, INVALID (e.g. empty)
        filepath = pytestdir.newfile(create=True)
        mock_get_data_open.reset_mock()
        mock_get_data_from_token.reset_mock()
        mock_get_data_from_userpass.reset_mock()
        self.config_overrides = {'restricted_data': os.path.abspath(filepath),
                                 'dataws': 'eida'}
        result = clirunner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        assert not clirunner.ok(result)
        assert ('Invalid token. If you passed a file path, '
                'check also that the file exists') in result.output


    # only last 4 patches are actually needed, the other are there
    # simply because this module was copied-pasted from other tests. too lazy to check/remove
    # them, and we might need those patches in the future
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_open')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_from_userpass')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_from_token')
    @patch('stream2segment.download.modules.segments.get_opener', side_effect=get_opener)
    def test_cmdline(self, mock_get_opener, mock_get_data_from_token,
                     mock_get_data_from_userpass,
                     mock_get_data_open, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                     mock_get_datacenters_df, mock_get_events_df,
                     # fixtures:
                     db, clirunner, pytestdir):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        # mseed unpack is mocked by accepting only first arg (so that time bounds are not considered)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())

        # patching class methods while preserving the original call requires storing once
        # the original methods (as class attributes). Sets the side effect of the mocked method
        # to those class attributes as to preserve the original functionality
        # and be able to assert mock_* functions are called and so on
        # For info see:
        # https://stackoverflow.com/a/29563665
        mock_get_data_open.side_effect = self.dc_get_data_open
        mock_get_data_from_userpass.side_effect = self.dc_get_data_from_userpass
        mock_get_data_from_token.side_effect = \
            lambda *a, **kw: self.dc_get_data_from_token([URLError('a'), 'abc'], *a, **kw)
        
        # TEST 1: provide a file with valid token:
        tokenfile = pytestdir.newfile(create=True)
        with open(tokenfile, 'w') as fh:
            fh.write('BEGIN PGP MESSAGE')
        # mock yaml_load to override restricted_data:
        self.config_overrides = {'restricted_data': os.path.abspath(tokenfile)}
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())
        result = clirunner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        assert clirunner.ok(result)
        assert 'restricted_data: enabled, with token' in result.output
        assert 'STEP 6 of 8: Acquiring credentials from token' in result.output
        # assert we print that we are downloading open data only (all errors):
        assert 'STEP 7 of 8: Downloading 12 segments (open data only)' in result.output
        assert not mock_get_data_open.called
        assert mock_get_data_from_token.called
        assert not mock_get_data_from_userpass.called
        assert not mock_get_opener.called
        # some assertions to check data properly written
        # These are important because they confirmthat data has been downloaded anyway
        # (the test does not differentiate between restricted or open data)
        assert len(db.session.query(Download.id).all()) == runs + 1
        runs += 1
        segments = db.session.query(Segment).all()
        assert len(segments) == 12
        segments = db.session.query(Segment).filter(Segment.has_data).all()
        assert len(segments) == 4        
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) == 2
        assert mock_updatedf.called  # called while saving inventories
        assert mock_insertdf.called



    # only last 4 patches are actually needed, the other are there
    # simply because this module was copied-pasted from other tests. too lazy to check/remove
    # them, and we might need those patches in the future
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_open')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_from_userpass')
    @patch('stream2segment.download.main.DcDataselectManager._get_data_from_token')
    @patch('stream2segment.download.modules.segments.get_opener', side_effect=get_opener)
    def test_cmdline(self, mock_get_opener, mock_get_data_from_token,
                     mock_get_data_from_userpass,
                     mock_get_data_open, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                     mock_get_datacenters_df, mock_get_events_df,
                     # fixtures:
                     db, clirunner, pytestdir):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        # mseed unpack is mocked by accepting only first arg (so that time bounds are not considered)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())

        # patching class methods while preserving the original call requires storing once
        # the original methods (as class attributes). Sets the side effect of the mocked method
        # to those class attributes as to preserve the original functionality
        # and be able to assert mock_* functions are called and so on
        # For info see:
        # https://stackoverflow.com/a/29563665
        mock_get_data_open.side_effect = self.dc_get_data_open
        mock_get_data_from_userpass.side_effect = self.dc_get_data_from_userpass
        mock_get_data_from_token.side_effect = \
            lambda *a, **kw: self.dc_get_data_from_token([URLError('a'), 'abc'], *a, **kw)
        
        # TEST 1: provide a file with valid token:
        tokenfile = pytestdir.newfile(create=True)
        with open(tokenfile, 'w') as fh:
            fh.write('BEGIN PGP MESSAGE')
        # mock yaml_load to override restricted_data:

        # launch two download runs with different responses for token auth query:
        for tokenquery_mocked_return_values, dc_token_failed in \
            ([[URLError('a'), 'uzer:pazzword'], "http://geofon.gfz-potsdam.de"],
             [['uzer:pazzword', URLError('a')], 'http://ws.resif.fr']):
            # set how many times self.mock_urlopen has been called:
            mock_urlopen_call_count = self.mock_urlopen.call_count
            # TEST 2: USERPASS good for just one datacenter:
            mock_get_data_open.reset_mock()
            mock_get_data_from_token.reset_mock()
            mock_get_data_from_userpass.reset_mock()
            mock_get_opener.reset_mock()
            mock_get_data_from_token.side_effect = \
                lambda *a, **kw: self.dc_get_data_from_token(tokenquery_mocked_return_values,
                                                             *a, **kw)
            self.config_overrides = {'restricted_data': os.path.abspath(tokenfile)}
            result = clirunner.invoke(cli , ['download',
                                           '-c', self.configfile,
                                            '--dburl', db.dburl,
                                           '--start', '2016-05-08T00:00:00',
                                           '--end', '2016-05-08T9:00:00'])
            assert clirunner.ok(result)
            assert 'restricted_data: enabled, with token' in result.output
            assert 'STEP 6 of 8: Acquiring credentials from token' in result.output
            # assert we print that we are downloading open and restricted data:
            assert re.search(r'STEP 7 of 8\: Downloading \d+ segments and saving to db',
                             result.output)
            assert not mock_get_data_open.called
            assert mock_get_data_from_token.called
            assert not mock_get_data_from_userpass.called
            
            assert "Downloading open data only from: %s" % dc_token_failed
            dc_token_ok = 'http://ws.resif.fr' \
                if dc_token_failed == "http://geofon.gfz-potsdam.de" else \
                "http://geofon.gfz-potsdam.de"
            assert mock_get_opener.call_count == 1
            assert mock_get_opener.call_args_list[0][0][:] == (dc_token_ok, 'uzer', 'pazzword')
            # assert urlopen has been called only once with query. Note that in the second
            # eownload run, urlopen might have been called ero times, if all retried segments
            # belong to the datacenter that did not fail requiring auth:
            # So first, build a dc_id dict which returns the dc id from its domain:
            dc_df = mock_get_channels_df.call_args_list[0][0][1]
            dc_id = {}
            for index, row in dc_df.iterrows():
                dc_id[Fdsnws(row['dataselect_url']).site] = row['id']
            # get the segments dataframe we (re)downloaded:
            segments_df_to_download = mock_download_save_segments.call_args_list[-1][0][1]
            dc2download = pd.unique(segments_df_to_download['datacenter_id']).tolist()
            # set the expected call count based on the datacenters of (re)downloaded segments:
            expected_call_count = 0 if dc_id[dc_token_failed] not in dc2download else \
                mock_urlopen_call_count + 1
            assert self.mock_urlopen.call_count == expected_call_count
            if expected_call_count > 0:
                assert self.mock_urlopen.call_args_list[-1][0][0].get_full_url() == \
                    dc_token_failed + "/fdsnws/station/1/query"
            
