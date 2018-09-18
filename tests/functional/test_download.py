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
from stream2segment.utils.url import read_async, URLError, HTTPError
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from stream2segment.utils.log import configlog4download

# from future.standard_library import install_aliases
# install_aliases()
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
        # self._logout_cache = ""
        
        # class-level patchers:
        with patch('stream2segment.utils.url.urlopen') as mock_urlopen:
            self.mock_urlopen = mock_urlopen
            with patch('stream2segment.utils.inputargs.get_session', return_value=db.session):
                # this mocks yaml_load and sets inventory to False, as tests rely on that
                with patch('stream2segment.main.closesession'):  # no-op (do not close session)

                    def yload(*a, **v):
                        dic = yaml_load(*a, **v)
                        if 'inventory' not in v:
                            dic['inventory'] = False
                        else:
                            sdf = 0
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

    @patch('stream2segment.io.db.pdsql._get_max')
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    def test_cmdline_dberr(self, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                           mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                           mock_get_datacenters_df, mock_get_events_df, mock_autoinc_db, db):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        # mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_autoinc_db.side_effect = lambda *a, **v: _get_db_autoinc_col_max(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())
        
        # mock insertdf to mess-up the ids so that we can check db errors
        def insdf(*a, **v):
            a = list(a)
            df = a[0]
            column = a[2]
            if column.class_ == Segment:
                df[Segment.id.key] = np.arange(len(df), dtype=int) + 1
            return insertdf(*a, **v)
        mock_insertdf.side_effect = insdf
                
        
        
        runner = CliRunner()
        result = runner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
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
        
        assert db.session.query(Station).count() == 4
        
        # assert log msg printed
        assert """duplicate key value violates unique constraint "segments_pkey"
DETAIL:  Key (id)=(1) already exists""" if db.is_postgres else \
"(UNIQUE constraint failed: segments.id)" in self.log_msg() 
    
        
        # get the excpeted segment we should have downloaded:
        segments_df = mock_download_save_segments.call_args_list[0][0][1]
        assert db.session.query(Segment).count() < len(segments_df)
        
        # get the first group written to the db. Note that as we mocked read_async (see above)
        # the first dataframe given to urlread should be also the first to be written to db,
        # and so on for the second, third. If the line below fails, check that maybe it's not
        # the case and we should be less strict. Actually, we will be less strict, 
        # turns out the check is undeterministic Comment out:
#         first_segments_df = segments_df.groupby(['datacenter_id', 'request_start', 'request_end'], sort=False).first()
        assert db.session.query(Segment).count()  == 3  # len(first_segments_df)
        # assert 
        assert db.session.query(Channel).count() == 12
        assert db.session.query(Event).count() == 2
        
        # assert run log has been written correctly, i.e. that the db error on the segments
        # has not affected further writing operations. To do this quickly, assert that
        # all run.log have something written in (not null, not empty)
        assert db.session.query(withdata(Download.log)).count() == db.session.query(Download).count()


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
                                 mock_get_datacenters_df, mock_get_events_df, db):
        
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())



        runner = CliRunner()
        result = runner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
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
        
        assert len(db.session.query(Download.id).all()) == runs + 1
        runs += 1
        segments = db.session.query(Segment).all()
        assert len(segments) == 12
        segments = db.session.query(Segment).filter(Segment.has_data).all()
        assert len(segments) == 0  # all out of bounds
        
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) == 0
        
        assert not mock_updatedf.called
        assert mock_insertdf.called
        
        dfres1 = dbquery2df(db.session.query(Segment.id, Segment.channel_id, Segment.datacenter_id,
                                               Segment.event_id,
                                         Segment.download_code, Segment.data,
                                         Segment.maxgap_numsamples, Segment.download_id,
                                         Segment.sample_rate, Segment.data_seed_id))
        dfres1.sort_values(by=Segment.id.key, inplace=True)  # for easier visual compare
        dfres1.reset_index(drop=True, inplace=True)  # we need to normalize indices for comparison later
        # just change the value of the bytes so that we can better 
        # visually inspect dataframe under clipse, in case:
        dfres1.loc[(~pd.isnull(dfres1[Segment.data.key])) & (dfres1[Segment.data.key].str.len()>0),
                  Segment.data.key] = b'data'
        # assert the segments we should have data for are actually out-of-time-bounds
        _, _, TBOUND_ERRCODE, TBOUND_WARNCODE = custom_download_codes()
        assert len(dfres1[dfres1[Segment.download_code.key] == TBOUND_ERRCODE]) == 4
        
    
    @pytest.mark.skip(reason=("no way of currently testing what we want to test, "
                              "error not reproducible. See comments below"))
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    def test_cmdline_unexpected_err(self, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                     mock_get_datacenters_df, mock_get_events_df, db):
        ''' we experienced once a UnicodeDecodeError in mseed_unpack, whcih, has expected
        raised. Fine, but we got also another error: attempt to write on an apparently already-closed logger,
        as if threads where trying to access concurrently to the same logger.
        Try to test this case even if unfortunately we did not experience this issue in this test'''
        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = lambda *a, **v: self.download_save_segments(None, *a, **v)
        # =========================
        # HERE IS THE IMPORTANT PART
        # mseed unpack is mocked by accepting only first arg (so that time bounds are not considered)
        # =========================
        def mockunpack(*a, **v):
            # https://gehrcke.de/2015/12/how-to-raise-unicodedecodeerror-in-python-3/
            raise UnicodeDecodeError('funnycodec', b'\x00\x00', 1, 2, 'This is just a fake reason!')
        mock_mseed_unpack.side_effect = mockunpack  # now is cought, and a MiniSeed Error is raised, but let's see
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())



        runner = CliRunner()
        result = runner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        h= 9
    
    
    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    def test_cmdline(self, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                     mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                     mock_get_datacenters_df, mock_get_events_df, db):
        
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
     
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())



        runner = CliRunner()
        result = runner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
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
        
        assert len(db.session.query(Download.id).all()) == runs + 1
        runs += 1
        segments = db.session.query(Segment).all()
        assert len(segments) == 12
        segments = db.session.query(Segment).filter(Segment.has_data).all()
        assert len(segments) == 4
        
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) == 0
        
        assert not mock_updatedf.called
        assert mock_insertdf.called
        
        dfres1 = dbquery2df(db.session.query(Segment.id, Segment.channel_id, Segment.datacenter_id,
                                               Segment.event_id,
                                         Segment.download_code, Segment.data,
                                         Segment.maxgap_numsamples, Segment.download_id,
                                         Segment.sample_rate, Segment.data_seed_id))
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
        mock_insertdf.reset_mock()
        self._seg_urlread_sideeffect = [413]
        idx = len(self.log_msg())
        runner = CliRunner()
        result = runner.invoke(cli , ['download',
                                       '-c', self.configfile,
                                        '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        
        dfres2 = dbquery2df(db.session.query(Segment.id, Segment.channel_id, Segment.datacenter_id,
                                               Segment.event_id,
                                         Segment.download_code, Segment.data,
                                         Segment.maxgap_numsamples, Segment.download_id,
                                         Segment.sample_rate, Segment.data_seed_id))
        dfres2.sort_values(by=Segment.id.key, inplace=True)  # for easier visual compare
        dfres2.reset_index(drop=True, inplace=True)  # we need to normalize indices for comparison later
        # just change the value of the bytes so that we can better 
        # visually inspect dataframe under clipse, in case:
        dfres2.loc[(~pd.isnull(dfres2[Segment.data.key])) & (dfres2[Segment.data.key].str.len()>0),
                  Segment.data.key] = b'data'
        
        assert mock_updatedf.called
        assert not mock_insertdf.called
        
        URLERROR, MSEEDERROR, OUTTIME_ERR, OUTTIME_WARN = custom_download_codes()
        
        assert len(dfres2) == len(dfres1)
        assert len(db.session.query(Download.id).all()) == runs + 1
        runs += 1
        # asssert we changed the download status code for segments which should be retried
        # WARNING: THIS TEST COULD FAIL IF WE CHANGE THE DEFAULTS. CHANGE `mask` here below IN CASE
        mask = dfres1[Segment.download_code.key].between(500, 599.999, inclusive=True) | \
                      (dfres1[Segment.download_code.key] == URLERROR) | \
                      pd.isnull(dfres1[Segment.download_code.key])
        retried = dfres2.loc[mask, :]
        assert (retried[Segment.download_code.key] == 413).all()
        # asssert we changed the run_id for segments which should be retried
        # WARNING: THIS TEST COULD FAIL IF WE CHANGE THE DEFAULTS. CHANGE THE `mask` IN CASE
        assert (retried[Segment.download_id.key] > dfres1.loc[retried.index, Segment.download_id.key]).all()
        
        assert mock_download_save_segments.called



        # Ok, now with the current config 413 is not retried: 
        # check that now we should skip all segments
        mock_download_save_segments.reset_mock()
        runner = CliRunner()
        result = runner.invoke(cli , ['download', 
                                       '-c', self.configfile,
                                       '--dburl', db.dburl,
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
        str_err = "Eida routing service error"
        assert str_err not in self.log_msg()
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(500, *a, **v) 
        mock_download_save_segments.reset_mock()
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        assert not mock_download_save_segments.called
        x = self.log_msg()
        assert str_err in self.log_msg()
        mock_get_datacenters_df.side_effect = lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        

        # test some edge cases, if run from eclipse, a debugger and inspection of self.log_msg()
        # might be needed to check that everything is printed right. IF WE CHANGE THE MESSAGES
        # TO BE DISPLAYED, THEN CHANGE THE STRING BELOW:
        str_err = "No channel found with sample rate"
        assert str_err not in self.log_msg()
        # assert str_err not in self.log_msg()
        def mgdf(*a, **v):
            aa = list(a)
            aa[9] = 100000  # change min sample rate to a huge number
            return self.get_channels_df(None, *aa, **v) 
        mock_get_channels_df.side_effect = mgdf
        
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
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
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        # assert we wrote again str_2:
        assert self.log_msg().rfind(str_2) > idx
        
        # reset to default:
        self._sta_urlread_sideeffect = rem
        
        
        # test with loading station inventories:
        
        # we should not have inventories saved:
        stainvs = db.session.query(Station).filter(Station.has_inventory).all()
        assert len(stainvs) == 0
        # calculate the expected stations:
        expected_invs_to_download_ids = [x[0] for x in db.session.query(Station.id).filter((~Station.has_inventory) &
                   (Station.segments.any(Segment.has_data))).all()]
        # test that we have data, but also errors
        num_expected_inventories_to_download = len(expected_invs_to_download_ids)
        assert num_expected_inventories_to_download == 2  # just in order to set the value below
        # and be more safe about the fact that we will have only ONE station inventory saved
        inv_urlread_ret_val = [self._inv_data, URLError('a')]
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(inv_urlread_ret_val, *a, **v)
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        stainvs = db.session.query(Station).filter(Station.has_inventory).all()
        assert len(stainvs) == 1
        assert "Unable to save inventory" in self.log_msg()
        ix = db.session.query(Station.id, Station.inventory_xml).filter(Station.has_inventory).all()
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
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        stainvs = db.session.query(Station).filter(Station.has_inventory).all()
        # assert we still have one station (the one we saved before):
        assert len(stainvs) == num_downloaded_inventories_first_try
        mock_save_inventories.reset_mock

        
        # now mock url returning always data (the default: it returns self._inv_data:
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        ix = db.session.query(Station.id, Station.inventory_xml).filter(Station.has_inventory).all()
        assert len(ix) == num_expected_inventories_to_download
        
        
        # check now that none is downloaded
        mock_save_inventories.reset_mock()
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
        
        stainvs2 = db.session.query(Station).filter(Station.has_inventory).all()
        assert len(stainvs2) == num_expected_inventories_to_download
        assert not mock_save_inventories.called  
                                                                                    
        
        # now test that if a station chanbges datacenter "owner", then the new datacenter
        # is used. Test also that if we remove a single miniseed component of a download that
        # miniseed only is downloaded again
        dfz = dbquery2df(db.session.query(Segment.id, Segment.data_seed_id,
                                            Segment.datacenter_id, Channel.station_id).
                         join(Segment.station, Segment.channel).filter(Segment.has_data))
        
        # dfz:
    #     id  data_seed_id    datacenter_id  Station.datacenter_id
    #  0  1   GE.FLT1..HHE    1              1            
    #  1  2   GE.FLT1..HHN    1              1            
    #  2  3   GE.FLT1..HHZ    1              1            
    #  3  6   IA.BAKI..BHZ    2              2 
        
        # remove the first one:
        deleted_seg_id = 1
        seed_to_redownload = dfz[dfz[Segment.id.key] == deleted_seg_id].iloc[0]
        db.session.query(Segment).filter(Segment.id == deleted_seg_id).delete()
        # be sure we deleted it:
        assert len(db.session.query(Segment.id).filter(Segment.has_data).all()) == len(dfz) - 1
        
        oldst_se = self._sta_urlread_sideeffect  # keep last side effect to restore it later
        self._sta_urlread_sideeffect = oldst_se[::-1]  # swap station return values from urlread
    
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False
            return
 
        # try to get
        dfz2 = dbquery2df(db.session.query(Segment.id, Segment.data_seed_id,
                                             Segment.datacenter_id, Channel.station_id,
                                             Station.network, Station.station, Channel.location, Channel.channel).
                         join(Segment.station, Segment.channel))
        
        # build manually the seed identifier id:
        
        
        dfz2[Segment.data_seed_id.key] = dfz2[Station.network.key].str.cat(dfz2[Station.station.key].str.cat(dfz2[Channel.location.key].str.cat(dfz2[Channel.channel.key], "."),"."), ".")
        seed_redownloaded = dfz2[dfz2[Segment.data_seed_id.key] == seed_to_redownload[Segment.data_seed_id.key]]
        assert len(seed_redownloaded) == 1
        seed_redownloaded = seed_redownloaded.iloc[0]
        
        # assert the seed_to_redownload and seed_redownloaded have still the same station_id:
        assert seed_redownloaded[Channel.station_id.key] == seed_to_redownload[Channel.station_id.key]
        # but different datacenters:
        assert seed_redownloaded[Segment.datacenter_id.key] != seed_to_redownload[Segment.datacenter_id.key]

        # restore default:
        self._sta_urlread_sideeffect =  oldst_se   
        
        # test update flag:
        
        
        
        # first change some values on the db, so that we can MOCK that the next download
        # has some metadata changed:
        sta1 = db.session.query(Station).filter(Station.has_inventory == False).first()
        sta_inv = db.session.query(Station).filter(Station.has_inventory == True).first()
        sta_inv_id = sta_inv.id
        cha = db.session.query(Channel).filter(Channel.id ==1).first()
        new_elevation = sta1.elevation + 5
        new_sitename = 'wow!!!!!!!!!!!--------------------'
        new_srate = 0
        new_sta_inv = b'abc------------------------'
        sta1.elevation= new_elevation
        sta_inv.site_name = new_sitename
        cha.sample_rate = new_srate
        sta_inv.inventory_xml = new_sta_inv
        db.session.commit()
        
        # assure some data is returned from inventoriy url:
        inv_urlread_ret_val = self._inv_data
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(inv_urlread_ret_val, *a, **v)
    
        # run without flag update on:
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        assert result.exit_code == 0
        
        # update_metadata False: assert nothing has been updated:
        assert db.session.query(Station).filter(Station.elevation == new_elevation).first()
        assert db.session.query(Station).filter(Station.site_name == new_sitename).first()
        assert db.session.query(Channel).filter(Channel.sample_rate == new_srate).first()
        # assert segment without inventory has still No inventory:
        assert db.session.query(Station).filter(Station.id == sta_inv_id).first().inventory_xml == new_sta_inv
        
        # NOW UPDATE METADATA
        
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--update-metadata',
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        assert result.exit_code == 0
        
        # assert that we overwritten the values set above, so we
        assert not db.session.query(Station).filter(Station.elevation == new_elevation).first()
        assert not db.session.query(Channel).filter(Channel.sample_rate == new_srate).first()
        # assert sta_inv has inventory re-downloaded:
        # assert segment without inventory has inventory:
        assert db.session.query(Station).filter(Station.id == sta_inv_id).first().inventory_xml != new_sta_inv
        # and now this:
        assert db.session.query(Station).filter(Station.site_name == new_sitename).first()
        # WHY? because site_name has been implemented for compatibility when the query level=station
        # is done. When querying level=channel (as we do) site_name is not returned (FDSN weird behaviour?)
        # so THAT attribute, and that only, is stille the old one
        
        
        # ------------------------------------------------------------------------
        # NOTE: THIS LAST TEST DELETES ALL SEGMENTS THUS EXECUTE IT AT THE REAL END
        # -----------------------------------------------------------------------
               
        # test a type error in the url_segment_side effect
        db.session.query(Segment).delete()
        assert len(db.session.query(Segment).all()) == 0
        errmsg_py2 = '_sre.SRE_Pattern object is not an iterator'  # python2
        errmsg_py3 = "'_sre.SRE_Pattern' object is not an iterator"  # python3
        assert errmsg_py2 not in self.log_msg()
        assert errmsg_py3 not in self.log_msg()
        suse = self._seg_urlread_sideeffect  # remainder (reset later)
        self._seg_urlread_sideeffect = re.compile(".*")  # just return something not number nor string
        runner = CliRunner()
        result = runner.invoke(cli , ['download', '-c', self.configfile,
                                       '--dburl', db.dburl,
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00', '--inventory'])
        if result.exception:
            assert result.exc_info[0] == SystemExit
            assert errmsg_py2 in self.log_msg() or errmsg_py3 in self.log_msg()
        else:
            print("DID NOT RAISE!!")
            assert False
        self._seg_urlread_sideeffect = suse  # restore default


    @patch('stream2segment.main.run_download')
    def test_yaml_optional_params(self, mock_run, pytestdir):
        with open(self.configfile) as fp:
            _yaml_dict = yaml.load(fp)
        
        # do not provide trailing spaces for easier comparison here below
        vals = [None, ['*'], ['HH?', 'AB']]
        
        params = ['net', 'networks', 'network', 'sta', 'stations', 'station',
                  'loc', 'location', 'locations', 'cha', 'channels', 'channel']
        for p in params:
            _yaml_dict.pop(p, None)
        
        
        for net, sta, loc, cha in product(vals, vals, vals, vals):
            yaml_dict = dict(_yaml_dict)  # copy, otherwise old params are still there ... 
            netname, staname, locname, chaname = None, None, None, None
            if net is not None:
                netname = params[random.randint(0, 2)]
                yaml_dict[netname] = net
            if sta is not None:
                staname = params[random.randint(3, 5)]
                yaml_dict[staname] = sta
            if loc is not None:
                locname = params[random.randint(6, 8)]
                yaml_dict[locname] = loc
            if cha is not None:
                chaname = params[random.randint(9, 11)]
                yaml_dict[chaname] = cha

            configfilename = pytestdir.newfile('.yaml')
            with open(configfilename, 'w') as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)

            mock_run.reset_mock()
                
            runner = CliRunner()
            result = runner.invoke(cli , ['download',
                                           '-c', configfilename,
                                           # '--dburl', db.dburl,
                                           #'--start', '2016-05-08T00:00:00',
                                           #'--end', '2016-05-08T9:00:00'
                                           ])
            args = mock_run.call_args_list
            assert len(args) == 1  # called just once (for safety)
            args = args[0]
            assert not args[0]  # no *args supplied (all kwargs)
            args = args[1]
            for name, val in zip(['networks', 'stations', 'locations', 'channels'],
                                 [net, sta, loc, cha]):
                if val is None or val == ['*']:
                    assert args[name] == []
#                     elif name=='locations' and val == ['--']:
#                         assert args[name] == ['']
                else:
                    assert args[name] == sorted(val)

        # now test errors (duplicates)
        for i in range(4):
            yaml_dict = dict(_yaml_dict)  # copy, otherwise old params are still there ... 
            
            for p1, p2 in combinations(params[i*3: (i+1)*3], 2):
                yaml_dict[p1] = []
                yaml_dict[p2] = []
        
                with open(configfilename, 'w') as outfile:
                    yaml.dump(yaml_dict, outfile, default_flow_style=False)

                mock_run.reset_mock()
                    
                runner = CliRunner()
                result = runner.invoke(cli , ['download',
                                               '-c', configfilename,
                                               # '--dburl', db.dburl,
                                               #'--start', '2016-05-08T00:00:00',
                                               #'--end', '2016-05-08T9:00:00'
                                               ])
                assert result.exit_code != 0
                assert 'Conflicting' in result.output
