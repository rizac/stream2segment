"""
Created on Feb 4, 2016

@author: riccardo
"""
import os
import re
from itertools import cycle
import socket
from io import BytesIO
from logging import StreamHandler
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stream2segment.download.modules.segments import DcDataselectManager
from stream2segment.cli import cli
from stream2segment.download.main import get_events_df, get_datacenters_df, \
    get_channels_df, \
    download_save_segments, save_inventories
from stream2segment.download.log import configlog4download
from stream2segment.io import Fdsnws
from stream2segment.download.db.models import DataCenter, Segment, Download, Station
from stream2segment.io.db.pdsql import dbquery2df, insertdf, updatedf
from stream2segment.download.modules.utils import s2scodes
from stream2segment.download.modules.mseedlite import unpack
from stream2segment.download.url import URLError, HTTPError, responses
from stream2segment.resources import get_templates_fpath


@pytest.fixture
def yamlfile(pytestdir):
    '''global fixture wrapping pytestdir.yamlfile'''
    def func(**overridden_pars):
        return pytestdir.yamlfile(get_templates_fpath('download.yaml'), **overridden_pars)

    return func


class Test:

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
        # if we called closing (we are testing the whole chain) the level will be reset
        # (to level.INFO) otherwise it stays what we set two lines above. Problems might arise
        # if closing sets a different level, but for the moment who cares
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

        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_gaps, 413, 500,
                                        self._seg_data[:2],
                                        self._seg_data_empty,  413, URLError("++urlerror++"),
                                        socket.timeout()]

        self._inv_data = data.read("inventory_GE.APE.xml")

        self.service = ''  # so get_datacenters_df accepts any row by default

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

        class patches:
            # paths container for class-level patchers used below. Hopefully
            # will mek easier debug when refactoring/move functions
            urlopen = 'stream2segment.download.url.urlopen'
            get_session = 'stream2segment.download.inputvalidation.get_session'
            close_session = 'stream2segment.download.main.close_session'
            yaml_load = 'stream2segment.download.inputvalidation.yaml_load'
            ThreadPool = 'stream2segment.download.url.ThreadPool'
            configlog4download = 'stream2segment.download.main.configlog4download'

        with patch(patches.urlopen) as mock_urlopen:
            self.mock_urlopen = mock_urlopen
            with patch(patches.get_session, return_value=db.session):
                # this mocks yaml_load and sets inventory to False, as tests rely on that
                with patch(patches.close_session):  # no-op (do not close session)

                    # mock ThreadPool (tp) to run one instance at a time, so we
                    # get deterministic results:
                    class MockThreadPool:

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
                    with patch(patches.ThreadPool, side_effect=MockThreadPool) \
                            as mock_thread_pool:

                        def c4d(logger, logfilebasepath, verbose):
                            # config logger as usual, but redirects to a temp file
                            # that will be deleted by pytest, instead of polluting the program
                            # package:
                            ret = configlog4download(logger, pytestdir.newfile('.log'),
                                                     verbose)
                            logger.addHandler(self.handler)
                            return ret
                        with patch(patches.configlog4download, side_effect=c4d) \
                                as mock_config4download:
                            self.mock_config4download = mock_config4download

                            yield

    def log_msg(self):
        return self.logout.getvalue()

    def setup_urlopen(self, urlread_side_effect):
        """setup urlopen return value.
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.read,
            that will be converted to an itertools.cycle(side_effect) REMEMBER that any
            element of urlread_side_effect which is a nonempty string must be followed by an EMPTY
            STRINGS TO STOP reading otherwise we fall into an infinite loop if the argument
            blocksize of url read is not negative !"""

        self.mock_urlopen.reset_mock()
        # convert returned values to the given urlread return value (tuple data, code, msg)
        # if k is an int, convert to an HTTPError
        retvals = []
        # Check if we have an iterable (where strings are considered not iterables):
        if not hasattr(urlread_side_effect, "__iter__") or \
                isinstance(urlread_side_effect, (bytes, str)):
            urlread_side_effect = [urlread_side_effect]

        for k in urlread_side_effect:
            a = Mock()
            if type(k) == int:
                a.read.side_effect = HTTPError('url', int(k),  responses[k], None, None)
            elif type(k) in (bytes, str):
                def func(k):
                    b = BytesIO(k.encode('utf8') if type(k) == str else k)
                    def rse(*a, **v):
                        rewind = not a and not v
                        if not rewind:
                            currpos = b.tell()
                        ret = b.read(*a, **v)
                        # hacky workaround to support cycle below: if reached the end,
                        # go back to start:
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
        self.mock_urlopen.side_effect = cycle(retvals)

    def get_events_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return get_events_df(*a, **v)

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return get_datacenters_df(*a, **v)

    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return get_channels_df(*a, **kw)

    # ============================================================================================

    def download_save_segments(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return download_save_segments(*a, **kw)

    def save_inventories(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._inv_data if url_read_side_effect is None
                           else url_read_side_effect)
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
    def test_opendata_and_errors(self, mock_get_data_from_token, mock_get_data_from_userpass,
                                 mock_get_data_open, mock_updatedf, mock_insertdf,
                                 mock_mseed_unpack, mock_download_save_segments,
                                 mock_save_inventories, mock_get_channels_df,
                                 mock_get_datacenters_df, mock_get_events_df,
                                 # fixtures:
                                 db, clirunner, pytestdir, yamlfile):

        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v) 
        mock_get_datacenters_df.side_effect = \
            lambda *a, **v: self.get_datacenters_df(None, *a, **v)
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = \
            lambda *a, **v: self.download_save_segments(None, *a, **v)
        # mseed unpack is mocked by accepting only first arg
        # (so that time bounds are not considered)
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
        yaml_file = yamlfile(restricted_data='')
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        assert clirunner.ok(result)
        assert 'Downloading 12 segments (open data only)' in result.output
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
        yaml_file = yamlfile(restricted_data=['user', 'password'], data_url='eida')
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
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
        yaml_file = yamlfile(restricted_data='abcdg465du97_Sdr4fvssgflero',
                             data_url='eida')
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        assert not clirunner.ok(result)
        assert ('invalid token. If you passed a file path') in result.output

        # TEST 2: TOKEN FILE EXISTS, INVALID (e.g. empty)
        filepath = pytestdir.newfile(create=True)
        mock_get_data_open.reset_mock()
        mock_get_data_from_token.reset_mock()
        mock_get_data_from_userpass.reset_mock()
        yaml_file = yamlfile(restricted_data=os.path.abspath(filepath),
                             data_url='eida')
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        assert not clirunner.ok(result)
        assert ('invalid token. If you passed a file path') in result.output

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
    # the following mock is important because: 1 the segments return responses can be set
    # in mock_download_save_segments and work, and 2: avoid performing a real queryauth to the
    # datacenter(s)
    @patch('stream2segment.download.modules.segments.get_opener',
           side_effect=lambda *a, **v: None)
    def test_restricted(self, mock_get_opener, mock_get_data_from_token,
                        mock_get_data_from_userpass,
                        mock_get_data_open, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                        mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                        mock_get_datacenters_df, mock_get_events_df,
                        # fixtures:
                        db, clirunner, pytestdir, yamlfile):

        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v)
        mock_get_datacenters_df.side_effect = \
            lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = \
            lambda *a, **v: self.download_save_segments(None, *a, **v)
        # mseed unpack is mocked by accepting only first arg
        # (so that time bounds are not considered)
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
        yaml_file = yamlfile(restricted_data=os.path.abspath(tokenfile))
        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        assert clirunner.ok(result)
        assert 'Downloading 12 segments (open data only)' in result.output
        assert 'STEP 5 of 8: Acquiring credentials from token' in result.output
        # Check message accounting for legacy dict entries orders:
        if not ('Downloading open data only from: http://geofon.gfz-potsdam.de, '
                'http://ws.resif.fr (Unable to acquire credentials for restricted data)') in \
                result.output:
            assert ('Downloading open data only from: http://ws.resif.fr, '
                    'http://geofon.gfz-potsdam.de (Unable to acquire credentials for restricted data)') in \
                    result.output
        # assert we print that we are downloading open data only (all errors):
        assert 'STEP 7 of 8: Downloading 12 segments (open data only)' in result.output
        assert not mock_get_data_open.called
        assert mock_get_data_from_token.called
        assert not mock_get_data_from_userpass.called
        assert not mock_get_opener.called
        # some assertions to check data properly written
        # These are important because they confirm that data has been downloaded anyway
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
    # the following mock is important because: 1 the segments return responses can be set
    # in mock_download_save_segments and work, and 2: avoid performing a real queryauth to the
    # datacenter(s)
    @patch('stream2segment.download.modules.segments.get_opener')
    def test_retry(self, mock_get_opener, mock_get_data_from_token,
                   mock_get_data_from_userpass,
                   mock_get_data_open, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                   mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                   mock_get_datacenters_df, mock_get_events_df,
                   # fixtures:
                   db, clirunner, pytestdir, yamlfile):

        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v)
        mock_get_datacenters_df.side_effect = \
            lambda *a, **v: self.get_datacenters_df(None, *a, **v)
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = \
            lambda *a, **v: self.download_save_segments([URLError('abc')], *a, **v)
        # mseed unpack is mocked by accepting only first arg (so that time bounds are
        # not considered)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())

        # mock our opener
        m = Mock()
        mockopen = Mock()
        mockopen.read = lambda *a, **v: b''
        mockopen.msg = 'abc'
        mockopen.code = 204
        m.open = lambda *a, **v: mockopen
        # m.read = lambda *a, **v: ''
        mock_get_opener.side_effect = lambda *a, **v: m

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
            yaml_file = yamlfile(restricted_data=os.path.abspath(tokenfile),
                                 retry_client_err=False)
            result = clirunner.invoke(cli, ['download',
                                            '-c', yaml_file,
                                            '--dburl', db.dburl,
                                            '--start', '2016-05-08T00:00:00',
                                            '--end', '2016-05-08T9:00:00'])
            assert clirunner.ok(result)
            assert 'restricted_data: %s' % os.path.abspath(tokenfile) in result.output
            assert 'STEP 5 of 8: Acquiring credentials from token' in result.output
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

            dc_id = {Fdsnws(i[1]).site: i[0] for i in
                     db.session.query(DataCenter.id, DataCenter.dataselect_url)}
            # assert urlopen has been called only once with query and not queryauth:
            # get the segments dataframe we (re)downloaded:
            segments_df_to_download = mock_download_save_segments.call_args_list[-1][0][1]
            dc2download = pd.unique(segments_df_to_download['datacenter_id']).tolist()
            # set the expected call count based on the datacenters of (re)downloaded segments:
            if dc_id[dc_token_failed] not in dc2download:
                assert self.mock_urlopen.call_count == 0
            else:
                assert self.mock_urlopen.call_count >= 1
                for i in range(self.mock_urlopen.call_count):
                    i+=1
                    assert self.mock_urlopen.call_args_list[-i][0][0].get_full_url() == \
                        dc_token_failed + "/fdsnws/dataselect/1/query"

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
    # the following mock is important because: 1 the segments return responses can be set
    # in mock_download_save_segments and work, and 2: avoid performing a real queryauth to the
    # datacenter(s)
    @patch('stream2segment.download.modules.segments.get_opener',
           side_effect=lambda *a, **v: None)
    def test_retry2(self, mock_get_opener, mock_get_data_from_token,
                    mock_get_data_from_userpass,
                    mock_get_data_open, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                    mock_download_save_segments, mock_save_inventories, mock_get_channels_df,
                    mock_get_datacenters_df, mock_get_events_df,
                    # fixtures:
                    db, clirunner, pytestdir, yamlfile):

        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v)
        mock_get_datacenters_df.side_effect = \
            lambda *a, **v: self.get_datacenters_df(None, *a, **v)
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        RESPONSES = [URLError('abc')]
        mock_download_save_segments.side_effect = \
            lambda *a, **v: self.download_save_segments(RESPONSES, *a, **v)
        # mseed unpack is mocked by accepting only first arg (so that time bounds are not
        # considered)
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
            lambda *a, **kw: self.dc_get_data_from_token(['a:b', 'c:d'], *a, **kw)

        # TEST 1: provide a file with valid token:
        tokenfile = pytestdir.newfile(create=True)
        with open(tokenfile, 'w') as fh:
            fh.write('BEGIN PGP MESSAGE')
        # mock yaml_load to override restricted_data:

        # USERPASS good for both  datacenter:
        mock_get_data_open.reset_mock()
        mock_get_data_from_token.reset_mock()
        mock_get_data_from_userpass.reset_mock()
        mock_get_opener.reset_mock()
        mock_get_data_from_token.side_effect = \
            lambda *a, **kw: self.dc_get_data_from_token(['uzer:pazzword', 'uzer:pazzword'],
                                                         *a, **kw)
        yaml_file = yamlfile(restricted_data=os.path.abspath(tokenfile),
                             retry_client_err=False)
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        assert clirunner.ok(result)
        # get db data, sort by index and reset index to assure comparison across data frames:
        seg_df = dbquery2df(db.session.query(Segment.id, Segment.download_code,
                                             Segment.queryauth, Segment.download_id))\
            .sort_values(by=[Segment.id.key]).reset_index(drop=True)
        # seg_df:
        # id  download_code  queryauth  download_id
        # 1  -1              True       2
        # 2  -1              True       2
        # 3  -1              True       2
        # 4  -1              True       2
        # 5  -1              True       2
        # 6  -1              True       2
        # 7  -1              True       2
        # 8  -1              True       2
        # 9  -1              True       2
        # 10 -1              True       2
        # 11 -1              True       2
        # 12 -1              True       2
        urlerr, mseederr = s2scodes.url_err, s2scodes.mseed_err
        # according to our mock, we should have all urlerr codes:
        assert (seg_df[Segment.download_code.key] == urlerr).all()
        assert (seg_df[Segment.queryauth.key] == True).all()
        DOWNLOADID = 2
        assert (seg_df[Segment.download_id.key] == DOWNLOADID).all()
        # other assertions:
        assert 'restricted_data: %s' % os.path.abspath(tokenfile) in result.output
        assert 'STEP 5 of 8: Acquiring credentials from token' in result.output
        # assert we print that we are downloading open and restricted data:
        assert re.search(r'STEP 7 of 8\: Downloading \d+ segments and saving to db',
                         result.output)
        assert not mock_get_data_open.called
        assert mock_get_data_from_token.called
        assert not mock_get_data_from_userpass.called
        # no credentials failed:
        assert "Downloading open data only from: " not in result.output

        # Ok, test retry:
        new_seg_df = seg_df.copy()
        # first get run id
        # we have 12 segments, change the download codes. The second boolean
        # value denotes queryauth (True or False):
        code_queryauth = [(204, False), (204, True), (404, False), (404, True),
                          (401, False), (401, True), (403, False), (403, True),
                          (400, True), (400, False), (None, False), (None, True)]
        for id_, (dc_, qa_) in zip(seg_df[Segment.id.key].tolist(), code_queryauth):
            seg = db.session.query(Segment).filter(Segment.id == id_).first()
            seg.download_code = dc_
            seg.queryauth = qa_
            # set expected values (see also yamlfile below)
            # remember that any segment download will give urlerr as code
            expected_new_download_code = dc_
            expected_download_id = DOWNLOADID
            if dc_ in (204, 404, 401, 403) and qa_ is False:
                # to retry becaue they failed due to authorization problems
                # (or most likely they did)
                expected_new_download_code = urlerr
                expected_download_id = DOWNLOADID + 1
            elif dc_ is None or (dc_ < 400 and dc_ >= 500):
                # to retry because of the flags (see yamlfile below)
                expected_new_download_code = urlerr
                expected_download_id = DOWNLOADID + 1
            expected_query_auth = qa_ if dc_ == 400 else True

            new_seg_df.loc[new_seg_df[Segment.id.key] == id_, :] = \
                (id_, expected_new_download_code, expected_query_auth, expected_download_id)
            db.session.commit()

        # re-download and check what we have retried:
        yaml_file = yamlfile(restricted_data=os.path.abspath(tokenfile),
                             retry_seg_not_found=True,
                             retry_client_err=False)
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        DOWNLOADID += 1
        assert clirunner.ok(result)
        # get db data, sort by index and reset index to assure comparison across data frames:
        seg_df2 = dbquery2df(db.session.query(Segment.id, Segment.download_code, Segment.queryauth,
                                              Segment.download_id))\
            .sort_values(by=[Segment.id.key]).reset_index(drop=True)
        # seg_df2:
        # id  download_code  queryauth  download_id
        # 1  -1              True       3
        # 2   204            True       2
        # 3  -1              True       3
        # 4   404            True       2
        # 5  -1              True       3
        # 6   401            True       2
        # 7  -1              True       3
        # 8   403            True       2
        # 9   400            True       2
        # 10  400            False      2
        # 11 -1              True       3
        # 12 -1              True       3
        pd.testing.assert_frame_equal(seg_df2, new_seg_df)

        # Another retry without modifyiung the segments but setting retry_client_err to True
        # re-download and check what we have retried:
        yaml_file = yamlfile(restricted_data=os.path.abspath(tokenfile),
                             retry_seg_not_found=True,
                             retry_client_err=True)
        result = clirunner.invoke(cli, ['download',
                                        '-c', yaml_file,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        DOWNLOADID += 1
        assert clirunner.ok(result)
        # get db data, sort by index and reset index to assure comparison across data frames:
        seg_df3 = dbquery2df(db.session.query(Segment.id, Segment.download_code, Segment.queryauth,
                                              Segment.download_id))\
            .sort_values(by=[Segment.id.key]).reset_index(drop=True)
        expected_df = seg_df2.copy()
        # modify all 4xx codes as they are updated. Note that old urlerr codes have the old
        # download id (do not override)
        old_4xx = expected_df[Segment.download_code.key].between(400, 499.999)
        expected_df.loc[old_4xx, Segment.download_id.key] = DOWNLOADID
        expected_df.loc[old_4xx, Segment.queryauth.key] = True
        expected_df.loc[old_4xx, Segment.download_code.key] = urlerr
        # seg_df3:
        # id  download_code  queryauth  download_id
        # 1  -1              True       3
        # 2   204            True       2
        # 3  -1              True       3
        # 4  -1              True       4
        # 5  -1              True       3
        # 6  -1              True       4
        # 7  -1              True       3
        # 8  -1              True       4
        # 9  -1              True       4
        # 10 -1              True       4
        # 11 -1              True       3
        # 12 -1              True       3
        pd.testing.assert_frame_equal(seg_df3, expected_df)
        old_urlerr_segids = seg_df2[seg_df2[Segment.download_code.key] == urlerr][Segment.id.key]
        new_urlerr_df = expected_df[expected_df[Segment.id.key].isin(old_urlerr_segids)]
        assert (new_urlerr_df[Segment.download_id.key] == 3).all()
