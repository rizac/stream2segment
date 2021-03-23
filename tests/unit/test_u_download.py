# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from builtins import str
import socket
from itertools import cycle, product
import logging
from logging import StreamHandler
from io import BytesIO
from mock import patch
from mock import Mock
# this can apparently not be avoided neither with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
try:
    from cStringIO import StringIO  # python2.x
except ImportError:
    from io import StringIO

import pandas as pd
import pytest

from stream2segment.io.utils import Fdsnws
from stream2segment.download.db import Download, Station
from stream2segment.utils.url import URLError, HTTPError, responses
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from stream2segment.download.utils import dblog


query_logger = logger = logging.getLogger("stream2segment")


@pytest.fixture(scope='module')
def tt_ak135_tts(request, data):
    return data.read_tttable('ak135_tts+_5.npz')


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)
        # setup a run_id:
        rdw = Download()
        db.session.add(rdw)
        db.session.commit()
        self.run = rdw

        # side effects:
        self._evt_urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
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
        self._sta_urlread_sideeffect = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
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

        self._mintraveltime_sideeffect = cycle([1])
        self._seg_data = data.read("GE.FLT1..HH?.mseed")
        self._seg_data_gaps = data.read("IA.BAKI..BHZ.D.2016.004.head")
        self._seg_data_empty = b''
        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_gaps, 413, 500,
                                        self._seg_data[:2], self._seg_data_empty,  413,
                                        URLError("++urlerror++"), socket.timeout()]
        self.service = ''  # so get_datacenters_df accepts any row by default
        self.db_buf_size = 1
        self.routing_service = yaml_load(get_templates_fpath("download.yaml"))\
            ['advanced_settings']['routing_service_url']

        # NON db stuff (logging, patchers, pandas...):
        self.logout = StringIO()
        handler = StreamHandler(stream=self.logout)
        self._logout_cache = ""
        # THIS IS A HACK:
        query_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset
        # (to level.INFO) otherwise it stays what we set two lines above. Problems might arise
        # if closing sets a different level, but for the moment who cares
        query_logger.addHandler(handler)

        # define class level patchers (we do not use a yiled as we need to do more stuff in the
        # finalizer, see below
        patchers = []

        patchers.append(patch('stream2segment.utils.url.urlopen'))
        self.mock_urlopen = patchers[-1].start()

        # mock ThreadPool (tp) to run one instance at a time, so we get deterministic results:
        class MockThreadPool(object):

            def __init__(self, *a, **kw):
                pass

            def imap(self, func, iterable, *args):
                # make imap deterministic: same as standard python map:
                # everything is executed in a single thread the right input order
                return map(func, iterable)

            def imap_unordered(self, func_, iterable, *args):
                # make imap_unordered deterministic: same as standard python map:
                # everything is executed in a single thread in the right input order
                return map(func_, iterable)

            def close(self, *a, **kw):
                pass
        # assign patches and mocks:
        patchers.append(patch('stream2segment.utils.url.ThreadPool'))
        self.mock_tpool = patchers[-1].start()
        self.mock_tpool.side_effect = MockThreadPool

        # add finalizer:
        def delete():

            for patcher in patchers:
                patcher.stop()

            hndls = query_logger.handlers[:]
            handler.close()
            for h in hndls:
                if h is handler:
                    query_logger.removeHandler(h)
        request.addfinalizer(delete)

    def log_msg(self):
        idx = len(self._logout_cache)
        self._logout_cache = self.logout.getvalue()
        if len(self._logout_cache) == idx:
            idx = None  # do not slice
        return self._logout_cache[idx:]

    def setup_urlopen(self, urlread_side_effect):
        """setup urlopen return value.
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.read,
            that will be converted to an itertools.cycle(side_effect) REMEMBER that any
            element of urlread_side_effect which is a nonempty string must be followed by an
            EMPTY STRINGS TO STOP reading otherwise we fall into an infinite loop if the
            argument blocksize of url read is not negative !"""

        self.mock_urlopen.reset_mock()
        # convert returned values to the given urlread return value (tuple data, code, msg)
        # if k is an int, convert to an HTTPError
        retvals = []
        # Check if we have an iterable (where strings are considered not iterables):
        if not hasattr(urlread_side_effect, "__iter__") or \
                isinstance(urlread_side_effect, (bytes, str)):
            # it's not an iterable (wheere str/bytes/unicode are considered NOT iterable
            # in both py2 and 3)
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
                        # hacky workaround to support cycle below: if reached the end,
                        # go back to start
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


def test_models_fdsn_url_1():
    for url in ["https://mock/fdsnws/station/1/query",
                "http://mock/fdsnws/station/1/query?",
                "https://mock/fdsnws/station/1/",
                "https://mock/fdsnws/station/1",
                "http://mock/fdsnws/station/1/query?h=8&b=76",
                "https://mock/fdsnws/station/1/auth?h=8&b=76",
                "mock/station/fdsnws/station/1/"]:  # this is not fdsn but we relax conditions
        fdsn = Fdsnws(url)
        expected_scheme = 'https' if url.startswith('https://') else 'http'
        assert fdsn.site == '%s://mock' % expected_scheme
        assert fdsn.service == Fdsnws.STATION
        assert str(fdsn.majorversion) == str(1)
        normalizedurl = fdsn.url()
        assert normalizedurl == '%s://mock/fdsnws/station/1/query' % expected_scheme
        for service in [Fdsnws.STATION, Fdsnws.DATASEL, Fdsnws.EVENT, 'abc']:
            assert fdsn.url(service) == normalizedurl.replace('station', service)

        assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')
        assert fdsn.url(majorversion='1.1') == normalizedurl.replace('1', '1.1')

        for method in [Fdsnws.QUERY, Fdsnws.QUERYAUTH, Fdsnws.APPLWADL, Fdsnws.VERSION,
                       'abcdefg']:
            assert fdsn.url(method=method) == normalizedurl.replace('query', method)

    for url in ["fdsnws/station/1/query",
                "/fdsnws/station/1/query",
                "http:mysite.org/fdsnws/dataselect/1",  # Note: this has invalid scheme
                "http:mysite.org/and/another/path/fdsnws/dataselect/1",
                "http://www.google.com",
                "https://mock/fdsnws/station/abc/1/whatever/abcde?h=8&b=76",
                "https://mock/fdsnws/station/", "https://mock/fdsnws/station",
                "https://mock/fdsnws/station/1/abcde?h=8&b=76",
                "https://mock/fdsnws/station/1/whatever/abcde?h=8&b=76"]:
        with pytest.raises(ValueError):
            Fdsnws(url)


def test_models_fdsn_url():
    url_ = 'abc.org/fdsnws/station/1'
    for (pre, post, slash) in product(['', 'http://', 'https://'],
                                      ['', Fdsnws.QUERY, Fdsnws.QUERYAUTH,
                                       Fdsnws.AUTH,
                                       Fdsnws.APPLWADL, Fdsnws.VERSION],
                                      ['', '/', '?']
                                      ):
        if not post and slash == '?':
            continue  # do not test "abc.org/fdsnws/station/1?" it's invalid
        elif slash == '?':
            asd = 6
        url = pre + url_ + ('/' if post else '') + post + slash
        fdsn = Fdsnws(url)
        if url.startswith('https'):
            assert fdsn.site == 'https://abc.org'
        else:
            assert fdsn.site == 'http://abc.org'
        assert fdsn.service == Fdsnws.STATION
        assert fdsn.majorversion == '1'

        normalizedurl = fdsn.url()
        for service in [Fdsnws.STATION, Fdsnws.DATASEL, Fdsnws.EVENT, 'abc']:
            assert fdsn.url(service) == normalizedurl.replace('station', service)

        assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')

        for method in [Fdsnws.QUERY, Fdsnws.QUERYAUTH, Fdsnws.APPLWADL, Fdsnws.VERSION,
                       'abcdefg']:
            assert fdsn.url(method=method) == normalizedurl.replace('query', method)


@pytest.mark.parametrize(['url_'],
                         [
                          ('',),
                          ('/fdsnws/station/1',),
                          ('fdsnws/station/1/',),
                          ('fdsnws/station/1/query',),
                          ('fdsnws/station/1/query/',),
                          ('abc.org',),
                          ('abc.org/',),
                          ('abc.org/fdsnws',),
                          ('abc.org/fdsnws/',),
                          ('abc.org/fdsnws/bla',),
                          ('abc.org/fdsnws/bla/',),
                          ('abc.org/fdsnws/bla/1',),
                          ('abc.org/fdsnws/bla/1r',),
                          ('abc.org/fdsnws/station/a',),
                          ('abc.org/fdsnws/station/b/',),
                          ('abc.org//fdsnws/station/1.1/',),
                          # ('abc.org/fdsnws/station/1?',),
                          ('abc.org/fdsnws/station/1.1//',),
                          ('abc.org/fdsnws/station/1.1/bla',),
                          ('abc.org/fdsnws/station/1.1/bla/',),])
def test_models_bad_fdsn_url(url_):
    for url in [url_, 'http://' + url_, 'https://'+url_]:
        with pytest.raises(ValueError):
            Fdsnws(url)
