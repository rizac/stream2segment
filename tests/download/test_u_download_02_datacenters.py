# -*- coding: utf-8 -*-
"""
Created on Feb 4, 2016

@author: riccardo
"""
from builtins import open as oopen

from datetime import datetime, timedelta
import socket
from itertools import cycle, product
import logging
from logging import StreamHandler
from io import BytesIO, StringIO
from unittest.mock import Mock, patch, MagicMock

import pytest

from stream2segment.download.db.models import WebService, Download
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.utils import fdsn_url as original_fdsn_url
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import URLError, HTTPError, responses
from stream2segment.resources import get_templates_fpath
from stream2segment.io import yaml_load

query_logger = logger = logging.getLogger("stream2segment")


@pytest.fixture(scope='module')
def tt_ak135_tts(request, data):
    return data.read_tttable('ak135_tts+_5.npz')


class Test:

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
        self._mintraveltime_sideeffect = cycle([1])
        self._seg_data = data.read("GE.FLT1..HH?.mseed")
        self._seg_data_gaps = data.read("IA.BAKI..BHZ.D.2016.004.head")
        self._seg_data_empty = b''
        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_gaps, 413, 500,
                                        self._seg_data[:2], self._seg_data_empty,  413,
                                        URLError("++urlerror++"), socket.timeout()]
        self.service = 'eida'  # NOT USED
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
        # (to level.INFO) otherwise it stays what we set two lines above. Problems might
        # arise if closing sets a different level, but for the moment who cares
        query_logger.addHandler(handler)

        # define class level patchers (we do not use a yiled as we need to do more stuff
        # in the finalizer, see below
        patchers = [
            patch('stream2segment.download.url.urlopen'),
            patch('stream2segment.download.url.ThreadPool')
        ]

        self.mock_urlopen = patchers[-2].start()

        # mock ThreadPool (tp) to run sequentially, so we get deterministic results:
        class MockThreadPool:

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

        if urlread_side_effect is None:
            from urllib.request import urlopen as u
            self.mock_urlopen.side_effect = u
            return

        # convert returned values to the given urlread return value
        # (tuple data, code, msg)
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
            ret = MagicMock()
            ret.__enter__.return_value = a
            retvals.append(ret)

        self.mock_urlopen.side_effect = cycle(retvals)

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(url_read_side_effect)
        return get_datacenters_df(*a, **v)

    @patch('stream2segment.download.modules.datacenters.fdsn_url', return_value='a')
    def test_get_dcs_general(self, mock_urljoin, db):
        """test fetching datacenters eida, iris, custom url"""
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01 *"""]  # <- test also dates (no datetimes) and *

        # provide defaults for arguments not tested here:
        net, sta, loc, cha, start, end = [], [], [], [], None, None

        urljoin_expected_callcount = 0
        # no fdsn service ("http://myservice")
        with pytest.raises(FailedDownload):
            dc_df = self.get_datacenters_df(urlread_sideeffect, db.session,
                                            "http://myservice", self.routing_service,
                                            net, sta, loc, cha, start, end,
                                            db_bufsize=self.db_buf_size)

        assert mock_urljoin.call_count == urljoin_expected_callcount

        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        data = self.get_datacenters_df(urlread_sideeffect, db.session,
                                       "https://mock/fdsnws/station/1/query",
                                       self.routing_service,
                                       net, sta, loc, cha, start, end,
                                       db_bufsize=self.db_buf_size)
        assert mock_urljoin.call_count == urljoin_expected_callcount
        assert len(data) == 1
        assert len(db.session.query(WebService).all()) == 2

        # iris:
        data = self.get_datacenters_df(urlread_sideeffect, db.session,
                                       "iris", self.routing_service,
                                       net, sta, loc, cha, start, end,
                                       db_bufsize=self.db_buf_size)
        assert mock_urljoin.call_count == urljoin_expected_callcount
        assert len(db.session.query(WebService).all()) == 4
        assert len(data) == 1

        # eida:
        data = self.get_datacenters_df(urlread_sideeffect, db.session,
                                       "eida", self.routing_service,
                                       net, sta, loc, cha, start, end,
                                       db_bufsize=self.db_buf_size)
        urljoin_expected_callcount += 1
        assert mock_urljoin.call_count == urljoin_expected_callcount
        # we had two already written, 1 written now:
        assert len(db.session.query(WebService).all()) == 6
        assert len(data) == 2

        # now re-launch and assert we did not write anything to the db cause we already
        # did:
        dcslen = len(db.session.query(WebService).all())
        self.get_datacenters_df(urlread_sideeffect, db.session,
                                "https://mock/fdsnws/station/1/query", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(db.session.query(WebService).all())
        self.get_datacenters_df(urlread_sideeffect, db.session,
                                "iris", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(db.session.query(WebService).all())

        self.get_datacenters_df(urlread_sideeffect, db.session,
                                "eida", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(db.session.query(WebService).all())

    @patch('stream2segment.download.modules.datacenters.fdsn_url',
           side_effect=lambda *a, **v: original_fdsn_url(*a, **v))
    def test_eida_postdata(self, mock_urljoin, db):  # , mock_urljoin):
        """test fetching datacenters eida, iris, custom url and test that postdata is what we
        expected (which is eida/iris/whatever independent)"""
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
ZZ A * * 2002-09-01T00:00:00 2005-10-20T00:00:00
ZZ * 00 * 2002-09-01T00:00:00 2005-10-20T00:00:00
ZZ * 02 * 2002-09-01T00:00:00 2005-10-20T00:00:00
VV * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
VV K * * 2002-09-01T00:00:00 2005-10-20T00:00:00
VV K * BHZ 2002-09-01T00:00:00 2005-10-20T00:00:00
VV K * BHE 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]

        mock_urljoin.reset_mock()
        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        # post data above not used
        data = self.get_datacenters_df(urlread_sideeffect, db.session,
                                       "https://mock/fdsnws/station/1/query",
                                       self.routing_service,
                                       None, None, None, None, None, None,
                                       db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called

        # eida: use post data above ("https://mocked_domain/fdsnws/station/1/query")
        data = self.get_datacenters_df(urlread_sideeffect, db.session,
                                       "eida",
                                       self.routing_service,
                                       None, None, None, None, None, None,
                                       db_bufsize=self.db_buf_size)
        assert mock_urljoin.called
        nslc = sorted(data['net'].str.cat(data[['sta', 'loc', 'cha']], sep='.'))
        assert nslc == ['UP.ARJ.*.BHW', 'VV.K.*.*', 'ZZ,VV.*.*.*']
        assert all(
            len(set(data[c])) == 1 for c in
            ['dataselect_url', 'dataselect_ws_id', 'station_url', 'station_ws_id']
        )

    def test_get_dcs_routingerror(self,
                                  # fixtures:
                                  db):
        """test errors in the routing service(s)"""
        # this is the output when using eida as service:
        urlread_sideeffect = [URLError('wat?')]

        # we might set the following params as defaults because not used, let's provide anyway
        # something meaningful:
        net, sta, loc, cha = ['*'], [], [], ['HH?', 'BH?']
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1.1)

        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        # we should not call self.mock_urlopen and not mock_fileopen (no eida)
        expected_ws = 0
        for url in ["https://mock/fdsnws/station/1/query", "iris"]:
            dcdf = self.get_datacenters_df(urlread_sideeffect, db.session,
                                           url,
                                           self.routing_service,
                                           net, sta, loc, cha, starttime, endtime,
                                           db_bufsize=self.db_buf_size)
            assert not self.mock_urlopen.called
            assert len(dcdf) == 1
            expected_ws += 2
            assert db.session.query(WebService).count() == expected_ws

        # eida:
        with pytest.raises(FailedDownload) as fdwn:
            dcdf = self.get_datacenters_df(urlread_sideeffect, db.session, "eida",
                                           self.routing_service,
                                           net, sta, loc, cha, starttime, endtime,
                                           db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        # msg = self.log_msg()
        assert 'None of the EIDA routing services returned valid data.' \
               in str(fdwn.value)
        assert db.session.query(WebService).count() == expected_ws

    @pytest.mark.parametrize("fdsn_strict", [True, False])
    @patch('stream2segment.download.modules.datacenters.fdsn_url',
           side_effect = lambda *a, **v: original_fdsn_url(*a, **v))
    def test_recif_bug(self, mock_urljoin, fdsn_strict, # fixtures
                       db):  # , mock_urljoin):
        """test a case where we got invalid recif URLs in the routing service list"""

        from stream2segment.download.modules.datacenters import Fdsnws
        # Use mock.patch to mock the constructor of Fdsnws
        with patch('stream2segment.download.modules.datacenters.Fdsnws',
                   side_effect=lambda url: Fdsnws(url, fdsn_strict)) as mock_fdsnws:

            # Make the patched Fdsnws build correct URLs (otherwise get '<mock ...>'
            # instead of 'dataselect' in the URLs):
            mock_fdsnws.DATASEL = Fdsnws.DATASEL
            mock_fdsnws.STATION = Fdsnws.STATION

            # this is the output when using eida as service:
            urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/dataselect/1/query
    ZV * * * 2018-01-01T00:00:00 2019-12-31T23:59:59
    ZU * * * 2015-01-01T00:00:00 2017-12-31T23:59:59

    http://ws.resif.fr/ph5/fdsnws/dataselect/1/query
    ZO * * * 2018-01-01T00:00:00 2018-12-31T23:59:59
    ZO * * * 2014-01-01T00:00:00 2016-12-31T23:59:59
    ZO * * * 2008-01-01T00:00:00 2009-12-31T23:59:59
    Z7 * * * 2018-01-01T00:00:00 2018-12-31T23:59:59
    6J * * * 2018-01-01T00:00:00 2018-12-31T23:59:59
    3C * * * 2019-01-01T00:00:00 2021-12-31T23:59:59
    1F * * * 2018-01-01T00:00:00 2018-12-31T23:59:59
    1D * * * 2019-01-01T00:00:00 2020-12-31T23:59:59

    http://webservices.ingv.it/fdsnws/dataselect/1/query
    ZM * * * 2017-08-26T00:00:00 2020-10-20T00:00:00
    Z3 A319A * * 2015-12-11T12:06:34 2019-04-10T23:59:00
    Z3 A318A * * 2015-11-17T10:32:52 2019-02-02T23:59:00"""]

            mock_urljoin.reset_mock()
            # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
            data = self.get_datacenters_df(urlread_sideeffect, db.session,
                                           "eida",
                                           self.routing_service,
                                           None, None, None, None, None, None,
                                           db_bufsize=self.db_buf_size)
            assert mock_urljoin.called

            dcs = sorted(_.url for _ in db.session.query(WebService))
            log_msg = self.log_msg()
            if fdsn_strict:
                assert dcs == sorted([
                    'http://ws.resif.fr/fdsnws/dataselect/1/query',
                    'http://webservices.ingv.it/fdsnws/dataselect/1/query',
                    'http://ws.resif.fr/fdsnws/station/1/query',
                    'http://webservices.ingv.it/fdsnws/station/1/query'
                ])
                assert "1 data center(s) discarded" in log_msg
                assert 'Discarding data center (Non-standard FDSN URL: ' \
                       'invalid "/ph5" before "fdsnws"). url:' in log_msg
            else:
                assert dcs == sorted([
                    'http://ws.resif.fr/fdsnws/dataselect/1/query',
                    'http://ws.resif.fr/ph5/fdsnws/dataselect/1/query',
                    'http://webservices.ingv.it/fdsnws/dataselect/1/query',
                    'http://ws.resif.fr/fdsnws/station/1/query',
                    'http://ws.resif.fr/ph5/fdsnws/station/1/query',
                    'http://webservices.ingv.it/fdsnws/station/1/query'
                ])
                assert "1 data center(s) discarded" not in log_msg
                assert 'Discarding data center' not in log_msg

    @patch('stream2segment.download.modules.datacenters.fdsn_url',
           side_effect=lambda *a, **v: original_fdsn_url(*a, **v))
    def test_same_dc_in_routingservice(self, mock_urljoin,  # fixtures
                                       db):  # , mock_urljoin):
        """test same WebService in eida routing service
        """
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/dataselect/1/query
ZV * * * 2018-01-01T00:00:00 2019-12-31T23:59:59
ZU * * * 2015-01-01T00:00:00 2017-12-31T23:59:59

http://ws.resif.fr/fdsnws/dataselect/1/
ZO * * * 2018-01-01T00:00:00 2018-12-31T23:59:59
ZO * * * 2014-01-01T00:00:00 2016-12-31T23:59:59

http://webservices.ingv.it/fdsnws/dataselect/1/query
ZM * * * 2017-08-26T00:00:00 2020-10-20T00:00:00
Z3 A319A * * 2015-12-11T12:06:34 2019-04-10T23:59:00
Z3 A318A * * 2015-11-17T10:32:52 2019-02-02T23:59:00"""]

        d0 = datetime.utcnow()
        d1 = d0 + timedelta(minutes=1.1)

        # run tests for all these cases is actually not needed, but they are fast
        # nsl = [['ABC'], []]
        # chans = [['HH?'], ['HH?', 'BH?'], []]

        # for net, sta, loc, cha, stime, etime in product(nsl, nsl, nsl, chans,
        #                                                 [None, d0], [None, d1]):

        net, sta, loc, cha, stime, etime = None, None, None, None, None, None

        mock_urljoin.reset_mock()
        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        data = self.get_datacenters_df(urlread_sideeffect,
                                       db.session,
                                       "eida",
                                       self.routing_service,
                                       net, sta, loc, cha, stime,
                                       etime,
                                       db_bufsize=self.db_buf_size)
        assert mock_urljoin.called

        dcs = sorted(_.url for _ in db.session.query(WebService))
        assert dcs == sorted([
            'http://ws.resif.fr/fdsnws/dataselect/1/query',
            'http://ws.resif.fr/fdsnws/station/1/query',
            'http://webservices.ingv.it/fdsnws/dataselect/1/query',
            'http://webservices.ingv.it/fdsnws/station/1/query'
        ])
        lmsg = self.log_msg()
        # legacy test because we probably discarded resif... now we don't (add 'not'):
        assert "2 data center(s) discarded" not in lmsg

    def test_adarray(self, #fixtures:
                     db):
        from urllib.request import urlopen as original_urlopen
        from stream2segment.download.url import HTTPError, urlread, HTTPException
        try:
            with original_urlopen("https://geofon.gfz-potsdam.de/") as _o:
                _ = _o.read(1)
            no_connection = len(_) < 1
        except Exception:  # noqa
            no_connection = True

        if no_connection:
            pytest.skip("No internet connection")

        data = self.get_datacenters_df(None,  # <- do not mock urlopen(_).read
                                       db.session,
                                       "eida",
                                       self.routing_service,
                                       ["_ADARRAY"], ["A*"], None, None, None, None,
                                       db_bufsize=self.db_buf_size)
        assert len(data) > 20

