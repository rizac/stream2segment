# -*- coding: utf-8 -*-
"""
Created on Feb 4, 2016

@author: riccardo
"""
from builtins import str

try:
    from __builtin__ import open as oopen  # noqa
except:
    from builtins import open as oopen

from datetime import datetime, timedelta
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

import pytest

from stream2segment.download.db.models import DataCenter, Download
from stream2segment.download.modules.datacenters import get_datacenters_df,\
    _get_local_routing_service
from stream2segment.download.modules.utils import urljoin as original_urljoin
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import URLError, HTTPError, responses
from stream2segment.resources import get_templates_fpath
from stream2segment.io import yaml_load, Fdsnws

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
        self._dc_urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25

http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999

"""
        self._mintraveltime_sideeffect = cycle([1])
        self._seg_data = data.read("GE.FLT1..HH?.mseed")
        self._seg_data_gaps = data.read("IA.BAKI..BHZ.D.2016.004.head")
        self._seg_data_empty = b''
        self._seg_urlread_sideeffect = [self._seg_data, self._seg_data_gaps, 413, 500,
                                        self._seg_data[:2], self._seg_data_empty,  413,
                                        URLError("++urlerror++"), socket.timeout()]
        self.service = 'eida'  # NOT USED (should be so get_datacenters_df accepts any row by default)
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
        patchers = [
            patch('stream2segment.download.url.urlopen'),
            patch('stream2segment.download.url.ThreadPool')
        ]

        self.mock_urlopen = patchers[-2].start()

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

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return get_datacenters_df(*a, **v)

    @patch('stream2segment.download.modules.datacenters.urljoin', return_value='a')
    def test_get_dcs_general(self, mock_urljoin, db):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query

ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]

        # provide defaults for arguments not tested here:
        net, sta, loc, cha, start, end = [], [], [], [], None, None

        urljoin_expected_callcount = 0
        # no fdsn service ("http://myservice")
        with pytest.raises(FailedDownload):
            data, _ = self.get_datacenters_df(urlread_sideeffect, db.session,
                                              "http://myservice", self.routing_service,
                                              net, sta, loc, cha, start, end,
                                              db_bufsize=self.db_buf_size)

        # NOTE: below we test that mock_urljoin.called and
        # ediavalidator != None because of legacy code, where they were true
        # conditionally depending on the input (iris or eida), whreas now both
        # expressions are always True
        urljoin_expected_callcount += 1
        assert mock_urljoin.call_count == urljoin_expected_callcount

        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session,
                                    "https://mock/fdsnws/station/1/query", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        urljoin_expected_callcount += 1
        assert mock_urljoin.call_count == urljoin_expected_callcount
        assert len(db.session.query(DataCenter).all()) == len(data) == 1
        assert db.session.query(DataCenter).first().organization_name is None
        assert eidavalidator is not None

        # iris:
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session,
                                    "iris", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        urljoin_expected_callcount += 1
        assert mock_urljoin.call_count == urljoin_expected_callcount
        assert len(db.session.query(DataCenter).all()) == 2  # we had one already (added above)
        assert len(data) == 1
        assert len(db.session.query(DataCenter).
                   filter(DataCenter.organization_name == 'iris').all()) == 1
        assert eidavalidator is not None

        # eida:
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session,
                                    "eida", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        urljoin_expected_callcount += 1
        assert mock_urljoin.call_count == urljoin_expected_callcount
        # we had two already written, 1 written now:
        assert len(db.session.query(DataCenter).all()) == 3
        assert len(data) == 1
        assert len(db.session.query(DataCenter).filter(DataCenter.organization_name ==
                                                       'eida').all()) == 1
        # assert we wrote just resif (the first one, the other one are malformed):
        assert db.session.query(DataCenter).filter(DataCenter.organization_name ==
                                                   'eida').first().station_url == \
            "http://ws.resif.fr/fdsnws/station/1/query"
        assert eidavalidator is not None

        # now re-launch and assert we did not write anything to the db cause we already did:
        dcslen = len(db.session.query(DataCenter).all())
        self.get_datacenters_df(urlread_sideeffect, db.session,
                                "https://mock/fdsnws/station/1/query", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(db.session.query(DataCenter).all())
        self.get_datacenters_df(urlread_sideeffect, db.session,
                                "iris", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(db.session.query(DataCenter).all())

        self.get_datacenters_df(urlread_sideeffect, db.session,
                                "eida", self.routing_service,
                                net, sta, loc, cha, start, end,
                                db_bufsize=self.db_buf_size)
        assert dcslen == len(db.session.query(DataCenter).all())

    @patch('stream2segment.download.modules.datacenters.urljoin',
           side_effect = lambda *a, **v: original_urljoin(*a, **v))
    def test_get_dcs_postdata(self, mock_urljoin, db):  # , mock_urljoin):
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
                                                              [None, d0], [None, d1]):
            mock_urljoin.reset_mock()
            # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                           "https://mock/fdsnws/station/1/query",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            assert eida_validator is not None
            assert mock_urljoin.called

            # iris:
            mock_urljoin.reset_mock()
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, db.session, "iris",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            assert eida_validator is not None
            assert mock_urljoin.called

            # eida:
            mock_urljoin.reset_mock()
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, db.session, "eida",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)

            geofon_id = data[data[DataCenter.station_url.key] ==
                             'http://geofon.gfz-potsdam.de/fdsnws/station/1/query'].iloc[0].id
            resif_id = data[data[DataCenter.station_url.key] ==
                            'http://ws.resif.fr/fdsnws/station/1/query'].iloc[0].id

            j = mock_urljoin.call_args_list
            assert len(j) == 1
            call_ = j[0]
            args = call_[0]
            kwargs = call_[1]
            assert len(args) == 1
            # assert args[0] == 'http://rz-vm258.gfz-potsdam.de/eidaws/routing/1/query'
            assert args[0] == "http://www.orfeus-eu.org/eidaws/routing/1/query"
            assert kwargs['service'] == 'dataselect'
            assert kwargs['format'] == 'post'

            # urljoin is not called with any other argument. Thus:
            assert len(kwargs) == 2

            # previously, we passed FDSN arguments (uncomment code below
            # in case the support for fdsn parameters will be restored:

#             if net:
#                 assert kwargs['net'] == ','.join(net)
#             else:
#                 assert 'net' not in kwargs
#             if sta:
#                 assert kwargs['sta'] == ','.join(sta)
#             else:
#                 assert 'sta' not in kwargs
#             if loc:
#                 assert kwargs['loc'] == ','.join(loc)
#             else:
#                 assert 'loc' not in kwargs
#             if cha:
#                 assert kwargs['cha'] == ','.join(cha)
#             else:
#                 assert 'cha' not in kwargs
#             if starttime:
#                 assert kwargs['start'] == starttime.isoformat()
#             else:
#                 assert 'start' not in kwargs
#             if endtime:
#                 assert kwargs['end'] == endtime.isoformat()
#             else:
#                 assert 'end' not in kwargs

    @patch('stream2segment.download.modules.datacenters.open',
           side_effect = lambda *a, **v: oopen(*a, **v))
    def test_get_dcs_routingerror(self, mock_fileopen,
                                  # fixtures:
                                  db):
        '''test fetching datacenters eida, iris, custom url'''
        # this is the output when using eida as service:
        urlread_sideeffect = [URLError('wat?')]

        # we might set the following params as defaults because not used, let's provide anyway
        # something meaningful:
        net, sta, loc, cha = ['*'], [], [], ['HH?', 'BH?']
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1.1)

        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        # we should not call self.mock_urlopen and not mock_fileopen (no eida)
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                      "https://mock/fdsnws/station/1/query",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert mock_fileopen.called
        mock_fileopen.reset_mock()
        assert eidavalidator is not None
        assert len(dcdf) == 1
        assert db.session.query(DataCenter).count() == 1

        # iris:
        # we should not call self.mock_urlopen and not mock_fileopen (no eida)
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session, "iris",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert mock_fileopen.called
        mock_fileopen.reset_mock()
        assert eidavalidator is not None
        assert len(dcdf) == 1
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'iris').count() == 1

        # eida:
        # we should call self.mock_urlopen and mock_fileopen (eida error => read from file)
        # first set the expected datacenters we get from teh local file.
        # (see resources/eidars.txt). The datacenters currently there are 13 but one
        # is not fdsn, thus 12:
        EXPECTED_EIDA_DCS_FROMFILE = 13 - 1
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session, "eida",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert mock_fileopen.called
        mock_fileopen.reset_mock()
        msg = self.log_msg()
        _, last_mod_time = _get_local_routing_service()
        expected_str = ("Eida routing service error, reading routes from file "
                        "(last updated: %s") % last_mod_time
        assert expected_str in msg
        assert eidavalidator is not None
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'eida').count() == EXPECTED_EIDA_DCS_FROMFILE
        assert len(dcdf) == EXPECTED_EIDA_DCS_FROMFILE

#         with pytest.raises(FailedDownload) as qdown:
#             data, _ = self.get_datacenters_df(urlread_sideeffect, db.session, "eida",
#                                               self.routing_service,
#                                               net, sta, loc, cha, starttime, endtime,
#                                               db_bufsize=self.db_buf_size)
#         assert self.mock_urlopen.called
#         assert "Eida routing service error, no eida data-center saved in database" \
#             in str(qdown.value)

        # now let's mock a valid response from the eida routing service
        self.mock_urlopen.reset_mock()
        mock_fileopen.reset_mock()
        urlread_sideeffect = ["""http://ws.resif.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://geofon.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                      "eida",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert not mock_fileopen.called  # no err => no read from file
        mock_fileopen.reset_mock()
        # datacenters on the mocked response are two:
        assert len(dcdf) == 2
        # on the database, we did not add any new data center:
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'eida').count() \
               == EXPECTED_EIDA_DCS_FROMFILE
        assert "Eida routing service error, reading from file (last updated: " \
            not in self.log_msg()[len(msg):]


        # write two new eida data centers
        self.mock_urlopen.reset_mock()
        mock_fileopen.reset_mock()
        urlread_sideeffect = ["""http://ws.NEWDC1.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://ws.NEWDC2.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                      "eida",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert not mock_fileopen.called  # no err => no read from file
        mock_fileopen.reset_mock()
        # datacenters on the mocked response are two:
        # Note that according to :func:`stream2segment.download.modules.datacenters.eidarsiter`
        # the second line in in `urlread_sideeffect` above is interpreted as a station of
        # the first datacenter NEWDC1. Thus we have only two datacenters NEWDC1 and NEWDC2
        assert len(dcdf) == 2
        assert 'NEWDC1' in sorted(dcdf.dataselect_url)[0]
        assert 'NEWDC2' in sorted(dcdf.dataselect_url)[1]
        # on the database, we added two more data center:
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'eida').count() == \
            2 + EXPECTED_EIDA_DCS_FROMFILE

    @pytest.mark.parametrize("fdsn_strict", [True, False])
    @patch('stream2segment.download.modules.datacenters.urljoin',
           side_effect = lambda *a, **v: original_urljoin(*a, **v))
    def test_recif_bug(self, mock_urljoin, fdsn_strict, # fixtures
                       db):  # , mock_urljoin):
        """test fetching datacenters eida, iris, custom url and test that postdata is what we
        expected (which is eida/iris/whatever independent)
        """
        class Fdsn2(Fdsnws):

            def __init__(self, url):
                super().__init__(url, fdsn_strict)

        # patch.object is a mess. I tried, hard, and failed. So use normal patch:
        with patch('stream2segment.download.modules.datacenters.Fdsnws',
                    side_effect=lambda *a, **w: Fdsn2(*a, **w)) as mock_fdsnws:
            # The patched still does not build correct URLs (it uses '<mock bla bla>'
            # instead of 'dataselect' in the URLs). So:
            mock_fdsnws.DATASEL = Fdsnws.DATASEL

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

            d0 = datetime.utcnow()
            d1 = d0 + timedelta(minutes=1.1)

            # run tests for all these cases is useless, as these arguments are not used
            # So comment it:

            # nsl = [['ABC'], []]
            # chans = [['HH?'], ['HH?', 'BH?'], []]
            #
            # for net, sta, loc, cha, stime, etime in product(nsl, nsl, nsl, chans,
            #                                                 [None, d0], [None, d1]):

            net, sta, loc, cha, stime, etime = None, None, None, None, None, None

            mock_urljoin.reset_mock()
            # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                           "eida",
                                                           self.routing_service,
                                                           net, sta, loc, cha, stime, etime,
                                                           db_bufsize=self.db_buf_size)
            assert eida_validator is not None
            assert mock_urljoin.called

            dcs = sorted(_.dataselect_url for _ in db.session.query(DataCenter))
            log_msg = self.log_msg()
            if fdsn_strict:
                assert dcs == sorted(['http://ws.resif.fr/fdsnws/dataselect/1/query',
                                      'http://webservices.ingv.it/fdsnws/dataselect/1/query'])
                assert "1 data center(s) discarded" in log_msg
                assert 'Discarding data center (Non-standard FDSN URL: ' \
                       'invalid "/ph5" before "fdsnws"). url:' in log_msg
            else:
                assert dcs == sorted(['http://ws.resif.fr/fdsnws/dataselect/1/query',
                                      'http://ws.resif.fr/ph5/fdsnws/dataselect/1/query',
                                      'http://webservices.ingv.it/fdsnws/dataselect/1/query'])
                assert "1 data center(s) discarded" not in log_msg
                assert 'Discarding data center' not in log_msg

    @patch('stream2segment.download.modules.datacenters.urljoin',
           side_effect=lambda *a, **v: original_urljoin(*a, **v))
    def test_same_dc_in_routingservice(self, mock_urljoin,  # fixtures
                                       db):  # , mock_urljoin):
        """test same datacenter in eida routing service
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
        data, eida_validator = self.get_datacenters_df(urlread_sideeffect,
                                                       db.session,
                                                       "eida",
                                                       self.routing_service,
                                                       net, sta, loc, cha, stime,
                                                       etime,
                                                       db_bufsize=self.db_buf_size)
        assert eida_validator is not None
        assert mock_urljoin.called

        dcs = sorted(_.dataselect_url for _ in db.session.query(DataCenter))
        assert dcs == sorted([# 'http://ws.resif.fr/fdsnws/dataselect/1/query',
                              # 'http://ws.resif.fr/ph5/fdsnws/dataselect/1/query',
                              'http://webservices.ingv.it/fdsnws/dataselect/1/query'])
        lmsg = self.log_msg()
        assert "2 data center(s) discarded" in lmsg