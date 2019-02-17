# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from builtins import str

try:
    from __builtin__ import open as oopen  # @UnresolvedImport
except:
    from builtins import open as oopen

import os
from datetime import datetime, timedelta
import random
import math
import sys
import socket
from itertools import cycle, repeat, count, product
import logging
from logging import StreamHandler
from io import BytesIO
import threading
from mock import patch
from mock import Mock
# this can apparently not be avoided neither with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
try:
    from cStringIO import StringIO  # python2.x
except ImportError:
    from io import StringIO

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.sql.expression import func
from click.testing import CliRunner
from obspy.core.stream import Stream, read
from obspy.taup.helper_classes import TauModelError

from stream2segment.io.db.models import Base, Event, Class, Fdsnws, DataCenter, Segment, \
    Download, Station, Channel, WebService
from stream2segment.download.modules.events import get_events_df
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df, chaid2mseedid_dict
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import prepare_for_download, \
    download_save_segments, DcDataselectManager
from stream2segment.download.utils import NothingToDownload, FailedDownload, Authorizer
from stream2segment.io.db.pdsql import dbquery2df, insertdf, updatedf
from stream2segment.download.utils import s2scodes
from stream2segment.download.modules.mseedlite import MSeedError, unpack
from stream2segment.utils.url import read_async, URLError, HTTPError, responses
from stream2segment.utils.resources import get_templates_fpath, yaml_load, get_ttable_fpath
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.download.utils import dblog
from stream2segment.utils import urljoin as original_urljoin

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

        # when debugging, I want the full dataframe with to_string(), not truncated
        # NOTE: this messes up right alignment of numbers in DownloadStats (see utils.py)
        # FIRST, remember current settings and restore them in cleanup:
        _pd_display_maxcolwidth = pd.get_option('display.max_colwidth')
        pd.set_option('display.max_colwidth', -1)

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
            pd.set_option('display.max_colwidth', _pd_display_maxcolwidth)

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

        # no fdsn service ("http://myservice")
        with pytest.raises(FailedDownload):
            data, _ = self.get_datacenters_df(urlread_sideeffect, db.session,
                                              "http://myservice", self.routing_service,
                                              net, sta, loc, cha, start, end,
                                              db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called  # is called only when supplying eida

        # normal fdsn service ("https://mocked_domain/fdsnws/station/1/query")
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session,
                                    "https://mock/fdsnws/station/1/query", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called  # is called only when supplying eida
        assert len(db.session.query(DataCenter).all()) == len(data) == 1
        assert db.session.query(DataCenter).first().organization_name is None
        assert eidavalidator is None  # no eida

        # iris:
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session,
                                    "iris", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        assert not mock_urljoin.called  # is called only when supplying eida
        assert len(db.session.query(DataCenter).all()) == 2  # we had one already (added above)
        assert len(data) == 1
        assert len(db.session.query(DataCenter).
                   filter(DataCenter.organization_name == 'iris').all()) == 1
        assert eidavalidator is None  # no eida

        # eida:
        data, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session,
                                    "eida", self.routing_service,
                                    net, sta, loc, cha, start, end,
                                    db_bufsize=self.db_buf_size)
        assert mock_urljoin.called  # is called only when supplying eida
        # we had two already written, 1 written now:
        assert len(db.session.query(DataCenter).all()) == 3
        assert len(data) == 1
        assert len(db.session.query(DataCenter).filter(DataCenter.organization_name ==
                                                       'eida').all()) == 1
        # assert we wrote just resif (the first one, the other one are malformed):
        assert db.session.query(DataCenter).filter(DataCenter.organization_name ==
                                                   'eida').first().station_url == \
            "http://ws.resif.fr/fdsnws/station/1/query"
        assert eidavalidator is not None  # no eida

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
            assert eida_validator is None
            assert not mock_urljoin.called

            # iris:
            mock_urljoin.reset_mock()
            data, eida_validator = self.get_datacenters_df(urlread_sideeffect, db.session, "iris",
                                                           self.routing_service,
                                                           net, sta, loc, cha, starttime, endtime,
                                                           db_bufsize=self.db_buf_size)
            assert eida_validator is None
            assert not mock_urljoin.called

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
        assert not self.mock_urlopen.called
        assert not mock_fileopen.called
        assert eidavalidator is None
        assert len(dcdf) == 1
        assert db.session.query(DataCenter).count() == 1

        # iris:
        # we should not call self.mock_urlopen and not mock_fileopen (no eida)
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session, "iris",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert not self.mock_urlopen.called
        assert not mock_fileopen.called
        assert eidavalidator is None
        assert len(dcdf) == 1
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'iris').count() == 1

        # eida:
        # we should call self.mock_urlopen and mock_fileopen (eida error => read from file)
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session, "eida",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert mock_fileopen.called
        msg = self.log_msg()
        assert "Eida routing service error, reading from file (last updated: " in msg
        assert eidavalidator is not None
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'eida').count() == 10
        assert len(dcdf) == 10

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
        assert not mock_fileopen.called
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'eida').count() == 10
        assert len(dcdf) == 2
        assert "Eida routing service error, reading from file (last updated: " \
            not in self.log_msg()[len(msg):]


        # write two new eida data centers
        self.mock_urlopen.reset_mock()
        mock_fileopen.reset_mock()
        urlread_sideeffect = ["""http://ws.NEWDC1.fr/fdsnws/station/1/query
http://geofon.gfz-potsdam.de/fdsnws/station/1/query

http://NEWDC2.gfz-potsdam.de/fdsnws/station/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * BHW 2013-08-01T00:00:00 2017-04-25"""]
        dcdf, eidavalidator = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                      "eida",
                                                      self.routing_service,
                                                      net, sta, loc, cha, starttime, endtime,
                                                      db_bufsize=self.db_buf_size)
        assert self.mock_urlopen.called
        assert not mock_fileopen.called
        assert db.session.query(DataCenter).\
            filter(DataCenter.organization_name == 'eida').count() == 12
        assert len(dcdf) == 2

