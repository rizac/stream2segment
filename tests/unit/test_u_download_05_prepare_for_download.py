# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from builtins import str
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

    def get_events_df(self, url_read_side_effect, session):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return get_events_df(session, "http://eventws", {}, datetime.utcnow(), datetime.utcnow())

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return get_datacenters_df(*a, **v)

    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return get_channels_df(*a, **kw)
    # def get_channels_df(session, datacenters_df, eidavalidator, # <- can be none
    #                     net, sta, loc, cha, starttime, endtime,
    #                     min_sample_rate, update,
    #                     max_thread_workers, timeout, blocksize, db_bufsize,
    #                     show_progress=False):

    def test_prepare_for_download(self, db, tt_ak135_tts):
        # prepare:
        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, db.session)
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)
        channels_df = self.get_channels_df(urlread_sideeffect, db.session,
                                           datacenters_df,
                                           eidavalidator,
                                           net, sta, loc, cha, None, None, 100,
                                           False, None, None, -1, self.db_buf_size)
        assert len(channels_df) == 12  # just to be sure. If failing, we might have changed the class default
    # events_df
#    id  magnitude  latitude  longitude  depth_km                    time
# 0  1   3.0        1.0       1.0        60.0     2016-05-08 05:17:11.500
# 1  2   4.0        90.0      90.0       2.0      2016-05-08 01:45:30.300

    # channels_df:
#    id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# 0  1   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHE
# 1  2   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHN
# 2  3   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHZ
# 3  4   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c1
# 4  5   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c2
# 5  6   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c3
# 6   7   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHE
# 7   8   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHN
# 8   9   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHZ
# 9   10  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c1
# 10  11  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c2
# 11  12  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c3

        # take all segments:
        segments_df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                            minmag_radius=100, maxmag_radius=200,
                                            tttable=tt_ak135_tts)

# segments_df:
#    channel_id  station_id  datacenter_id network station location channel  event_distance_deg  event_id  depth_km                    time               arrival_time
# 0  1           1           1              GE      FLT1             HHE     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 1  2           1           1              GE      FLT1             HHN     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 2  3           1           1              GE      FLT1             HHZ     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 3  4           2           1              n1      s                c1      89.000              1         60.0     2016-05-08 05:17:11.500 NaT
# 4  5           2           1              n1      s                c2      89.000              1         60.0     2016-05-08 05:17:11.500 NaT
# 5  6           2           1              n1      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 6  7           3           2              IA      BAKI             BHE     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT
# 7  8           3           2              IA      BAKI             BHN     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT
# 8  9           3           2              IA      BAKI             BHZ     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT
# 9  10          4           2              n2      s                c1      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 10  11          4           2              n2      s                c2      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 11  12          4           2              n2      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 12  1           1           1              GE      FLT1             HHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 13  2           1           1              GE      FLT1             HHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 14  3           1           1              GE      FLT1             HHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 15  4           2           1              n1      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 16  5           2           1              n1      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 17  6           2           1              n1      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 18  7           3           2              IA      BAKI             BHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 19  8           3           2              IA      BAKI             BHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 20  9           3           2              IA      BAKI             BHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 21  10          4           2              n2      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 22  11          4           2              n2      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 23  12          4           2              n2      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT

        # make a copy of evts_stations_df cause we will modify in place the data frame
#         segments_df =  self.get_arrivaltimes(urlread_sideeffect, evts_stations_df.copy(),
#                                                    [1,2], ['P', 'Q'],
#                                                         'ak135', mp_max_workers=1)

        expected = len(segments_df)  # no segment on db, we should have all segments to download
        wtimespan = [1,2]
        assert Segment.id.key not in segments_df.columns
        assert Segment.download_id.key not in segments_df.columns
        orig_seg_df = segments_df.copy()
        # define a dc_dataselect_manager for open data only:
        dc_dataselect_manager = DcDataselectManager(datacenters_df, Authorizer(None), False)
        segments_df, request_timebounds_need_update = \
            prepare_for_download(db.session, orig_seg_df, dc_dataselect_manager, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True,
                                 retry_timespan_warn=True)
        assert request_timebounds_need_update is False

# segments_df: (not really the real dataframe, some columns are removed but relevant data is ok):
#    channel_id  datacenter_id network station location channel  event_distance_deg  event_id            arrival_time          start_time            end_time
# 0  1           1              GE      FLT1             HHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 1  2           1              GE      FLT1             HHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 2  3           1              GE      FLT1             HHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 3  4           1              n1      s                c1      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 4  5           1              n1      s                c2      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 5  6           1              n1      s                c3      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 6  7           2              IA      BAKI             BHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 7  8           2              IA      BAKI             BHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 8  9           2              IA      BAKI             BHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 9  10          2              n2      s                c1      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 10  11          2              n2      s                c2      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 11  12          2              n2      s                c3      89.0                1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12
# 12  1           1              GE      FLT1             HHE     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 13  2           1              GE      FLT1             HHN     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 14  3           1              GE      FLT1             HHZ     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 15  4           1              n1      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 16  5           1              n1      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 17  6           1              n1      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 18  7           2              IA      BAKI             BHE     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 19  8           2              IA      BAKI             BHN     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 20  9           2              IA      BAKI             BHZ     89.0                2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 21  10          2              n2      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 22  11          2              n2      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31
# 23  12          2              n2      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31

        assert Segment.id.key in segments_df.columns
        assert Segment.download_id.key not in segments_df.columns
        assert len(segments_df) == expected
        # assert len(db.session.query(Segment.id).all()) == len(segments_df)

        assert all(x[0] is None for x in db.session.query(Segment.download_code).all())
        assert all(x[0] is None for x in db.session.query(Segment.data).all())

        # mock an already downloaded segment.
        # Set the first 7 to have a particular download status code
        urlerr, mseederr, outtime_err, outtime_warn = \
            s2scodes.url_err, s2scodes.mseed_err, s2scodes.timespan_err, s2scodes.timespan_warn
        downloadstatuscodes = [None, urlerr, mseederr, 413, 505, outtime_err, outtime_warn]
        for i, download_code in enumerate(downloadstatuscodes):
            dic = segments_df.iloc[i].to_dict()
            dic['download_code'] = download_code
            dic['download_id'] = self.run.id
            # hack for deleting unused columns:
            for col in [Station.network.key, Station.station.key,
                        Channel.location.key, Channel.channel.key]:
                if col in dic:
                    del dic[col]
            # convet numpy values to python scalars:
            # pandas 20+ seems to keep numpy types in to_dict
            # https://github.com/pandas-dev/pandas/issues/13258
            # this was not the case in pandas 0.19.2
            # sql alchemy does not like that
            # (Curiosly, our pd2sql methods still work fine (we should check why)
            # So, quick and dirty:
            for k in dic.keys():
                if hasattr(dic[k], "item"):
                    dic[k] = dic[k].item()
            # postgres complains about nan primary keys
            if math.isnan(dic.get(Segment.id.key, 0)):
                del dic[Segment.id.key]

            # now we can safely add it:
            db.session.add(Segment(**dic))

        db.session.commit()

        assert len(db.session.query(Segment.id).all()) == len(downloadstatuscodes)

        # Now we have an instance of all possible errors on the db (5 in total) and a new
        # instance (not on the db). Assure all work:
        # set the num of instances to download anyway. Their number is the not saved ones, i.e.:
        to_download_anyway = len(segments_df) - len(downloadstatuscodes)
        for c in product([True, False], [True, False], [True, False], [True, False], [True, False],
                         [True, False], [True, False]):
            s_df, request_timebounds_need_update = \
                prepare_for_download(db.session, orig_seg_df, dc_dataselect_manager, wtimespan,
                                     retry_seg_not_found=c[0],
                                     retry_url_err=c[1],
                                     retry_mseed_err=c[2],
                                     retry_client_err=c[3],
                                     retry_server_err=c[4],
                                     retry_timespan_err=c[5],
                                     retry_timespan_warn=c[6])
            to_download_in_this_case = sum(c)  # count the True's (bool sum works in python)
            assert len(s_df) == to_download_anyway + to_download_in_this_case
            assert request_timebounds_need_update is False

        # now change the window time span and see that everything is to be downloaded again:
        # do it for any retry combinations, as it should ALWAYS return "everything has to be
        # re-downloaded"
        wtimespan[1] += 5
        for c in product([True, False], [True, False], [True, False], [True, False],
                         [True, False], [True, False], [True, False]):
            s_df, request_timebounds_need_update = \
                prepare_for_download(db.session, orig_seg_df, dc_dataselect_manager, wtimespan,
                                     retry_seg_not_found=c[0],
                                     retry_url_err=c[1],
                                     retry_mseed_err=c[2],
                                     retry_client_err=c[3],
                                     retry_server_err=c[4],
                                     retry_timespan_err=c[5],
                                     retry_timespan_warn=c[6])
            assert len(s_df) == len(orig_seg_df)
            assert request_timebounds_need_update is True  # because we changed wtimespan
        # this hol

        # now test that we raise a NothingToDownload
        # first, write all remaining segments to db, with 204 code so they will not be
        # re-downloaded
        for i in range(len(segments_df)):
            download_code = 204
            dic = segments_df.iloc[i].to_dict()
            dic['download_code'] = download_code
            dic['download_id'] = self.run.id
            # hack for deleting unused columns:
            for col in [Station.network.key, Station.station.key,
                        Channel.location.key, Channel.channel.key]:
                if col in dic:
                    del dic[col]
            # convet numpy values to python scalars:
            # pandas 20+ seems to keep numpy types in to_dict
            # https://github.com/pandas-dev/pandas/issues/13258
            # this was not the case in pandas 0.19.2
            # sql alchemy does not like that
            # (Curiosly, our pd2sql methods still work fine (we should check why)
            # So, quick and dirty:
            for k in dic.keys():
                if hasattr(dic[k], "item"):
                    dic[k] = dic[k].item()
            # postgres complains about nan primary keys
            if math.isnan(dic.get(Segment.id.key, 0)):
                del dic[Segment.id.key]

            # now we can safely add it:
            # brutal approach: add and commit, if error, rollback
            # if error it means we already wrote the segment on db and an uniqueconstraint
            # is raised
            try:
                db.session.add(Segment(**dic))
                db.session.commit()
            except SQLAlchemyError as _err:
                db.session.rollback()
        # test that we have the correct number of segments saved:
        assert db.session.query(Segment.id).count() == len(orig_seg_df)
        # try to test a NothingToDownload:
        # reset the old wtimespan otherwise everything will be flagged to be redownloaded:
        wtimespan[1] -= 5
        with pytest.raises(NothingToDownload):
            segments_df, request_timebounds_need_update = \
                prepare_for_download(db.session, orig_seg_df, dc_dataselect_manager, wtimespan,
                                     retry_seg_not_found=False,
                                     retry_url_err=False,
                                     retry_mseed_err=False,
                                     retry_client_err=False,
                                     retry_server_err=False,
                                     retry_timespan_err=False,
                                     retry_timespan_warn=False)

    def test_prepare_for_download_sametimespans(self, db, tt_ak135_tts):
        # prepare. event ws returns two events very close by
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|2.01|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 05:17:12.300000|1.001|1.001|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        events_df = self.get_events_df(urlread_sideeffect, db.session)
        net, sta, loc, cha = [], [], [], []
        urlread_sideeffect = None
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)
        channels_df = self.get_channels_df(urlread_sideeffect, db.session,
                                           datacenters_df,
                                           eidavalidator,
                                           net, sta, loc, cha, None, None, 100,
                                           False, None, None, -1, self.db_buf_size)
        assert len(channels_df) == 12  # just to be sure. If failing, we might have changed the class default
    # events_df
#    id  magnitude  latitude  longitude  depth_km                    time
# 0  1   3.0        1.0       1.0        60.0     2016-05-08 05:17:11.500
# 1  2   4.0        90.0      90.0       2.0      2016-05-08 01:45:30.300

    # channels_df:
#    id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# 0  1   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHE
# 1  2   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHN
# 2  3   1           1.0       1.0        1             2003-01-01 NaT       GE      FLT1             HHZ
# 3  4   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c1
# 4  5   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c2
# 5  6   2           90.0      90.0       1             2009-01-01 NaT       n1      s                c3
# 6   7   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHE
# 7   8   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHN
# 8   9   3           1.0       1.0        2             2003-01-01 NaT       IA      BAKI             BHZ
# 9   10  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c1
# 10  11  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c2
# 11  12  4           90.0      90.0       2             2009-01-01 NaT       n2      s                c3

        # take all segments:
        segments_df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                            minmag_radius=100, maxmag_radius=200,
                                            tttable=tt_ak135_tts)

# segments_df:
#    channel_id  station_id  datacenter_id network station location channel  event_distance_deg  event_id  depth_km                    time               arrival_time
# 0  1           1           1              GE      FLT1             HHE     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 1  2           1           1              GE      FLT1             HHN     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 2  3           1           1              GE      FLT1             HHZ     500.555             1         60.0     2016-05-08 05:17:11.500 2017-05-10 12:39:13.463745
# 3  4           2           1              n1      s                c1      89.000              1         60.0     2016-05-08 05:17:11.500 NaT
# 4  5           2           1              n1      s                c2      89.000              1         60.0     2016-05-08 05:17:11.500 NaT
# 5  6           2           1              n1      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 6  7           3           2              IA      BAKI             BHE     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT
# 7  8           3           2              IA      BAKI             BHN     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT
# 8  9           3           2              IA      BAKI             BHZ     0.0                 1         60.0     2016-05-08 05:17:11.500 NaT
# 9  10          4           2              n2      s                c1      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 10  11          4           2              n2      s                c2      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 11  12          4           2              n2      s                c3      89.0                1         60.0     2016-05-08 05:17:11.500 NaT
# 12  1           1           1              GE      FLT1             HHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 13  2           1           1              GE      FLT1             HHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 14  3           1           1              GE      FLT1             HHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 15  4           2           1              n1      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 16  5           2           1              n1      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 17  6           2           1              n1      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 18  7           3           2              IA      BAKI             BHE     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 19  8           3           2              IA      BAKI             BHN     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 20  9           3           2              IA      BAKI             BHZ     89.0                2         2.0      2016-05-08 01:45:30.300 NaT
# 21  10          4           2              n2      s                c1      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 22  11          4           2              n2      s                c2      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT
# 23  12          4           2              n2      s                c3      0.0                 2         2.0      2016-05-08 01:45:30.300 NaT

        # define a dc_dataselect_manager for open data only:
        dc_dataselect_manager = DcDataselectManager(datacenters_df, Authorizer(None), False)

        expected = len(segments_df)  # no segment on db, we should have all segments to download
        wtimespan = [1,2]
        assert Segment.id.key not in segments_df.columns
        assert Segment.download_id.key not in segments_df.columns
        orig_seg_df = segments_df.copy()
        segments_df, request_timebounds_need_update = \
            prepare_for_download(db.session, orig_seg_df, dc_dataselect_manager, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True,
                                 retry_timespan_warn=True)

        assert request_timebounds_need_update is False

        logmsg = self.log_msg()
        # the dupes should be the number of segments divided by the events set (2) which are
        # very close
        expected_dupes = len(segments_df) / len(events_df)
        assert ("%d suspiciously duplicated segments found" % expected_dupes) in logmsg
