# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from builtins import str
from datetime import datetime
import socket
from itertools import cycle
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

import numpy as np
import pandas as pd
import pytest
from obspy.core.stream import read

from stream2segment.download.db import Segment, Download, Station, Channel
from stream2segment.download.modules.events import get_events_df
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df, chaid2mseedid_dict
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import prepare_for_download, \
    download_save_segments, DcDataselectManager, get_counts
from stream2segment.download.utils import Authorizer
from stream2segment.io.db.pdsql import dbquery2df, insertdf, updatedf
from stream2segment.download.utils import s2scodes
from stream2segment.download.modules.mseedlite import unpack
from stream2segment.download.url import URLError, HTTPError, responses
from stream2segment.utils.resources import get_templates_fpath, yaml_load


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
        self.service = 'eida'  # so get_datacenters_df accepts any row by default
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

    def download_save_segments(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return download_save_segments(*a, **kw)

    @patch("stream2segment.download.modules.segments.mseedunpack")
    @patch("stream2segment.io.db.pdsql.insertdf")
    @patch("stream2segment.io.db.pdsql.updatedf")
    def test_download_save_segments(self, mock_updatedf, mock_insertdf, mseed_unpack, db,
                                    tt_ak135_tts):
        # prepare:
        # mseed unpack takes no starttime and endtime arguments, so that
        # we do not discard any correct chunk
        mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)

        urlread_sideeffect = None  # use defaults from class
        events_df = self.get_events_df(urlread_sideeffect, db.session)
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session, self.service,
                                    self.routing_service, net, sta, loc, cha,
                                    db_bufsize=self.db_buf_size)
        channels_df = self.get_channels_df(urlread_sideeffect, db.session,
                                           datacenters_df,
                                           eidavalidator,
                                           net, sta, loc, cha, None, None, 10,
                                           False, None, None, -1, self.db_buf_size)
        assert len(channels_df) == 12  # just to be sure. If failing, we might have changed the class default
    # events_df
#                  id  magnitude  latitude  longitude  depth_km  time
# 0  20160508_0000129        3.0       1.0        1.0      60.0  2016-05-08 05:17:11.500
# 1  20160508_0000004        4.0       2.0        2.0       2.0  2016-05-08 01:45:30.300

# channels_df (index not shown):
# columns:
# id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# data (not aligned with columns):
# 1   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHE
# 2   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHN
# 3   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHZ
# 4   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c1
# 5   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c2
# 6   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c3
# 7   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHE
# 8   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHN
# 9   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHZ
# 10  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c1
# 11  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c2
# 12  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c3

        assert all(_ in channels_df.columns for _ in [Station.network.key, Station.station.key,
                                                      Channel.location.key, Channel.channel.key])
        chaid2mseedid = chaid2mseedid_dict(channels_df)
        # check that we removed the columns:
        assert not any(_ in channels_df.columns for _ in
                       [Station.network.key, Station.station.key,
                        Channel.location.key, Channel.channel.key])

        # take all segments:
        # use minmag and maxmag
        ttable = tt_ak135_tts
        segments_df = merge_events_stations(events_df, channels_df, dict(minmag=10, maxmag=10,
                                            minmag_radius=10, maxmag_radius=10), tttable=ttable)

        assert len(pd.unique(segments_df['arrival_time'])) == 2

        h = 9

# segments_df (index not shown). Note that
# cid sid did n   s    l  c    ed   event_id          depth_km                time  <- LAST TWO ARE Event related columns that will be removed after arrival_time calculations
# 1   1   1   GE  FLT1    HHE  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 2   1   1   GE  FLT1    HHN  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 3   1   1   GE  FLT1    HHZ  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 7   3   2   IA  BAKI    BHE  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 8   3   2   IA  BAKI    BHN  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 9   3   2   IA  BAKI    BHZ  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 4   2   1   n1  s       c1   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 5   2   1   n1  s       c2   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 6   2   1   n1  s       c3   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 10  4   2   n2  s       c1   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 11  4   2   n2  s       c2   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 12  4   2   n2  s       c3   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300

# LEGEND:
# cid = channel_id
# sid = station_id
# scid = datacenter_id
# n, s, l, c = network, station, location, channel
# ed = event_distance_deg

        # define a dc_dataselect_manager for open data only:
        dc_dataselect_manager = DcDataselectManager(datacenters_df, Authorizer(None), False)

        wtimespan = [1,2]
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        orig_segments_df = segments_df.copy()
        segments_df, request_timebounds_need_update = \
            prepare_for_download(db.session, orig_segments_df, dc_dataselect_manager, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True,
                                 retry_timespan_warn=True)

# segments_df
# COLUMNS:
# channel_id  datacenter_id network station location channel event_distance_deg event_id arrival_time start_time end_time id download_status_code run_id
# DATA (not aligned with columns):
#               channel_id  datacenter_id network station location channel  event_distance_deg  event_id            arrival_time          start_time            end_time    id download_status_code  run_id
# GE.FLT1..HHE  1           1              GE      FLT1             HHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# GE.FLT1..HHN  2           1              GE      FLT1             HHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# GE.FLT1..HHZ  3           1              GE      FLT1             HHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# IA.BAKI..BHE  7           2              IA      BAKI             BHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# IA.BAKI..BHN  8           2              IA      BAKI             BHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# IA.BAKI..BHZ  9           2              IA      BAKI             BHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# n1.s..c1      4           1              n1      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n1.s..c2      5           1              n1      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n1.s..c3      6           1              n1      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n2.s..c1      10          2              n2      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n2.s..c2      11          2              n2      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n2.s..c3      12          2              n2      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1

        # self._segdata is the folder file of a "valid" 3-channel miniseed
        # The channels are:
        # Thus, no match will be found and all segments will be written with a None
        # download status code

        # setup urlread: first three rows: ok
        # rows[3:6]: 413, retry them
        # rows[6:9]: malformed_data
        # rows[9:12] 413, retry them
        # then retry:
        # rows[3]: empty_data
        # rows[4]: data_with_gaps (but seed_id should notmatch)
        # rows[5]: data_with_gaps (seed_id should notmatch)
        # rows[9]: URLError
        # rows[10]: Http 500 error
        # rows[11]: 413

        # NOTE THAT THIS RELIES ON THE FACT THAT THREADS ARE EXECUTED IN THE ORDER OF THE DATAFRAME
        # WHICH SEEMS TO BE THE CASE AS THERE IS ONE SINGLE PROCESS
        # self._seg_data[:2] is a way to mock data corrupted
        urlread_sideeffect = [self._seg_data, 413, self._seg_data[:2], 413,
                              '', self._seg_data_gaps, self._seg_data_gaps,
                              URLError("++urlerror++"), 500, 413]
        # Let's go:
        ztatz = self.download_save_segments(urlread_sideeffect, db.session, segments_df,
                                            dc_dataselect_manager,
                                            chaid2mseedid,
                                            self.run.id, False,
                                            request_timebounds_need_update,
                                            1, 2, 3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.id, Segment.channel_id, Segment.datacenter_id,
                Segment.download_code, Segment.maxgap_numsamples, \
                Segment.sample_rate, Segment.data_seed_id, Segment.data, Segment.download_id,
                Segment.request_start, Segment.request_end, Segment.start_time, Segment.end_time
                ]
        db_segments_df = dbquery2df(db.session.query(*cols))
        assert Segment.download_id.key in db_segments_df.columns

        # change data column otherwise we cannot display db_segments_df.
        # When there is data just print "data"
        db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
                           (db_segments_df[Segment.data.key].str.len() > 0),
                           Segment.data.key] = b'data'

        # assert we have 4 segments with "data" properly set:
        assert len(db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
                                      (db_segments_df[Segment.data.key].str.len() > 0),
                                      Segment.data.key]) == 4

        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

# db_segments_df:
#    id  channel_id  datacenter_id  download_status_code  max_gap_ovlap_ratio  sample_rate data_seed_id     data  run_id          start_time            end_time
# 0  1   1           1              200.0                 0.0001               100.0        GE.FLT1..HHE    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 1  2   2           1              200.0                 0.0001               100.0        GE.FLT1..HHN    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 2  3   3           1              200.0                 0.0001               100.0        GE.FLT1..HHZ    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 6  7   7           2              200.0                 NaN                  NaN          None                  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 7  8   8           2              NaN                   NaN                  NaN          None            None  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 8  9   9           2              200.0                 20.0                 20.0         IA.BAKI..BHZ    data  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 3  4   4           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 4  5   5           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 5  6   6           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 9  10  10          2              -1.0                  NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 10 11  11          2              500.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 11 12  12          2              413.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31

        assert len(ztatz) == len(datacenters_df)
        assert len(db_segments_df) == len(segments_df)
        assert mock_updatedf.call_count == 0

        dsc = db_segments_df[Segment.download_code.key]
        exp_dsc = np.array([200, 200, 200, 200, np.nan, 200, -2, -2, -2, -1, 500, 413])
        assert ((dsc == exp_dsc) | (np.isnan(dsc) & np.isnan(exp_dsc))).all()
        # as we have 12 segments and a buf size of self.db_buf_size(=1, but it might change in the
        # future), this below is two
        # it might change if we changed the buf size in the future

        # test that we correctly called mock_insertdf. Note that we assume that the
        # latter is called ONLY inside DbManager. To test that, as the number of stuff
        # to be added (length of the dataframes) varies, we need to implement a counter here:
        mock_insertdf_call_count = 0
        _bufzise = 0
        for c in mock_insertdf.call_args_list:
            c_args = c[0]
            df_ = c_args[0]
            _bufzise += len(df_)
            if _bufzise >= self.db_buf_size:
                mock_insertdf_call_count += 1
                _bufzise = 0

        assert mock_insertdf.call_count == mock_insertdf_call_count

        # assert data is consistent
        COL = Segment.data.key
        assert (db_segments_df.iloc[:3][COL] == b'data').all()
        assert (db_segments_df.iloc[3:4][COL] == b'').all()
        assert pd.isnull(db_segments_df.iloc[4:5][COL]).all()
        assert (db_segments_df.iloc[5:6][COL] == b'data').all()
        assert pd.isnull(db_segments_df.iloc[6:][COL]).all()

        # assert downdload status code is consistent
        URLERR_CODE, MSEEDERR_CODE = s2scodes.url_err, s2scodes.mseed_err

        # also this asserts that we grouped for dc starttime endtime
        COL = Segment.download_code.key
        assert (db_segments_df.iloc[:4][COL] == 200).all()
        assert pd.isnull(db_segments_df.iloc[4:5][COL]).all()
        assert (db_segments_df.iloc[5:6][COL] == 200).all()
        assert (db_segments_df.iloc[6:9][COL] == MSEEDERR_CODE).all()
        assert (db_segments_df.iloc[9][COL] == URLERR_CODE).all()
        assert (db_segments_df.iloc[10][COL] == 500).all()
        assert (db_segments_df.iloc[11][COL] == 413).all()

        # assert gaps are only in the given position
        COL = Segment.maxgap_numsamples.key
        assert (db_segments_df.iloc[:3][COL] < 0.01).all()
        assert pd.isnull(db_segments_df.iloc[3:5][COL]).all()
        assert (db_segments_df.iloc[5][COL] == 20).all()
        assert pd.isnull(db_segments_df.iloc[6:][COL]).all()

        # now mock retry:
        segments_df, request_timebounds_need_update = \
            prepare_for_download(db.session, orig_segments_df, dc_dataselect_manager, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True,
                                 retry_timespan_warn=True)

        assert request_timebounds_need_update is False

        COL = Segment.download_code.key
        mask = (db_segments_df[COL] >= 400) | pd.isnull(db_segments_df[COL]) \
            | (db_segments_df[COL].isin([URLERR_CODE, MSEEDERR_CODE]))
        assert len(segments_df) == len(db_segments_df[mask])

        urlread_sideeffect = [413]
        mock_updatedf.reset_mock()
        mock_insertdf.reset_mock()
        # define a dc_dataselect_manager for open data only:
        dc_dataselect_manager = DcDataselectManager(datacenters_df, Authorizer(None), False)
        # Let's go:
        ztatz = self.download_save_segments(urlread_sideeffect, db.session, segments_df,
                                            dc_dataselect_manager,
                                            chaid2mseedid,
                                            self.run.id, False,
                                            request_timebounds_need_update,
                                            1, 2, 3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.download_code, Segment.channel_id]
        db_segments_df = dbquery2df(db.session.query(*cols))

        # change data column otherwise we cannot display db_segments_df. When there is data
        # just print "data"
        # db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
        # (db_segments_df[Segment.data.key].str.len() > 0), Segment.data.key] = b'data'

        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

        assert (db_segments_df[COL] == 413).all()
        assert len(ztatz) == len(datacenters_df)
        assert len(db_segments_df) == len(segments_df)

        # same as above: but with updatedf: test that we correctly called mock_insertdf_napkeys.
        # Note that we assume that the latter is called ONLY inside download.main.DbManager.
        # To test that, as the number of stuff to be added (length of the dataframes) varies,
        # we need to implement a counter here:
        mock_updatedf_call_count = 0
        _bufzise = 0
        for c in mock_updatedf.call_args_list:
            c_args = c[0]
            df_ = c_args[0]
            _bufzise += len(df_)
            if _bufzise >= self.db_buf_size:
                mock_updatedf_call_count += 1
                _bufzise = 0

        assert mock_updatedf.call_count == mock_updatedf_call_count

        assert mock_insertdf.call_count == 0

    @patch("stream2segment.download.modules.segments.mseedunpack")
    @patch("stream2segment.io.db.pdsql.insertdf")
    @patch("stream2segment.io.db.pdsql.updatedf")
    def test_download_save_segments_timebounds(self, mock_updatedf, mock_insertdf, mseed_unpack,
                                               db, tt_ak135_tts):
        # prepare:
        # mseed unpack takes no starttime and endtime arguments, so that
        mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)

        # mock event response: it's the same as self._evt_urlread_sideeffect but modify the dates
        # as NOW. This means, any segment downloaded later will
        # be out-of-bound
        utcnow = datetime.utcnow()
        utcnow_iso = utcnow.isoformat().replace("T", " ")
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|%s|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|%s|90|90|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
""" % (utcnow_iso, utcnow_iso)
        events_df = self.get_events_df(urlread_sideeffect, db.session)
        # restore urlread_side_effect:
        urlread_sideeffect = None
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session, self.service,
                                    self.routing_service, net, sta, loc, cha,
                                    db_bufsize=self.db_buf_size)
        channels_df = self.get_channels_df(urlread_sideeffect, db.session,
                                           datacenters_df,
                                           eidavalidator,
                                           net, sta, loc, cha, None, None, 10,
                                           False, None, None, -1, self.db_buf_size)
        # just to be sure. If failing, we might have changed the class default:
        assert len(channels_df) == 12
    # events_df
#                  id  magnitude  latitude  longitude  depth_km  time
# 0  20160508_0000129        3.0       1.0        1.0      60.0  2016-05-08 05:17:11.500
# 1  20160508_0000004        4.0       2.0        2.0       2.0  2016-05-08 01:45:30.300

# channels_df (index not shown):
# columns:
# id  station_id  latitude  longitude  datacenter_id start_time end_time network station location channel
# data (not aligned with columns):
# 1   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHE
# 2   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHN
# 3   1  1.0   1.0   1 2003-01-01 NaT  GE  FLT1    HHZ
# 4   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c1
# 5   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c2
# 6   2  90.0  90.0  1 2009-01-01 NaT  n1  s       c3
# 7   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHE
# 8   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHN
# 9   3  1.0   1.0   2 2003-01-01 NaT  IA  BAKI    BHZ
# 10  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c1
# 11  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c2
# 12  4  90.0  90.0  2 2009-01-01 NaT  n2  s       c3

        assert all(_ in channels_df.columns for _ in [Station.network.key, Station.station.key,
                                                      Channel.location.key, Channel.channel.key])
        chaid2mseedid = chaid2mseedid_dict(channels_df)
        # check that we removed the columns:
        assert not any(_ in channels_df.columns for _ in
                       [Station.network.key, Station.station.key,
                        Channel.location.key, Channel.channel.key])

        # take all segments:
        # use minmag and maxmag
        ttable = tt_ak135_tts
        segments_df = merge_events_stations(events_df, channels_df, dict(minmag=10, maxmag=10,
                                            minmag_radius=10, maxmag_radius=10), tttable=ttable)

        assert len(pd.unique(segments_df['arrival_time'])) == 2

        h = 9

# segments_df (index not shown). Note that
# cid sid did n   s    l  c    ed   event_id          depth_km                time  <- LAST TWO ARE Event related columns that will be removed after arrival_time calculations
# 1   1   1   GE  FLT1    HHE  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 2   1   1   GE  FLT1    HHN  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 3   1   1   GE  FLT1    HHZ  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 7   3   2   IA  BAKI    BHE  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 8   3   2   IA  BAKI    BHN  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 9   3   2   IA  BAKI    BHZ  0.0  20160508_0000129  60.0 2016-05-08 05:17:11.500
# 4   2   1   n1  s       c1   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 5   2   1   n1  s       c2   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 6   2   1   n1  s       c3   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 10  4   2   n2  s       c1   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 11  4   2   n2  s       c2   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300
# 12  4   2   n2  s       c3   0.0  20160508_0000004  2.0  2016-05-08 01:45:30.300

# LEGEND:
# cid = channel_id
# sid = station_id
# scid = datacenter_id
# n, s, l, c = network, station, location, channel
# ed = event_distance_deg

        # define a dc_dataselect_manager for open data only:
        dc_dataselect_manager = DcDataselectManager(datacenters_df, Authorizer(None), False)

        wtimespan = [1, 2]  # in minutes
        expected = len(segments_df)  # no segment on db, we should have all segments to download
        orig_segments_df = segments_df.copy()
        segments_df, request_timebounds_need_update = \
            prepare_for_download(db.session, orig_segments_df, dc_dataselect_manager, wtimespan,
                                 retry_seg_not_found=True,
                                 retry_url_err=True,
                                 retry_mseed_err=True,
                                 retry_client_err=True,
                                 retry_server_err=True,
                                 retry_timespan_err=True,
                                 retry_timespan_warn=True)

# segments_df
# COLUMNS:
# channel_id  datacenter_id network station location channel event_distance_deg event_id arrival_time start_time end_time id download_status_code run_id
# DATA (not aligned with columns):
#               channel_id  datacenter_id network station location channel  event_distance_deg  event_id            arrival_time          start_time            end_time    id download_status_code  run_id
# GE.FLT1..HHE  1           1              GE      FLT1             HHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# GE.FLT1..HHN  2           1              GE      FLT1             HHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# GE.FLT1..HHZ  3           1              GE      FLT1             HHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# IA.BAKI..BHE  7           2              IA      BAKI             BHE     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# IA.BAKI..BHN  8           2              IA      BAKI             BHN     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# IA.BAKI..BHZ  9           2              IA      BAKI             BHZ     0.0                 1        2016-05-08 05:17:12.500 2016-05-08 05:16:12 2016-05-08 05:19:12  None  None                 1
# n1.s..c1      4           1              n1      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n1.s..c2      5           1              n1      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n1.s..c3      6           1              n1      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n2.s..c1      10          2              n2      s                c1      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n2.s..c2      11          2              n2      s                c2      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1
# n2.s..c3      12          2              n2      s                c3      0.0                 2        2016-05-08 01:45:31.300 2016-05-08 01:44:31 2016-05-08 01:47:31  None  None                 1

        # self._segdata is the folder file of a "valid" 3-channel miniseed
        # The channels are:
        # Thus, no match will be found and all segments will be written with a None
        # download status code

        # setup urlread: first three rows: ok
        # rows[3:6]: 413, retry them
        # rows[6:9]: malformed_data
        # rows[9:12] 413, retry them
        # then retry:
        # rows[3]: empty_data
        # rows[4]: data_with_gaps (but seed_id should notmatch)
        # rows[5]: data_with_gaps (seed_id should notmatch)
        # rows[9]: URLError
        # rows[10]: Http 500 error
        # rows[11]: 413

        # NOTE THAT THIS RELIES ON THE FACT THAT THREADS ARE EXECUTED IN THE ORDER OF THE DATAFRAME
        # WHICH SEEMS TO BE THE CASE AS THERE IS ONE SINGLE PROCESS
        # self._seg_data[:2] is a way to mock data corrupted
        urlread_sideeffect = [self._seg_data, 413, self._seg_data[:2], 413,
                              '', self._seg_data_gaps, self._seg_data_gaps,
                              URLError("++urlerror++"), 500, 413]
        # Let's go:
        ztatz = self.download_save_segments(urlread_sideeffect, db.session, segments_df,
                                            dc_dataselect_manager,
                                            chaid2mseedid,
                                            self.run.id, False,
                                            request_timebounds_need_update,
                                            1, 2, 3, db_bufsize=self.db_buf_size)
        # get columns from db which we are interested on to check
        cols = [Segment.id, Segment.channel_id, Segment.datacenter_id,
                Segment.download_code, Segment.maxgap_numsamples,
                Segment.sample_rate, Segment.data_seed_id, Segment.data, Segment.download_id,
                Segment.request_start, Segment.request_end, Segment.start_time, Segment.end_time
                ]
        db_segments_df = dbquery2df(db.session.query(*cols))
        assert Segment.download_id.key in db_segments_df.columns

        OUTTIME_ERR, OUTTIME_WARN = s2scodes.timespan_err, s2scodes.timespan_warn
        # assert no segment has data (time out of bounds):
        assert len(db_segments_df.loc[(~pd.isnull(db_segments_df[Segment.data.key])) &
                                      (db_segments_df[Segment.data.key].str.len() > 0),
                                      Segment.data.key]) == 0
        # assert the number of "correctly" downloaded segments, i.e. with data (4) has now
        # code = TIMEBOUND_ERR
        assert len(db_segments_df[db_segments_df[Segment.download_code.key] == OUTTIME_ERR]) == 4

        # re-sort db_segments_df to match the segments_df:
        ret = []
        for cha in segments_df[Segment.channel_id.key]:
            ret.append(db_segments_df[db_segments_df[Segment.channel_id.key] == cha])
        db_segments_df = pd.concat(ret, axis=0)

# db_segments_df:
#    id  channel_id  datacenter_id  download_status_code  max_gap_ovlap_ratio  sample_rate data_seed_id     data  run_id          start_time            end_time
# 0  1   1           1              -3                    0.0001               100.0        GE.FLT1..HHE    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 1  2   2           1              -3                    0.0001               100.0        GE.FLT1..HHN    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 2  3   3           1              -3                    0.0001               100.0        GE.FLT1..HHZ    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 6  7   7           2              200.0                 NaN                  NaN          None                  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 7  8   8           2              NaN                   NaN                  NaN          None            None  1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 8  9   9           2              -3                 20.0                 20.0         IA.BAKI..BHZ    b''   1      2016-05-08 05:16:12 2016-05-08 05:19:12
# 3  4   4           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 4  5   5           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 5  6   6           1             -2.0                   NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 9  10  10          2              -1.0                  NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 10 11  11          2              500.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31
# 11 12  12          2              413.0                 NaN                  NaN          None            None  1      2016-05-08 01:44:31 2016-05-08 01:47:31

        # now modify the first row time bounds:
        # first we need to assign the database id to our segments_df, to prevent
        # db contraint error when writing to db:
        # `download_save_segments` below needs toi UPDATE the segments and it does it by
        # checking if an id is present.
        # check that the channel_ids align:
        assert (segments_df[Segment.channel_id.key].values ==
                db_segments_df[Segment.channel_id.key].values).all()
        # so that we can simply do this:
        segments_df[Segment.id.key] = db_segments_df[Segment.id.key]

        # first read the miniseed:
        stream = read(BytesIO(self._seg_data))
        tstart = stream[0].stats.starttime.datetime
        tend = stream[0].stats.endtime.datetime
        segments_df.loc[segments_df[Segment.channel_id.key] == 1,
                        Segment.request_start.key] = tstart
        segments_df.loc[segments_df[Segment.channel_id.key] == 1,
                        Segment.request_end.key] = tstart + (tend-tstart)/2

        segments_df.loc[segments_df[Segment.channel_id.key] == 2,
                        Segment.request_start.key] = tstart
        segments_df.loc[segments_df[Segment.channel_id.key] == 2,
                        Segment.request_end.key] = tend

        # build a segments_df of the three segments belonging to the same channel
        # copy at the end to avoid pandas settingwithcopy warning
        new_segments_df = \
            segments_df.loc[segments_df[Segment.channel_id.key].isin([1, 2, 3]), :].copy()
        # change urlread_side_effect to provide, for the first three segments, the same
        # sequence of bytes. The sequence actually is OK, but in the first case it will be
        # PARTIALLY saved in the second case TOTALLY, and in the thrid case NOT AT ALL:
        urlread_sideeffect = [self._seg_data, self._seg_data, self._seg_data]
        # define a dc_dataselect_manager for open data only:
        dc_dataselect_manager = DcDataselectManager(datacenters_df, Authorizer(None), False)
        ztatz = self.download_save_segments(urlread_sideeffect, db.session, new_segments_df,
                                            dc_dataselect_manager,
                                            chaid2mseedid,
                                            self.run.id, False,
                                            request_timebounds_need_update,
                                            1, 2, 3, db_bufsize=self.db_buf_size)
        db_segments_df = dbquery2df(db.session.query(*cols))
        # re-sort db_segments_df to match the segments_df:
        ret = [db_segments_df[db_segments_df[Segment.channel_id.key] == cha]
               for cha in segments_df[Segment.channel_id.key]]
        db_segments_df = pd.concat(ret, axis=0)

        # assert the 1st segment whose time range has been modified has data, BUT
        # download_status_code still TIMEBOUNDS_ERROR
        df__ = db_segments_df.loc[db_segments_df[Segment.channel_id.key] == 1, :]
        assert len(df__) == 1
        row__ = df__.iloc[0]
        assert row__[Segment.download_code.key] == OUTTIME_WARN
        assert len(row__[Segment.data.key]) > 0

        # assert the 2nd segment whose time range has been modified has data, AND
        # download_status_code 200 (ok)
        df__ = db_segments_df.loc[db_segments_df[Segment.channel_id.key] == 2, :]
        assert len(df__) == 1
        row__ = df__.iloc[0]
        assert row__[Segment.download_code.key] == 200
        assert len(row__[Segment.data.key]) > 0

        # assert the 3rd segment whose time range has NOT been modified has no data,
        # AND download_status_code is still TIMEBOUNDS_ERROR
        df__ = db_segments_df.loc[db_segments_df[Segment.channel_id.key] == 3, :]
        assert len(df__) == 1
        row__ = df__.iloc[0]
        assert row__[Segment.download_code.key] == OUTTIME_ERR
        assert len(row__[Segment.data.key]) == 0


def test_get_counts():
    '''tests get_counts in segments.py'''
    dframe = pd.DataFrame([
        {'a': 1.1},
        {'a': 1.1},
        {'a': 5},
        {'a': None}
    ])
    d = dict(get_counts(dframe, 'a', 'bla'))
    assert d[1.1] == 2
    assert d[5] == 1
    assert d['bla'] == 1
    assert len(d) == 3

    dframe = pd.DataFrame([
        {'a': None},
        {'a': None},
        {'a': None}
    ])
    d = dict(get_counts(dframe, 'a', None))
    assert d[None] == 3
    assert len(d) == 1

    dframe = pd.DataFrame([
        {'a': 5},
        {'a': 1.1},
        {'a': 1.1}
    ])
    d = dict(get_counts(dframe, 'a', 'bla'))
    assert d[1.1] == 2
    assert d[5] == 1
    assert len(d) == 2
