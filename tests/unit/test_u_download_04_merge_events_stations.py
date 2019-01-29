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

    def test_merge_event_stations(self, db, tt_ak135_tts):
        # get events with lat lon (1,1), (2,2,) ... (n, n)
        urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|2|2|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
"""
        events_df = self.get_events_df(urlread_sideeffect, db.session)

        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(None, db.session, None, self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)

        # url read for channels: Note: first response data raises, second has an error and
        # that error is skipped (other channels are added), and last two channels are from two
        # stations (BLA|BLA|...) with only different start time (thus stations should both be
        # added)
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|a||HHZ|1|1|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
A|b||HHE|2|2|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""",
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|c||HHZ|3|3|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
BLA|e||HHZ|7|7|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|2019-01-01T00:00:00
BLA|e||HHZ|8|8|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  URLError('wat'), socket.timeout()]

        channels_df = self.get_channels_df(urlread_sideeffect, db.session,
                                           datacenters_df,
                                           eidavalidator,
                                           net, sta, loc, cha, None, None, 10,
                                           False, None, None, -1, self.db_buf_size)
        assert len(channels_df) == 5

    # events_df
#    id  magnitude  latitude  longitude  depth_km                    time
# 0  1   3.0        1.0       1.0        60.0     2016-05-08 05:17:11.500
# 1  2   4.0        90.0      90.0       2.0      2016-05-08 01:45:30.300

    # channels_df:
#     id station_id  latitude  longitude  datacenter_id start_time   end_time
# 0   1           1       1.0        1.0              1 2008-02-12        NaT
# 1   2           2       2.0        2.0              1 2009-01-01        NaT
# 2   3           3       3.0        3.0              2 2008-02-12        NaT
# 3   4           4       7.0        7.0              2 2009-01-01 2019-01-01
# 4   5           5       8.0        8.0              2 2019-01-01        NaT

        tt_table = tt_ak135_tts
        # for magnitude <10, max_radius is 0. For magnitude >10, max_radius is 200
        # we have only magnitudes <10, we have two events exactly on a station (=> dist=0)
        # which will be taken (the others dropped out)
        df = merge_events_stations(events_df, channels_df, minmag=10, maxmag=10,
                                   minmag_radius=0, maxmag_radius=200, tttable=tt_table)

        assert len(df) == 2

        # for magnitude <1, max_radius is 100. For magnitude >1, max_radius is 200
        # we have only magnitudes <10, we have all event-stations closer than 100 deg
        # So we might have ALL channels taken BUT: one station start time is in 2019, thus
        # it will not fall into the case above!
        df = merge_events_stations(events_df, channels_df, minmag=1, maxmag=1,
                                   minmag_radius=100, maxmag_radius=2000, tttable=tt_table)

        assert len(df) == (len(channels_df)-1) * len(events_df)
        # assert channel outside time bounds was in:
        assert not channels_df[channels_df[Station.start_time.key] ==
                               datetime(2019, 1, 1)].empty
        # we need to get the channel id from channels_df cause in df we removed unnecessary
        # columns (including start end time)
        ch_id = channels_df[channels_df[Station.start_time.key] ==
                            datetime(2019, 1, 1)][Channel.id.key].iloc[0]
        # old Channel.id.key is Segment.channel_id.key in df:
        assert df[df[Segment.channel_id.key] == ch_id].empty

        # this is a more complex case, we want to drop the first event by setting a very low
        # threshold (sraidus_minradius=1) for magnitudes <=3 (the first event magnitude)
        # and maxradius very high for the other event (magnitude=4)
        df = merge_events_stations(events_df, channels_df, minmag=3, maxmag=4,
                                   minmag_radius=1, maxmag_radius=40, tttable=tt_table)

        # assert we have only the second event except the first channel which is from the 1st event.
        # The first event is retrievable by its latitude (2)
        # FIXME: more fine grained tests based on distance?
        evid = events_df[events_df[Event.latitude.key] == 2][Event.id.key].iloc[0]
        assert np.array_equal((df[Segment.event_id.key] == evid),
                              [False, True, True, True, True])

        # test arrival times are properly set: Set all event locations to [0,0] as well
        # as stations locations. This should result in all arrival times equal to event time
        #
        _events_df = events_df
        _channels_df = channels_df
        events_df = events_df.copy()
        events_df.loc[:, Event.latitude.key] = 0
        events_df.loc[:, Event.longitude.key] = 0
        event_ids = pd.unique(events_df[Event.id.key])
        # We have two events, set the depth of the first one to zero the other to 60
        evtid1, evtid2 = event_ids[0], event_ids[1]
        evttime1 = events_df[events_df[Event.id.key] == evtid1][Event.time.key].iloc[0]
        evttime2 = events_df[events_df[Event.id.key] == evtid2][Event.time.key].iloc[0]
        events_df.loc[events_df[Event.id.key] == evtid1, Event.depth_km.key] = 0
        events_df.loc[events_df[Event.id.key] == evtid2, Event.depth_km.key] = 60

        channels_df = channels_df.copy()
        channels_df.loc[:, Station.latitude.key] = 0
        channels_df.loc[:, Station.longitude.key] = 0
        df = merge_events_stations(events_df, channels_df, minmag=3, maxmag=4,
                                   minmag_radius=1, maxmag_radius=40, tttable=tt_table)
        # assert for events of depth 0 arrival times are queal to event times
        assert (df[df[Segment.event_id.key] == evtid1][Segment.arrival_time.key]
                == evttime1).all()
        # assert for events of depth > 0 arrival times are GREATER than event times
        assert (df[df[Segment.event_id.key] == evtid2][Segment.arrival_time.key]
                > evttime2).all()

        # now set the first event time out-of bounds:
        events_df.loc[events_df[Event.id.key] == evtid1, Event.depth_km.key] = 600000
        df = merge_events_stations(events_df, channels_df, minmag=3, maxmag=4,
                                   minmag_radius=1, maxmag_radius=40, tttable=tt_table)
        # assert for events of depth 0 arrival times are queal to event times
        # as nans are dropped from the returned dataframe, assert we do not have segments with
        # event_id == evtid1:
        assert df[df[Segment.event_id.key] == evtid1][Segment.arrival_time.key].empty
        # still assert for events of depth > 0 arrival times are GREATER than event times
        assert (df[df[Segment.event_id.key] == evtid2][Segment.arrival_time.key] > evttime2).all()
