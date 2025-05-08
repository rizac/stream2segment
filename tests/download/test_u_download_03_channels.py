# -*- coding: utf-8 -*-
"""
Created on Feb 4, 2016

@author: riccardo
"""
from datetime import datetime
import socket
from itertools import cycle, product
import logging
from logging import StreamHandler
from io import BytesIO, StringIO
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from stream2segment.download.db.models import Download, Station, Channel
from stream2segment.download.modules.events import get_events_df
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df
from stream2segment.download.exc import FailedDownload
from stream2segment.io.db.pdsql import dbquery2df
from stream2segment.download.url import URLError, HTTPError, responses
from stream2segment.resources import get_templates_fpath
from stream2segment.io import yaml_load

query_logger = logger = logging.getLogger("stream2segment")


@pytest.fixture(scope='module')
def tt_ak135_tts(request, data):
    return data.read_tttable('ak135_tts+_5.npz')


class Test:

    # execute this fixture always even if not provided as argument:
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)
        # setup a run_id:
        rdw = Download()
        db.session.add(rdw)
        db.session.commit()
        self.run = rdw

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
        # (to level.INFO) otherwise it stays what we set two lines above. Problems might
        # arise
        # if closing sets a different level, but for the moment who cares
        query_logger.addHandler(handler)

        # define class level patchers (we do not use a yiled as we need to do more stuff
        # in the finalizer, see below
        patchers = [
            patch('stream2segment.download.url.urlopen'),
            patch('stream2segment.download.url.ThreadPool')
        ]

        self.mock_urlopen = patchers[-2].start()

        # mock ThreadPool (tp) to run sequenctially so we get deterministic results:
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
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.
            read, that will be converted to an itertools.cycle(side_effect) REMEMBER that
            any element of urlread_side_effect which is a nonempty string must be
            followed by an EMPTY STRINGS TO STOP reading otherwise we fall into an i
            nfinite loop if the argument blocksize of url read is not negative !"""

        self.mock_urlopen.reset_mock()

        if urlread_side_effect is None:
            from urllib.request import urlopen as u
            self.mock_urlopen.side_effect = u
            return

        # convert returned values to the given urlread return value (data, code, msg)
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

    def get_events_df(self, url_read_side_effect, session):
        """1st arg. mocks `urllib.urlopen.read`: None, no mocking. Otherwise, it is the
        sequence of returned values of each url opened within this function call"""
        self.setup_urlopen(url_read_side_effect)
        now = datetime.utcnow()
        return get_events_df(session, "http://eventws", {}, now, datetime.utcnow())

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        """1st arg. mocks `urllib.urlopen.read`: None, no mocking. Otherwise, it is the
        sequence of returned values of each url opened within this function call"""
        self.setup_urlopen(url_read_side_effect)
        return get_datacenters_df(*a, **v)

    def get_channels_df(self, url_read_side_effect, *a, **kw):
        """1st arg. mocks `urllib.urlopen.read`: None, no mocking. Otherwise, it is the
        sequence of returned values of each url opened within this function call"""
        self.setup_urlopen(url_read_side_effect)
        return get_channels_df(*a, **kw)

    def test_get_channels_df(self, db):
        urlread_sideeffect = """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""  # noqa
        events_df = self.get_events_df(urlread_sideeffect, db.session)

        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
ZZ * * * 2002-09-01T00:00:00 2005-10-20T00:00:00
UP ARJ * * 2013-08-01T00:00:00 2017-04-25

http://ws.resif.fr/fdsnws/dataselect/1/query
ZU * * HHZ 2015-01-01T00:00:00 2016-12-31T23:59:59.999999
"""
        # we tried to add two events with the same id, check we printed out the msg:
        assert "Duplicated instances violate db constraint" in self.log_msg()

        net, sta, loc, cha = [], [], [], []
        datacenters_df = self.get_datacenters_df(urlread_sideeffect, db.session,
                                                 self.service, self.routing_service,
                                                 net, sta, loc, cha,
                                                 db_bufsize=self.db_buf_size)
        # mock url errors in all queries. We still did not write anything in the db
        # so we should quit:
        with pytest.raises(FailedDownload) as qd:
            _ = self.get_channels_df(URLError('urlerror_wat'), db.session,
                                     datacenters_df,
                                     net, sta, loc, cha, None, None, 100,
                                     False, None, None, -1, self.db_buf_size)
        log_msg = self.log_msg()
        assert 'urlerror_wat' in log_msg
        assert "Unable to fetch stations" in log_msg
        assert "Fetching stations from database: " in log_msg
        # Test that the exception message is correct
        # note that this message is in the log if we run the method from the main
        # function (which is not the case here):
        assert ("Unable to fetch stations from all data-centers, "
                "no data to fetch from the database. "
                "Check config and log for details") in str(qd.value)

        # now get channels with a mocked custom urlread_sideeffect below:
        # IMPORTANT: url read for channels: Note: first response data raises, second has
        # an error and that error is skipped (the other channels are added)
        urlread_sideeffect = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
--- ERROR --- MALFORMED|12T00:00:00|
HT|AGG||HHZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|
""",  # noqa
# NOTE THAT THE CHANNELS ABOVE WILL BE OVERRIDDEN BY THE ONES BELOW (MULTIPLE NAMES< WE
# SHOULD NOT HAVE THIS CASE WITH THE EDIAWS ROUTING SERVICE BUT WE TEST HERE THE CASE)
# NOTE THE USE OF HTß as SensorDescription (to check non-asci characters do not raise)
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
HT|AGG||HHE|--- ERROR --- NONNUMERIC |22.336|622.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|70.0|2008-02-12T00:00:00|
HT|AGG||HLE|95.6|22.336|622.0|0.0|90.0|0.0|GFZ:HTß1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|AGG||HLZ|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2008-02-12T00:00:00|
HT|LKD2||HHE|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|90.0|2009-01-01T00:00:00|
HT|LKD2||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|90.0|2009-01-01T00:00:00|
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2009-01-01T00:00:00|2019-01-01T00:00:00
BLA|BLA||HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2018-01-01T00:00:00|
"""  # noqa
                              ]

        cha_df = self.get_channels_df(urlread_sideeffect, db.session,
                                      datacenters_df,
                                      net, sta, loc, cha, None, None, 90,
                                      False, None, None, -1, self.db_buf_size)
        # assert we have a message for discarding the response data
        # (first arg of urlread):
        assert "Discarding response data" in self.log_msg()
        # we should have called mock_urlopen_in_async times the datacenters
        assert self.mock_urlopen.call_count == len(datacenters_df)
        assert len(db.session.query(Station.id).all()) == 4
        # the last two channels of the second item of `urlread_sideeffect` are from two
        # stations (BLA|BLA|...) with only different start time. Thus they should
        # both be added:
        assert len(db.session.query(Channel.id).all()) == 6
        # as net, sta, loc, cha are all empty lists and start = end = None
        # (all default=>no filter),
        # this is the post data passed to urlread for the 1st datacenter:
        assert self.mock_urlopen.call_args_list[0][0][0].data == b"""format=text
level=channel
* * * * * *"""
        # as net, sta, loc, cha are all empty lists and start = end = None (all default
        # => no filter),this is the post data passed to urlread for the 2nd datacenter:
        assert self.mock_urlopen.call_args_list[1][0][0].data == b"""format=text
level=channel
* * * * * *"""
        assert self.mock_urlopen.call_args_list[0][0][0].get_full_url() == \
            "http://geofon.gfz-potsdam.de/fdsnws/station/1/query"
        assert self.mock_urlopen.call_args_list[1][0][0].get_full_url() == \
            "http://ws.resif.fr/fdsnws/station/1/query"
        # assert all downloaded stations have datacenter_id of the second datacenter:
        dcid = datacenters_df.iloc[1].id
        assert all(sid[0] == dcid for sid in db.session.query(Station.datacenter_id).
                   all())
        # assert all downloaded channels have station_id in the set of downloaded
        # stations only:
        sta_ids = [x[0] for x in db.session.query(Station.id).all()]
        assert all(c_staid[0] in sta_ids for c_staid in db.session.query(
            Channel.station_id).all())

        # now mock again url errors in all queries. As we wrote something in the db
        # so we should NOT quit
        cha_df2 = self.get_channels_df(URLError('urlerror_wat'), db.session,
                                       datacenters_df,
                                       net, sta, loc, cha, datetime(2020, 1, 1), None,
                                       100, False, None, None, -1, self.db_buf_size)

        # Note above that min sample rate = 100 and a starttime proivded should return
        # 3 channels:
        assert len(cha_df2) == 3
        assert "Fetching stations from database for 2 (of 2) data center(s)" in \
               self.log_msg()

        # now test again with a socket timeout
        cha_df2 = self.get_channels_df(socket.timeout(), db.session,
                                       datacenters_df,
                                       net, sta, loc, cha, None, None, 100,
                                       False, None, None, -1, self.db_buf_size)
        assert 'timeout' in self.log_msg() or 'TimeoutError' in self.log_msg()
        assert "Fetching stations from database for 2 (of 2) data center(s)" in \
               self.log_msg()

        # now mixed case:

        # now change min sampling rate and see that we should get one channel less
        cha_df3 = self.get_channels_df(urlread_sideeffect, db.session,
                                       datacenters_df,
                                       net, sta, loc, cha, None, None,  100,
                                       False, None, None, -1, self.db_buf_size)
        assert len(cha_df3) == len(cha_df)-2
        assert "2 channel(s) discarded according to current configuration filters" \
               in self.log_msg()

        # now change this:

        urlread_sideeffect  = [URLError('wat'),
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A|B|10|HBE|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2003-02-12T00:00:00|2010-02-12T00:00:00
E|F|11|HHZ|38.7889|20.6578|485.0|0.0|90.0|0.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|100.0|2019-01-01T00:00:00|
""",  # noqa
                               URLError('wat'), socket.timeout()]

        # now change channels=['B??']. In the urlread_sideeffect above, for the 1st,
        # 3rd and 4th case we fallback to a db query, but we do not have such a channel,
        # so nothing is returned
        # The dataframe currently saved on db is:
        #    id channel start_time   end_time  sample_rate  datacenter_id
        # 0  1   HLE    2008-02-12 NaT         100.0        2
        # 1  2   HLZ    2008-02-12 NaT         100.0        2
        # 2  3   HHE    2009-01-01 NaT         90.0         2
        # 3  4   HHZ    2009-01-01 NaT         90.0         2
        # 4  5   HHZ    2009-01-01 2019-01-01  100.0        2
        # 5  6   HHZ    2018-01-01 NaT         100.0        2
        # for the second case, the mocked response returns two channels and in this case
        # we might put whatever filter here below. Assert that the number of channels
        # returned is 2
        cha_df = self.get_channels_df(urlread_sideeffect, db.session,
                                      datacenters_df,
                                      net, sta, loc, ['B??'], None, None, 10,
                                      False, None, None, -1, self.db_buf_size)
        assert len(cha_df) == 2

        # test channels and startime + entimes provided when querying the db (postdata
        # None) by iussuing the command:
        # dbquery2df(db.session.query(Channel.id, Station.network, Station.station,
        #  Channel.location, Channel.channel, Station.start_time,Station.end_time,
        #  Channel.sample_rate, Station.datacenter_id).join(Channel.station))
        # This is the actual state of the db:
        # ----------------------------------------------
        # channel_id network station location channel start_time    end_time   sample_rate  datacenter_id      # noqa
        #          1      HT     AGG              HLE    2008-02-12 NaT              100.0              2      # noqa
        #          2      HT     AGG              HLZ    2008-02-12 NaT              100.0              2      # noqa
        #          3      HT     LKD2             HHE    2009-01-01 NaT              90.0               2      # noqa
        #          4      HT     LKD2             HHZ    2009-01-01 NaT              90.0               2      # noqa
        #          5      BLA    BLA              HHZ    2009-01-01 2019-01-01       100.0              2      # noqa
        #          6      BLA    BLA              HHZ    2018-01-01 NaT              100.0              2      # noqa
        #          7      A      B         10     HBE    2003-02-12 2010-02-12       100.0              2      # noqa
        #          8      E      F         11     HHZ    2019-01-01 NaT              100.0              2      # noqa
        # ----------------------------------------------
        # Now according to the table above set a list of arguments:
        # Each key is: the argument, each value IS A LIST OF BOOLEAN MAPPED TO EACH ROW
        # OF THE DATAFRAME ABOVE, telling if the row matches according to the argument:
        nets = {('*',): [1, 1, 1, 1, 1, 1, 1, 1],
                # ('HT', 'BLA'): [1, 1, 1, 1, 1, 1, 0, 0],
                ('*A*',): [0, 0, 0, 0, 1, 1, 1, 0]
                }
        stas = {('B*',): [0, 0, 0, 0, 1, 1, 1, 0],
                ('B??',): [0, 0, 0, 0, 1, 1, 0, 0]}
        # note that we do NOT assume '--' can be given, as this should be the parsed
        # output of `nslc_lists`:
        locs = {('',): [1, 1, 1, 1, 1, 1, 0, 0],
                ('1?',): [0, 0, 0, 0, 0, 0, 1, 1]}
        chans = {('?B?',): [0, 0, 0, 0, 0, 0, 1, 0],
                 ('HL?', '?B?'): [1, 1, 0, 0, 0, 0, 1, 0],
                 ('HHZ',): [0, 0, 0, 1, 1, 1, 0, 1]}
        stimes = {None: [1, 1, 1, 1, 1, 1, 1, 1],
                  datetime(2002, 1, 1): [1, 1, 1, 1, 1, 1, 1, 1],
                  datetime(2099, 1, 1): [1, 1, 1, 1, 0, 1, 0, 1]}
        etimes = {None: [1, 1, 1, 1, 1, 1, 1, 1],
                  datetime(2002, 1, 1): [0, 0, 0, 0, 0, 0, 0, 0],
                  datetime(2011, 1, 1): [1, 1, 1, 1, 1, 0, 1, 0],
                  datetime(2099, 1, 1): [1, 1, 1, 1, 1, 1, 1, 1]}
        minsr = {90: [1, 1, 1, 1, 1, 1, 1, 1],
                 # 95: [1, 1, 0, 0, 1, 1, 1, 1],
                 100: [1, 1, 0, 0, 1, 1, 1, 1],
                 105: [0, 0, 0, 0, 0, 0, 0, 0]}
        # no url read: set socket.tiomeout as urlread side effect. This will force
        # querying the database to test that the filtering works as expected:
        for n, s, l, c, st, e, m in product(nets, stas, locs, chans, stimes, etimes,
                                            minsr):
            matches = np.array(nets[n]) * np.array(stas[s]) * np.array(locs[l]) * \
                np.array(chans[c]) * np.array(stimes[st]) * np.array(etimes[e]) * \
                np.array(minsr[m])
            expected_length = matches.sum()
            # Now: if expected length is zero, it means we do not have data matches on
            # the db. This raises a quitdownload (avoiding pytest.raises cause in this
            # case it's easier like done below):
            try:
                __dc_df = datacenters_df.loc[datacenters_df[DataCenter.id.key] == 2]
                cha_df = self.get_channels_df(socket.timeout(), db.session, __dc_df,
                                              eidavalidator, n, s, l, c, st, e, m,
                                              False, None, None, -1, self.db_buf_size)
                assert len(cha_df) == expected_length
            except FailedDownload as qd:
                assert expected_length == 0
                assert "Unable to fetch stations from all data-centers" in str(qd)

        # Same test as above, but test negative assertions with "!". Reminder: data on
        # db is:
        # ----------------------------------------------
        # channel_id network station location channel start_time    end_time   sample_rate  datacenter_id     # noqa
        #          1      HT     AGG              HLE    2008-02-12 NaT              100.0              2     # noqa
        #          2      HT     AGG              HLZ    2008-02-12 NaT              100.0              2     # noqa
        #          3      HT     LKD2             HHE    2009-01-01 NaT              90.0               2     # noqa
        #          4      HT     LKD2             HHZ    2009-01-01 NaT              90.0               2     # noqa
        #          5      BLA    BLA              HHZ    2009-01-01 2019-01-01       100.0              2     # noqa
        #          6      BLA    BLA              HHZ    2018-01-01 NaT              100.0              2     # noqa
        #          7      A      B         10     HBE    2003-02-12 2010-02-12       100.0              2     # noqa
        #          8      E      F         11     HHZ    2019-01-01 NaT              100.0              2     # noqa
        # ----------------------------------------------
        # Now according to the table above set a list of arguments:
        # Each key is: the argument, each value IS A LIST OF BOOLEAN MAPPED TO EACH ROW
        # OF THE DATAFRAME ABOVE, telling if the row matches according to the argument:
        nets = {('!*A*', 'A'): [1, 1, 1, 1, 0, 0, 1, 1],
                ('E', 'A'): [0, 0, 0, 0, 0, 0, 1, 1]
                }
        stas = {('!*B*', 'B'): [1, 1, 1, 1, 0, 0, 1, 1],
                ('!???2',): [1, 1, 0, 0, 1, 1, 1, 1]}
        # note that we do NOT assume '--' can be given, as this should be the parsed
        # output of `nslc_lists`:
        locs = {('',): [1, 1, 1, 1, 1, 1, 0, 0],
                ('!',): [0, 0, 0, 0, 0, 0, 1, 1]}
        chans = {('HHZ', '!*E'): [0, 1, 0, 1, 1, 1, 0, 1],
                 ('!?H?',): [1, 1, 0, 0, 0, 0, 1, 0]}
        stimes = {None: [1, 1, 1, 1, 1, 1, 1, 1]}
        etimes = {None: [1, 1, 1, 1, 1, 1, 1, 1]}
        minsr = {-1: [1, 1, 1, 1, 1, 1, 1, 1]}
        # no url read: set socket.tiomeout as urlread side effect. This will force
        # querying the database to test that the filtering works as expected:
        for n, s, l, c, st, e, m in product(nets, stas, locs, chans, stimes, etimes,
                                            minsr):
            matches = np.array(nets[n]) * np.array(stas[s]) * np.array(locs[l]) * \
                np.array(chans[c]) * np.array(stimes[st]) * np.array(etimes[e]) * \
                      np.array(minsr[m])
            expected_length = matches.sum()
            # Now: if expected length is zero, it means we do not have data matches on t
            # he db. This raises a quitdownload (avoiding pytest.raises cause in this
            # case it's easier like done below):
            try:
                __dc_df = datacenters_df.loc[datacenters_df[DataCenter.id.key] == 2]
                cha_df = self.get_channels_df(socket.timeout(), db.session, __dc_df,
                                              eidavalidator, n, s, l, c, st, e, m,
                                              False, None, None, -1, self.db_buf_size)
                assert len(cha_df) == expected_length
            except FailedDownload as qd:
                assert expected_length == 0
                assert "Unable to fetch stations from all data-centers" in str(qd)

        # now make the second url_side_effect raise => force query from db, and the
        # first good => fetch from the web
        # We want to test the mixed case: some fetched from db, some from the web
        # ---------------------------------------------------
        # first we query the db to check what we have:
        cha_df = dbquery2df(db.session.query(Channel.id, Station.datacenter_id,
                                             Station.network).join(Station))
        # build a new network:
        newnetwork = 'U'
        while newnetwork in cha_df[Station.network.key]:
            newnetwork += 'U'
        urlread_sideeffect2  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
%s|W||HBE|39.0211|22.336|622.0|0.0|0.0|-90.0|GFZ:HT1980:CMG-3ESP/90/g=2000|838860800.0|0.1|M/S|50.0|2008-02-12T00:00:00|2010-02-12T00:00:00
""" % newnetwork,  socket.timeout()]  # noqa
        # now note: the first url read raised, now it does not: write the channel
        # above with network = newnetwork (surely non existing to the db). The second
        # url read did not raise, now it does (socket.timeout): fetch from the db we
        # issue a ['???'] as 'channel' argument in order to fetch everything from the db
        # (we would have got the same by passing None as 'channel' argument). The three
        # [] before ['???'] are net, sta, loc and mean: no filter on those params
        cha_df_ = self.get_channels_df(urlread_sideeffect2, db.session,
                                       datacenters_df,
                                       [], [], [], ['???'], None, None, 10,
                                       False, None, None, -1, self.db_buf_size)

        # we should have the channel with network 'U' to the first datacenter
        dcid = datacenters_df.iloc[0][DataCenter.id.key]
        assert len(cha_df_[cha_df_[Station.datacenter_id.key] == dcid]) == 1
        assert cha_df_[cha_df_[Station.datacenter_id.key]
                       == dcid][Station.network.key][0] == newnetwork
        # but we did not query other channels for datacenter id = dcid, as the web
        # response was successful, we rely on that. Conversely, for the other
        # datacenter we should have all channels fetched from db
        dcid = datacenters_df.iloc[1][DataCenter.id.key]
        chaids_of_dcid = \
            cha_df_[cha_df_[Station.datacenter_id.key] == dcid][Channel.id.key].tolist()
        db_chaids_of_dcid = \
            cha_df[cha_df[Station.datacenter_id.key] == dcid][Channel.id.key].tolist()
        assert chaids_of_dcid == db_chaids_of_dcid

    def test_get_channels_df_eidavalidator_station_and_channel_duplicates(self, db):
        urlread_sideeffect = """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""  # noqa
        events_df = self.get_events_df(urlread_sideeffect, db.session)

        # urlread for datacenters: will be called only if we have eida (the case here)
        urlread_sideeffect = """http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query
A1 * * * 2002-09-01T00:00:00 2015-10-20T00:00:00
A2 a2 * * 2013-08-01T00:00:00 2017-04-25
XX xx * * 2013-08-01T00:00:00 2017-04-25
YY yy * HH? 2013-08-01T00:00:00 2017-04-25

http://ws.resif.fr/fdsnws/dataselect/1/query
B1 * * HH? 2002-09-01T00:00:00 2005-10-20T00:00:00
XX xx * * 2013-08-01T00:00:00 2017-04-25
YY yy * DE? 2013-08-01T00:00:00 2017-04-25
"""
        net, sta, loc, cha = [], [], [], []
        datacenters_df, eidavalidator = \
            self.get_datacenters_df(urlread_sideeffect, db.session, self.service,
                                    self.routing_service,
                                    net, sta, loc, cha, db_bufsize=self.db_buf_size)

        # MOCK THE CASE OF DUPLICATED STATIONS (I.E., RETURNED BY MORE THAN ONE
        # DATACENTER). Look at the Sensor description (starting with "OK: " or "NO: "
        # to know if the given channel should be taken or not
        # EXCEPT the first two channels (IV.BOTM)
        # because they are real cases of conflicts that should NOT be saved
        urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
IV|BOTM||HNZ|45.5416|10.3213|157|0|0|-90|KINEMETRICS EPISENSOR-FBA-ES-T-CL-2G-FS-40-VPP|320770|0.2|M/S**2|100|2014-11-14T14:00:00|2014-11-14T14:00:00
IV|BOTM||HNZ|45.5416|10.3213|157|0|0|-90|KINEMETRICS EPISENSOR-FBA-ES-T-CL-2G-FS-40-VPP|320770|0.2|M/S**2|100|2014-11-14T14:00:00|2019-08-30T15:01:00
A1|aa||DEL|3|4|6|0|0|0|OK:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|ww||NNL|3|4|6|0|0|0|OK:                                                |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|xx||DNL|3|4|6|0|0|0|NO: station also in other dc, eida_rs says not found      |8|0.1|M/S|50.0|2008-02-12T00:00:00|
XX|xx||DEL|3|4|6|0|0|0|NO: station also in other dc, eida_rs says should be in both (conflict)        |8|0.1|M/S|50.0|2008-02-12T00:00:00|2020-02-12
YY|yy||DEL|3|4|6|0|0|0|NO: station also in other dc, eida_rs says this dc is wrong  |8|0.1|M/S|50.0|2008-02-12T00:00:00|
""",  # noqa
"""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
A1|aa||DNZ|3|4|6|0|0|0|NO: station also in other dc, eida_rs says this dc is wrong |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A1|aa||NNZ|3|4|6|0|0|0|OK: station also in other dc, but starttime changed -> new station               |8|0.1|M/S|50.0|2018-02-12T00:00:00|
B1|bb||NEZ|3|4|6|0|0|0|OK:                                                                 |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|xx||DNL|3|4|6|0|0|0|NO: station also in other dc, eida_rs says not found       |8|0.1|M/S|50.0|2008-02-12T00:00:00|
A2|a2||NNL|3|4|6|0|0|0|OK: note that eida_rs says this dc is wrong but we ignore eida rs because we dont have conflicts                                               |8|0.1|M/S|50.0|2008-02-12T00:00:00|
XX|xx||DEL|3|4|6|0|0|0|NO: station also in other dc, eida_rs says should be in both (conflict)       |8|0.1|M/S|50.0|2008-02-12T00:00:00|2020-02-12
YY|yy||DEL|3|4|6|0|0|0|NO: station also in other dc, eida_rs says this dc is ok but station cannot be saved (some channels in one dc, some not) |8|0.1|M/S|50.0|2008-02-12T00:00:00|
B1|bb||NEZ|3|4|6|0|0|0|OK:                                                                 |8|0.1|M/S|50.0|2008-02-12T00:00:00|
"""]  # NOTE: the last line (B1|bb...) is ON PURPOSE THE SAME AS THE THIRD, IT SHOULD BE IGNORED   # noqa

        EXPECTED_SAVED_CHANNELS = 5

        # get channels with the above implemented urlread_sideeffect:
        cha_df = self.get_channels_df(urlread_sideeffect, db.session,
                                      datacenters_df, eidavalidator,
                                      net, sta, loc, cha, None, None, 10,
                                      False, None, None, -1, self.db_buf_size)
        # if u want to check what has been taken, issue in the debugger:
        # str(dbquery2df(db.session.query(Channel.id, Station.network, Station.station,
        # Channel.channel, Channel.station_id, Station.datacenter_id).join(Station)))
        csd = dbquery2df(db.session.query(Channel.sensor_description))
        assert len(csd) == EXPECTED_SAVED_CHANNELS
        logmsg = self.log_msg()
        assert ("4 station(s) and 7 channel(s) not saved to db (wrong datacenter"
                " detected using either Routing services or already saved "
                "stations)") in logmsg
        assert "2 channel(s) not saved to db (conflicting data" \
            in logmsg
        assert "BOTM" in logmsg[logmsg.index("2 channel(s) not saved to db"):]
        # assert that the OK string is in the sensor description
        assert all(c[:3] == "OK:" for c in csd[Channel.sensor_description.key])

        # what happens if we need to query the db? We should get the same isn't it?
        # then set eidavalidator = None
        cha_df2 = self.get_channels_df(urlread_sideeffect, db.session,
                                       datacenters_df,
                                       None,
                                       net, sta, loc, cha, None, None, 10,
                                       False, None, None, -1, self.db_buf_size)

        # assert that we get the same result as when eidavalidator is None:
        assert cha_df2.equals(cha_df)

        # now test when the response is different
        urlread_sideeffect[0], urlread_sideeffect[1] = \
            urlread_sideeffect[1], urlread_sideeffect[0]
        # get channels with the above implemented urlread_sideeffect:
        cha_df3 = self.get_channels_df(urlread_sideeffect, db.session,
                                       datacenters_df,
                                       eidavalidator,
                                       net, sta, loc, cha, None, None, 10,
                                       False, None, None, -1, self.db_buf_size)
        # we tested visually that everything is ok by issuing a
        # str(dbquery2df(db.session.query(Channel.id, Station.network, Station.station,
        # Channel.channel, Channel.station_id, Station.datacenter_id).join(Station)))
        # we might add some more specific assert here
        assert len(cha_df3) == EXPECTED_SAVED_CHANNELS

        # get channels with the above implemented urlread_sideeffect:
        cha_df4 = self.get_channels_df(urlread_sideeffect, db.session,
                                       datacenters_df,
                                       None,
                                       net, sta, loc, cha, None, None, 10,
                                       False, None, None, -1, self.db_buf_size)

        assert not cha_df3.equals(cha_df4)  # WHY?? cause they are not sorted, we
        # might have updated some portion of dataframe and inserted the rest on
        # dbmanager.close, or the other way around
        # sort them by id, reset dataframes indices... and they should be the same:
        assert cha_df3.sort_values(by=['id']).\
            reset_index(drop=True).equals(cha_df4.sort_values(by=['id']).
                                          reset_index(drop=True))
