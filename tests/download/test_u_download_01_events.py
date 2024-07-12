# -*- coding: utf-8 -*-
"""
Created on Feb 4, 2016

@author: riccardo
"""
from datetime import datetime, timedelta
import socket
from itertools import cycle
import logging
import shutil
from logging import StreamHandler
from io import BytesIO, StringIO
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from stream2segment.download.db.models import Event, Download, WebService
from stream2segment.download.modules.events import (get_events_df,
                                                    _get_freq_mag_distrib,
                                                    islocalfile as o_islocalfile,
                                                    ERR_FETCH_FDSN, ERR_READ_FDSN,
                                                    ERR_FETCH, ERR_FETCH_NODATA)
from stream2segment.download.modules.utils import get_dataframe_from_fdsn, urljoin
from stream2segment.download.exc import FailedDownload, NothingToDownload
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
        self._evt_urlread_sideeffect = """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID | MagType | Magnitude | MagAuthor | EventLocationName
20160508_0000129|2016-05-08 05:17:11.500000|1|1|60.0|AZER|EMSC-RTS|AZER|505483|ml|3|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|90|90|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|4|EMSC|CROATIA
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
        self.loghandler = StreamHandler(stream=StringIO())

        # THIS IS A HACK:
        query_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset
        # (to level.INFO) otherwise it stays what we set two lines above. Problems might arise
        # if closing sets a different level, but for the moment who cares
        query_logger.addHandler(self.loghandler)

        # define class level patchers (we do not use a yiled as we need to do more stuff in the
        # finalizer, see below
        patchers = [
            patch('stream2segment.download.url.urlopen'),
            patch('stream2segment.download.url.ThreadPool')
        ]

        self.mock_urlopen = patchers[-2].start()

        # mock ThreadPool (tp) to run one instance at a time, so we get deterministic results:
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
            for h in hndls:
                if h is self.loghandler:
                    self.loghandler.close()
                    query_logger.removeHandler(h)
        request.addfinalizer(delete)

    def log_msg(self):
        ret = self.loghandler.stream.getvalue()
        self.loghandler.stream.seek(0)
        self.loghandler.stream.truncate(0)
        return ret

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
            urlread_side_effect = [urlread_side_effect]

        for k in urlread_side_effect:
            mymock = Mock()
            if type(k) == int:
                mymock.read.side_effect = HTTPError('url', int(k),  responses[k], None, None)
            elif type(k) in (bytes, str):
                def func(k):
                    bio = BytesIO(k.encode('utf8') if type(k) == str else k)

                    def rse(*mymock, **v):
                        rewind = not mymock and not v
                        if not rewind:
                            currpos = bio.tell()
                        ret = bio.read(*mymock, **v)
                        # hacky workaround to support cycle below: if reached the end,
                        # go back to start
                        if not rewind:
                            cp = bio.tell()
                            rewind = cp == currpos
                        if rewind:
                            bio.seek(0, 0)
                        return ret
                    return rse
                mymock.read.side_effect = func(k)
                mymock.code = 200
                mymock.msg = responses[mymock.code]
            else:
                mymock.read.side_effect = k
            ret = MagicMock()
            ret.__enter__.return_value = mymock
            retvals.append(ret)

        self.mock_urlopen.side_effect = cycle(retvals)

    def get_events_df(self, url_read_side_effect, session, url, evt_query_args, start, end,
                      db_bufsize=30, timeout=15,
                      show_progress=False):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return get_events_df(session, url, evt_query_args, start, end,
                             db_bufsize, timeout,
                             show_progress)

    @patch('stream2segment.download.modules.events.urljoin', side_effect=urljoin)
    def test_get_events(self, mock_urljoin, db):
        urlread_sideeffect = ["""#1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
"""]
        data = self.get_events_df(urlread_sideeffect, db.session, "http://eventws", {},
                                  datetime.utcnow() - timedelta(seconds=1), datetime.utcnow(),
                                  db_bufsize=self.db_buf_size)
        # assert only first two events events were successfully saved
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == \
            len(data) == 3
        # check that log has notified:
        log1 = self.log_msg()
        assert "20160508_0000113" in log1
        assert "Adding missing nullable column(s) in data: event_type" in log1
        assert "1 database row(s) not inserted" in log1
        assert "1 row(s) discarded (malformed text data)" in log1
        assert mock_urljoin.call_count == 1
        mock_urljoin.reset_mock()

        # now download again, with an url error:
        urlread_sideeffect = [504, """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", URLError('blabla23___')]
        with pytest.raises(FailedDownload) as fld:
            data = self.get_events_df(urlread_sideeffect, db.session, "http://eventws", {},
                                      datetime.utcnow() - timedelta(seconds=1), datetime.utcnow(),
                                      db_bufsize=self.db_buf_size)
        # assert the error message is of type:
        # "Unable to fetch events (urlopeneror"...
        # (check only for the prefix present
        assert str(fld.value).startswith(ERR_FETCH + " (")

        # assert we got the same result as above:
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == \
            len(data) == 3
        log2 = self.log_msg()

        # log text has the message about the second (successful) download, with the
        # two rows discarded: LEGACY CODE, not true anymore: the parsing happens after fetching data,
        # which raised, so no parsing and thus no discarded rows:
        # assert "2 row(s) discarded" in log2

        # We called urljoin once, plus two times because we split url time bounds once,
        # plus one time at the end to provide the url which raised in the error message:
        assert mock_urljoin.call_count == 4
        mock_urljoin.reset_mock()


        # now download again, with a recursion error (max iterations reached):
        urlread_sideeffect = [413]
        with pytest.raises(FailedDownload) as fld:
            data = self.get_events_df(urlread_sideeffect, db.session, "http://eventws", {},
                                      datetime.utcnow() - timedelta(seconds=1), datetime.utcnow(),
                                      db_bufsize=self.db_buf_size)
        # assert we got the same result as above:
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == \
            len(data) == 3
        log2 = self.log_msg()

        # nothing written to log:
        assert "Request seems to be too large" in log2
        # assertion on exception:
        assert "Unable to fetch events" in str(fld)
        assert "maximum recursion depth reached" in str(fld)

    def test_get_events_eventws_not_saved(self, db):
        '''test request splitted, but failing due to a http error'''
        urlread_sideeffect = [socket.timeout, 500]

        # we want to return all times 413, and see that we raise a ValueError:
        with pytest.raises(FailedDownload) as fldl:
            # now it should raise because of a 413:
            data = self.get_events_df(urlread_sideeffect, db.session, "abcd", {},
                                      start=datetime(2010, 1, 1),
                                      end=datetime(2011, 1, 1),
                                      db_bufsize=self.db_buf_size)
        # test that we raised the proper message:
        assert 'Unable to fetch events' in str(fldl)
        # assert we wrote the url
        assert len(db.session.query(WebService.url).filter(WebService.url == 'abcd').all()) == 1
        # assert only three events were successfully saved to db (two have same id)
        assert not db.session.query(Event).all()
        # we cannot assert anything has been written to logger cause the exception are caucht
        # if we raun from main. This should be checked in functional tests where we test the whole
        # chain
        # assert "request entity too large" in self.log_msg()

    def get_pbar_total_steps(self):
        return _get_freq_mag_distrib({})[2].sum()

    @patch('stream2segment.download.modules.events.get_progressbar')
    @patch('stream2segment.download.modules.events.urljoin', side_effect=urljoin)
    def test_pbar1(self, mock_urljoin, mock_pbar, db):
        '''test request splitted, but failing due to a http error'''

        class Pbar:

            def __init__(self, *a, **kw):
                self.updates = []

            def __enter__(self, *a, **kw):
                return self

            def __exit__(self, *a, **kw):
                pass

            def update(self, increment):
                self.updates.append(increment)

        mock_pbar.return_value = Pbar()

        urlread_sideeffect = [socket.timeout, 500]
        mock_pbar.reset_mock()
        mock_pbar.return_value.updates = []
        with pytest.raises(FailedDownload) as fldl:
            # now it should raise because of a 413:
            _ = self.get_events_df(urlread_sideeffect, db.session, "abcd", {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)
        # test that we did not increment the pbar (exceptions)
        assert mock_pbar.call_args[1]['length'] == self.get_pbar_total_steps()
        assert mock_pbar.return_value.updates == []

        # Now let's supply a bad response response, the
        # progressabr should not be called
        urlread_sideeffect = ['']
        mock_pbar.reset_mock()
        mock_pbar.return_value.updates = []
        with pytest.raises(NothingToDownload) as fldl:
            # returning empty data
            _ = self.get_events_df(urlread_sideeffect, db.session, "abcd", {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)
        # test that we did not increment the pbar (exceptions)
        expected_str = 'No event received, search parameters might be too strict'
        assert str(fldl.value).startswith(expected_str)
        # the string will be in the log during a download routine, but not here because
        # we do not call get_events_df from the parent main function. So:
        assert expected_str not in self.log_msg()

        # we do not notify empty responses anymore, we simply pass:
        # assert "Discarding response (Empty input data)" in self.log_msg()
        assert not mock_pbar.called

        # Now let's supply a successful response:
        urlread_sideeffect = ['''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''']
        mock_pbar.reset_mock()
        mock_pbar.return_value.updates = []
        _ = self.get_events_df(urlread_sideeffect, db.session, "abcd", {},
                               start=datetime(2010, 1, 1),
                               end=datetime(2011, 1, 1),
                               db_bufsize=self.db_buf_size)
        assert not mock_pbar.called
        assert "Discarding response (Empty input data)" not in self.log_msg()
        assert "Request seems to be too large, splitting into" not in self.log_msg()

        # Now let's supply a successful response:
        urlread_sideeffect = [413,
                              '''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''',
                              413,
                              '''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''',
                              '''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''',
                              ]
        mock_pbar.reset_mock()
        mock_pbar.return_value.updates = []
        data = self.get_events_df(urlread_sideeffect, db.session, "abcd", {},
                                  start=datetime(2010, 1, 1),
                                  end=datetime(2011, 1, 1),
                                  db_bufsize=self.db_buf_size)
        logmsg = self.log_msg()
        assert 'Duplicated instances' in logmsg
        # test that we did not increment the pbar (exceptions)
        assert mock_pbar.call_args[1]['length'] == self.get_pbar_total_steps()
        # the first 413 produces a magnitude split 1part vs 9parts,
        # the second 413 produces a split 1vs9 on the 9 parts, thus 1*9 and 9*9:
        assert sum(mock_pbar.return_value.updates) == mock_pbar.call_args[1]['length']
        assert "Request seems to be too large, splitting into" in logmsg

        # =================================================================
        # The test below check th same for different magnitude bound values
        # =================================================================
        mock_pbar.reset_mock()
        mock_urljoin.reset_mock()
        mock_pbar.return_value.updates = []
        data = self.get_events_df(urlread_sideeffect, db.session, "abcd", {'minmag': 2},
                                  start=datetime(2010, 1, 1),
                                  end=datetime(2011, 1, 1),
                                  db_bufsize=self.db_buf_size)
        assert 'Duplicated instances' in self.log_msg()
        # test that we did not increment the pbar (exceptions)
        assert mock_pbar.call_args[1]['length'] < self.get_pbar_total_steps()
        assert sum(mock_pbar.return_value.updates) == mock_pbar.call_args[1]['length']
        # assert that we do not have maxmagnitude in the first request,
        # but in the first sub-request (index 1) (do not test other sub requests)
        req_kwargs = [_[1] for _ in mock_urljoin.call_args_list]
        assert not any(['maxmagnitude' in req_kwargs[0],
                        'maxmag' in req_kwargs[0]])
        assert any(['maxmagnitude' in req_kwargs[1],
                    'maxmag' in req_kwargs[1]])


        mock_pbar.reset_mock()
        mock_urljoin.reset_mock()
        mock_pbar.return_value.updates = []
        data = self.get_events_df(urlread_sideeffect, db.session, "abcd", {'maxmag': 5},
                                  start=datetime(2010, 1, 1),
                                  end=datetime(2011, 1, 1),
                                  db_bufsize=self.db_buf_size)
        assert 'Duplicated instances' in self.log_msg()
        # test that we did not increment the pbar (exceptions)
        assert mock_pbar.call_args[1]['length'] < self.get_pbar_total_steps()
        assert sum(mock_pbar.return_value.updates) == mock_pbar.call_args[1]['length']
        # assert that we do not have minmagnitude in the first two sub-request (from index 1),
        # but in the third (do not test other sub requests)
        req_kwargs = [_[1] for _ in mock_urljoin.call_args_list]
        assert not any(['minmagnitude' in req_kwargs[0],
                        'minmag' in req_kwargs[0]])
        assert not any(['minmagnitude' in req_kwargs[1],
                        'minmag' in req_kwargs[1]])
        assert any(['minmagnitude' in req_kwargs[2],
                    'minmag' in req_kwargs[2]])


    @pytest.mark.parametrize('args', [{'minmag': 2.1}, {'minmag': 2.11},
                                       {'minmag': 0, 'maxmag': 1.9},
                                       {'minmag': 2, 'maxmag': 8}])
    @patch('stream2segment.download.modules.events.get_progressbar')
    def test_pbar2(self, mock_pbar, args, db):
        '''test request splitted, but failing due to a http error'''

        urlread_sideeffect = [413,
                              '''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''',
                              413,
                              '''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''',
                              '''20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN''',
                              ]

        class Pbar:

            def __init__(self, *a, **kw):
                self.updates = []

            def __enter__(self, *a, **kw):
                return self

            def __exit__(self, *a, **kw):
                pass

            def update(self, increment):
                self.updates.append(increment)

        mock_pbar.return_value = Pbar()
        data = self.get_events_df(urlread_sideeffect, db.session, "abcd", args,
                                  start=datetime(2010, 1, 1),
                                  end=datetime(2011, 1, 1),
                                  db_bufsize=self.db_buf_size)
        assert 'Duplicated instances' in self.log_msg()
        # test that we did not increment the pbar (exceptions)
        assert mock_pbar.call_args[1]['length'] < self.get_pbar_total_steps()
        assert sum(mock_pbar.return_value.updates) == mock_pbar.call_args[1]['length']

    def test_get_events_eventws_from_file(self,
                                          # fixtures:
                                          db, pytestdir):
        '''test request splitted, but reading from events file'''
        urlread_sideeffect = [socket.timeout, 500]

        filepath = pytestdir.newfile('.txt', create=True)
        with open(filepath, 'w') as _fpn:
            _fpn.write("""1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""")
        log1 = self.log_msg()

        data = self.get_events_df(urlread_sideeffect, db.session, filepath, {},
                                  start=datetime(2010, 1, 1),
                                  end=datetime(2011, 1, 1),
                                  db_bufsize=self.db_buf_size)
        # assert we got the same result as above:
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == \
            len(data) == 3
        log2 = self.log_msg()
        # since one row is discarded, the message is something like:
        # 1 row(s) discarded (malformed server response data, e.g. NaN's). url: file:////private/var/folders/l9/zpp7wn1n4r7bt4vs39gylk4w0000gn/T/pytest-of-riccardo/pytest-442/test_get_events_eventws_from_f0/368e6e99-171c-40e1-ad8e-3afc40ebeeab.txt
        # however, we test the bare minimum:
        assert 'url: file:///' in log2
        assert not self.mock_urlopen.called

    def test_get_events_errors(self,
                               # fixtures:
                               db, pytestdir):
        '''test request splitted, but reading from BAD events file'''
        urlread_sideeffect = [socket.timeout, 500]

        # create a invalid FDSN file (with a column length mismatch):
        filepath = pytestdir.newfile('.txt', create=True)
        with open(filepath, 'w') as _fpn:
            _fpn.write("""000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""")

        # provide a valid file (that exists but is malformed) and tst the err message:
        expected_err_msg = ERR_READ_FDSN

        with pytest.raises(FailedDownload) as fdl:
            _ = self.get_events_df(urlread_sideeffect, db.session, filepath, {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)

        assert expected_err_msg in str(fdl.value)
        assert not self.mock_urlopen.called
        
        # Now provide a url and test the error message. Test that we get a 500 error
        # (Note that `urlread_sideeffect` above should raise a socket timeout and
        # an THHP error 500, but the first socket.timeout forces by design the split of
        # the request into sub-request, so the first exception caught is the HTTP error 500
        with pytest.raises(FailedDownload) as fdl:
            _ = self.get_events_df(urlread_sideeffect, db.session, 'iris', {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)

        # the string above might change across python versions: Thus:
        # assert expected_err_msg in str(fdl)
        # might fail and thus cause annoying debugs.
        # We then test that our message is there and a '500' is found in the error:
        assert (ERR_FETCH + ' ') in str(fdl.value)
        assert '500' in str(fdl.value)
        assert self.mock_urlopen.called  # urlopen HAS been called!
        self.mock_urlopen.reset_mock()

        with pytest.raises(FailedDownload) as fdl:
            _ = self.get_events_df([413, '', ''], db.session, 'iris', {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)

        # the string above might change across python versions: Thus:
        # assert expected_err_msg in str(fdl)
        # might fail and thus cause annoying debugs.
        # We then test that our message is there and a '500' is found in the error:
        assert (ERR_FETCH + ' ') in str(fdl.value)
        assert self.mock_urlopen.called  # urlopen HAS been called!
        self.mock_urlopen.reset_mock()

        # NO_DATA_ERR = "No event received, search parameters might be too strict. "
        # INVALID_DATA_ERR = ", ".join([E_FETCH_EVTS, E_PARSE_FDSN])
        # Now mock empty or invalid data:
        for url_read_side_effect in [b'', b'!invalid!']:

            expected_exc = NothingToDownload if not url_read_side_effect else FailedDownload
            expected_err = ERR_FETCH_NODATA if not url_read_side_effect else ERR_FETCH_FDSN

            # Now provide a FDSN url:
            with pytest.raises(expected_exc) as fdl:
                _ = self.get_events_df([url_read_side_effect], db.session, 'iris', {},
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)

            assert str(fdl.value).startswith(expected_err)
            assert self.mock_urlopen.called
            self.mock_urlopen.reset_mock()

            # Now provide a custom url (don't know if FDSN):
            with pytest.raises(expected_exc) as fdl:
                _ = self.get_events_df([url_read_side_effect], db.session,
                                       'http://custom_service', {},
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)

            assert str(fdl.value).startswith(expected_err)

            # assert ('No event found, try to change your search parameters. Check '
            #         'also that the service returns parsable data (FDSN-compliant)') in str(fdl)
            assert self.mock_urlopen.called
            self.mock_urlopen.reset_mock()
            
            # Now provide a custom "string" (url? file? if url, don't know if FDSN):
            with pytest.raises(expected_exc) as fdl:
                _ = self.get_events_df([url_read_side_effect], db.session, 'filepath', {},
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)

            assert str(fdl.value).startswith(expected_err)

            # assert ('No event found. If you supplied a file, the file was not found: '
            #         'check path and typos. Otherwise, try to change your search parameters: '
            #         'check also that the service returns parsable data (FDSN-compliant)') \
            #     in str(fdl.value)
            assert self.mock_urlopen.called
            self.mock_urlopen.reset_mock()

    def test_get_events_eventws_from_isc(self,
                                         # fixtures:
                                         db, data):
        '''test bad events from isc'''

        # normal query from emsc, data expected as FDSN, returned as FDSN
        _ = self.get_events_df(None, db.session, 'emsc', {},
                               start=datetime(2010, 1, 1),
                               end=datetime(2011, 1, 1),
                               db_bufsize=self.db_buf_size)
        assert db.session.query(Event.id).count() == 2

        _ = self.get_events_df(None, db.session, 'isc', {},
                                start=datetime(2010, 1, 1),
                                end=datetime(2011, 1, 1),
                                db_bufsize=self.db_buf_size)
        assert db.session.query(Event.id).count() == 4

    @patch('stream2segment.download.modules.events.urljoin', side_effect=urljoin)
    def test_get_events_response_has_one_col_more(self, mock_urljoin, db):
        """WARNING: THIS TEST MIGHT FAIL IN THE FUTURE IF NEW COLUMNS ARE ADDED TO OUR
        Event MODEL. TO FIX THIS, EDIT THE RESPONSE BELOW IN ORDER TO HAVE ALWAYS ONE
        COLUMN MORE THAN IN OUR Event MODEL
         """
        urlread_sideeffect = ["""#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType||
gfz2021edty|2021-02-28T23:37:15.211956|-17.565212|167.572067|10.0|||GFZ|gfz2021edty|M|5.787024361||Vanuatu Islands|earthquake||
gfz2021edpn|2021-02-28T21:23:50.840903|-22.500320|172.554474|26.75543594|||GFZ|gfz2021edpn|mb|4.907085435||Southeast of Loyalty Islands|earthquake||
gfz2021edoa|2021-02-28T20:37:40.931643|-22.658522|172.432373|30.70357132|||GFZ|gfz2021edoa|Mw|5.755797284||Southeast of Loyalty Islands|earthquake||
"""]
        with pytest.raises(FailedDownload) as fdw:
            data = self.get_events_df(urlread_sideeffect, db.session, "http://eventws", {},
                                      datetime.utcnow() - timedelta(seconds=1), datetime.utcnow(),
                                      db_bufsize=self.db_buf_size)
        assert str(fdw.value).startswith(ERR_FETCH_FDSN)
        assert "column(s)" in str(fdw.value)  # test it's a columns problem
        log1 = self.log_msg()
        # the message should be in the log in a normal routine, but here it should NOT
        # be because we do not run get_events_df from the parent main function, where
        # the faileddownload is logged:
        assert ERR_FETCH_FDSN not in log1
