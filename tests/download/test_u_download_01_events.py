# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from builtins import str
from datetime import datetime, timedelta
import socket
from itertools import cycle
import logging
import shutil
from logging import StreamHandler
from io import BytesIO
# import threading
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

from stream2segment.download.db.models import Event, Download, WebService
from stream2segment.download.modules.events import get_events_df, isf2text_iter,\
    _get_freq_mag_distrib, islocalfile as o_islocalfile
from stream2segment.download.modules.utils import response2normalizeddf, urljoin
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import URLError, HTTPError, responses
from stream2segment.resources import get_templates_fpath
from stream2segment.io import yaml_load

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
            # it's not an iterable (wheere str/bytes/unicode are considered NOT iterable
            # in both py2 and 3)
            urlread_side_effect = [urlread_side_effect]

        for k in urlread_side_effect:
            mymock = Mock()
            if type(k) == int:
                mymock.read.side_effect = HTTPError('url', int(k),  responses[k], None, None)
            elif type(k) in (bytes, str):
                def func(k):
                    bio = BytesIO(k.encode('utf8') if type(k) == str else k)  # py2to3 compatible

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
            retvals.append(mymock)

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
        assert "1 database row(s) not inserted" in log1
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
        # assert we got the same result as above:
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == \
            len(data) == 3
        log2 = self.log_msg()

        # log text has the message about the second (successful) dwnload, with the
        # two rows discarded:
        assert "2 row(s) discarded" in log2
        # test that the exception has expected mesage:
        assert "Unable to fetch events" in str(fld)
        # check that we splitted once, thus we called 2 times mock_urljoin
        # (plus the first call):
        assert mock_urljoin.call_count == 3
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

        class Pbar(object):

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
        with pytest.raises(FailedDownload) as fldl:
            # now it should raise because of a 413:
            _ = self.get_events_df(urlread_sideeffect, db.session, "abcd", {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)
        # test that we did not increment the pbar (exceptions)
        assert "Discarding response (Empty input data)" in self.log_msg()
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

        class Pbar(object):

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

        filepath = pytestdir.newfile('.txt', create=True)
        with open(filepath, 'w') as _fpn:
            _fpn.write("""000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""")

        # provide a valid file (that exists but is malformed) and tst the err message:
        expected_err_msg = ('No event found. Check that the file is non empty '
                            'and its content is valid')
        with pytest.raises(FailedDownload) as fdl:
            _ = self.get_events_df(urlread_sideeffect, db.session, filepath, {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)

        assert expected_err_msg in str(fdl)
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

        expected_err_msg = 'Unable to fetch events (HTTP Error 500: Internal Server Error)'
        # the string above might change across python versions: Thus:
        # assert expected_err_msg in str(fdl)
        # might fail and thus cause annoying debugs.
        # We then test that our message is there and a '500' is found in the error:
        assert 'Unable to fetch events' in str(fdl)
        assert '500' in str(fdl)
        assert self.mock_urlopen.called  # urlopen HAS been called!
        self.mock_urlopen.reset_mock()

        # Now mock empty or invalid data:
        for url_read_side_effect in [b'', b'!invalid!']:
            # Now provide a FDSN url:
            with pytest.raises(FailedDownload) as fdl:
                _ = self.get_events_df([url_read_side_effect], db.session, 'iris', {},
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)

            assert 'No event found, try to change your search parameters' in str(fdl)
            assert self.mock_urlopen.called
            self.mock_urlopen.reset_mock()

            # Now provide a custom url (don't know if FDSN):
            with pytest.raises(FailedDownload) as fdl:
                _ = self.get_events_df([url_read_side_effect], db.session,
                                       'http://custom_service', {},
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)

            assert ('No event found, try to change your search parameters. Check '
                    'also that the service returns parsable data (FDSN-compliant)') in str(fdl)
            assert self.mock_urlopen.called
            self.mock_urlopen.reset_mock()
            
            # Now provide a custom "string" (url? file? if url, don't know if FDSN):
            with pytest.raises(FailedDownload) as fdl:
                _ = self.get_events_df([url_read_side_effect], db.session, 'filepath', {},
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)
        
            assert ('No event found. If you supplied a file, the file was not found: '
                    'check path and typos. Otherwise, try to change your search parameters: '
                    'check also that the service returns parsable data (FDSN-compliant)') \
                in str(fdl.value)
            assert self.mock_urlopen.called
            self.mock_urlopen.reset_mock()

    @patch('stream2segment.download.modules.events.isf2text_iter', side_effect=isf2text_iter)
    def test_get_events_eventws_from_isc(self, mock_isf_to_text,
                                         # fixtures:
                                         db, data):
        '''test request splitted, but reading from BAD events file'''

        # now it should raise because of a 413:
        _ = self.get_events_df(None, db.session, 'emsc', {},
                               start=datetime(2010, 1, 1),
                               end=datetime(2011, 1, 1),
                               db_bufsize=self.db_buf_size)
        assert not mock_isf_to_text.called
        assert db.session.query(Event.id).count() == 2

        with pytest.raises(FailedDownload) as fld:
            # now it should raise because of a 413:
            _ = self.get_events_df(None, db.session, 'isc', {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)
        assert "No event found, try to change your search parameters" in str(fld.value)
        assert mock_isf_to_text.called
        mock_isf_to_text.reset_mock()
        assert not mock_isf_to_text.called

        # now supply a valid isf file:
        _ = self.get_events_df([data.read('event_request_sample_isc.isf').decode('utf8')],
                               db.session, 'isc', {},
                               start=datetime(2010, 1, 1),
                               end=datetime(2011, 1, 1),
                               db_bufsize=self.db_buf_size)
        assert mock_isf_to_text.called
        assert db.session.query(Event.id).count() == 5
        # looking at the file, these three events should be written
        assert db.session.query(Event.id).\
            filter(Event.event_id.in_(['16868827', '600516599', '600516598'])).count() == 3
        assert db.session.query(Event.contributor_id).\
            filter(Event.event_id.in_(['16868827', '600516599', '600516598'])).count() == 3
        # and this not:
        assert db.session.query(Event.id).\
            filter(Event.event_id.in_(['15916121'])).count() == 0
        assert db.session.query(Event.contributor_id).\
            filter(Event.event_id.in_(['15916121'])).count() == 0

    @patch('stream2segment.download.modules.events.islocalfile', 
           side_effect=o_islocalfile)
    def test_get_events_eventws_format_param(self, mock_islocalfile,
                                             # fixtures:
                                             db, data, pytestdir):
        '''test that format is inferred, unless explicitly set, and all combination
            of these cases'''

        isf_file = pytestdir.newfile(create=True)
        shutil.copy(data.path('event_request_sample_isc.isf'), isf_file)

        txt_file = pytestdir.newfile(create=True)
        with open(txt_file, 'w') as _opn:
            _opn.write(self._evt_urlread_sideeffect)
        shutil.copy(data.path('event_request_sample_isc.isf'), isf_file)

        # valid isf file, no format => infer it
        for filepath, expected_events, evt_query_args in \
            [(txt_file, 2, ({}, {'format': 'txt'})),
             (isf_file, 3, ({}, {'format': 'isf'}))]:
            for evt_query_arg in evt_query_args:
                db.session.query(Event).delete()
                _ = self.get_events_df([None],
                                       db.session, filepath, evt_query_arg,
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)
                assert mock_islocalfile.call_args_list[-1][0][0] == \
                    filepath
                assert db.session.query(Event.id).count() == expected_events

        for filepath, expected_events, evt_query_arg in \
            [(txt_file, 0, {'format': 'isf'}),
             (isf_file, 0, {'format': 'txt'})]:
            db.session.query(Event).delete()
            with pytest.raises(FailedDownload) as fdwl:
                _ = self.get_events_df([None],
                                       db.session, filepath, evt_query_arg,
                                       start=datetime(2010, 1, 1),
                                       end=datetime(2011, 1, 1),
                                       db_bufsize=self.db_buf_size)
            assert "No event found. Check that the file is non empty and its content is valid" \
                in str(fdwl)
            assert mock_islocalfile.call_args_list[-1][0][0] == filepath
            assert db.session.query(Event.id).count() == expected_events


    def test_isf2text(self, data):
        '''test isc format=isf with iris equivalent'''
        # this file is stored in test data  dir and represents the iris request:
        # https://service.iris.edu/fdsnws/event/1/query?starttime=2011-01-08T00:00:00&endtime=2011-01-08T00:05:00&format=text
        iris_req_file = 'event_request_sample_iris.txt'

        # this file is stored in test data dir and represents the same request
        # on isc:
        # http://www.isc.ac.uk/fdsnws/event/1/query?starttime=2011-01-08T00:00:00&endtime=2011-01-08T00:05:00&format=isf
        isc_req_file = 'event_request_sample_isc.isf'

        iris_df = response2normalizeddf('',
                                        data.read(iris_req_file).decode('utf8'),
                                        'event')
        ret = []
        with open(data.path(isc_req_file)) as opn:
            for lst in isf2text_iter(opn, 'ISC', 'ISC'):
                ret.append('|'.join(lst))

        isc_df = response2normalizeddf('', '\n'.join(ret), 'event')

        # sort values
        iris_df.sort_values(by=[Event.contributor_id.key], inplace=True)
        isc_df.sort_values(by=[Event.event_id.key], inplace=True)
        # Now, Event with event_location_name 'POLAND' has no magnitude
        # in isc_df, so first:
        iris_df = iris_df[iris_df[Event.event_location_name.key].str.lower() != 'poland']

        iris_df.reset_index(inplace=True, drop=True)
        isc_df.reset_index(inplace=True, drop=True)

        # 1. assert a value has correctly been parsed (by looking at the file content):
        assert isc_df[isc_df[Event.event_id.key] == '16868827'].loc[0, Event.magnitude.key] == 2.1
        # and set the value to the corresponding iris value, which IN THIS CASE
        # differs (maybe due to the 'Err' field =0.2 reported in the isc file?):
        isc_df.at[isc_df.loc[isc_df[Event.event_id.key] == '16868827'].index,
                  Event.magnitude.key] = 2.0
        # test we set the value:
        assert isc_df[isc_df[Event.event_id.key] == '16868827'].loc[0, Event.magnitude.key] == 2.0

        # 2. assert a value has correctly been parsed (by looking at the file content):
        assert isc_df[isc_df[Event.event_id.key] == '16868827'].loc[0, Event.mag_author.key] \
            == 'THE'
        # and set the value to the corresponding iris value, which IN THIS CASE
        # differs (why?):
        isc_df.at[isc_df.loc[isc_df[Event.event_id.key] == '16868827'].index,
                  Event.mag_author.key] = 'ATH'
        # test we set the value:
        assert isc_df[isc_df[Event.event_id.key] == '16868827'].loc[0, Event.mag_author.key] \
            == 'ATH'

        assert (isc_df[Event.event_id.key].values == iris_df[Event.contributor_id.key].values).all()
        assert (isc_df[Event.event_id.key].values == isc_df[Event.contributor_id.key].values).all()

        # assert the following columns are equal:. We omit columns where the values
        # differ by spaces/upper cases / other minor stuff, like Event.event_location_name.key
        # or because they MUST differ (Event.event_id):
        for col in iris_df.columns:
            if col not in (Event.event_id.key, # Event.time.key,
                           Event.event_location_name.key,):
                assert (iris_df[col].values == isc_df[col].values).all()
