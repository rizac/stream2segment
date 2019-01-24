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
from stream2segment.download.modules.events import get_events_df, isf2text
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df, chaid2mseedid_dict
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import prepare_for_download, \
    download_save_segments, DcDataselectManager
from stream2segment.download.utils import NothingToDownload, FailedDownload, Authorizer,\
    response2normalizeddf
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
                      db_bufsize=30, max_downloads=30, timeout=15,
                      show_progress=False):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None else
                           url_read_side_effect)
        return get_events_df(session, url, evt_query_args, start, end,
                             db_bufsize, max_downloads, timeout,
                             show_progress)

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    def test_get_events(self, mock_query, db):
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
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == 2
        # AND data to save has length 2:
        assert len(data) == 2
        # check that log has notified:
        log1 = self.log_msg()
        assert "20160508_0000113" in log1
        assert "2 database rows not inserted" in log1

        # now download again, with an url error:
        urlread_sideeffect = [504, """1|2|3|4|5|6|7|8|9|10|11|12|13
20160508_0000129|2016-05-08 05:17:11.500000|40.57|52.23|60.0|AZER|EMSC-RTS|AZER|505483|ml|3.1|AZER|CASPIAN SEA, OFFSHR TURKMENISTAN
20160508_0000004|2016-05-08 01:45:30.300000|44.96|15.35|2.0|EMSC|EMSC-RTS|EMSC|505183|ml|3.6|EMSC|CROATIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
20160508_0000113|2016-05-08 22:37:20.100000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""", URLError('blabla23___')]
        data = self.get_events_df(urlread_sideeffect, db.session, "http://eventws", {},
                                  datetime.utcnow() - timedelta(seconds=1), datetime.utcnow(),
                                  db_bufsize=self.db_buf_size)
        # assert we got the same result as above:
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == 2
        assert len(data) == 2
        log2 = self.log_msg()

        # and since first response is 413, that having split the request into two, the
        # second response is our URLError (we could test it better, anyway):
        assert "blabla23___" in log2
        # assert also that we splitted the request:
        assert "Request split into 2 sub-requests" in log2
        # also second request failed, see message:
        assert "Some sub-request failed, some available events might not have been fetched" in log2
        # aslso we did not inserted anything new:
        assert "Db table 'events': no new row to insert, no row to update" in log2

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    def test_get_events_toomany_requests_raises(self, mock_query, db):
        '''test request splitted, but failing due to max recursion'''
        urlread_sideeffect = [413]
        # as urlread returns alternatively a 413 and a good string, also sub-queries
        # will return that, so that we will end up having a 413 when the string is not
        # further splittable:
        with pytest.raises(FailedDownload) as fldwl:
            data = self.get_events_df(urlread_sideeffect, db.session, "http://eventws",
                                      {},
                                      start=datetime(2010, 1, 1),
                                      end=datetime(2011, 1, 1),
                                      db_bufsize=self.db_buf_size,
                                      max_downloads=30)
        assert 'max download' in str(fldwl)
        # assert only three events were successfully saved to db (two have same id)
        assert not db.session.query(Event).all()
        # AND data to save has length 3: (we skipped last or next-to-last cause they are dupes)
        with pytest.raises(NameError):
            assert len(data) == 3
        # assert only three events were successfully saved to db (two have same id)
        assert not db.session.query(Event).all()
        log = self.log_msg()
        assert "Calculating the required sub-requests" in log

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    def test_get_events_eventws_not_saved(self, mock_query, db):
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

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    def test_get_events_eventws_from_file(self, mock_query, db, pytestdir):
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
        assert len(db.session.query(Event).all()) == len(pd.unique(data['id'])) == 2
        assert len(data) == 2
        log2 = self.log_msg()
        # since one row is discarded, the message is something like:
        # 1 row(s) discarded (malformed server response data, e.g. NaN's). url: file:////private/var/folders/l9/zpp7wn1n4r7bt4vs39gylk4w0000gn/T/pytest-of-riccardo/pytest-442/test_get_events_eventws_from_f0/368e6e99-171c-40e1-ad8e-3afc40ebeeab.txt
        # however, we test the bare minimum:
        assert 'url: file:///' in log2
        assert not self.mock_urlopen.called

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    def test_get_events_eventws_from_file_err(self, mock_query, db, pytestdir):
        '''test request splitted, but reading from BAD events file'''
        urlread_sideeffect = [socket.timeout, 500]

        filepath = pytestdir.newfile('.txt', create=True)
        with open(filepath, 'w') as _fpn:
            _fpn.write("""000|45.68|26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
--- ERRROR --- THIS IS MALFORMED 20160508_abc0113|2016-05-08 22:37:20.100000| --- ERROR --- |26.64|163.0|BUC|EMSC-RTS|BUC|505351|ml|3.4|BUC|ROMANIA
""")

        # we want to return all times 413, and see that we raise a ValueError:

        with pytest.raises(FailedDownload) as fdl:
            # now it should raise because of a 413:
            data = self.get_events_df(urlread_sideeffect, db.session, filepath, {},
                                      start=datetime(2010, 1, 1),
                                      end=datetime(2011, 1, 1),
                                      db_bufsize=self.db_buf_size)

        assert 'Is the file content FDSN compliant?' in str(fdl)
        assert not self.mock_urlopen.called

    @patch('stream2segment.download.modules.events.urljoin', return_value='a')
    @patch('stream2segment.download.modules.events.isf2text', side_effect=isf2text)
    def test_get_events_eventws_from_isc(self, mock_isf_to_text,
                                         mock_query, db, data):
        '''test request splitted, but reading from BAD events file'''

        # now it should raise because of a 413:
        _ = self.get_events_df(None, db.session, 'emsc', {},
                               start=datetime(2010, 1, 1),
                               end=datetime(2011, 1, 1),
                               db_bufsize=self.db_buf_size)
        assert not mock_isf_to_text.called

        with pytest.raises(FailedDownload) as fld:
            # now it should raise because of a 413:
            _ = self.get_events_df(None, db.session, 'isc', {},
                                   start=datetime(2010, 1, 1),
                                   end=datetime(2011, 1, 1),
                                   db_bufsize=self.db_buf_size)
        assert "Malformed response data" in str(fld)
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
            for lst in isf2text(opn, 'ISC', 'ISC'):
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
