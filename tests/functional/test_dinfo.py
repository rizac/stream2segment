# -*- encoding: utf-8 -*-
'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

import os
import sys
import json
import re
from datetime import datetime, timedelta
from mock import patch
import pytest

from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.download.utils import custom_download_codes
from future.utils import PY2


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=True)

        # setup a run_id:
        r = Download()
        db.session.add(r)
        db.session.commit()
        self.run = r

        ws = WebService(id=1, url='eventws')
        db.session.add(ws)
        db.session.commit()
        self.ws = ws
        # setup an event:
        e1 = Event(id=1, webservice_id=ws.id, event_id='ev1', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e2 = Event(id=2, webservice_id=ws.id, event_id='ev2', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e3 = Event(id=3, webservice_id=ws.id, event_id='ev3', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e4 = Event(id=4, webservice_id=ws.id, event_id='ev4', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e5 = Event(id=5, webservice_id=ws.id, event_id='ev5', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        db.session.add_all([e1, e2, e3, e4, e5])
        db.session.commit()

        d1 = DataCenter(station_url='asd', dataselect_url='www.dc1/dataselect/query')
        d2 = DataCenter(station_url='asd', dataselect_url='www.dc2/dataselect/query')
        db.session.add_all([d1, d2])
        db.session.commit()

        # d1 has one station
        s_d1 = Station(datacenter_id=d1.id, latitude=11, longitude=11, network='N1', station='S1',
                       start_time=datetime.utcnow())
        s_d2 = Station(datacenter_id=d1.id, latitude=22.1, longitude=22.1, network='N1',
                       station='S2a', start_time=datetime.utcnow())
        s2_d2 = Station(datacenter_id=d1.id, latitude=22.2, longitude=22.2, network='N2',
                        station='S2b', start_time=datetime.utcnow())
        db.session.add_all([s_d1, s_d2, s2_d2])
        db.session.commit()

        url_err, mseed_err, timespan_err, timespan_warn = custom_download_codes()
        # we are about to add 3 stations * 4 channels = 12 channels
        # we add also 1 segment pre channel
        # the segments data is as follows (data, download_code, maxgap)
        seg_data = ([None, url_err, None],
                    [None, mseed_err, None],
                    [None, None, None],
                    [None, timespan_err, None],
                    # station s_d2:
                    [b'x', 200, 0.2],
                    [b'x', timespan_warn, 3.9],
                    [b'x', 200, 0.6],
                    [b'x', 200, 0.3],
                    # station s_d3:
                    [b'x', 200, 0.1],
                    [b'x', timespan_warn, 3.9],
                    [b'x', 400, None],
                    [b'x', 500, None],
                    )

        i = 0
        for s in [s_d1, s_d2, s2_d2]:
            for cha in ['HHZ', 'HHE', 'HHN', 'ABC']:
                c = Channel(station_id=s.id, location='', channel=cha, sample_rate=56.7)
                db.session.add(c)
                db.session.commit()

                data, code, gap = seg_data[i]
                i += 1
                seg = Segment(channel_id=c.id, datacenter_id=s.datacenter_id,
                              event_id=e1.id, download_id=r.id,
                              event_distance_deg=35, request_start=datetime.utcnow(),
                              arrival_time=datetime.utcnow(),
                              request_end=datetime.utcnow() + timedelta(seconds=5), data=data,
                              download_code=code, maxgap_numsamples=gap)
                db.session.add(seg)
                db.session.commit()

        with patch('stream2segment.utils.inputargs.get_session',
                   return_value=db.session) as mock_session:
            yield


# ## ======== ACTUAL TESTS: ================================

    @patch('stream2segment.main.open_in_browser')
    @patch('stream2segment.main.gettempdir')
    def test_simple_dinfo(self, mock_gettempdir, mock_open_in_browser, db, pytestdir):
        '''test a case where save inventory is True, and that we saved inventories'''

        runner = CliRunner()

        # text output, to file
        outfile = pytestdir.newfile('.txt')
        result = runner.invoke(cli, ['utils', 'dinfo', '--dburl', db.dburl, outfile])

        assert not result.exception
        content = open(outfile).read()
        assert """
                              OK        OK         Time                 Segment           Internal       
                              Gaps      Partially  Span   MSeed  Url    Not      Bad      Server         
                          OK  Overlaps  Saved      Error  Error  Error  Found    Request  Error     TOTAL
------------------------  --  --------  ---------  -----  -----  -----  -------  -------  --------  -----
www.dc1/dataselect/query   3         1          2      1      1      1        1        1         1     12
TOTAL                      3         1          2      1      1      1        1        1         1     12""" in content
        assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download info and statistics written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # text output, to stdout
        result = runner.invoke(cli, ['utils', 'dinfo', '--dburl', db.dburl])
        assert not result.exception
        assert """
                              OK        OK         Time                 Segment           Internal       
                              Gaps      Partially  Span   MSeed  Url    Not      Bad      Server         
                          OK  Overlaps  Saved      Error  Error  Error  Found    Request  Error     TOTAL
------------------------  --  --------  ---------  -----  -----  -----  -------  -------  --------  -----
www.dc1/dataselect/query   3         1          2      1      1      1        1        1         1     12
TOTAL                      3         1          2      1      1      1        1        1         1     12""" in result.output

        assert not mock_open_in_browser.called
        expected_string = '\n'.join(('╔════════════════╗',
                                     '║ Download id: 1 ║',
                                     '╚════════════════╝'))
        # result.output below is uncicode in PY2, whereas expected_string is str
        # Thus
        if PY2:
            expected_string = expected_string.decode('utf8')
        assert expected_string in result.output
        assert not mock_gettempdir.called

        # Test html output.
        # First, implement function that parses the sta_data content into dict and check stuff:
        def jsonloads(varname, content):
            start = re.search('%s\\s*:\\s*' % varname, content).end()
            end = start
            while content[end] != '[':
                end += 1
            end += 1
            brakets = 1
            while brakets > 0:
                if content[end] == "[":
                    brakets += 1
                elif content[end] == ']':
                    brakets -= 1
                end += 1
            return json.loads(content[start:end])

        # html output, to file
        outfile = pytestdir.newfile('.html')
        result = runner.invoke(cli, ['utils', 'dinfo', '--html',  '--dburl', db.dburl, outfile])

        assert not result.exception
        content = open(outfile).read()

        sta_data = jsonloads('sta_data', content)
        networks = jsonloads('networks', content)
        assert len(sta_data) == db.session.query(Station.id).count() * 2
        for i in range(0, len(sta_data), 2):
            staname = sta_data[i]
            data = sta_data[i+1]
            # Each sta is: [sta_name, [staid, stalat, stalon, sta_dcid,
            #                        d_id1, [code1, num_seg1 , ..., codeN, num_seg],
            #                        d_id2, [code1, num_seg1 , ..., codeN, num_seg],
            #                       ]
            if staname == 'S1':
                assert data[0] == 1
                assert data[1] == 11  # lat
                assert data[2] == 11  # lon
                assert data[3] == 1  # dc id
                assert data[4] == networks.index('N1')
                assert data[5] == 1  # download id (only 1)
                # assert the correct segments categories
                # (list length is the number of categories,
                # each element is the num of segments for that category,
                # and the sum of all elements must be 4)
                assert sorted(data[6][1::2]) == [1, 1, 1, 1]
            elif staname == 'S2a':
                assert data[0] == 2
                assert data[1] == 22.1  # lat
                assert data[2] == 22.1  # lon
                assert data[3] == 1  # dc id
                assert data[4] == networks.index('N1')
                assert data[5] == 1  # download id (only 1)
                # assert the correct segments categories
                # (list length is the number of categories,
                # each element is the num of segments for that category,
                # and the sum of all elements must be 4)
                assert sorted(data[6][1::2]) == [1, 1, 2]
            elif staname == 'S2b':
                assert data[0] == 3
                assert data[1] == 22.2  # lat
                assert data[2] == 22.2  # lon
                assert data[3] == 1  # dc id
                assert data[4] == networks.index('N2')
                assert data[5] == 1  # download id (only 1)
                # assert the correct segments categories
                # (list length is the number of categories,
                # each element is the num of segments for that category,
                # and the sum of all elements must be 4)
                assert sorted(data[6][1::2]) == [1, 1, 1, 1]
            else:
                raise Exception('station should not be there: %s' % staname)

        assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download info and statistics written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # html output, to file, setting maxgap to 0.2, so that S1a' has all three ok segments with gaps
        result = runner.invoke(cli, ['utils', 'dinfo', '-g', '0.15', '--html',  '--dburl', db.dburl,
                                     outfile])

        assert not result.exception

        content = open(outfile).read()
        # parse the sta_data content into dict and check stuff:
        sta_data = jsonloads('sta_data', content)
        networks = jsonloads('networks', content)
        assert len(sta_data) == db.session.query(Station.id).count() * 2
        for i in range(0, len(sta_data), 2):
            staname = sta_data[i]
            data = sta_data[i+1]
            # Each sta is: [sta_name, [staid, stalat, stalon, sta_dcid,
            #                        d_id1, [code1, num_seg1 , ..., codeN, num_seg],
            #                        d_id2, [code1, num_seg1 , ..., codeN, num_seg],
            #                       ]
            if staname == 'S1':
                assert data[0] == 1
                assert data[1] == 11  # lat
                assert data[2] == 11  # lon
                assert data[3] == 1  # dc id
                assert data[4] == networks.index('N1')
                assert data[5] == 1  # download id (only 1)
                # assert the correct segments categories
                # (list length is the number of categories,
                # each element is the num of segments for that category,
                # and the sum of all elements must be 4)
                assert sorted(data[6][1::2]) == [1, 1, 1, 1]
            elif staname == 'S2a':
                assert data[0] == 2
                assert data[1] == 22.1  # lat
                assert data[2] == 22.1  # lon
                assert data[3] == 1  # dc id
                assert data[4] == networks.index('N1')
                assert data[5] == 1  # download id (only 1)
                # assert the correct segments categories
                # (list length is the number of categories,
                # each element is the num of segments for that category,
                # and the sum of all elements must be 4)
                assert sorted(data[6][1::2]) == [1, 3]
            elif staname == 'S2b':
                assert data[0] == 3
                assert data[1] == 22.2  # lat
                assert data[2] == 22.2  # lon
                assert data[3] == 1  # dc id
                assert data[4] == networks.index('N2')
                assert data[5] == 1  # download id (only 1)
                # assert the correct segments categories
                # (list length is the number of categories,
                # each element is the num of segments for that category,
                # and the sum of all elements must be 4)
                assert sorted(data[6][1::2]) == [1, 1, 1, 1]
            else:
                raise Exception('station should not be there: %s' % staname)

        assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download info and statistics written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # html output, to temp file
        mytmpdir = pytestdir.makedir()
        assert not os.listdir(mytmpdir)
        mock_gettempdir.side_effect = lambda *a, **v: mytmpdir
        result = runner.invoke(cli, ['utils', 'dinfo', '--html', '--dburl',  db.dburl])
        assert not result.exception
        assert mock_open_in_browser.called
        assert mock_gettempdir.called
        assert os.listdir(mytmpdir) == ['s2s_dinfo.html']

