# -*- encoding: utf-8 -*-
"""
Created on Feb 14, 2017

@author: riccardo
"""
import os
import json
import re
import sys
from datetime import datetime, timedelta
from unittest.mock import patch
import pytest
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.download.db.models import (Event, Station, WebService, Segment,
                                               Channel, Download)
from stream2segment.io.cli import ascii_decorate
from stream2segment.download.modules.utils import s2scodes


def readfile(outfile):
    with open(outfile) as _:
        return _.read()


class patches:
    # paths container for patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    open_in_browser = 'stream2segment.download.db.inspection.main.open_in_browser'
    gettempdir = 'stream2segment.download.db.inspection.main.gettempdir'
    get_session = 'stream2segment.download.db.inspection.main.get_session'


class Test:
    __test__ = False  # FIXME: Disabled pytest, because of DataCenter refactoring

    # define ONCE HERE THE command name, so if we change it in the cli it will be easier to fix here
    CMD_PREFIX = ['dl', 'stats']

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

        # we are about to add 3 stations * 4 channels = 12 channels
        # we add also 1 segment pre channel
        # the segments data is as follows (data, download_code, maxgap)
        seg_data = ([None, s2scodes.url_err, None],
                    [None, s2scodes.mseed_err, None],
                    [None, None, None],
                    [None, s2scodes.timespan_err, None],
                    # station s_d2:
                    [b'x', 200, 0.2],
                    [b'x', s2scodes.timespan_warn, 3.9],
                    [b'x', 200, 0.6],
                    [b'x', 200, 0.3],
                    # station s_d3:
                    [b'x', 200, 0.1],
                    [b'x', s2scodes.timespan_warn, 3.9],
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

        with patch(patches.get_session, return_value=db.session) as mock_session:
            yield


# ## ======== ACTUAL TESTS: ================================

    @patch(patches.open_in_browser)
    @patch(patches.gettempdir)
    def test_simple_dstats(self, mock_gettempdir, mock_open_in_browser, db, pytestdir):
        '''test a case where save inventory is True, and that we saved inventories'''

        runner = CliRunner()

        # text output, to file
        outfile = pytestdir.newfile('.txt')
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', db.dburl,
                                                       '-o', outfile])

        assert not result.exception
        content = readfile(outfile)
        assert """
                              OK        OK         Time                 Segment           Internal       
                              Gaps      Partially  Span   MSeed  Url    Not      Bad      Server         
                          OK  Overlaps  Saved      Error  Error  Error  Found    Request  Error     TOTAL
------------------------  --  --------  ---------  -----  -----  -----  -------  -------  --------  -----
www.dc1/dataselect/query   3         1          2      1      1      1        1        1         1     12
TOTAL                      3         1          2      1      1      1        1        1         1     12""" in content
        assert result.output.startswith("""Fetching data, please wait""")
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # text output, to stdout
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', db.dburl])
        assert not result.exception
        assert """
                              OK        OK         Time                 Segment           Internal       
                              Gaps      Partially  Span   MSeed  Url    Not      Bad      Server         
                          OK  Overlaps  Saved      Error  Error  Error  Found    Request  Error     TOTAL
------------------------  --  --------  ---------  -----  -----  -----  -------  -------  --------  -----
www.dc1/dataselect/query   3         1          2      1      1      1        1        1         1     12
TOTAL                      3         1          2      1      1      1        1        1         1     12""" in result.output

        assert not mock_open_in_browser.called
        expected_string = ascii_decorate("Download id: 1")
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
        result = runner.invoke(cli, self.CMD_PREFIX + ['--html',  '--dburl',
                                                       db.dburl, '-o', outfile])

        assert not result.exception
        content = readfile(outfile)

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

        assert result.output.startswith("""Fetching data, please wait""")
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # html output, to file, setting maxgap to 0.2, so that S1a' has all three ok segments
        # with gaps
        result = runner.invoke(cli, self.CMD_PREFIX + ['-g', '0.15', '--html',
                                                       '--dburl', db.dburl,
                                                       '--outfile', outfile])

        assert not result.exception

        content = readfile(outfile)
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

        assert result.output.startswith("""Fetching data, please wait""")
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # html output, to temp file
        mytmpdir = pytestdir.makedir()
        assert not os.listdir(mytmpdir)
        mock_gettempdir.side_effect = lambda *a, **v: mytmpdir
        result = runner.invoke(cli, self.CMD_PREFIX + ['--html', '--dburl',
                                                       db.dburl])
        assert not result.exception
        assert mock_open_in_browser.called
        assert mock_gettempdir.called
        assert os.listdir(mytmpdir) == ['s2s_dstats_' +
                                        os.path.basename(db.dburl) + '.html']

    @pytest.mark.parametrize('download_index', [None, -1])
    @patch(patches.open_in_browser)
    @patch(patches.gettempdir)
    def test_dstats_no_segments(self, mock_gettempdir, mock_open_in_browser,
                                download_index,
                                # fixtures:
                                db, pytestdir):
        """test a case where save inventory is True, and that we saved inventories"""

        # mock  a download with only inventories, i.e. with no segments downloaded
        dwnl = Download()
        dwnl.run_time = datetime(2018, 12, 2, 16, 46, 56, 472330)
        db.session.add(dwnl)
        db.session.commit()

        runner = CliRunner()

        # text output, to file
        outfile = pytestdir.newfile('.txt')
        args = self.CMD_PREFIX + ['--dburl', db.dburl, '-o', outfile]
        if download_index is not None:
            args += [str(download_index)]
        result = runner.invoke(cli, args)

        assert not result.exception
        content = readfile(outfile)
        assert """
                              OK        OK         Time                 Segment           Internal       
                              Gaps      Partially  Span   MSeed  Url    Not      Bad      Server         
                          OK  Overlaps  Saved      Error  Error  Error  Found    Request  Error     TOTAL
------------------------  --  --------  ---------  -----  -----  -----  -------  -------  --------  -----
www.dc1/dataselect/query   3         1          2      1      1      1        1        1         1     12
TOTAL                      3         1          2      1      1      1        1        1         1     12""" in content
        assert result.output.startswith("""Fetching data, please wait""")
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        expected_string = ascii_decorate("Download id: 2")
        assert expected_string in content
        expected_string2 = """
Executed: 2018-12-02T16:46:56.472330
Event query parameters: N/A

No segments downloaded
"""
        assert expected_string2 in content[content.index(expected_string):]

        # run with html, test just that everything works fine
        result = runner.invoke(cli, self.CMD_PREFIX + ['--html', '--dburl',
                                                       db.dburl, '-o', outfile])
        assert not result.exception
