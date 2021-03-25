# -*- encoding: utf-8 -*-
'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

from datetime import datetime, timedelta

from mock import patch
import pytest
from future.utils import PY2
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.download.db import (Event, Station, WebService, Segment,
                                        Channel, Download, DataCenter)
from stream2segment.io.cli import ascii_decorate
from stream2segment.download.modules.utils import s2scodes


def readfile(outfile):
    with open(outfile) as _:
        return _.read()


class Test(object):
    # define ONCE HERE THE command name, so if we change it in the cli it will be easier to fix here
    CMD_PREFIX = ['dl', 'report']

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

        with patch('stream2segment.main.valid_session',
                   return_value=db.session) as mock_session:
            yield


# ## ======== ACTUAL TESTS: ================================

    @patch('stream2segment.main.open_in_browser')
    @patch('stream2segment.main.gettempdir')
    def test_simple_dreport(self, mock_gettempdir, mock_open_in_browser, db, pytestdir):
        '''test a case where save inventory is True, and that we saved inventories'''

        # test "no flags" case:
        runner = CliRunner()
        # text output, to file
        outfile = pytestdir.newfile('.txt')
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', db.dburl,
                                                       outfile])
        assert not result.exception
        expected_string = ascii_decorate("Download id: 1 (%s)" %
                                         str(db.session.query(Download.run_time).first()[0]))
        # result.output below is uncicode in PY2, whereas expected_string is str
        # Thus
        if PY2:
            expected_string = expected_string.decode('utf8')

        content = readfile(outfile)
        assert expected_string.strip() == content.strip()

        # test "normal" case:
        runner = CliRunner()
        # text output, to file
        outfile = pytestdir.newfile('.txt')
        result = runner.invoke(cli, self.CMD_PREFIX + ['--config', '--log',
                                     '--dburl', db.dburl, outfile])
        assert not result.exception
        expected_string = ascii_decorate("Download id: 1 (%s)" %
                                         str(db.session.query(Download.run_time).first()[0]))
        expected_string += """

Configuration: N/A

Log messages: N/A"""
        # result.output below is uncicode in PY2, whereas expected_string is str
        # Thus
        if PY2:
            expected_string = expected_string.decode('utf8')

        content = readfile(outfile)
        assert expected_string in content
        assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download report written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # calling with no ouptut file (print to screen, i.e. result.output):
        result = runner.invoke(cli, self.CMD_PREFIX + ['--log',
                                     '--config', '--dburl', db.dburl])
        assert not result.exception
        content = result.output
        assert expected_string in content
        assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
""")
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        expected = """Fetching data, please wait (this might take a while depending on the db size and connection)
"""
        # try with flags:
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', '--log',
                                                       db.dburl])
        assert expected in result.output
        assert not result.output[result.output.index(expected)+len(expected):]

        # try with flags:
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', '--config',
                                                       db.dburl])
        assert expected in result.output
        assert not result.output[result.output.index(expected)+len(expected):]
