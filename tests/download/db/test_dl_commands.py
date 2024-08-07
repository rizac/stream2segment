# -*- encoding: utf-8 -*-
"""
Created on Feb 14, 2017

@author: riccardo
"""
import re
from datetime import datetime, timedelta
from unittest.mock import patch
import pytest
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.download.db.models import (Event, Station, WebService, Segment,
                                               Channel, Download)
from stream2segment.download.modules.utils import s2scodes


class patches:
    # paths container for patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    # open_in_browser = 'stream2segment.download.db.inspection.main.open_in_browser'
    # gettempdir = 'stream2segment.download.db.inspection.main.gettempdir'
    get_session = 'stream2segment.download.db.inspection.main.get_session'


class Test:
    __test__ = False  # FIXME: Disabled pytest, because of DataCenter refactoring

    # define ONCE HERE THE command name, so if we change it in the cli it will be easier to fix here
    CMD_PREFIX_CONFIG = ['dl', 'config']
    CMD_PREFIX_LOG = ['dl', 'log']
    CMD_PREFIX_SUMMARY = ['dl', 'summary']

    config_content = 'this_is_a_yam;l_config: a'
    log_content = 'this_is_a_yam;l_config: a'
    d_time0 = datetime.utcnow() - timedelta(days=1)  # assure it's in the past so its download
    # is at index 0
    d_time1 = datetime.utcnow()

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=True)

        # fake run id with no associated segments but defined log, config and time:
        r = Download(id=1, log=self.log_content, config=self.config_content,
                     run_time=self.d_time0)
        db.session.add(r)
        db.session.commit()

        # setup a run_id:
        r = Download(id=2, run_time=self.d_time1)
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

    def test_config(self, db):
        CMD_PREFIX = self.CMD_PREFIX_CONFIG
        expected_re1 = "\n".join([
            "#+",
            "# Download id: 2.*",
            "#+"
        ])
        expected_re0 = "\n".join([
            "#+",
            "# Download id: 1.*",
            "#+"
            "\\s*",
            re.escape(self.config_content)
        ])
        default_dindex_when_missing = -1
        for dindex in [None, 0, 1]:
            # test "no flags" case:
            runner = CliRunner(mix_stderr=False)
            # text output, to file
            args = [] if dindex is None else [str(dindex)]
            result = runner.invoke(cli, CMD_PREFIX + ['--dburl', db.dburl] + args)
            assert not result.exception
            output = result.output.strip()
            if dindex == 0:
                assert re.match(expected_re0, output)
            elif dindex == 1:
                assert re.match(expected_re1, output)
            else:
                # the default when missing is -1, i.e. dindex 1:
                # assert re.search(expected_re0, output)
                assert re.search(expected_re1, output)

        # test providing download id and not download indices:
        result = runner.invoke(cli, CMD_PREFIX + ['--dburl', db.dburl, '-did', 1])
        assert not result.exception

    def test_log(self, db):
        CMD_PREFIX = self.CMD_PREFIX_LOG
        "╔", "═", "╗", "║", "║", "╚", "═", "╝"
        expected_re1 = "\n".join([
            "╔═+╗",
            "║ Download id: 2.*",
            "╚═+╝"
            "\\s*",
            "\\[[a-zA-Z ]+\\]"  # <- end of log file tag
        ])
        expected_re0 = "\n".join([
            "╔═+╗",
            "║ Download id: 1.*",
            "╚═+╝"
            "\\s*",
            re.escape(self.log_content),
            "\\[[a-zA-Z ]+\\]"  # <- end of log file tag
        ])
        default_dindex_when_missing = -1
        for dindex in [None, 0, 1]:
            # test "no flags" case:
            runner = CliRunner(mix_stderr=False)
            # text output, to file
            args = [] if dindex is None else [str(dindex)]
            result = runner.invoke(cli, CMD_PREFIX + ['--dburl', db.dburl] + args)
            assert not result.exception
            output = result.output.strip()
            if dindex == 0:
                assert re.match(expected_re0, output)
            elif dindex == 1:
                assert re.match(expected_re1, output)
            else:
                # the default when missing is -1, i.e. dindex 1:
                # assert re.search(expected_re0, output)
                assert re.search(expected_re1, output)


    def test_summary(self, db):
        CMD_PREFIX = self.CMD_PREFIX_SUMMARY
        t_0 = self.d_time0.replace(microsecond=0)
        t_1 = self.d_time1.replace(microsecond=0)
        expected_re1 = "\n".join([
            'Download id\\s+Execution time\\s+Index',
            '\\s*2\\s+' + re.escape(t_1.isoformat()) + '\\s+1',
        ])
        expected_re0 = "\n".join([
            'Download id\\s+Execution time\\s+Index',
            '\\s*1\\s+' + re.escape(t_0.isoformat()) + '\\s+0'
        ])
        expected_re_all = "\n".join([
            'Download id\\s+Execution time\\s+Index',
            '\\s*1\\s+' + re.escape(t_0.isoformat()) + '\\s+0',
            '\\s*2\\s+' + re.escape(t_1.isoformat()) + '\\s+1'
        ])
        default_dindex_when_missing = -1
        for dindex in [None, 0, 1]:
            # test "no flags" case:
            runner = CliRunner(mix_stderr=False)
            # text output, to file
            args = [] if dindex is None else [str(dindex)]
            result = runner.invoke(cli, CMD_PREFIX + ['--dburl', db.dburl] + args)
            assert not result.exception
            output = result.output.strip()
            if dindex == 0:
                assert re.match(expected_re0, output)
            elif dindex == 1:
                assert re.match(expected_re1, output)
            else:
                # the default when missing is all:
                assert re.search(expected_re_all, output)
