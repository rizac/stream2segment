# -*- encoding: utf-8 -*-
'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

import os
import sys
from itertools import cycle, chain
from collections import defaultdict
from datetime import datetime, timedelta

from mock import patch
import pytest
from click.testing import CliRunner

from stream2segment.cli import cli
from stream2segment.download.db import (Event, Station, WebService, Segment,
                                        Channel, Download, DataCenter)
from stream2segment.download.utils import s2scodes


class Test(object):

    # define ONCE HERE THE command name, so if we change it in the cli it will be easier to fix here
    CMD_PREFIX = ['db', 'drop']

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=True)

        # setup a run_id:
        self.downloads = [Download(id=1), Download(id=2), Download(id=3)]
        db.session.add_all(self.downloads)
        db.session.commit()

        wss = WebService(id=1, url='eventws')
        db.session.add(wss)
        db.session.commit()

        # setup an event:
        ev1 = Event(id=1, webservice_id=wss.id, event_id='ev1', latitude=8, longitude=9,
                    magnitude=5, depth_km=4, time=datetime.utcnow())
        db.session.add_all([ev1])
        db.session.commit()

        dc1 = DataCenter(station_url='asd', dataselect_url='www.dc1/dataselect/query')
        db.session.add_all([dc1])
        db.session.commit()

        # d1 has one station
        s_d1 = Station(datacenter_id=dc1.id, latitude=11, longitude=11, network='N1', station='S1',
                       start_time=datetime.utcnow())
        s_d2 = Station(datacenter_id=dc1.id, latitude=22.1, longitude=22.1, network='N1',
                       station='S2a', start_time=datetime.utcnow())
        s2_d2 = Station(datacenter_id=dc1.id, latitude=22.2, longitude=22.2, network='N2',
                        station='S2b', start_time=datetime.utcnow())
        db.session.add_all([s_d1, s_d2, s2_d2])
        db.session.commit()

        # we are about to add 3 stations * 4 channels = 12 channels
        # we add also 1 segment pre channel
        # the segments data is as follows (data, download_code, maxgap)
        # the first 4 SEGMENTS are download id = self.downloads[0].id
        # the last 8 are download id = self.downloads[1].id
        seg_data = ([None, s2scodes.url_err, None, self.downloads[0].id],
                    [None, s2scodes.mseed_err, None, self.downloads[0].id],
                    [None, None, None, self.downloads[0].id],
                    [None, s2scodes.timespan_err, None, self.downloads[0].id],
                    # station s_d2:
                    [b'x', 200, 0.2, self.downloads[1].id],
                    [b'x', s2scodes.timespan_warn, 3.9, self.downloads[1].id],
                    [b'x', 200, 0.6, self.downloads[1].id],
                    [b'x', 200, 0.3, self.downloads[1].id],
                    # station s_d3:
                    [b'x', 200, 0.1, self.downloads[1].id],
                    [b'x', s2scodes.timespan_warn, 3.9, self.downloads[1].id],
                    [b'x', 400, None, self.downloads[1].id],
                    [b'x', 500, None, self.downloads[1].id],
                    )

        i = 0
        for s in [s_d1, s_d2, s2_d2]:
            for cha in ['HHZ', 'HHE', 'HHN', 'ABC']:
                c = Channel(station_id=s.id, location='', channel=cha, sample_rate=56.7)
                db.session.add(c)
                db.session.commit()

                data, code, gap, did = seg_data[i]
                i += 1
                seg = Segment(channel_id=c.id, datacenter_id=s.datacenter_id,
                              event_id=ev1.id, download_id=did,
                              event_distance_deg=35, request_start=datetime.utcnow(),
                              arrival_time=datetime.utcnow(),
                              request_end=datetime.utcnow() + timedelta(seconds=5), data=data,
                              download_code=code, maxgap_numsamples=gap)
                db.session.add(seg)
                db.session.commit()

        with patch('stream2segment.main.valid_session',
                   return_value=db.session) as mock_session:
            yield

    def get_db_status(self, session):
        ddic = defaultdict(list)
        for did, segid in session.query(Segment.download_id, Segment.id).order_by(Segment.download_id):
            ddic[did].append(segid)
        for did in session.query(Download.id):
            if did[0] not in ddic:
                ddic[did[0]]
        return dict(ddic)

# ## ======== ACTUAL TESTS: ================================

    @patch('stream2segment.main.input', return_value='y')
    def test_simple_ddrop_boundary_cases(self, mock_input,
                                         # oytest fixtures:
                                         db):
        '''test boundary cases for ddrop'''

        runner = CliRunner()
        # text no download id provided
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', db.dburl])
        assert result.exception
        assert result.exit_code != 0
        assert not mock_input.called
        # click outputs slightly different messages depending on version:
        assert ('Missing option "-did" / "--download-id"' in result.output) \
            or ("Missing option '-did' / '--download-id'" in result.output)

        # text output, to file
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', db.dburl, '-did', 4])
        assert not result.exception
        assert "Nothing to delete" in result.output
        assert result.exit_code == 0
        assert not mock_input.called


    @pytest.mark.parametrize('ids_to_delete', [(1,), (2,), (3,),
                                               (1, 2), (1, 3), (2, 3),
                                               (1, 2, 3)])
    @patch('stream2segment.main.input', return_value='y')
    def test_simple_ddrop(self, mock_input, ids_to_delete,
                          # oytest fixtures:
                          db):
        '''test ddrop with different cases'''

        db_status = self.get_db_status(db.session)
        runner = CliRunner()
        expected_deleted_seg_ids = [segid for id2delete in ids_to_delete
                                    for segid in db_status[id2delete]]
        # add list of the form ['--did', 1, '--did',  2, ...]:
        dids_args = [item for pair in zip(cycle(['-did']), ids_to_delete) for item in pair]
        #  db.session.query(Download.id, ).join(Download.segments)
        result = runner.invoke(cli, self.CMD_PREFIX + ['--dburl', db.dburl] +
                               dids_args)
        assert not result.exception
        assert mock_input.called
        for ddd in ids_to_delete:
            expected_str = 'Download id=%d: DELETED (%d associated segments deleted)' % \
                (ddd, len(db_status[ddd]))
            assert expected_str in result.output
        expected_dids_remained = sorted(set(db_status) - set(ids_to_delete))
        assert sorted(set(_[0] for _ in db.session.query(Download.id))) == \
            expected_dids_remained
        expected_segids_remained = sorted(chain(*[db_status[_] for _ in expected_dids_remained]))
        assert sorted(_[0] for _ in db.session.query(Segment.id)) == \
                expected_segids_remained



