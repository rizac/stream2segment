"""
Created on Feb 4, 2016

@author: riccardo
"""
import sys

from click.testing import CliRunner
from sqlalchemy.testing.plugin.plugin_base import FixtureFunctions

from stream2segment.download.modules.utils import NothingToDownload
from io import StringIO
from unittest.mock import patch
import socket

import pandas as pd
import pytest

from stream2segment.cli import cli
from stream2segment.io.db.models import WebService, Event, Channel
from stream2segment.io.db.pdsql import get_row_count


def no_connection():
    try:
        # 8.8.8.8 → Google Public DNS
        # Port 53 is almost always reachable if internet works.
        # No DNS resolution needed → faster and avoids local resolver issues.
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return False
    except OSError:
        return True


download_save_segments_path = 'stream2segment.download.main.download_and_save'
get_channels_path = 'stream2segment.download.main.get_channels'
get_events_path = 'stream2segment.download.main.get_events'
mock_merge_event_stations_path = 'stream2segment.download.main.merge_events_stations'


@pytest.mark.skipif(no_connection(), reason="no internet connection")
@pytest.mark.skipif(sys.version_info < (3,7), reason="requires python3.7+")
@patch(download_save_segments_path)
@patch(get_events_path)
# @patch(patches.get_post_data)  # FIXME REMOVE?
def test_real_download_channels(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    db, log_capture, test_data_dir
):
    """This tess a REAL download to test channels conflicts and stuff during download
    It is a legacy code to test negative network filter bug, now renamed for the new
    purpose
    """
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    ws_id = 1
    with db.engine.begin() as conn:
        conn.execute(
            WebService.__table__.insert(),
            {'url': 'http://www.isc.ac.uk/fdsnws/event/1/query', 'ws_id': ws_id}
        )

    # mock just one event downloaded. The event below is a RELa event (we took  the
    # 1st one only):
    d = pd.read_csv(StringIO("""event_id,time,latitude,longitude,depth_km,author,catalog,contributor,contributor_id,mag_type,magnitude,mag_author,event_location_name,event_type,webservice_id,id
750359 P,2000-01-03T18:28:35,42.2585,2.5413,6.9,MDD,ISC,ISC,1750359 P,mb,4.3,MDD,yrenees,,1,1"""), sep=',')
    d['time'] = pd.to_datetime(d['time'])
    d['event_type'] = d['event_type'].astype(str)
    d['webservice_id'].at[0] = ws_id

    mock_get_events_df.return_value = d

    # mock download save segments: raise NothingToDownload to speed up things:
    custom_message = 'custom message!'
    def func_(*a, **kw):
        raise NothingToDownload(custom_message)
    mock_download_save_segments.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert 'No station within' in result.output
    num_channels = get_row_count(db.engine, Channel)
    num_events = get_row_count(db.engine, Event)

    # do it again (test what's been written):
    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert 'No station within' in result.output
    # nothing new has been written:
    assert num_channels == get_row_count(db.engine, Channel)
    assert num_events == get_row_count(db.engine, Event)

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url, '-ds', 'iris', '-ds', 'eida',
        ]
    )
    assert result.exit_code == 0
    assert custom_message in result.output
    # nothing new has been written:
    assert num_channels < get_row_count(db.engine, Channel)
    assert num_events == get_row_count(db.engine, Event)


@pytest.mark.skipif(no_connection(), reason="no internet connection")
@pytest.mark.skipif(sys.version_info < (3,7), reason="requires python3.7+")
@patch(download_save_segments_path)
@patch(get_channels_path)
# @patch(patches.get_post_data)  # FIXME REMOVE?
def test_real_download_events(
    mock_get_channels_df, mock_download_save_segments,
    # fixtures:
    db, log_capture, test_data_dir
):
    """This tess a REAL download to test channels conflicts and stuff during download
    It is a legacy code to test negative network filter bug, now renamed for the new
    purpose
    """
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    # mock download save segments: raise NothingToDownload to speed up things:
    custom_message = 'custom message!'
    def func_(*a, **kw):
        raise NothingToDownload(custom_message)
    mock_get_channels_df.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
        ]
    )
    assert result.exit_code == 0
    assert 'custom message!' in result.output
    num_channels = get_row_count(db.engine, Channel)
    num_events = get_row_count(db.engine, Event)
    assert num_channels == 0
    assert num_events > 0

    # do it again (test what's been written):
    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--start', '2010-01-01', '--end', '2010-06-01',
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert 'custom message!' in result.output
    num_channels = get_row_count(db.engine, Channel)
    num_events2 = get_row_count(db.engine, Event)
    assert num_channels == 0
    assert num_events2 >= num_events

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url, '-ds', 'iris', '-ds', 'eida',
        ]
    )
    assert result.exit_code == 0
    assert custom_message in result.output
    # nothing new has been written:
    assert num_channels < get_row_count(db.engine, Channel)
    assert num_events == get_row_count(db.engine, Event)


