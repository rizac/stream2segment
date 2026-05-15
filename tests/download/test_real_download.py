"""
Real download test scenarios
"""
# Feb 4, 2016
from datetime import datetime

from click.testing import CliRunner

from stream2segment.download.url import Response, read_url as original_read_url
from stream2segment.download.utils import NothingToDownload
from unittest.mock import patch

import pandas as pd

from stream2segment.cli import cli
from stream2segment.io.db.models import WebService, Event, Channel
from stream2segment.io.db.pdsql import get_row_count


mock_url = patch("stream2segment.download.url.read_url")
mock_read_url = mock_url.start()
mock_url.side_effect = original_read_url

def teardown_module():
    mock_url.stop()


download_save_segments_path = 'stream2segment.download.main.download_and_save'
get_channels_path = 'stream2segment.download.main.get_channels'
get_events_path = 'stream2segment.download.main.get_events'
# merge_event_stations_path = 'stream2segment.download.main.merge_events_stations'
read_url_path = "stream2segment.download.url.read_url"


@patch(get_channels_path)
def test_real_download_events(
    mock_get_channels_df,
    # fixtures:
    online_only, db, log_capture, test_data_dir
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


@patch(download_save_segments_path)
@patch(get_events_path)
def test_real_download_channels(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    online_only, db, log_capture, test_data_dir
):
    """This tess a REAL download to test channels conflicts and stuff during download
    It is a legacy code to test negative network filter bug, now renamed for the new
    purpose
    """
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    mock_get_events_df.return_value = pd.DataFrame([{
        'time': datetime.fromisoformat('2000-01-03T18:28:35'),
        'mag_type': 'mb',
        'magnitude': 4.3,
        'latitude': 42.2585,
        'longitude': 2.5413,
        'depth_km': 6.9,
        'id': 1,  # random number, needs only to be present
        'webservice_id': 1  # same as above
    }])

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
    assert num_channels > 0
    assert num_events == 0  # we mocked get_events, nothing inserted on db

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


@patch(download_save_segments_path)
@patch(get_events_path)
def test_real_download_segments(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    online_only, db, log_capture, test_data_dir
):
    """This tess a REAL download to test iris (split request over liong time range)
     and segments download
    """
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    mock_get_events_df.return_value = pd.DataFrame([{
        'time': datetime.fromisoformat('2000-01-03T18:28:35'),
        'mag_type': 'mb',
        'magnitude': 4.3,
        'latitude': 42.2585,
        'longitude': 2.5413,
        'depth_km': 6.9,
        'id': 1,  # random number, needs only to be present
        'webservice_id': 1  # same as above
    }])

    # mock download save segments: raise NothingToDownload to speed up things:
    custom_message = 'custom message!'

    def func_(*a, **kw):
        raise NothingToDownload(custom_message)

    mock_download_save_segments.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    def mocked_read_url(url, *args, **kwargs):
        if "/dataselect/" in url:
            return Response("custom 500", 500, url)
        else:
            return original_read_url(url, *args, **kwargs)

    mock_read_url.side_effect=mocked_read_url

    try:
        result = CliRunner().invoke(
            cli, [
                'download', '-c', str(cfg_file), '--dburl', db.url, '-ds', 'iris',
                '--start', '1999-12-31T18:28:35', '--start', '2000-12-31T18:28:35',
                '-t', '-0.1', '0.1'
            ]
        )
        assert result.exit_code == 0
        assert custom_message in result.output

    finally:
        mock_read_url.side_effect = original_read_url
    # nothing new has been written:
    # assert num_channels < get_row_count(db.engine, Channel)
    # assert num_events == get_row_count(db.engine, Event)


@patch(download_save_segments_path)
@patch(get_events_path)
# @patch(patches.get_post_data)  # FIXME REMOVE?
def test_download_channels_adarray(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    online_only, db, log_capture, test_data_dir
):
    """This tess _ADARRAY private network"""
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    mock_get_events_df.return_value = pd.DataFrame([{
        'time': datetime.fromisoformat('2000-01-03T18:28:35'),
        'mag_type': 'mb',
        'magnitude': 4.3,
        'latitude': 42.2585,
        'longitude': 2.5413,
        'depth_km': 6.9,
        'id': 1,  # random number, needs only to be present
        'webservice_id': 1  # same as above
    }])

    # mock download save segments: raise NothingToDownload to speed up things:
    custom_message = 'custom message!'
    def func_(*a, **kw):
        raise NothingToDownload(custom_message)
    mock_download_save_segments.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--net', "_ADARRAY", '--sta', 'A*,B*', '--data_url', 'eida'
        ]
    )
    assert result.exit_code == 0
    assert 'custom message' in result.output
    num_channels = get_row_count(db.engine, Channel)
    num_events = get_row_count(db.engine, Event)
    assert num_channels > 0
