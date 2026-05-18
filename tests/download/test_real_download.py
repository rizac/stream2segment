"""
Real download test scenarios
"""
# Feb 4, 2016
from datetime import datetime

from click.testing import CliRunner

from stream2segment.download.url import (
    Response, read_url as original_read_url, read_urls as original_read_urls
)
from stream2segment.download.utils import NothingToDownload
from unittest.mock import patch

import pandas as pd

from stream2segment.cli import cli
from stream2segment.io.db.models import WebService, Event, Channel, Segment, StationXML, \
    QuakeML, SkippedSegment
from stream2segment.io.db.pdsql import get_row_count


# DEFINE PATHS GLOBALLY (SO IN CASE OF REFACTORING, WE CHANGE STR HERE ONCE):
download_save_segments_path = 'stream2segment.download.main.download_and_save'
get_channels_path = 'stream2segment.download.main.get_channels'
get_events_path = 'stream2segment.download.main.get_events'
read_url_path = "stream2segment.download.url.read_url"


# # WE PATCH THIS AT THE BEGINNING, TO BE SURE WE INTERCEPT ALL READ_URLS
# # WE DO NOT PATCH PER FUNCTION BECAUSE SOME FIXTURES MIGHT IMPORT read_url
# # BEFORE WE MOCK IT. BY DEFAULT, THE MOCKED FUNCTION DOES WHAT THE MOCKED FUNCTION DOES
# read_url_patch = patch(read_url_path)
# mock_read_url = read_url_patch.start()
# mock_read_url.side_effect = original_read_url
# def teardown_module():
#     read_url_patch.stop()


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


@patch(download_save_segments_path)
@patch(get_events_path)
def test_download_channels_all(
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
            '--data_url', 'eida', '--data_url', 'iris'
        ]
    )
    assert result.exit_code == 0
    assert 'custom message' in result.output
    num_channels = get_row_count(db.engine, Channel)
    num_events = get_row_count(db.engine, Event)
    assert num_channels > 0

@patch("stream2segment.download.segments.read_urls")
def test_real_download_segments(
    mock_download_segments_read_urls,
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

    cfg_file = test_data_dir / "download-network-filter.yaml"


    mock_download_segments_read_urls.side_effect = original_read_urls

    # Start tests:

    # No station within search radia:
    mock_download_segments_read_urls.reset_mock()
    cli_options = [
        'download', '-c', str(cfg_file), '--dburl', db.url,
        '--data_url', 'iris',
        '--events_url', 'www.seismicportal.eu/fdsnws/event/1/query',
        '--minmag', '4', '--maxmag', '5',
        '--net', 'CAVN,CAVN,CAVN,CFON,CLLI,MAHO',
        '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
        '--time_window', '0.1', '0.2'
    ]
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    num_segments = get_row_count(db.engine, Segment)
    num_stations = get_row_count(db.engine, StationXML)
    num_events = get_row_count(db.engine, QuakeML)
    assert num_segments == num_stations == num_events == 0
    assert not mock_download_segments_read_urls.called

    # All 4xx errors, see if we have our conditions and download logic working:
    ################
    def mocked_download_segments_read_urls(iterable, **kwargs):
        """test read_urls conditions and retry logic by lowering settings default"""
        kwargs['timeout'] = 0.0001
        # kwargs['max_concurrency'] = 1
        # kwargs['error_limit'] = 2
        # use lists cause is easier to debug:
        return original_read_urls(list(u for u in iterable), **kwargs)
    mock_download_segments_read_urls.side_effect = mocked_download_segments_read_urls
    mock_download_segments_read_urls.reset_mock()
    cli_options = [
        'download', '-c', str(cfg_file), '--dburl', db.url,
        '--data_url', 'iris',
        '--events_url', 'www.seismicportal.eu/fdsnws/event/1/query',
        '--minmag', '4', '--maxmag', '5',
        '--net', '*',
        '--sta', 'CAVN,CAVN,CAVN,CFON,CLLI,MAHO',
        '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
        '--time_window', '0.1', '0.2'
    ]
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    num_segments = get_row_count(db.engine, Segment)
    num_channels = get_row_count(db.engine, Channel)
    num_skip_segments = get_row_count(db.engine, SkippedSegment)
    num_stations = get_row_count(db.engine, StationXML)
    num_events = get_row_count(db.engine, QuakeML)
    assert num_segments == num_stations == num_events == 0
    assert num_channels > 1
    assert num_skip_segments > 1
    assert mock_download_segments_read_urls.called

    # Same config, with real download (no mocked 4xx, we should have all No data found):
    ################
    mock_download_segments_read_urls.side_effect = original_read_urls
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    num_segments = get_row_count(db.engine, Segment)
    num_channels = get_row_count(db.engine, Channel)
    num_skip_segments = get_row_count(db.engine, SkippedSegment)
    num_stations = get_row_count(db.engine, StationXML)
    num_events = get_row_count(db.engine, QuakeML)
    assert num_segments == num_stations == num_events == 0
    assert num_channels > 1
    assert num_skip_segments > 1
    assert mock_download_segments_read_urls.called

    # Same download as above (thus, nothing to download)
    ####################################################
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    # everything as before:
    assert num_segments == get_row_count(db.engine, Segment)
    assert num_channels == get_row_count(db.engine, Channel)
    assert num_skip_segments == get_row_count(db.engine, SkippedSegment)
    assert num_stations == get_row_count(db.engine, StationXML)
    assert num_events == get_row_count(db.engine, QuakeML)
    # but now we called read urls (previously not called):
    assert not mock_download_segments_read_urls.called



    # TO TEST HERE BLEOW!


    # No data found:
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--data_url', 'iris',
            '--events_url', 'www.seismicportal.eu/fdsnws/event/1/query',
            '--minmag', '4', '--maxmag', '5',
            '--quakeml',
            '--net', '*',
            '--sta', 'CAVN,CAVN,CAVN,CFON,CLLI,MAHO',
            '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
            '--time_window', '0.1', '0.2'
        ]
    )
    assert result.exit_code == 0
    num_segments2 = get_row_count(db.engine, Segment)
    num_skipped_segments2 = get_row_count(db.engine, SkippedSegment)
    num_stations2 = get_row_count(db.engine, StationXML)
    num_events2 = get_row_count(db.engine, QuakeML)
    assert num_skipped_segments2 > 0
    assert num_segments2 == num_stations2 == num_events2 == 0

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--data_url', 'iris',
            '--events_url', 'www.seismicportal.eu/fdsnws/event/1/query',
            '--minmag', '4', '--maxmag', '5',
            '--quakeml',
            '--net', '*',
            '--sta', 'CAVN,CAVN,CAVN,CFON,CLLI,MAHO',
            '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
            '--time_window', '0.1', '0.2'
        ]
    )
    assert result.exit_code == 0
    assert 'no new segments' in result.output.lower()
    num_segments2 = get_row_count(db.engine, Segment)
    num_skipped_segments2 = get_row_count(db.engine, SkippedSegment)
    num_stations2 = get_row_count(db.engine, StationXML)
    num_events2 = get_row_count(db.engine, QuakeML)
    # assert num_skipped_segments2 > 0
    assert num_segments2 == num_stations2 == num_events2 == 0

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--data_url', 'eida',
            '--events_url', 'www.seismicportal.eu/fdsnws/event/1/query',
            '--minmag', '3', '--maxmag', '4',
            '--quakeml',
            '--net', 'GE',
            '--sta', 'STU, PSZ,MORC,KMBO',
            '--minlat', '45', '--maxlat', '55',
            '--minlon', '10', '--maxlon', '20',
            '--start', '2020-01-01T00:00:00', '--end', '2020-01-31T23:59:59',
            '--time_window', '0.1', '0.2'
        ]
    )
    assert result.exit_code == 0
    num_segments3 = get_row_count(db.engine, Segment)
    num_skipped_segments3 = get_row_count(db.engine, SkippedSegment)
    num_stations3 = get_row_count(db.engine, StationXML)
    num_events3 = get_row_count(db.engine, QuakeML)
    # assert num_skipped_segments2 > 0
    assert num_segments3 == num_stations3 == num_events3 == 0





