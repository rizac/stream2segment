"""
Real download test scenarios
"""
from collections import namedtuple
# Feb 4, 2016
from datetime import datetime

from click.testing import CliRunner
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError

from stream2segment.download.url import read_urls as original_read_urls
from stream2segment.download.segments import _nothing_to_download_msg  # noqa
from stream2segment.download.utils import NothingToDownload
from unittest.mock import patch

import pandas as pd

from stream2segment.cli import cli
from stream2segment.io.db.models import (
    WebService, Event, Channel, Segment, StationXML, QuakeML, SkippedSegment
)
from stream2segment.io.db.pdsql import get_row_count, fetch_df

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

Count = namedtuple('count', [
    'segment',
    'skipped_segment',
    'channel',
    'event',
    'quakeml',
    'stationxml'
])


def count_from_db(engine):
    return Count(
        get_row_count(engine, Segment),
        get_row_count(engine, SkippedSegment),
        get_row_count(engine, Channel),
        get_row_count(engine, Event),
        get_row_count(engine, QuakeML),
        get_row_count(engine, StationXML)
    )

@patch("stream2segment.download.segments.read_urls")
def test_real_download_segments(
    mock_download_segments_read_urls,
    # fixtures:
    online_only, db, log_capture, test_data_dir
):
    """This segments download test"""
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
        '--quakeml',
        '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
        '--time_window', '0.1', '0.2'
    ]
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    count = count_from_db(db.engine)
    assert (count.segment == count.skipped_segment == count.stationxml ==
            count.quakeml == count.channel == 0)
    assert count.event > 0
    assert not mock_download_segments_read_urls.called

    # Widen the stations range to get some channel. Mock read_urls to test a
    # repeat "same error" type whilst keeping the other default args
    ################
    def mocked_download_segments_read_urls(iterable, **kwargs):
        """forward the original read_urls after changing some args"""
        kwargs['timeout'] = 0.00001
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
        '--quakeml',
        '--sta', 'CAVN,CAVN,CAVN,CFON,CLLI,MAHO',
        '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
        '--time_window', '0.1', '0.2'
    ]
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    prev_count = count
    count = count_from_db(db.engine)
    assert count.event == prev_count.event
    assert count.channel > 0
    assert (count.segment == count.skipped_segment == count.stationxml ==
            count.quakeml == 0)
    assert mock_download_segments_read_urls.called

    # same as before, but lower to the bare minimum the retry settings, to execute
    # read_urls code paths not yet hit:
    def mocked_download_segments_read_urls(iterable, **kwargs):
        """forward the original read_urls after changing some args"""
        kwargs['timeout'] = 0.00001
        kwargs['consecutive_error_limit'] = 1
        kwargs['error_limit'] = 2
        # use lists cause is easier to debug:
        return original_read_urls(list(u for u in iterable), **kwargs)
    mock_download_segments_read_urls.side_effect = mocked_download_segments_read_urls
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    assert 'download not performed for' in result.output.lower()
    prev_count = count
    count = count_from_db(db.engine)
    assert count.event == prev_count.event
    assert count.channel  == prev_count.channel
    assert (count.segment == count.skipped_segment == count.stationxml ==
            count.quakeml == 0)
    assert mock_download_segments_read_urls.called

    # Same as before, but read_urls behaves normally. We will get all 204 No content
    mock_download_segments_read_urls.side_effect = original_read_urls
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    assert 'no content' in result.output.lower()
    prev_count = count
    count = count_from_db(db.engine)
    assert count.event == prev_count.event
    assert count.channel == prev_count.channel
    assert count.skipped_segment > 0
    assert (count.segment == count.stationxml == count.quakeml == 0)
    assert mock_download_segments_read_urls.called

    # now adjust the config to get some real segments (Http 200):
    cli_options = [
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
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    prev_count = count
    count = count_from_db(db.engine)
    assert count.event > prev_count.event
    assert count.channel > prev_count.channel
    assert count.skipped_segment == prev_count.skipped_segment
    assert count.segment > 0
    assert count.stationxml > 0
    assert count.quakeml > 0
    assert mock_download_segments_read_urls.called

    # test compressed binary function in models.py also when reading back:
    stmt = select(StationXML.data).limit(1)
    with db.engine.connect() as conn:
        result = conn.execute(stmt).scalar_one_or_none()
    assert b"<?xml " in result  # check it is not compressed

    # same as before, we check that nothing is downloaded again:
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    prev_count = count
    count = count_from_db(db.engine)
    assert count.event == prev_count.event
    assert count.channel == prev_count.channel
    assert count.skipped_segment == prev_count.skipped_segment
    assert count.segment == prev_count.segment
    assert count.stationxml == prev_count.stationxml
    assert count.quakeml == prev_count.quakeml
    assert _nothing_to_download_msg in result.output
    assert not mock_download_segments_read_urls.called  # NOTE: NOT CALLED!

    # get channels with stationxm_id, set to null one stationxml id
    # and check that we have 1 stationxml more. Also delete one quakeml from db
    # so that we will download 1 stationxml and 1 quakeml

    # get cha ids and quakeml ids:
    with db.engine.connect() as conn:
        channel_ids_0 = conn.execute(
            select(Channel.id).where(Channel.stationxml_id.is_not(None))
        ).scalars().all()
        quakeml_ids_0 = conn.execute(select(QuakeML.id)).scalars().all()

    with db.engine.begin() as conn:
        with conn.begin_nested():
            try:
                conn.execute(update(Channel).where(
                    Channel.id == list(channel_ids_0)[0]
                ).values({'stationxml_id': None}))
                conn.execute(delete(QuakeML).where(
                    QuakeML.id == list(quakeml_ids_0)[0]
                ))
            except IntegrityError as e:
                raise

    # recompute cha ids and quakeml ids:
    with db.engine.connect() as conn:
        channel_ids = conn.execute(
            select(Channel.id).where(Channel.stationxml_id.is_not(None))
        ).scalars().all()
        quakeml_ids = conn.execute(select(QuakeML.id)).scalars().all()


    # same as before, we check that nothing is downloaded again:
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    prev_count = count
    count = count_from_db(db.engine)
    assert count.event == prev_count.event
    assert count.channel == prev_count.channel
    assert count.skipped_segment == prev_count.skipped_segment
    assert count.segment == prev_count.segment
    assert count.stationxml == prev_count.stationxml
    assert count.quakeml == prev_count.quakeml
    assert _nothing_to_download_msg in result.output
    assert not mock_download_segments_read_urls.called  # NOTE: NOT CALLED!

    # recompute cha ids and quakeml ids:
    with db.engine.connect() as conn:
        channel_ids2 = conn.execute(
            select(Channel.id).where(Channel.stationxml_id.is_not(None))
        ).scalars().all()
        quakeml_ids2 = conn.execute(select(QuakeML.id)).scalars().all()

    assert len(set(channel_ids2) - set(channel_ids)) == 1
    assert len(set(quakeml_ids2) - set(quakeml_ids)) == 1