"""
Real download test scenarios
"""
from collections import namedtuple
# Feb 4, 2016
from datetime import datetime, timedelta

from click.testing import CliRunner
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError

from stream2segment.download.url import read_urls as original_read_urls
from stream2segment.download.segments import _nothing_to_download_msg  # noqa
from stream2segment.download.stationsearch import _no_station_found_within_search_area_msg  # noqa
from stream2segment.download.utils import NoSegmentsToDownload
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
load_input_path = 'stream2segment.download.main.load_input'


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
    online_only, db, log_capture, test_data_dir, tmp_path
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
        raise NoSegmentsToDownload(custom_message)
    mock_get_channels_df.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    cli_args = [
        'download', '-c', str(cfg_file), '--dburl', db.url,
        '--start', '2019-11-26T00:00:00', '--end', '2019-11-27T00:00:00',
    ]
    result = CliRunner().invoke(cli, cli_args)
    # we should have 3 events:
    # eventid   time                     latitude longitude depth_km mag_type magnitude
    # 617436519 2019-11-26 03:13:24.485  41.2633  18.9648   10.0     mb       4.17
    # 616921219 2019-11-26 09:19:27.174  43.2096  18.0153   23.0     MW       5.37
    # 621394962 2019-11-26 16:30:43.400  43.1931  18.0287   21.7     ML       3.40
    assert result.exit_code == 0
    assert custom_message in result.output
    num_channels = get_row_count(db.engine, Channel)
    num_events = get_row_count(db.engine, Event)
    assert num_channels == 0
    assert num_events > 0

    # do it again (test that no new event is written:
    result = CliRunner().invoke(cli, cli_args)
    assert result.exit_code == 0
    assert custom_message in result.output
    num_channels2 = get_row_count(db.engine, Channel)
    num_events2 = get_row_count(db.engine, Event)
    assert num_channels2 == 0
    assert num_events2 == num_events

    # mock the event_df case with duplicates
    evts = pd.concat(fetch_df(db.engine, select(Event))).iloc[[0]]
    evts = pd.concat([evts, evts, evts], ignore_index=True)
    evts.reset_index(drop=True, inplace=True)
    assert evts.index.tolist() == [0, 1, 2]
    evts.at[1, Event.magnitude.key] = evts.at[0, Event.magnitude.key] - 0.5
    evts.loc[[1,2], Event.latitude.key] = evts.at[0, Event.latitude.key] + 0.000001
    evts.loc[[1,2], Event.longitude.key] = evts.at[0, Event.longitude.key] - 0.000001
    evts.loc[[1,2], Event.depth_km.key] = evts.at[0, Event.depth_km.key] - 0.000001
    evts.loc[[1,2], Event.time.key] = (
        evts.at[0, Event.time.key] + timedelta(seconds=0.1)
    )
    evts.at[2, Event.mag_type.key] = 'preferred_magtype'
    evts.at[2, Event.mag_type.key] = 'some_other_magtype'

    cat_file = tmp_path / 'events.csv'
    evts.to_csv(cat_file, index=False)

    from stream2segment.download.inputvalidation import load_input as original_load_input
    for on_event_conflict in ['discard', 'preferred_magtype,mw', 'keep']:

        with patch(load_input_path) as _:

            def mock_load_input(*args, **kwargs):
                config, kwargs = original_load_input(*args, **kwargs)
                kwargs['advanced_settings']['on_event_conflict'] = on_event_conflict
                return config, kwargs
            _.side_effect = mock_load_input

            cli_args.extend(['--events_url', str(cat_file.resolve())])
            result = CliRunner().invoke(cli, cli_args)
            if on_event_conflict == 'discard':
                assert result.exit_code == 1
            else:
                assert result.exit_code == 0
            assert custom_message in result.output
            num_channels2 = get_row_count(db.engine, Channel)
            num_events2 = get_row_count(db.engine, Event)
            assert num_channels2 == 0
            assert num_events2 == num_events



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
        raise NoSegmentsToDownload(custom_message)
    mock_download_save_segments.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert _no_station_found_within_search_area_msg in result.output
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
    assert _no_station_found_within_search_area_msg in result.output
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
        raise NoSegmentsToDownload(custom_message)
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
        raise NoSegmentsToDownload(custom_message)
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