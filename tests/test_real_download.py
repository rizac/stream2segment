"""
Real download test scenarios
"""
# Feb 4, 2016
import re
import os
import shutil
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError

from stream2segment.download.url import (
    build_and_read_urls as original_read_urls, Response
)
from stream2segment.download.channels import (
    download_channels as original_download_channels
)
from stream2segment.download.main import _nothing_to_download_msg
from stream2segment.download.stationsearch import (
    _no_station_found_within_search_area_msg
)
from stream2segment.download.inputvalidation import (
    get_engine as original_get_engine
)
from stream2segment.download.main import (
    start_logging as original_start_logging
)
from stream2segment.download.utils import NoSegmentsToDownload, FailedDownload
from stream2segment.download.inputvalidation import load_input as original_load_input
from stream2segment.cli import cli
from stream2segment.io.db import is_sqlite, is_postgres
from stream2segment.io.db.models import (
    Event, Channel, Segment, StationXML, QuakeML, SkippedSegment
)
from stream2segment.io.db.pdsql import get_row_count, fetch_df
from stream2segment.download.xml import (
    build_and_read_urls as xml_build_and_read_urls
)


# DEFINE PATHS GLOBALLY (SO IN CASE OF REFACTORING, WE CHANGE STR HERE ONCE):
download_save_segments_path = 'stream2segment.download.main.download_and_save'
get_channels_path = 'stream2segment.download.main.get_channels'
download_channels_path = 'stream2segment.download.channels.download_channels'
get_events_path = 'stream2segment.download.main.get_events'
read_url_path = "stream2segment.download.url.read_url"
load_input_path = 'stream2segment.download.main.load_input'


@pytest.fixture
def db_engine(db_url):
    """
    Intercept the db engine created in tested functions to be accessible in tests
    """
    db_eng = original_get_engine(db_url)
    with patch(
        "stream2segment.download.inputvalidation.get_engine",
        return_value=db_eng,
    ):
        yield db_eng


@pytest.fixture
def log_capture(tmp_path: Path):
    """
    Intercept the log file Path created in tested functions to be accessible in tests
    """

    log_file = tmp_path / 'test.log'
    
    def wrapper(logger, log_file_path, verbose):
        return original_start_logging(logger, log_file, verbose)

    with patch(
        "stream2segment.download.main.start_logging", side_effect=wrapper
    ):
        yield log_file


# ======== ACTUAL TESTS: ================================


@patch(get_channels_path)
def test_real_download_events(
    mock_get_channels_df,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir, tmp_path
):
    """This tess a REAL download to test channels conflicts and stuff during download
    It is a legacy code to test negative network filter bug, now renamed for the new
    purpose
    """
    db_url = str(db_engine.url)
    if is_postgres(db_url):
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    # mock download save segments: raise NothingToDownload to speed up things:
    no_channels_msg = 'no channels found!'
    def func_(*a, **kw):
        raise NoSegmentsToDownload(no_channels_msg)
    mock_get_channels_df.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    cli_args = [
        'download', '-c', str(cfg_file), '--dburl', db_url,
        '--start', '2019-11-26T00:00:00', '--end', '2019-11-27T00:00:00',
    ]
    result = CliRunner().invoke(cli, cli_args)
    # we should have 3 events:
    # eventid   time                     latitude longitude depth_km mag_type magnitude
    # 617436519 2019-11-26 03:13:24.485  41.2633  18.9648   10.0     mb       4.17
    # 616921219 2019-11-26 09:19:27.174  43.2096  18.0153   23.0     MW       5.37
    # 621394962 2019-11-26 16:30:43.400  43.1931  18.0287   21.7     ML       3.40
    assert result.exit_code == 0
    assert no_channels_msg in result.output
    num_channels = get_row_count(db_engine, Channel)
    num_events = get_row_count(db_engine, Event)
    assert num_channels == 0
    assert num_events > 0

    # do it again (test that no new event is written:
    result = CliRunner().invoke(cli, cli_args)
    assert result.exit_code == 0
    assert no_channels_msg in result.output
    num_channels2 = get_row_count(db_engine, Channel)
    num_events2 = get_row_count(db_engine, Event)
    assert num_channels2 == 0
    assert num_events2 == num_events

    # mock the event_df case with duplicates
    evts = pd.concat(fetch_df(db_engine, select(Event))).iloc[[0]]
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
    evts.at[1, Event.mag_type.key] = 'preferred_magtype'
    evts.at[2, Event.mag_type.key] = 'some_other_magtype'

    cat_file = tmp_path / 'events.csv'
    evts.to_csv(cat_file, index=False)

    discard, pref_mag, keep =  'discard', 'preferred_magtype,mw', 'keep'
    for on_event_conflict in [discard, pref_mag, keep]:

        with patch(load_input_path) as _:

            def mock_load_input(*args, **kwargs):
                config, kwargs = original_load_input(*args, **kwargs)
                kwargs['advanced_settings']['on_event_conflict'] = on_event_conflict
                return config, kwargs
            _.side_effect = mock_load_input

            cli_args.extend(['--events_url', str(cat_file.resolve())])
            result = CliRunner().invoke(cli, cli_args)
            assert result.exit_code == 0
            log_text = log_capture.read_text()
            # in all cases we issued a no segments to download
            assert NoSegmentsToDownload.prefix in result.output
            assert NoSegmentsToDownload.prefix in log_text
            if on_event_conflict == discard:
                # we did not get to the channel download step:
                assert no_channels_msg not in log_text
                assert no_channels_msg not in result.output
                # we found all events overlapping:
                assert '3 overlapping event(s)' in log_text
                assert '3 overlapping event(s)' in result.output
                # we did not get to the events save to db sub-step:
                assert 'event(s) replaced' not in log_text
            elif on_event_conflict == pref_mag:
                # we got to the channel download step:
                assert no_channels_msg in log_text
                assert no_channels_msg in result.output
                # we found 2 events overlapping:
                assert '2 overlapping event(s)' in result.output
                assert '2 overlapping event(s)' in log_text
                # we got to the events save to db sub-step, replacing with db records:
                assert '1 event(s) replaced' in log_text
            else:
                # we got to the channel download step:
                assert no_channels_msg in log_text
                assert no_channels_msg in result.output
                # we found no event overlapping (we said to keep them anyway):
                assert 'overlapping event(s)' not in log_text
                assert 'overlapping event(s)' not in result.output
                # we got to the events save to db sub-step, replacing with db records:
                assert '2 event(s) replaced' in log_text

            # we never save new events in any case:
            num_channels2 = get_row_count(db_engine, Channel)
            num_events2 = get_row_count(db_engine, Event)
            assert num_channels2 == 0
            assert num_events2 == num_events


@patch(download_save_segments_path)
@patch(get_events_path)
def test_real_download_channels(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir
):
    """This tess a REAL download to test channels conflicts and stuff during download
    It is a legacy code to test negative network filter bug, now renamed for the new
    purpose
    """
    db_url = str(db_engine.url)
    if is_postgres(db_url):
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
            'download', '-c', str(cfg_file), '--dburl', db_url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert _no_station_found_within_search_area_msg in result.output
    num_channels = get_row_count(db_engine, Channel)
    num_events = get_row_count(db_engine, Event)
    assert num_channels > 0
    assert num_events == 0  # we mocked get_events, nothing inserted on db

    # do it again (test what's been written):
    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db_url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert _no_station_found_within_search_area_msg in result.output
    # nothing new has been written:
    assert num_channels == get_row_count(db_engine, Channel)
    assert num_events == get_row_count(db_engine, Event)

    # do it again (test channel conflict):
    # first change 10 channels lat, to return conflicts:
    with db_engine.begin() as conn:
        first_10_ids = select(Channel.__table__.c.id).limit(10)

        conn.execute(
            update(Channel.__table__)
            .where(Channel.__table__.c.id.in_(first_10_ids))
            .values(latitude=Channel.__table__.c.latitude + 1)
        )
    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db_url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )
    assert result.exit_code == 0
    assert _no_station_found_within_search_area_msg in result.output
    # nothing new has been written:
    assert num_channels == get_row_count(db_engine, Channel)
    assert num_events == get_row_count(db_engine, Event)
    assert (
        'Replacing the following channels with matching database records'
        in log_capture.read_text()
    )


@patch(download_save_segments_path)
@patch(get_events_path)
# @patch(patches.get_post_data)  # FIXME REMOVE?
def test_download_channels_adarray(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir
):
    """This tess _ADARRAY private network"""
    db_url = str(db_engine.url)
    if is_postgres(db_url):
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    with urlopen(
        'https://www.orfeus-eu.org/eidaws/routing/1/query?network=_ADARRAY'
        '&service=dataselect&format=json'
    ) as _:
        req = b''
        if _.code == 200:
            req = _.read(1)
        if not req:
            pytest.skip('_ADARRAY returns no data (bugfix?)')

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
            'download', '-c', str(cfg_file), '--dburl', db_url,
            '--net', "_ADARRAY", '--sta', 'A*,B*', '--data_url', 'eida'
        ]
    )
    assert result.exit_code == 0
    assert 'custom message' in result.output
    num_channels = get_row_count(db_engine, Channel)
    num_events = get_row_count(db_engine, Event)
    assert num_channels > 0


@patch(download_save_segments_path)
@patch(download_channels_path)
@patch(get_events_path)
def test_download_channels_all(
    mock_get_events_df, mock_download_channels, mock_download_save_segments,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir
):
    """This tess _ADARRAY private network"""
    db_url = str(db_engine.url)
    if is_postgres(db_url):
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

    # mock channels urls to check stuff
    def func_(urls, *a, **kw):
        urls = list(urls)  # consume generator
        assert len(urls) > 1
        # no dupes in nets:
        all_nets = set()
        for url in urls:
            nets = re.search(r"[?&]net=([^&]+)", url).group(1).split(",")
            assert all_nets.isdisjoint(nets)
            all_nets.update(nets)
        # pass first element only, we do not want to spend more time on this:
        return original_download_channels(urls[:1], *a, **kw)
        # raise NoSegmentsToDownload(custom_message)
    mock_download_channels.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db_url,
            '--data_url', 'iris'  #  '--data_url', 'eida',
        ]
    )
    assert result.exit_code == 0
    assert mock_download_channels.called
    # we did not get to the segments download because no stations in the search area:
    assert custom_message not in result.output
    # or alternatively:
    assert not mock_download_save_segments.called
    # test stuf on db:
    num_channels = get_row_count(db_engine, Channel)
    num_events = get_row_count(db_engine, Event)
    assert num_channels > 0
    assert num_events == 0


@patch("stream2segment.download.segments.build_and_read_urls")
def test_real_download_segments(
    mock_download_segments_read_urls,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir
):
    """This segments download test"""
    db_url = str(db_engine.url)

    cfg_file = test_data_dir / "download-network-filter.yaml"

    mock_download_segments_read_urls.side_effect = original_read_urls

    # Start tests:

    # No station within search radia:
    mock_download_segments_read_urls.reset_mock()
    cli_options = [
        'download', '-c', str(cfg_file), '--dburl', db_url,
        '--data_url', 'iris',
        '--events_url', 'www.seismicportal.eu/fdsnws/event/1/query',
        '--minmag', '5', '--maxmag', '6',
        '--sta', 'MAHO',  # CAVN,CAVN,CFON,CLLI,MAHO',
        '--quakeml',
        '--start', '2000-01-01T00:00:00', '--end', '2000-12-31T23:59:59',
        '--time_window', '0.1', '0.2'
    ]

    # Widen the stations range to get some channel. Mock read_urls to test a
    # repeat "same error" type whilst keeping the other default args
    ################
    def mocked_download_segments_read_urls(url_builder, iterable, **kwargs):
        """forward the original read_urls after changing some args"""
        kwargs['timeout'] = 0.00001
        # use lists cause is easier to debug:
        return original_read_urls(url_builder, list(u for u in iterable), **kwargs)

    mock_download_segments_read_urls.side_effect = mocked_download_segments_read_urls
    mock_download_segments_read_urls.reset_mock()
    cli_options = [
        'download', '-c', str(cfg_file), '--dburl', db_url,
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
    count = count_from_db(db_engine)
    assert count.event > 0
    assert count.channel > 0
    assert count.stationxml > 0
    assert (
        count.segment == count.skipped_segment == count.quakeml == 0
    )
    assert mock_download_segments_read_urls.called

    # same as before, but lower to the bare minimum the retry settings, to execute
    # read_urls code paths not yet hit:
    def mocked_download_segments_read_urls(url_builder, iterable, **kwargs):
        """forward the original read_urls after changing some args"""
        kwargs['timeout'] = 0.00001
        kwargs['same_error_limit'] = 1
        kwargs['error_limit'] = 2
        # use lists cause is easier to debug:
        return original_read_urls(url_builder, list(u for u in iterable), **kwargs)

    mock_download_segments_read_urls.side_effect = mocked_download_segments_read_urls
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    assert 'download(s) skipped' in result.output.lower()
    assert count == count_from_db(db_engine)  # nothing changed on DB
    assert mock_download_segments_read_urls.called

    # Same as before, but read_urls behaves normally. We will get all 204 No content
    mock_download_segments_read_urls.side_effect = original_read_urls
    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    assert 'no content' in result.output.lower()
    prev_count = count
    count = count_from_db(db_engine)
    assert count.event == prev_count.event
    assert count.channel == prev_count.channel
    assert count.stationxml == prev_count.stationxml
    assert count.skipped_segment > 0
    assert (count.segment == count.quakeml == 0)
    assert mock_download_segments_read_urls.called

    # now adjust the config to get some real segments (Http 200):
    cli_options = [
        'download', '-c', str(cfg_file), '--dburl', db_url,
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
    count = count_from_db(db_engine)
    assert count.event > prev_count.event
    assert count.channel > prev_count.channel
    assert count.skipped_segment == prev_count.skipped_segment
    assert count.segment > 0
    assert count.stationxml > 0
    assert count.quakeml > 0
    assert mock_download_segments_read_urls.called

    # test compressed binary function in models.py also when reading back:
    stmt = select(StationXML.id, StationXML.data)
    with db_engine.connect() as conn:
        result = {_[0]: _[1] for _ in conn.execute(stmt).fetchall()}
    result_with_xml = {k :v for k, v in result.items() if v is not None}
    assert len(result_with_xml)
    # check XML is not compressed (should only be saved as compressed)
    assert  b"<?xml " in list(result_with_xml.values())[0]

    # xml_build_and_read_urls should be called when we have XML to download.
    # So mock it and check we call it or not depending on our input:
    with patch(
        'stream2segment.download.xml.build_and_read_urls',
        side_effect=xml_build_and_read_urls
    ) as mock_download_xml:

        # same as before, we check that nothing is downloaded again:
        mock_download_segments_read_urls.reset_mock()
        result = CliRunner().invoke(cli, cli_options)
        assert result.exit_code == 0
        assert count == count_from_db(db_engine)  # nothing changed on DB
        assert _nothing_to_download_msg in result.output
        assert not mock_download_segments_read_urls.called  # NOTE: NOT CALLED!
        assert not mock_download_xml.called

        # get channels with stationxm_id, set to null one stationxml id
        # and check that we have 1 stationxml more. Also delete one quakeml from db
        # so that we will download 1 stationxml and 1 quakeml

        # get cha ids and quakeml ids:
        def get_ids() -> tuple[list[int], dict[int, int]]:
            with db_engine.connect() as conn:
                cha_ids_ = {_[0]: _[1] for _ in conn.execute(
                    select(Channel.id, StationXML.id).join(
                        StationXML, StationXML.id == Channel.station_id
                    ).where(StationXML.data.is_not(None))
                )}
                ev_ids_ = sorted(set(
                    conn.execute(select(QuakeML.id)).scalars().all()
                ))
            return ev_ids_, cha_ids_

        ev_ids, cha_ids = get_ids()
        # check three cases where we should call download_xml again:
        for change in ['event', 'sta_none', 'sta_time']:
            with db_engine.begin() as conn:
                with conn.begin_nested():
                    try:
                        if change == 'event':
                            conn.execute(delete(QuakeML).where(
                                QuakeML.id == ev_ids[0]
                            ))
                        elif change == 'sta_none':
                            conn.execute(update(StationXML).where(
                                StationXML.id == list(cha_ids.values())[0]
                            ).values({'data': None}))
                        else:
                            past = datetime.fromisoformat(
                                '1900-01-01T00:00:00Z'
                            ).replace(tzinfo=None)
                            conn.execute(update(StationXML).where(
                                StationXML.id == list(cha_ids.values())[0]
                            ).values({
                                'last_updated':past
                            }))

                    except IntegrityError as e:
                        raise

            # same as before, we check that nothing is downloaded again:
            mock_download_segments_read_urls.reset_mock()
            mock_download_xml.reset_mock()
            result = CliRunner().invoke(cli, cli_options)
            assert result.exit_code == 0
            assert count == count_from_db(db_engine)  # nothing changed on DB
            assert _nothing_to_download_msg in result.output
            assert not mock_download_segments_read_urls.called  # NOTE: NOT CALLED!
            assert mock_download_xml.called

            # recompute cha ids and quakeml ids:
            ev_ids2, cha_ids2 = get_ids()
            assert sorted(cha_ids2) == sorted(cha_ids)
            assert sorted(ev_ids2) == sorted(ev_ids)


@patch("stream2segment.download.segments.build_and_read_urls")
def test_real_download_segments_with_credentials(
    mock_download_segments_read_urls,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir
):
    """This segments download test"""
    db_url = str(db_engine.url)
    if is_postgres(db_url):
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    cfg_file = test_data_dir / "download-network-filter.yaml"


    mock_download_segments_read_urls.side_effect = original_read_urls

    # Start tests, try download restricted data with no token
    # actually, data is not restricted anymore, so we should download 1 segment
    mock_download_segments_read_urls.reset_mock()
    cli_options = [
        'download', '-c', str(cfg_file), '--dburl', db_url,
        '--data_url', 'https://www.orfeus-eu.org/fdsnws/dataselect/1/query',
        '--events_url', 'isc',
        # '--minmag', '4', '--maxmag', '5',
        '--start', '2015-01-24T00:00:00',
        '--end', '2015-04-30T23:59:59',
        '--net', 'Z3', '--sta', 'A009A', '--cha', 'HHE',
        '--time_window', '0.1', '0.2',
        '--minlatitude', '45', '--maxlatitude', '52',
        '--minlongitude', '14', '--maxlongitude', '20',
        '--minmag', '3.3', '--maxdepth', '10'
    ]
    result = CliRunner().invoke(cli, cli_options)
    assert result.exit_code == 0
    count = count_from_db(db_engine)
    assert (count.skipped_segment == count.quakeml == 0)
    assert count.channel > 0
    assert count.event > 0
    assert count.segment > 0
    assert count.stationxml > 0
    assert mock_download_segments_read_urls.called

    mock_download_segments_read_urls.reset_mock()
    result = CliRunner().invoke(cli, cli_options)
    assert 'all segments already downloaded' in result.output
    assert 'all segments already downloaded' in log_capture.read_text()
    assert not mock_download_segments_read_urls.called
    assert result.exit_code == 0

    # TEST WITH CREDENTIALS NOW
    # (error downloading user password cause token invalid)
    # mock load config to add the token from data dir:

    # delete segments table otheerwise we do not hit segments download
    with db_engine.begin() as conn:
        conn.execute(Segment.__table__.delete())

    with (patch(load_input_path) as _):
        def load_input(config_file_path: str, **override_params):
            override_params['credentials'] = str(test_data_dir / 'eidatoken')
            return original_load_input(config_file_path, **override_params)

        _.side_effect = load_input

        # run tests:
        result = CliRunner().invoke(cli, cli_options)
        assert result.exit_code == 1
        assert FailedDownload.prefix in result.output
        assert FailedDownload.prefix in log_capture.read_text()
        assert count_from_db(db_engine).segment == 0

        # now mock url read to return a valid user password:
        from stream2segment.download.segments import read_url as original_read_url
        with patch('stream2segment.download.segments.read_url') as mock_read_url:

            def read_url(req, *a, **kw):
                if isinstance(req, Request) and '/auth' in req.full_url:
                    return Response('uS3r:pa55uu0Rd', 200, req.full_url)
                return original_read_url(req, *a, **kw)

            mock_read_url.side_effect = read_url

            # run tests:
            result = CliRunner().invoke(cli, cli_options)
            assert result.exit_code == 0
            # we got a 500 from trying to download the segment with the user password
            # returned by the mock function above:
            assert 'Download unsuccessful' in log_capture.read_text()
            assert count_from_db(db_engine).segment == 0


@patch(download_save_segments_path)
def test_download_iris_caltec_up_to_segments(
    mock_download_save_segments,
    # fixtures:
    online_only, db_engine, log_capture, test_data_dir
):
    """This tess _ADARRAY private network"""
    db_url = str(db_engine.url)
    if is_postgres(db_url):
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    # mock download save segments: raise NothingToDownload to speed up things:
    custom_message = 'custom message!'
    def func_(*a, **kw):
        raise NoSegmentsToDownload(custom_message)
    mock_download_save_segments.side_effect = func_

    # check  the log files in the dir:

    cfg_file = test_data_dir / "download-iris-caltec-500.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db_url,
        ]
    )
    assert result.exit_code == 0
    assert custom_message in log_capture.read_text()
    assert custom_message in result.output


@pytest.mark.skip('Huge download - tested only once in debug mode with pydev')
def test_download_iris_caltec_up_to_segments_skip(
    # fixtures:
    online_only, db, log_capture, test_data_dir
):
    """"""
    db_url = str(db_engine.url)
    if is_postgres(db_url):
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    # check  the log files in the dir:

    cfg_file = test_data_dir / "download-iris-caltec-500.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db_url,
            # '--events_url', 'https://service.ncedc.org/fdsnws/event/1/query'
        ]
    )
    assert result.exit_code == 1
    asd = 9


def test_download_real_db(db_engine, tmp_path, test_data_dir, log_capture):
    
    db_url = str(db_engine.url)
    cfg_file = tmp_path / "download-iris-caltec-500.yaml"
    shutil.copyfile(test_data_dir / cfg_file.name, cfg_file)

    curr_dir = os.getcwd()
    os.chdir(str(tmp_path))

    # now we have sqlite (to file) + anything else passed as option in the cli
    try:
        if is_sqlite(db_url, in_memory=True):  # skip in mem sqlite
            pytest.skip('Im memory sqlite')

        result = CliRunner().invoke(
            cli, [
                'download', '-c', str(cfg_file),
                '--dburl', db_url,
                '--start', "2019-07-06T00:00:00",
                '--end', "2019-07-07T00:00:00",
                '--sta', "VOC,VOB,25282",
                '--cha', 'HNZ'
            ]
        )
        assert result.exit_code == 0

    finally:
        os.chdir(curr_dir)



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
