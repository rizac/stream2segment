"""
Core download routine
"""
import sys
import json
import time
from datetime import datetime, UTC, timedelta
import os
import logging
from pathlib import Path

import psutil
import yaml
from sqlalchemy import Engine

from stream2segment.io.utils import start_logging, create_log_handlers, BadParam, \
    ascii_decorate
from stream2segment.io.db.pdsql import (get_col_max, insert, update)
from stream2segment.io.db import models, close_engine, secure_dburl
from stream2segment.download.inputvalidation import load_input
from stream2segment.download.utils import NoSegmentsToDownload, FailedDownload
from stream2segment.download.events import get_events
from stream2segment.download.channels import get_channels
from stream2segment.download.stationsearch import merge_events_stations
from stream2segment.download.segments import (
    prepare_for_download, download_and_save
)
from stream2segment.download.xml import save_stationxml, save_quakeml
from stream2segment.resources import get_resource_abspath


# make the logger refer to the parent of this package (`rfind` below. For info:
# https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__[:__name__.rfind('.')])
logger.setLevel(logging.INFO)  # not strictly necessary, but we want to avoid debug


def download(
    config_file: str | Path, log2file=True, verbose=False, **override_params
):
    """
    Start an event-based download routine, fetching segment data and
    metadata from FDSN web services and saving it in an SQL database

    :param config_file: str or Path denoting a configuration
        file in YAML syntax that will be read as `dict` of config. parameters
    :param log2file: bool or str (default: True). If string, it is the path to
        the log file (whose parent directory must exist). If True, `config` can
        not be a `dict` (raise `ValueError` otherwise) and the log file path
        will be built as `config` + ".[now].log" (where [now] = current date
        and time in ISO format). If False, logging is disabled.
        When logging is enabled, the file will be used to catch all warnings,
        errors and critical messages (=Python exceptions): if the download
        routine exits with no exception, the file content is written to the
        database (`Download` table) and the file deleted. Otherwise, the file
        will be left on the system for inspection
    :param verbose: if True (default: False) print some log information also on
        the standard output (usually the screen), as well as progress bars
        showing the estimated remaining time for each sub-task. This option is
        set to True when this function is invoked from the command line
        interface (`cli.py`)
    :param override_params: additional parameter(s) for the YAML `config`. The
        value of existing config parameters will be overwritten, e.g. if
        `config` is {'a': 1} and `param_overrides` is `a=2`, the result is
        {'a': 2}. Note however that when both parameters are dictionaries, the
        result will be merged. E.g. if `config` is {'a': {'b': 1, 'c': 1}} and
        `param_overrides` is `a={'c': 2, 'd': 2}`, the result is
        {'a': {'b': 1, 'c': 2, 'd': 2}}
    """
    ret = 0
    engine = None
    d_stats = {}
    config = {}

    if verbose:
        print(ascii_decorate("Stream2segment download"))
        print(f"\nConfiguration file: {config_file}")
        if override_params:
            print(
                f'(explicitly overwritten parameter(s): '
                f'{", ".join(override_params)})'
            )

    # configure logger and handlers:
    if log2file is True:  # noqa
        _now = datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat('T')
        log_file_path = f'{config_file}.{_now}.log'
    else:
        log_file_path = log2file or ''  # assure we have a string

    if log_file_path and verbose:
        # this is not going to the logger (and not saved to db):
        print(f"\nLog file: '{log_file_path}")
        print("(if the download ends with no errors, the file will be deleted "
              "and its content written to the database)")

    try:
        config, kwargs = load_input(config_file, **override_params)
        if verbose:
            print(f"\nOutput database: {secure_dburl(str(kwargs['engine'].url))}")
    except BadParam as bpar:
        raise bpar from None

    with start_logging(logger, create_log_handlers(log_file_path, verbose)):
        try:
            engine = kwargs['engine']
            stime = time.time()
            d_stats = _download(isterminal=verbose, **kwargs)
            logger.info(
                f"\nCompleted in {timedelta(seconds=round((time.time()) - stime))}"
            )
        except NoSegmentsToDownload as ntd_exc:
            logger.info(str(ntd_exc))
        except FailedDownload as fd_exc:
            logger.error(str(fd_exc))
            ret = 1
        except KeyboardInterrupt as ki:
            logger.critical("Download aborted by user")
            ret = 2
        except:  # noqa
            logger.critical("Download aborted", exc_info=True)
            # by raising, we execute finally but not what's afterward:
            raise
        finally:
            save_download_run(engine, config, log_file_path, d_stats)
            close_engine(engine)

    try:
        if ret == 0 and os.path.isfile(log_file_path):
            os.remove(log_file_path)
    except Exception:
        pass

    return ret


def _download(
    engine: Engine,
    events_url,
    start: datetime,
    end: datetime,
    data_url: str | list[str],
    event_params: dict,
    network: list[str],
    station: list[str],
    location: list[str],
    channel: list[str],
    min_sample_rate: float,
    search_radius,
    stationxml: bool,
    quakeml: bool,
    time_window,
    advanced_settings: dict,
    credentials: tuple[str, str] | bytes | None,
    isterminal=False
) -> dict:
    """Download waveforms related to events to a specific path"""

    max_download_concurrency = advanced_settings['max_concurrent_downloads']
    download_blocksize = advanced_settings['download_blocksize']
    tt_table = advanced_settings['traveltimes_model']
    text_download_timeout = advanced_settings['text_download_timeout']
    data_download_timeout = advanced_settings['data_download_timeout']

    process = psutil.Process(os.getpid()) if isterminal else None

    max_steps = 5 + quakeml + stationxml

    # custom function for logging.info different steps:
    def log_step_header(text, step_num:int):
        # prefix = f'STEP {step_num} of {max_steps}:'
        prefix = ("●" * step_num) + ("○" * (max_steps - step_num))
        logger.info(f"\n{prefix} {text}")
        if process is not None:
            percent = process.memory_percent()
            logger.warning("(%.1f%% memory used)", percent)

    # FIXME in events and channels, warn if i constantly get 4xx (failed download?)
    #  or 5xx (please retry alter?)

    log_step_header("Downloading events", 1)
    events = get_events(
        engine=engine,
        urls=events_url,
        evt_query_args=event_params,
        start=start,
        end=end,
        download_timeout=text_download_timeout,
        event_overlap_tolerance=advanced_settings['event_overlap_tolerance'],
        on_event_conflict=advanced_settings['on_event_conflict'],
        show_progress=isterminal
    )
    logger.info(f'Working with {len(events):,} event(s)')

    # Get datacenters, store them in the db, returns the dc instances
    # (db rows) correctly added
    log_step_header("Downloading station channels", 2)
    channels = get_channels(
        engine=engine,
        datacenter_urls=data_url,
        network=network,
        station=station,
        location=location,
        channel=channel,
        start=start,
        end=end,
        min_sample_rate=min_sample_rate,
        eida_rs_urls=advanced_settings['routing_service_url'],
        restricted_download=credentials is not None,
        download_timeout=text_download_timeout,
        show_progress=isterminal
    )
    logger.info(f'Working with {len(channels):,} channel(s)')

    log_step_header(f"Finding nearby stations for each event",3)
    # merge vents and stations (might raise FailedDownload):
    segments = merge_events_stations(
        events=events,
        channels=channels,
        search_radius=search_radius,
        tttable=tt_table,
        show_progress=isterminal
    )
    logger.info(f'{len(segments):,} potential segment(s) detected')

    del events  # help gc?
    del channels  # help gc?

    log_step_header(f"Checking already downloaded segments",4)

    # logger.warning(
    #                 f"Discarding {already_saved.sum():,} already downloaded segment(s)"
    #             )

    # raises NothingToDownload, but store variable because if we have stationxml or quakeml
    # we want to check those as well and potentially download them (imagine a failed previous
    # attempt, and one wants just to update XMLs)

    segments = prepare_for_download(
        engine=engine, segments=segments, restricted_download=credentials is not None
    )

    if segments.empty:
        if not stationxml and not quakeml:
            raise NoSegmentsToDownload(_nothing_to_download_msg)
        else:
            download_seg_msg = _nothing_to_download_msg
    else:
        download_seg_msg = f"Downloading {len(segments):,} segments"
        if credentials is None:
            download_seg_msg += ' (no credentials, open data only)'

    log_step_header(download_seg_msg, 5)
    download_stats = {}

    if not segments.empty:
        d_stats = download_and_save(
            engine=engine,
            segments=segments,
            time_window=time_window,
            credentials=credentials,
            max_download_concurrency=max_download_concurrency,
            download_timeout=data_download_timeout,
            download_blocksize=download_blocksize,
            show_progress=isterminal
        )
        del segments  # help gc?
        # logger.info("")
        stats_str = str(d_stats) or "Nothing to show"
        logger.info(
            "** Segments (miniSEED) download summary **\n"
            "Number of segments per data center url (row) and download status (column):\n"
            f"{stats_str}"
        )
        download_stats = d_stats.to_dict()

    if stationxml:
        log_step_header("Downloading Stations", 6)
        s_stats = save_stationxml(
            engine=engine,
            max_download_concurrency_per_domain=max_download_concurrency,
            download_timeout=data_download_timeout,
            download_blocksize=download_blocksize,
            show_progress=isterminal
        )
        logger.info(
            f"** Stations (StationXML) download summary **\n{str(s_stats)}"
        )
    if quakeml:
        log_step_header("Downloading Events", 7)
        e_stats = save_quakeml(
            engine=engine,
            max_download_concurrency_per_domain=max_download_concurrency,
            download_timeout=data_download_timeout,
            download_blocksize=download_blocksize,
            show_progress=isterminal
        )
        logger.info(
            f"** Events (QuakeML) download summary **\n{str(e_stats)}"
        )
    return download_stats


# used for testing
_nothing_to_download_msg = 'all segments already downloaded'


def save_download_run(engine, config: dict, log_file_path: str, d_stats: dict) -> int:
    if config is None:
        config = {}
    config_str = yaml.safe_dump(config, default_flow_style=True, sort_keys=False)

    if isinstance(config_str, bytes):
        config_str = config_str.decode('utf-8')  # legacy py2 code?

    try:
        with (open(log_file_path, 'r') as _):
            tmp_log = _ .read()
    except Exception as e:  # noqa
        tmp_log = (
            'Log N/A: either logger not configured, file was corrupted or an '
            'unexpected error interrupted the process'
        )

    try:
        with open(get_resource_abspath('program_version')) as _:
            version =_.read().strip()
    except Exception as e:  # noqa
        version = "N/A"

    download_id = get_col_max(engine, models.DownloadRun.id) + 1
    with engine.begin() as conn:
        conn.execute(
            insert(models.DownloadRun),
            dict(
                id=download_id,
                time=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
                config=config_str,
                log=tmp_log,
                summary=json.dumps(d_stats or {}),
                s2s_version=version
            )
        )

    return download_id
