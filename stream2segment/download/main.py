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

from stream2segment.io.log import LevelFilter
from stream2segment.io.db.pdsql import (
    get_col_max, execute_sql, create_insert_statement
)
from stream2segment.io.log import close_logger
from stream2segment.io.db import models
from stream2segment.download.inputvalidation import extract_download_args
from stream2segment.download.modules.utils import NothingToDownload, FailedDownload
from stream2segment.download.modules.events import get_events
from stream2segment.download.modules.channels import get_channels
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import (
    prepare_for_download, download_and_save
)
from stream2segment.download.modules.xml import save_stationxml, save_quakeml
from stream2segment.resources import get_resource_abspath


# make the logger refer to the parent of this package (`rfind` below. For info:
# https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__[:__name__.rfind('.')])


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
    log_file_path = ''

    try:
        config, kwargs = extract_download_args(config_file, **override_params)

        engine = kwargs['engine']

        if verbose:
            print(f"Configuration file: {config_file}")
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
        configure_logging(log_file_path, verbose)

        if log_file_path and verbose:
            print(f"Log file: '{log_file_path}'\n"
                  "(if the download ends with no errors, the file will be deleted"
                  "and its content written to the db table "
                  f"'{models.DownloadRun.__tablename__}')")

        stime = time.time()
        d_stats = _download(isterminal=verbose, **kwargs)
        logger.info(f"Completed in {timedelta(seconds=round((time.time()) - stime))}")

    except NothingToDownload as nothing_to_download_exc:
        logger.info(f'Nothing to download: {nothing_to_download_exc}')
    except FailedDownload as failed_download_exc:
        logger.error(f'Download failed: {failed_download_exc}')
        ret = 1
    except KeyboardInterrupt:
        # https://stackoverflow.com/q/5191830
        logger.critical("Download aborted by user")
        raise
    except:  # noqa
        # https://stackoverflow.com/q/5191830
        logger.critical("Download aborted", exc_info=True)
        raise
    finally:
        close_logger(logger)
        if engine is not None:
            engine.dispose()

    save_download_run(engine, config, log_file_path, d_stats)

    try:
        if os.path.isfile(log_file_path):
            os.remove(log_file_path)
    except Exception:
        pass

    return ret


def configure_logging(logfile_path='', verbose=False):
    """
    Configure the logger for download
    """
    # https://docs.python.org/2/howto/logging.html#optimization:  # FIXME really needed?
    logging._srcfile = None  # noqa
    logging.logThreads = 0
    logging.logProcesses = 0

    logger.setLevel(logging.INFO)  # necessary to forward to handlers

    if logfile_path:
        db_streamer = logging.FileHandler(logfile_path, mode='w+')
        db_streamer.setLevel(logging.INFO)  # do not print debug, print others
        db_streamer.setFormatter(logging.Formatter('[%(levelname).1s]  %(message)s'))
        logger.addHandler(db_streamer)

    if verbose:
        sysout_streamer = logging.StreamHandler(sys.stdout)
        sysout_streamer.setFormatter(logging.Formatter('%(message)s'))
        # configure the levels we want to print (20: info, 40: error, 50: critical)
        l_filter = LevelFilter((logging.INFO, logging.ERROR, logging.CRITICAL))
        sysout_streamer.addFilter(l_filter)
        # set minimum level (for safety):
        sysout_streamer.setLevel(min(l_filter.levels))
        logger.addHandler(sysout_streamer)


def _download(
    engine: Engine,
    events_url,
    start: datetime,
    end: datetime,
    data_url,
    event_params,
    network,
    station,
    location,
    channel,
    min_sample_rate,
    search_radius,
    stationxml: bool,
    quakeml: bool,
    time_window,
    advanced_settings,
    credentials: tuple[str, str] | bytes | None,
    isterminal=False
):
    """Download waveforms related to events to a specific path.

    :raise: :class:`FailedDownload` exceptions
    """
    dbbufsize = advanced_settings['db_buf_size']
    max_thread_workers = advanced_settings['max_concurrent_downloads']
    download_blocksize = advanced_settings['download_blocksize']
    tt_table = advanced_settings['traveltimes_model']

    process = psutil.Process(os.getpid()) if isterminal else None

    max_steps = 5 + quakeml + stationxml

    # custom function for logging.info different steps:
    def log_step_header(text, step_num:int):
        logger.info(f"\nSTEP {step_num} of {max_steps}: {text}")
        if process is not None:
            percent = process.memory_percent()
            logger.warning("(%.1f%% memory used)", percent)

    log_step_header("Fetching events", 1)

    events = get_events(
        engine,
        events_url,
        event_params,
        start,
        end,
        True
    )

    # Get datacenters, store them in the db, returns the dc instances
    # (db rows) correctly added
    log_step_header("Fetching channels urls", 2)

    channels = get_channels(
        engine,
        data_url,
        network,
        station,
        location,
        channel,
        start,
        end,
        min_sample_rate,
        advanced_settings['routing_service_url'],
        credentials is not None,
        True
    )

    log_step_header(
        f"Selecting nearby station channels "
        f"for each event via input search parameters",
        3
    )
    # merge vents and stations (might raise FailedDownload):
    segments = merge_events_stations(
        events, channels, search_radius, tt_table, isterminal
    )

    del events  # help gc?
    del channels  # help gc?

    log_step_header(
        f"{len(segments):,} segments found. Checking already downloaded segments",
        4
    )
    # raises NothingToDownload
    segments = prepare_for_download(engine, segments, credentials is not None)

    # prepare_for_download raises a NothingToDownload if there is no
    # data, so if we are here segments is not empty
    log_step_header(
        f"Downloading {len(segments):,} segments and saving to db " +
        '(no credentials, open data only)' if credentials is None else '',
        5
    )

    d_stats = download_and_save(
        engine,
        segments,
        time_window,
        credentials,
        max_thread_workers,
        advanced_settings['w_timeout'],
        download_blocksize,
        dbbufsize,
        isterminal
    )
    del segments  # help gc?
    logger.info("")
    logger.info(("** Segments download summary **\n"
                 "Number of segments per data center url (row) and response "
                 "type (column):\n%s") %
                str(d_stats) or "Nothing to show")

    if stationxml:
        log_step_header("Downloading Stations (StationXML)", 6)
        n_downloaded, n_saved, n_errors = \
            save_stationxml(
                engine,
                max_thread_workers,
                advanced_settings['i_timeout'],
                download_blocksize,
                isterminal
            )
        logger.info(
            f"** Stations StationXML download summary **\n"
            f"- downloaded     {n_downloaded:,} \n"
            f"- saved          {n_saved:,}\n"
            f"- not downloaded {n_errors:,} (empty data, download error)"
        )
    if quakeml:
        log_step_header("Downloading Events (QuakeML)", 7)
        n_downloaded, n_saved, n_errors = \
            save_quakeml(
                engine,
                max_thread_workers,
                advanced_settings['i_timeout'],
                download_blocksize,
                isterminal
            )
        logger.info(
            f"** Events QuakeML download summary **\n"
            f"- downloaded     {n_downloaded:,} \n"
            f"- saved          {n_saved:,}\n"
            f"- not downloaded {n_errors:,} (empty data, download error)"
        )
    return d_stats


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
    _ = list(  # list will consume the iterable `execute_sql` FIXME better?
        execute_sql(
            engine,
            [create_insert_statement(models.DownloadRun)],
            [dict(
                id=download_id,
                time=datetime.now(UTC).replace(tzinfo=None),
                config=config_str,
                log=tmp_log,
                summary=json.dumps(d_stats or {}),
                s2s_version=version
            )]
        )
    )

    return download_id
