"""
Core download routine
"""
import sys
import json
import time
from datetime import datetime, UTC, timedelta
import os
import logging

import psutil
from sqlalchemy import Engine

from stream2segment.io.log import LevelFilter
from stream2segment.io.db.models import DownloadRun
from stream2segment.io.db.pdsql import (
    get_col_max, execute_sql, create_insert_statement, create_update_statement
)
from stream2segment.io.log import close_logger
from stream2segment.io import yaml_safe_dump
from stream2segment.io.db import secure_dburl, models
from stream2segment.download.inputvalidation import load_config_for_download, pop_param
from stream2segment.download.exc import NothingToDownload, FailedDownload
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
    config, log2file=True, verbose=False, print_config_only=False, **param_overrides
):
    """Start an event-based download routine, fetching segment data and
    metadata from FDSN web services and saving it in an SQL database

    :param config: str or dict: If str, it is valid path to a configuration
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
        showing the estimated remaining time for each sub task. This option is
        set to True when this function is invoked from the command line
        interface (`cli.py`)
    :param print_config_only: boolean (default False). Set to True to test
        the configuration only (no download): merge input and overridden parameters,
        validate them, print the final configuration in YAML syntax on the screen,
        and then simply return skipping the download routine
    :param param_overrides: additional parameter(s) for the YAML `config`. The
        value of existing config parameters will be overwritten, e.g. if
        `config` is {'a': 1} and `param_overrides` is `a=2`, the result is
        {'a': 2}. Note however that when both parameters are dictionaries, the
        result will be merged. E.g. if `config` is {'a': {'b': 1, 'c': 1}} and
        `param_overrides` is `a={'c': 2, 'd': 2}`, the result is
        {'a': {'b': 1, 'c': 2, 'd': 2}}
    """

    # short check (this should just raise, so execute this before configuring loggers):
    isfile = isinstance(config, str) and os.path.isfile(config)
    if not isfile and log2file is True:
        raise ValueError('`log2file` can be True only if `config` is a '
                         'string denoting an existing file')

    # Validate params converting them in dict of args for the download function. Also in
    # this case do it before configuring loggers, we simply need to raise `BadParam`s in
    # case of problems:
    d_kwargs, session, authorizer = load_config_for_download(
        config, True, **param_overrides
    )

    engine = session.get_bind()  # FIXME remove session!
    ret = 0
    download_id = None

    # configure logger and handlers:
    if log2file is True:
        _now = datetime.now(UTC).replace(microsecond=0).isoformat('T')
        log_file_path = f'{config}.{_now}.log'
    else:
        log_file_path = log2file or ''  # assure we have a string

    try:
        # Download a YAML dict for printing to the screen and saving to the db (No need
        # to validate parameters again)
        real_yaml_dict = load_config_for_download(config, False, **param_overrides)
        # Still, some parameters should be printed/saved as validated. E.g. starttime
        # and endtime might be integers. In case, they need to be saved as date times
        # otherwise the config depends on when it is launched
        for keys in [['starttime', 'start'], ['endtime', 'end']]:
            key = [k for k in keys if k in real_yaml_dict][0]
            real_yaml_dict[key] = d_kwargs[keys[0]]

        if verbose or print_config_only:
            print("%s\n" % _pretty_printed_str(real_yaml_dict))

        if print_config_only:
            return ret

        configure_logging(log_file_path, verbose)
        download_id = new_download_run(engine, real_yaml_dict)

        if log_file_path and verbose:
            print(f"Log file: '{log_file_path}'\n"
                  "(if the download ends with no errors, the file will be deleted"
                  "and its content written to the db table "
                  f"'{models.DownloadRun.__tablename__}')")

        stime = time.time()
        d_stats = _download(
            isterminal=verbose,
            authorizer=authorizer,
            engine=engine,
            **d_kwargs
        )
        logger.info(f"Completed in {timedelta(seconds=round((time.time()) - stime))}")
        _write_download_summary(engine, download_id, d_stats)

    except NothingToDownload as nothing_to_download_exc:
        logger.info(nothing_to_download_exc)
    except FailedDownload as failed_download_exc:
        logger.error(failed_download_exc)
        ret = 1
    except KeyboardInterrupt:
        # https://stackoverflow.com/q/5191830
        logger.critical("Aborted by user")
        raise
    except:  # noqa
        # https://stackoverflow.com/q/5191830
        logger.critical("Download aborted", exc_info=True)
        raise
    finally:
        close_logger(logger)
        _write_download_log(engine, download_id, log_file_path, rm_file=True)
    return ret


def _pretty_printed_str(yaml_dict):
    """Return a pretty printed string from yaml_dict"""
    # print yaml_dict to terminal if needed. Unfortunately we need a bit of
    # workaround just to print relevant params first (YAML sorts by key)
    tmp_cfg = dict(yaml_dict)
    # provide sorting in the printed yaml by splitting into subdicts:
    dburl_name, dburl_val = pop_param(tmp_cfg, 'dburl')
    dburl_val = secure_dburl(dburl_val)  # hide passwords
    tmp_cfg_pre = [(dburl_name, dburl_val),
                   pop_param(tmp_cfg, ('starttime', 'start')),
                   pop_param(tmp_cfg, ('endtime', 'end'))]
    tmp_cfg_post = [pop_param(tmp_cfg, 'advanced_settings', {})]
    return "\n".join(_.strip() for _ in [
        "####################",
        "# Input parameters #",
        "####################",
        yaml_safe_dump(dict(tmp_cfg_pre)),
        yaml_safe_dump(tmp_cfg),
        yaml_safe_dump(dict(tmp_cfg_post)),
    ]).strip()


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


def new_download_run(engine, params=None) -> int:
    if params is None:
        params = {}
    config = yaml_safe_dump(params)
    if isinstance(config, bytes):
        config = config.decode('utf-8')  # legacy py2 code?
    tmp_log = ('N/A: either logger not configured, or an '
               'unexpected error interrupted the process')
    download_id = get_col_max(engine, models.DownloadRun.id) + 1
    rows = list(
        execute_sql(
            engine,
            [create_insert_statement(models.DownloadRun)],
            [dict(
                id=download_id,
                time=datetime.now(UTC).replace(tzinfo=None),
                config=config,
                log=tmp_log,
                summary="",
                s2s_version=version()
            )]
        )
    )
    if len(rows) != 1:
        raise FailedDownload('Unable to write to the DB')
    return download_id


def version():
    with open(get_resource_abspath('program_version')) as _:
        return _.read().strip()


def _download(engine: Engine, events_url, starttime, endtime, data_url,
         events_extra_params, network, station, location, channel, min_sample_rate,
         search_radius, stationxml, quakeml, time_window,
         advanced_settings, authorizer, isterminal=False):
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
        events_extra_params,
        starttime,
        endtime,
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
        starttime,
        endtime,
        min_sample_rate,
        advanced_settings['routing_service_url'],
        authorizer is not None,
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
    # help gc by deleting the (only) refs to unused dataframes
    del events
    del channels

    log_step_header(
        f"{len(segments):,} segments found. Checking already downloaded segments", 4
    )
    # raises NothingToDownload
    segments = prepare_for_download(engine, segments, authorizer is not None)

    # prepare_for_download raises a NothingToDownload if there is no
    # data, so if we are here segments is not empty
    log_step_header(
        f"Downloading {len(segments):,} segments and saving to db" +
        ' (no credentials, open data only)' if authorizer is not None else '', 5
    )

    d_stats = download_and_save(
        engine,
        segments,
        time_window,
        authorizer,
        # download_id,
        # update_metadata,
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
            save_stationxml(engine,
                            max_thread_workers,
                            advanced_settings['i_timeout'],
                            download_blocksize,
                            isterminal)
        logger.info(
            f"** Stations StationXML download summary **\n"
            f"- downloaded     {n_downloaded:,} \n"
            f"- saved          {n_saved:,}\n"
            f"- not downloaded {n_errors:,} (empty data, download error)"
        )
    if quakeml:
        log_step_header("Downloading Events (QuakeML)", 7)
        n_downloaded, n_saved, n_errors = \
            save_quakeml(engine,
                         max_thread_workers,
                         advanced_settings['i_timeout'],
                         download_blocksize,
                         isterminal)
        logger.info(
            f"** Events QuakeML download summary **\n"
            f"- downloaded     {n_downloaded:,} \n"
            f"- saved          {n_saved:,}\n"
            f"- not downloaded {n_errors:,} (empty data, download error)"
        )
    return d_stats


def _write_download_summary(engine: Engine, download_id: int | None, stats: dict):

    if download_id is None:
        return

    execute_sql(
        engine,
        [
            create_update_statement(
                DownloadRun,
                DownloadRun.id.key,
                [DownloadRun.summary.key]
            )
        ],
        {
            DownloadRun.id.key: download_id,
            DownloadRun.summary.key: json.dumps(stats)
        }
    )


def _write_download_log(
    engine: Engine, download_id: int | None, log_file_path: str, rm_file: bool
):
    is_file = log_file_path and os.path.isfile(log_file_path)
    if download_id and is_file:

        with open(log_file_path, 'r') as f:
            execute_sql(
                engine,
                [
                    create_update_statement(
                        DownloadRun,
                        DownloadRun.id.key,
                        [DownloadRun.log.key]
                    )
                ],
                {
                    DownloadRun.id.key: download_id,
                    DownloadRun.log.key: f.read()
                }
            )

    if is_file and rm_file:
        try:
            os.remove(log_file_path)
        except Exception:  # noqa
            pass