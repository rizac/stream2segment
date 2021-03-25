# -*- coding: utf-8 -*-
"""
Core functions and classes for the download routine

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

# make the following(s) behave like python3 counterparts if running from
# python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
import sys
import time
from builtins import next, range

import os
import logging

import psutil
from future.utils import string_types

from stream2segment.download import db as ddb
from stream2segment.io.log import logfilepath, close_logger, LevelFilter, elapsed_time
from stream2segment.io import yaml_safe_dump
from stream2segment.io.db.pdsql import dbquery2df
from stream2segment.io.db import secure_dburl, close_session
from stream2segment.io.inputvalidation import load_config_for_download, pop_param
from stream2segment.download.db import Download
from stream2segment.download.exc import NothingToDownload, FailedDownload
from stream2segment.download.modules.events import get_events_df
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df, chaid2mseedid_dict
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import prepare_for_download,\
    download_save_segments, DcDataselectManager
from stream2segment.download.modules.stations import (save_inventories,
                                                      query4inventorydownload)



# make the logger refer to the root of this package ('stream2segment.download')
# This is necessary to make all loggers oif this package refer to the one below:
logger = logging.getLogger(__name__[:__name__.rfind('.')])


def download(config, log2file=True, verbose=False, **param_overrides):
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
    :param param_overrides: additional parameter(s) for the YAML `config`. The
        value of existing config parameters will be overwritten, e.g. if
        `config` is {'a': 1} and `param_overrides` is `a=2`, the result is
        {'a': 2}. Note however that when both parameters are dictionaries, the
        result will be merged. E.g. if `config` is {'a': {'b': 1, 'c': 1}} and
        `param_overrides` is `a={'c': 2, 'd': 2}`, the result is
        {'a': {'b': 1, 'c': 2, 'd': 2}}
    """
    # Implementation details: this function can:
    # - raise, in case of an error usually a user/code error (e.g., bad input
    #   param)
    # - return 1 in case of FailedDownload, e.g. an error independent from the
    #   user (no internet connection, bad data received)
    # - return 0 otherwise (meaning: success). This includes the case where,
    #   acording to our config, there are not segments to download

    # short check (this should just raise, so execute this before configuring loggers):
    isfile = isinstance(config, string_types) and os.path.isfile(config)
    if not isfile and log2file is True:
        raise ValueError('`log2file` can be True only if `config` is a '
                         'string denoting an existing file')

    # Validate params converting them in dict of args for the download function. Also in
    # this case do it before configuring loggers, we simply need to raise `BadParam`s in
    # case of problems:
    d_kwargs, session, authorizer, tt_table = \
        load_config_for_download(config, True, **param_overrides)

    ret = 0
    noexc_occurred = True
    db_streamer = None  # handler logging to db upon successful completion
    download_id = None
    try:
        real_yaml_dict = load_config_for_download(config, False, **param_overrides)
        if verbose:
            print("%s\n" % _pretty_printed_str(real_yaml_dict))

        # configure logger and handlers:
        if log2file is True:
            log2file = logfilepath(config)  # auto create log file
        else:
            log2file = log2file or ''  # assure we have a string
        db_streamer, _ = configlog4download(logger, log2file, verbose)

        # create download row with unprocessed config (yaml_load function)
        # Note that we call again load_config with parseargs=False:
        download_id = new_db_download(session, real_yaml_dict)
        if log2file and verbose:  # (=> loghandlers not empty)
            print("Log file: '%s'"
                  "\n(if the download ends with no errors, the file will be "
                  "deleted\nand its content written "
                  "to the table '%s', column '%s')" % (log2file,
                                                       Download.__tablename__,
                                                       Download.log.key))

        stime = time.time()
        _run(download_id=download_id, isterminal=verbose,
             authorizer=authorizer, session=session, tt_table=tt_table,
             **d_kwargs)
        logger.info("Completed in %s", str(elapsed_time(stime)))
        if log2file:
            errs, warns = db_streamer.errors, db_streamer.warnings
            logger.info("%d error%s, %d warning%s", errs,
                        '' if errs == 1 else 's', warns,
                        '' if warns == 1 else 's')
    except FailedDownload as fdwnld:
        # we logged the exception in `run_download`, just set ret=1:
        ret = 1
    except KeyboardInterrupt:
        # https://stackoverflow.com/q/5191830
        logger.critical("Aborted by user")
        raise
    except:  # @IgnorePep8 pylint: disable=broad-except
        # log the (last) exception traceback and raise
        noexc_occurred = False
        # https://stackoverflow.com/q/5191830
        logger.critical("Download aborted", exc_info=True)
        raise
    finally:
        if session is not None:
            close_session(session)  # help gc?
            # write log to db if default handlers are provided:
            if log2file and db_streamer is not None and download_id is not None:
                # remove file if no exceptions occurred:
                db_streamer.finalize(session, download_id, removefile=noexc_occurred)
                # the method above closes the logger, let's remove it manually
                # before calling closelogger below to avoid closing
                # loghandlers[0] twice:
                # logger.removeHandler(loghandlers[0])
            close_session(session, True)  # engine disposal
        close_logger(logger)

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
        "Input parameters",
        "----------------",
        yaml_safe_dump(dict(tmp_cfg_pre)),
        yaml_safe_dump(tmp_cfg),
        yaml_safe_dump(dict(tmp_cfg_post)),
    ]).strip()


def _run(session, download_id, eventws, starttime, endtime, dataws,
         eventws_params, network, station, location, channel, min_sample_rate,
         search_radius, update_metadata, inventory, timespan,
         retry_seg_not_found, retry_url_err, retry_mseed_err, retry_client_err,
         retry_server_err, retry_timespan_err, tt_table, advanced_settings,
         authorizer, isterminal=False):
    """Download waveforms related to events to a specific path.

    :raise: :class:`FailedDownload` exceptions
    """
    # NOTE: Any function here EXPECTS THEIR DATAFRAME INPUT TO BE NON-EMPTY.
    # Thus, any function returning a dataframe is responsible to return well
    # formed (non empty) data frames: if it would not be the case, the function
    # should raise either:
    # 1) a NothingToDownload to stop the routine silently (log to info) and
    #    proceed to the inventories (if set),
    # 2) a FailedDownload to stop the download immediately and raise the
    #    exception

    dbbufsize = advanced_settings['db_buf_size']
    max_thread_workers = advanced_settings['max_concurrent_downloads']
    download_blocksize = advanced_settings['download_blocksize']

    update_md_only = update_metadata == 'only'
    if update_md_only:
        update_metadata = True

    process = psutil.Process(os.getpid()) if isterminal else None
    # calculate steps (note that booleans work, e.g: 8 - True == 7):
    __steps = 3 if update_md_only else \
        (6 + inventory + (True if authorizer.token else False))
    stepiter = iter(range(1, __steps+1))

    # custom function for logging.info different steps:
    def stepinfo(text, *args, **kwargs):
        step = next(stepiter)
        logger.info("\nSTEP %d of %d: {}".format(text), step, __steps, *args,
                    **kwargs)
        if process is not None:
            percent = process.memory_percent()
            logger.warning("(%.1f%% memory used)", percent)

    try:
        if not update_md_only:

            stepinfo("Fetching events")
            events_df = get_events_df(session, eventws, eventws_params,
                                      starttime, endtime, dbbufsize,
                                      advanced_settings['e_timeout'], isterminal)

        # Get datacenters, store them in the db, returns the dc instances
        # (db rows) correctly added
        stepinfo("Fetching data-centers")
        # get dacatanters (might raise FailedDownload):
        datacenters_df, eidavalidator = \
            get_datacenters_df(session, dataws,
                               advanced_settings['routing_service_url'],
                               network, station, location, channel, starttime,
                               endtime, dbbufsize)

        stepinfo("Fetching stations and channels from %d data-center%s",
                 len(datacenters_df), "" if len(datacenters_df) == 1 else "s")
        # get datacenters (might raise FailedDownload):
        channels_df = get_channels_df(session, datacenters_df, eidavalidator,
                                      network, station, location, channel,
                                      starttime, endtime,
                                      min_sample_rate, update_metadata,
                                      max_thread_workers,
                                      advanced_settings['s_timeout'],
                                      download_blocksize,
                                      dbbufsize, isterminal)

        if not update_md_only:
            # get channel id to mseed id dict and purge channels_df. The dict
            # will be used to download the segments later, but we use it now to
            # drop unnecessary columns and save space (and time)
            chaid2mseedid = chaid2mseedid_dict(channels_df,
                                               drop_mseedid_columns=True)

            stepinfo("Selecting stations within search area from %d events",
                     len(events_df))
            # merge vents and stations (might raise FailedDownload):
            segments_df = merge_events_stations(events_df, channels_df,
                                                search_radius, tt_table,
                                                isterminal)
            # help gc by deleting the (only) refs to unused dataframes
            del events_df
            del channels_df

            if authorizer.token:
                stepinfo("Acquiring credentials from token in order to "
                         "download restricted data")
            dc_dataselect_manager = DcDataselectManager(datacenters_df,
                                                        authorizer, isterminal)

            stepinfo("%d segments found. Checking already downloaded segments",
                     len(segments_df))
            # raises NothingToDownload
            segments_df, request_timebounds_need_update = \
                prepare_for_download(session, segments_df,
                                     dc_dataselect_manager, timespan,
                                     retry_seg_not_found, retry_url_err,
                                     retry_mseed_err, retry_client_err,
                                     retry_server_err, retry_timespan_err,
                                     retry_timespan_warn=False)

            # prepare_for_download raises a NothingToDownload if there is no
            # data, so if we are here segments_df is not empty
            stepinfo("Downloading %d segments %sand saving to db", len(segments_df),
                     '(open data only) ' if dc_dataselect_manager.opendataonly else '')
            # frees memory. Although maybe unnecessary, let's do our best to
            # free stuff cause the next one is memory consuming:
            # https://stackoverflow.com/a/30022294/3526777
            session.expunge_all()
            session.close()

            d_stats = download_save_segments(session, segments_df,
                                             dc_dataselect_manager,
                                             chaid2mseedid, download_id,
                                             update_metadata,
                                             request_timebounds_need_update,
                                             max_thread_workers,
                                             advanced_settings['w_timeout'],
                                             download_blocksize,
                                             dbbufsize,
                                             isterminal)
            del segments_df  # help gc?
            session.close()  # frees memory?
            logger.info("")
            logger.info(("** Segments download summary **\n"
                         "Number of segments per data center url (row) and response "
                         "type (column):\n%s") %
                        str(d_stats) or "Nothing to show")

    except NothingToDownload as ntdexc:
        # we are here if some function raised a NothingToDownload (e.g., in
        # prepare_for_download there is nothing according to current config).
        # Print message as info, not that inventory might be downloaded (see
        # finally clause below)
        logger.info(str(ntdexc))
        # comment out: DO NOT RAISE:
        # raise
    except FailedDownload as dexc:
        # We are here if we raised a FailedDownload. Same behaviour as
        # NothingToDownload, except we log an error message, and we prevent
        # downloading inventories by forcing the flag to be false
        inventory = False
        logger.error(dexc)
        raise
    except:  # @IgnorePep8
        inventory = False
        raise
    finally:
        if inventory:
            # frees memory. Although maybe unecessary, let's do our best to
            # free stuff cause the next one might be memory consuming:
            # https://stackoverflow.com/a/30022294/3526777
            session.expunge_all()
            session.close()

            # query station id, network station, datacenter_url
            # for those stations with empty inventory_xml
            # AND at least one segment non empty/null
            # Download inventories for those stations only
            sta_df = dbquery2df(query4inventorydownload(session, update_metadata))
            if sta_df.empty:
                stepinfo("Skipping: No station inventory to download")
            else:
                stepinfo("Downloading %d station inventories", len(sta_df))
                n_downloaded, n_empty, n_errors = \
                    save_inventories(session, sta_df,
                                     max_thread_workers,
                                     advanced_settings['i_timeout'],
                                     download_blocksize,
                                     dbbufsize, isterminal)
                logger.info(("** Station inventories download summary **\n"
                             "- downloaded     %7d \n"
                             "- discarded      %7d (empty response)\n"
                             "- not downloaded %7d (client/server errors)"),
                            n_downloaded, n_empty, n_errors)


def new_db_download(session, params=None):
    if params is None:
        params = {}
    # print local vars: use safe_dump to avoid python types. See:
    # http://stackoverflow.com/questions/1950306/pyyaml-dumping-without-tags
    download_inst = Download(config=tounicode(yaml_safe_dump(params)),
                             # log by default shows error. If everything works
                             # fine, we replace the content later
                             log=('N/A: either logger not configured, or an '
                                  'unexpected error interrupted the process'),
                             program_version=version())

    session.add(download_inst)
    session.commit()
    download_id = download_inst.id
    session.close()  # frees memory?
    return download_id


def tounicode(string, decoding='utf-8'):
    """Convert string to 'text' (unicode in python2, str in Python3). Function
    Python 2-3 compatible. If string is already a 'text' type, returns it

    :param string: a `str`, 'bytes' or (in py2) 'unicode' object.
    :param decoding: the decoding used if `string` has to be converted to text.
        Defaults to 'utf-8' when missing
    :return: the text (`str` in python3, `unicode` string in Python2)
        representing `string`
    """
    # Curiously, future.utils has no such a simple method. So instead of
    # checking when string is text, let's check when it is NOT, i.e. when it's
    # instance of bytes (str in py2 is instanceof bytes):
    return string.decode(decoding) if isinstance(string, bytes) else string


def version():
    dir_ = os.path.abspath(os.path.dirname(__file__))
    while True:
        filez = set(os.listdir(dir_))
        if 'setup.py' in filez and 'version' in filez:
            with open(os.path.join(dir_, "version")) as _:
                return _.read().strip()
        _  = os.path.dirname(dir_)
        if _ == dir_:  # root
            raise ValueError('No version file found')
        dir_ = _


def configlog4download(logger, logfile_path='', verbose=False):
    """"Configures the logger, setting it to a `INFO` level with a list of
    default handlers:

    - If `logfile_path` is not the empty str, a :class:`DbStreamHandler`
      (streaming to that file) will capture all INFO, ERROR and WARNING level
      messages, and when its finalize() method is called, flushes the file
      content to the database (deleting the file if needed. This assures that
      if `DbStreamHandler.finalize` is not called, possibly due to an
      exception, the file can be inspected). See :func:`logfilepath` if you
      want to create automatically a log file path in the same directory of a
      given download config file.

    - If `verbose` is True (False by default), a :class:`StreamHandler`
      (streaming to standard output) will capture ONLY messages of level INFO
      (20) and ERROR (40) and CRITICAL (50), ideal for showing relevant
      information to the user on a terminal

    The returned list can thus contain 0, 1 or 2 loggers depending on the
    arguments.

    Implementation detail: this method modifies these values for performance
    reason:
    ```
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0
    ```

    :return: a list of handlers added to the logger
    """
    # https://docs.python.org/2/howto/logging.html#optimization:
    logging._srcfile = None  # pylint: disable=protected-access
    logging.logThreads = 0
    logging.logProcesses = 0

    logger.setLevel(logging.INFO)  # necessary to forward to handlers

    # add handlers:
    db_streamer, sysout_streamer = None, None

    if logfile_path:
        db_streamer = DbStreamHandler(logfile_path)
        logger.addHandler(db_streamer)

    if verbose:
        sysout_streamer = logging.StreamHandler(sys.stdout)
        sysout_streamer.setFormatter(logging.Formatter('%(message)s'))
        # configure the levels we want to print (20: info, 40: error, 50: critical)
        l_filter = LevelFilter((20, 40, 50))
        sysout_streamer.addFilter(l_filter)
        # set minimum level (for safety):
        sysout_streamer.setLevel(min(l_filter.levels))
        logger.addHandler(sysout_streamer)

    return db_streamer, sysout_streamer

    # custom StreamHandler: count errors and warnings:
    # handlers = []
    # if logfile_path:
    #     handlers.append(DbStreamHandler(logfile_path))
    # if verbose:
    #     handlers.append(SysOutStreamHandler(sys.stdout))
    # for hand in handlers:
    #     logger.addHandler(hand)
    # return handlers


class DbStreamHandler(logging.FileHandler):
    """A `logging.FileHandler` which counts errors and warnings. See
    https://stackoverflow.com/q/812477. This class takes in the constructor an
    id of the table 'downloads' (referring to the current download), and when
    closed writes the content of the file to the database, deleting the
    handler's file. You should always explicitly call close() to assure the log
    is written to the database**. For an example using SQL-Alchemy log rows
    (slightly different case but informative) see:
    http://docs.pylonsproject.org/projects/pyramid_cookbook/en/latest/logging/sqlalchemy_logger.html
    """
    def __init__(self, filepath, min_level=20):
        """
        Initialize a DbStreamHandler

        :param min_level: this handlers level
        (https://docs.python.org/3/library/logging.html#logging.Handler.setLevel)
        """
        # w+: allows to read without closing first:
        super(DbStreamHandler, self).__init__(filepath, mode='w+')
        # access the stream with self.stream
        self.errors = 0
        self.warnings = 0
        self.criticals = 0  # one should be enough
        # configure level and formatter
        self.setLevel(min_level)
        self.setFormatter(logging.Formatter('[%(levelname)s]  %(message)s'))

    def emit(self, record):
        if record.levelno == 30:
            self.warnings += 1
        elif record.levelno == 40:
            self.errors += 1
        elif record.levelno == 50:
            self.criticals += 1
        super(DbStreamHandler, self).emit(record)
        # (superclass logging.FileHandler flushes every emit)

    def finalize(self, session, download_id, removefile=True):
        """Write to db, closes this handler
        and optionally removes the underlying file"""
        # the super-class sets the stream to None when closing, so we might
        # check this to see if we closed it already:
        if self.stream is None:
            return
        # we experienced the NoneType error which we could not test
        # deterministically so the if above serves to this, especially because
        # we know self.stream == None => already closed

        super(DbStreamHandler, self).flush()  # for safety
        self.stream.seek(0)  # offset of 0
        logcontent = self.stream.read()   # read again
        try:
            super(DbStreamHandler, self).close()
        except:
            pass
        if removefile:
            try:
                os.remove(self.baseFilename)
            except:
                pass
        Download = ddb.Download
        session.query(Download).filter(Download.id == download_id).\
            update({Download.log.key: logcontent,
                    Download.errors.key: self.errors,
                    Download.warnings.key: self.warnings})
        session.commit()
