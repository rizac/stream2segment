# -*- coding: utf-8 -*-
"""
Main module with all root functions (download, process, ...)

:date: Feb 2, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# make the following(s) behave like python3 counterparts if running from
# python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import round, open, input  # pylint: disable=redefined-builtin

import time
import logging
# import re
# from collections import OrderedDict
import sys
import os
import inspect
import shutil
from datetime import timedelta
from webbrowser import open as open_in_browser
import threading
from tempfile import gettempdir

from future.utils import PY2

import jinja2
# from sqlalchemy.orm import close_all_sessions

from sqlalchemy.sql.expression import func

from stream2segment.io.inputvalidation import load_config_for_process, valid_session, load_config_for_visualization, \
    validate_param
from stream2segment.utils.log import configlog4processing,\
    closelogger, logfilepath
import stream2segment.download.db as dbd
from stream2segment.process.main import run as run_process, run_and_yield, \
    fetch_segments_ids
from stream2segment.io.cli import ascii_decorate
from stream2segment.io.db import secure_dburl
from stream2segment.io import open2writetext
from stream2segment.resources import get_templates_fpaths
from stream2segment.gui.main import create_s2s_show_app, run_in_browser
from stream2segment.gui.dinfo import DReport, DStats
from stream2segment.resources.templates import DOCVARS
from stream2segment.process.writers import get_writer


if PY2:
    # https://stackoverflow.com/a/45946245
    import funcsigs  # @UnresolvedImport pylint: disable=import-error
    SIGNATURE = funcsigs.signature
else:
    SIGNATURE = inspect.signature  # @UndefinedVariable # pylint: disable=no-member

# set root logger if we are executing this module as script, otherwise as
# module name following logger conventions (https://stackoverflow.com/q/30824981)
# However, based on how we configured entry points in config, the name is (as
# november 2016) 'stream2segment.main', which messes up all inheritances. So
# basically setup a main logger with the package name:
logger = logging.getLogger("stream2segment")  # pylint: disable=invalid-name


def process(dburl, pyfile, funcname=None, config=None, outfile=None,
            log2file=False, verbose=False, append=False, **param_overrides):
    """Start a processing routine, fetching segments from the database at the
    given URL and optionally saving the processed data into `outfile`.
    See the doc-strings in stream2segment templates (command `s2s init`) for
    implementing a process module and configuration (arguments `pyfile` and
    `config`).

    :param dburl: str. The URL of the database where data has been previously
        downloaded. It can be the path of the YAML config file used for
        downloading data, in that case the file parameter 'dburl' will be taken
    :param pyfile: str: path to the processing module
    :param funcname: str or None (default: None). The function name in `pyfile`
        to be used (None means: use default name, currently "main")
    :param config: str. Path of the configuration file in YAML syntax
    :param outfile: str or None. The destination file where to write the
        processing output, either ".csv" or ".hdf". If not given, the returned
        values of `funcname` in `pyfile` will be ignored, if given.
    :param log2file: bool or str (default: False). If str, it is the log file
        path (whose directory must exist). If True, the log file path will be
        built as `outfile` + ".[now].log" or (if no output file is given) as
        `pyfile` + ".[now].log" ([now] = current date and time in ISO format).
        If False, logging is disabled.
    :param verbose: if True (default: False) print some log information also on
        the screen (messages of level info and critical), as well as a progress
        bar showing the estimated remaining time. This option is set to True
        when this function is invoked from the command line interface (`cli.py`)
    :param append: bool (default False) ignored if the output file is not given
        or non existing, otherwise: if False, overwrite the existing output
        file. If True, process unprocessed segments only (checking the segment
        id), and append to the given file, without replacing existing data.
    :param param_overrides: additional parameter(s) for the YAML `config`. The
        value of existing config parameters will be overwritten, e.g. if
        `config` is {'a': 1} and `param_overrides` is `a=2`, the result is
        {'a': 2}. Note however that when both parameters are dictionaries, the
        result will be merged. E.g. if `config` is {'a': {'b': 1, 'c': 1}} and
        `param_overrides` is `a={'c': 2, 'd': 2}`, the result is
        {'a': {'b': 1, 'c': 2, 'd': 2}}
    """
    # checks dic values (modify in place) and returns dic value(s) needed here:
    # Outside the try catch below as BadParam might be raised and need to
    # be caught by the caller (see `cli.py`)
    session, pyfunc, funcname, config_dict, segments_selection, multi_process, \
        chunksize = load_config_for_process(dburl, pyfile, funcname,
                                                             config, outfile,
                                                             **param_overrides)

    if log2file is True:
        log2file = logfilepath(outfile or pyfile)  # auto create log file
    else:
        log2file = log2file or ''  # assure we have a string

    try:
        loghandlers = configlog4processing(logger, log2file, verbose)
        abp = os.path.abspath
        info = [
            "Input database:      %s" % secure_dburl(dburl),
            "Processing function: %s:%s" % (abp(pyfile), funcname),
            "Config. file:        %s" % (abp(config) if config else 'n/a'),
            "Log file:            %s" % (abp(log2file) if log2file else 'n/a'),
            "Output file:         %s" % (abp(outfile) if outfile else 'n/a')
        ]
        logger.info(ascii_decorate("\n".join(info)))

        stime = time.time()
        writer_options = config_dict.get('advanced_settings', {}).\
            get('writer_options', {})
        run_process(session, pyfunc, get_writer(outfile, append, writer_options),
                    config_dict, segments_selection, verbose, multi_process,
                    chunksize, None)
        logger.info("Completed in %s", str(elapsedtime(stime)))
        return 0  # contrarily to download, an exception should always raise
        # and log as error with the stack trace
        # (this includes pymodule exceptions e.g. TypeError)
    except KeyboardInterrupt:
        logger.critical("Aborted by user")  # see comment above
        raise
    except:  # @IgnorePep8 pylint: disable=broad-except
        logger.critical("Process aborted", exc_info=True)  # see comment above
        raise
    finally:
        closesession(session, True)
        closelogger(logger)


def s2smap(pyfunc, dburl, segments_selection=None, config=None, *,
           logfile='', show_progress=False, multi_process=False, chunksize=None,
           skip_exceptions=None):
    """Return an iterator that applies the function `pyfunc` to every segment
    found on the database at the URL `dburl`, processing only segments matching
    the given selection (`segments_selection`), yielding the results in the form
    of the tuple:
    ```
        (output:Any, segment_id:int)
    ```
    (where output is the return value of `pyfunc`)

    :param pyfunc: a Python function with signature (= accepting arguments):
        `(segment:Segment, config:dict)`. The first argument is the segment
        object which will be automatically passed from this function
    :param dburl: the database URL. Supported formats are Sqlite and Postgres
        (See https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls).
    :param segments_selection: a dict[str, str] of Segments attributes mapped to
        a given selection expression, e.g.:
        ```
        {
            'event.magnitude': '<=6',
            'channel.channel': 'HHZ',
            'maxgap_numsamples': '(-0.5, 0.5)',
            'has_data': 'true'
        }
        ```
        (the last two keys assure segments with no gaps, and with data.
        'has_data': 'true' is basically a default to be provided in most cases)
    :param config: dict of additional needed arguments to be passed to `pyfunc`,
        usually defining configuration parameters
    :param safe_exceptions: tuple of Python exceptions that will not interrupt the whole
        execution. Instead, they will be logged to file, with the relative segment id.
        If `logfile` (see below) is empty, then safe exceptions will be yielded. In this
        case, the `output` variable can be either the returned value of `pyfunc`, or any
        given safe exception, and the user is responsible to check that
    :param logfile: string. When not empty, it denotes the path of the log file
        where exceptions will be logged, with the relative segment id
    :param show_progress: print progress bar to standard output (usually, the terminal
        window) and estimated remaining time
    :param multi_process: enable multi process (parallel sub-processes) to speed up
        execution. When not boolean, this parameter can be an integer denoting the
        exact number of subprocesses to be allocated (only for advanced users. True is
        fine in most cases)
    :param chunksize: the size, in number of segments, of each chunk of data that will
        be loaded from the database. Increasing this number speeds up the load but also
        increases memory consumption. None (the default) means: set size automatically
    """
    session = validate_param('dburl', dburl, valid_session, for_process=True)
    try:
        loghandlers = configlog4processing(logger, logfile, show_progress)
        stime = time.time()
        yield from run_and_yield(session,
                                 fetch_segments_ids(session, segments_selection),
                                 pyfunc, config, show_progress, multi_process,
                                 chunksize, skip_exceptions)
        logger.info("Completed in %s", str(elapsedtime(stime)))
    except KeyboardInterrupt:
        logger.critical("Aborted by user")  # see comment above
        raise
    except:  # @IgnorePep8 pylint: disable=broad-except
        logger.critical("Process aborted", exc_info=True)  # see comment above
        raise
    finally:
        closesession(session, True)
        closelogger(logger)


def elapsedtime(t0_sec, t1_sec=None):
    """Time elapsed from `t0_sec` until `t1_sec`, as `timedelta` object rounded
    to seconds. If `t1_sec` is None, it will default to `time.time()` (the
    current time since the epoch, in seconds)

    :param t0_sec: (float) the start time in seconds. Usually it is the result
        of a previous call to `time.time()`, before starting a process that
        had to be monitored
    :param t1_sec: (float) the end time in seconds. If None, it defaults to
        `time.time()` (current time since the epoch, in seconds)

    :return: a timedelta object, rounded to seconds
    """
    return timedelta(seconds=round((time.time() if t1_sec is None else t1_sec) - t0_sec))


def show(dburl, pyfile, configfile):
    """Show downloaded data plots in a system browser dynamic web page"""
    session, pymodule, config_dict, segments_selection = \
        load_config_for_visualization(dburl, pyfile, configfile)
    run_in_browser(create_s2s_show_app(session, pymodule, config_dict,
                                       segments_selection))
    return 0


def init(outpath, prompt=True, *filenames):
    """Initialize an output directory writing therein the given template files

    :param prompt: bool (default: True) telling if a prompt message (Python
        `input` function) should be issued to warn the user when overwriting
        files. The user should return a string or integer where '1' means
        'overwrite all files', '2' means 'overwrite only non-existing', and any
        other value will return without copying.
    """
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        if not os.path.isdir(outpath):
            raise Exception("Unable to create '%s'" % outpath)

    if prompt:
        existing_files = [f for f in filenames
                          if os.path.isfile(os.path.join(outpath, f))]
        non_existing_files = [f for f in filenames if f not in existing_files]
        if existing_files:
            suffix = ("Type:\n1: overwrite all files\n2: write only non-existing\n"
                      "0 or any other value: do nothing (exit)")
            msg = ("The following file(s) "
                   "already exist on '%s':\n%s"
                   "\n\n%s") % (outpath, "\n".join([_ for _ in existing_files]), suffix)
            val = input(msg)
            try:
                val = int(val)
                if val == 2:
                    if not non_existing_files:
                        raise ValueError()  # fall back to "exit" case
                    else:
                        filenames = non_existing_files
                elif val != 1:
                    raise ValueError()  # fall back to "exit" case
            except ValueError:
                return []

    srcfilepaths = get_templates_fpaths(*filenames)
    if srcfilepaths:
        basedir = os.path.dirname(srcfilepaths[0])
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(basedir),
                                 keep_trailing_newline=True)
        copied_files = []
        for srcfilepath in srcfilepaths:
            filename = os.path.basename(srcfilepath)
            outfilepath = os.path.join(outpath, filename)
            if os.path.splitext(filename)[1].lower() in ('.yaml', '.py'):
                env.get_template(filename).stream(DOCVARS).dump(outfilepath)
            else:
                shutil.copyfile(srcfilepath, outfilepath)
            copied_files.append(outfilepath)
    return copied_files


def dreport(dburl, download_ids=None, config=True, log=True, html=False,
            outfile=None):
    """Create a diagnostic html page (or text string) showing the status of the
    download. Note that html is not supported for the moment and will raise an
    Exception. (leaving the same signatire as dstats for compatibility and
    easing future implementations of the html page if needed)

    :param config: boolean (True by default)
    :param log: boolean (True by default)
    """
    _get_download_info(DReport(config, log), dburl, download_ids, html, outfile)


def dstats(dburl, download_ids=None, maxgap_threshold=0.5, html=False,
           outfile=None):
    """Create a diagnostic html page (or text string) showing the status of the
    download

    :param maxgap_threshold: the max gap threshold (float)
    """
    _get_download_info(DStats(maxgap_threshold), dburl, download_ids, html,
                       outfile)


def _get_download_info(info_generator, dburl, download_ids=None, html=False,
                       outfile=None):
    """Process dinfo or dstats"""
    session = validate_param('dburl', dburl, valid_session)
    if html:
        openbrowser = False
        if not outfile:
            openbrowser = True
            outfile = os.path.join(gettempdir(), "s2s_%s.html" %
                                   info_generator.__class__.__name__.lower())
        # get_dstats_html returns unicode characters in py2, str in py3,
        # so it is safe to use open like this (cf below):
        with open(outfile, 'w', encoding='utf8', errors='replace') as opn:
            opn.write(info_generator.html(session, download_ids))
        if openbrowser:
            open_in_browser('file://' + outfile)
        threading.Timer(1, lambda: sys.exit(0)).start()
    else:
        itr = info_generator.str_iter(session, download_ids)
        if outfile is not None:
            # itr is an iterator of strings in py2, and str in py3, so open
            # must be input differently (see utils module):
            with open2writetext(outfile, encoding='utf8', errors='replace') as opn:
                for line in itr:
                    line += '\n'
                    opn.write(line)
        else:
            for line in itr:
                print(line)


def ddrop(dburl, download_ids, prompt=True):
    """Drop data from the database by download id(s). Drops also all segments

    :return: None if prompt is True and the user decided not to drop via user
        input, otherwise a dict of deleted download ids mapped to either:
        - an int (the number of segments deleted)
        - an exception (if the download id could not be deleted)
    """
    Download, Segment = dbd.Download, dbd.Segment
    ret = {}
    session = validate_param('dburl', dburl, valid_session)
    try:
        ids = [_[0] for _ in
               session.query(Download.id).filter(Download.id.in_(download_ids))]
        if not ids:
            return ret
        if prompt:
            segs = session.query(func.count(Segment.id)).\
                filter(Segment.download_id.in_(ids)).scalar()
            val = input('Do you want to delete %d download execution(s) '
                        '(id=%s) and the associated %d segment(s) from the '
                        'database [y|n]?' % (len(ids), str(ids), segs))
            if val.lower().strip() != 'y':
                return None

        for did in ids:
            ret[did] = session.query(func.count(Segment.id)).\
                filter(Segment.download_id == did).scalar()
            try:
                session.query(Download).filter(Download.id == did).delete()
                session.commit()
                # be sure about how many segments we deleted:
                ret[did] -= session.query(func.count(Segment.id)).\
                    filter(Segment.download_id == did).scalar()
            except Exception as exc:
                session.rollback()
                ret[did] = exc
        return ret
    finally:
        closesession(session, True)
