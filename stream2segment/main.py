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
from builtins import range, round, open, input  # pylint: disable=redefined-builtin

import time
import logging
import re
import sys
import os
import inspect
import shutil
from collections import OrderedDict
from datetime import timedelta
from webbrowser import open as open_in_browser
import threading
from tempfile import gettempdir

from future.utils import PY2, string_types

import jinja2

from sqlalchemy.sql.expression import func

from stream2segment.utils.inputvalidation import load_config_for_process, pop_param,\
    load_config_for_download, valid_session, load_config_for_visualization, \
    validate_param
from stream2segment.utils.log import configlog4download, configlog4processing,\
    closelogger, logfilepath
from stream2segment.io.db.models import Download, Segment
from stream2segment.process.main import run as run_process, run_and_yield, \
    fetch_segments_ids
from stream2segment.download.main import run as run_download, new_db_download
from stream2segment.utils import secure_dburl, strconvert, iterfuncs, \
    open2writetext, ascii_decorate, yaml_safe_dump
from stream2segment.utils.resources import get_templates_dirpath
from stream2segment.gui.main import create_s2s_show_app, run_in_browser
from stream2segment.process import math as s2s_math
from stream2segment.download.utils import FailedDownload
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
    loghandlers = None
    download_id = None
    try:
        real_yaml_dict = load_config_for_download(config, False, **param_overrides)
        if verbose:
            # print yaml_dict to terminal if needed. Unfortunately we need a bit of
            # workaround just to print relevant params first (YAML sorts by key)
            tmp_cfg = dict(real_yaml_dict)
            # provide sorting in the printed yaml by splitting into subdicts:
            tmp_cfg_pre = [list(pop_param(tmp_cfg, 'dburl')),  # needs to be mutable
                           pop_param(tmp_cfg, ('starttime', 'start')),
                           pop_param(tmp_cfg, ('endtime', 'end'))]
            tmp_cfg_pre[0][1] = secure_dburl(tmp_cfg_pre[0][1])
            tmp_cfg_post = [pop_param(tmp_cfg, 'advanced_settings', {})]
            _pretty_printed_yaml = "\n".join(_.strip() for _ in [
                "Input parameters",
                "----------------",
                yaml_safe_dump(dict(tmp_cfg_pre)),
                yaml_safe_dump(tmp_cfg),
                yaml_safe_dump(dict(tmp_cfg_post)),
            ])
            print("%s\n" % _pretty_printed_yaml.strip())

        # configure logger and handlers:
        if log2file is True:
            log2file = logfilepath(config)  # auto create log file
        else:
            log2file = log2file or ''  # assure we have a string
        loghandlers = configlog4download(logger, log2file, verbose)

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
        run_download(download_id=download_id, isterminal=verbose,
                     authorizer=authorizer, session=session, tt_table=tt_table,
                     **d_kwargs)
        logger.info("Completed in %s", str(elapsedtime(stime)))
        if log2file:
            errs, warns = loghandlers[0].errors, loghandlers[0].warnings
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
            closesession(session)
            # write log to db if default handlers are provided:
            if log2file and loghandlers is not None and download_id is not None:
                # remove file if no exceptions occurred:
                loghandlers[0].finalize(session, download_id,
                                        removefile=noexc_occurred)
                # the method above closes the logger, let's remove it manually
                # before calling closelogger below to avoid closing
                # loghandlers[0] twice:
                logger.removeHandler(loghandlers[0])

        closelogger(logger)

    return ret


# def _to_pretty_str(yaml_dict, unparsed_yaml_dict):
#     """Return a pretty printed string from yaml_dict
#
#     :param yaml_dict: the PARSED yaml as dict. It might contain variables, such as
#         `session`, not in the original yaml file (these variables are not returned
#         in this function)
#     :param unparsed_yaml_dict: the UNPARSED yaml dict.
#     """
#
#     # the idea is: get the param value from yaml_dict, if not present get it from
#     # unparsed_yaml_dict. Use this list of params so we can control order
#     # (this works only in PyYaml>=5.1 and Python 3.6+, otherwise yaml_dict
#     # keys order can not be known):
#     params = ['dburl', 'starttime', 'endtime', 'eventws',
#               # These variables are merged into eventws_params in yaml_dict,
#               # so do not show them:
#               # 'minlatitude', 'maxlatitude', 'minlongitude', 'maxlongitude',
#               # 'mindepth', 'maxdepth', 'minmagnitude', 'maxmagnitude',
#               'eventws_params', 'channel', 'network', 'station', 'location',
#               'min_sample_rate', 'update_metadata', 'inventory',
#               'search_radius', 'dataws', 'traveltimes_model', 'timespan',
#               'restricted_data', 'retry_seg_not_found', 'retry_url_err',
#               'retry_mseed_err', 'retry_client_err', 'retry_server_err',
#               'retry_timespan_err', 'advanced_settings']
#
#     newdic = {}
#     for k in params:  # add yaml_dic[k] or unparsed_yaml_dict[k]:
#         if k in yaml_dict:
#             newdic[k] = yaml_dict[k]
#         elif k in unparsed_yaml_dict:
#             newdic[k] = unparsed_yaml_dict[k]
#     newdic['dburl'] = secure_dburl(newdic['dburl'])  # don't show passowrd, if present
#     ret = [
#         "Parsed input parameters",
#         "-----------------------",
#         yaml_safe_dump(newdic)
#     ]
#     return "%s\n" % ('\n'.join(ret)).strip()


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
    # implementation details: this function returns 0 on success and raises
    # otherwise.
    # First, it can raise Exceptions for a bad parameter (checked before
    # starting db session and logger),
    # Then, during processing, each segment ValueError is logged as warning
    # and the program continues. Other exceptions are raised, caught here and
    # logged with level CRITICAL, with the stack trace: this allows to help
    # users to discovers possible bugs in pyfile, without waiting for the whole
    # process to finish

    # checks dic values (modify in place) and returns dic value(s) needed here:
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
        closesession(session)
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
        closesession(session)
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


def closesession(session):
    """Close the SQL-Alchemy session. This method simply calls
    `session.close()`, passing all exceptions, if any. Useful for unit testing
    and mock
    """
    try:
        session.close()
    except:
        pass


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
    templates_dir = get_templates_dirpath()
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

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(templates_dir),
                             keep_trailing_newline=True)
    copied_files = []
    for filename in filenames:
        outfilepath = os.path.join(outpath, filename)
        if os.path.splitext(filename)[1].lower() in ('.yaml', '.py'):
            env.get_template(filename).stream(DOCVARS).dump(outfilepath)
        else:
            shutil.copyfile(os.path.join(templates_dir, filename), outfilepath)
        copied_files.append(outfilepath)
    return copied_files


def helpmathiter(type, filter):  # noqa
    """Iterator yielding the doc-string of
    :module:`stream2segment.process.math.ndarrays` or
    :module:`stream2segment.process.math.traces`

    :param type: select the module: 'numpy' for doc of
        :module:`stream2segment.process.math.ndarrays`, 'obspy' for the doc of
        :module:`stream2segment.process.math.traces`, 'all' for both
    :param filter: a filter (with wildcard expressions allowed) to filter by
        function name

    :return: doc-string for all matching functions and classes
    """
    if type == 'numpy':
        itr = [s2s_math.ndarrays]
    elif type == 'obspy':
        itr = [s2s_math.traces]
    else:
        itr = [s2s_math.ndarrays, s2s_math.traces]

    reg = re.compile(strconvert.wild2re(filter))
    _indent = "   "

    def render(string, indent_num=0):
        """Render a string stripping newlines at beginning and end and with the
        intended indent number"""
        if not indent_num:
            return string
        indent = _indent.join('' for _ in range(indent_num+1))
        return '\n'.join("%s%s" % (indent, s) for s in
                         string.replace('\r\n', '\n').split('\n'))

    for pymodule in itr:
        module_doc_printed = False
        for func in iterfuncs(pymodule, False):
            if func.__name__[0] != '_' and reg.search(func.__name__):
                if not module_doc_printed:
                    modname = pymodule.__name__
                    yield "=" * len(modname)
                    yield modname
                    yield "=" * len(modname)
                    yield pymodule.__doc__
                    module_doc_printed = True
                    yield "-" * len(modname) + "\n"
                yield "%s%s:" % (func.__name__, SIGNATURE(func))
                yield render(func.__doc__ or '(No documentation found)', indent_num=1)
                if inspect.isclass(func):
                    for funcname, func_ in inspect.getmembers(func):
                        if funcname != "__class__" and not funcname.startswith("_"):
                            # Consider anything that starts with _ private
                            # and don't document it
                            yield "\n"
                            yield "%s%s%s:" % (_indent, funcname, SIGNATURE(func_))
                            yield render(func_.__doc__, indent_num=2)

                yield "\n"


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
        closesession(session)
