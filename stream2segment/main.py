# -*- coding: utf-8 -*-
"""
Main module with all root functions (download, process, ...)

:date: Feb 2, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import range, round, open, input  # pylint: disable=redefined-builtin

import time
import logging
import re
import sys
import os
import inspect
from collections import OrderedDict
from datetime import timedelta
from webbrowser import open as open_in_browser
import threading
from tempfile import gettempdir

from future.utils import PY2, string_types

import jinja2

from sqlalchemy.sql.expression import func

from stream2segment.utils.inputargs import load_config_for_process, load_config_for_download,\
    load_session_for_dinfo
from stream2segment.utils.log import configlog4download, configlog4processing,\
    closelogger, logfilepath
from stream2segment.io.db.models import Download, Segment
from stream2segment.process.main import run as run_process
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
import shutil


if PY2:
    # https://stackoverflow.com/questions/45946051/signature-method-in-inspect-module-for-python-2
    import funcsigs  # @UnresolvedImport pylint: disable=import-error
    SIGNATURE = funcsigs.signature
else:
    SIGNATURE = inspect.signature  # @UndefinedVariable # pylint: disable=no-member

# set root logger if we are executing this module as script, otherwise as module name following
# logger conventions. Discussion here:
# http://stackoverflow.com/questions/30824981/do-i-need-to-explicitly-check-for-name-main-before-calling-getlogge
# howver, based on how we configured entry points in config, the name is (as november 2016)
# 'stream2segment.main', which messes up all hineritances. So basically setup a main logger
# with the package name
logger = logging.getLogger("stream2segment")  # pylint: disable=invalid-name


def download(config, log2file=True, verbose=False, **param_overrides):
    """Download the given segment providing a set of keyword arguments to match those of
    the config file `config`

    :param config: str or dict: If str, it is valid path to a configuration file in YAML
        syntax, or a dict of parameters reflecting a download configuration file
    :param log2file: bool or str. If string, it is the path to the log file (whose
        parent directory must exist). If True (the default), `config` can not be a
        `dict` (raise `ValueError` otherwise) and the log file path will be built as
        `config` + ".[now].log" (where [now] = current date and time in ISO format).
        If False, logging is disabled.
        When logging is enabled, the file will be used to catch all warnings, errors and
        critical messages (=Python exceptions): if the download routine exits with no
        exception, the file content is written in the database (`Download` table) and
        the file deleted. Otherwise, the file will be left on the system for inspection
    :param verbose: if True (False by default) print some log information also on the
        screen (messages of level info and critical), as well as a progress-bar showing
        also the estimated remaining time will be printed on screen. This option is set
        to True when this function is invoked from the command line interface (`cli.py`)
    """
    # implementation details: this function can return 0 on success and 1 on failure.
    # First, it can raise ValueError for a bad parameter (checked before starting db session and
    # logger),
    # Then, during download, if the process completed 0 is returned. This includes the case
    # when according to our config, there are no segments to download
    # For any other case where we cannot proceed (we do not have data, e.g. no stations,
    # for whatever reason it is), 1 is returned. We should actually check better if there
    # might be some of these cases where 0 should be returned instead of 1.
    # When 1 is returned, a FailedDownload is raised and logged to error.
    # Other exceptions are caught, logged with the stack trace as critical, and raised

    ret = 0
    noexc_occurred = True
    loghandlers = None
    session = None
    download_id = None
    try:
        # short check:
        isfile = isinstance(config, string_types) and os.path.isfile(config)
        if not isfile and log2file is True:
            raise ValueError('`log2file` can be True only if `config` is a '
                             'string denoting an existing file')

        # check and parse config values (modify in place):
        yaml_dict = load_config_for_download(config, True, **param_overrides)
        # get the session object and the tt_table object (needed separately, see below):
        session = yaml_dict['session']
        # print yaml_dict to terminal if needed. Do not use input_yaml_dict as
        # params needs to be shown as expanded/converted so the user can check their correctness
        # Do no use loggers yet:
        if verbose:
            print(_to_pretty_str(yaml_dict,
                                 load_config_for_download(config, False, **param_overrides)))

        # configure logger and habdlers:
        if log2file is True:
            log2file = logfilepath(config)  # auto create log file
        else:
            log2file = log2file or ''  # assure we have a string
        loghandlers = configlog4download(logger, log2file, verbose)

        # create download row with unprocessed config (yaml_load function)
        # Note that we call again load_config with parseargs=False:
        download_id = new_db_download(session,
                                      load_config_for_download(config, False, **param_overrides))
        if log2file and verbose:  # (=> loghandlers not empty)
            print("Log file:\n'%s'\n"
                  "(if the program does not quit for unexpected exceptions,\n"
                  "the file will be deleted before exiting and its content will be written\n"
                  "to the table '%s', column '%s')" % (log2file,
                                                       Download.__tablename__,
                                                       Download.log.key))

        stime = time.time()
        run_download(download_id=download_id, isterminal=verbose, **yaml_dict)
        logger.info("Completed in %s", str(totimedelta(stime)))
        if log2file:
            errs, warns = loghandlers[0].errors, loghandlers[0].warnings

            def frmt(n, text):
                '''stupid function to format 'No error', '1 error' , '2 errors', ...'''
                return "%s %s%s" % ("No" if n == 0 else str(n), text, '' if n == 1 else 's')
            logger.info("%s, %s", frmt(errs, 'error'), frmt(warns, 'warning'))
    except FailedDownload as fdwnld:
        # we logged the exception in `run_download`, just set return value as 1:
        ret = 1
    except KeyboardInterrupt:
        # https://stackoverflow.com/questions/5191830/best-way-to-log-a-python-exception:
        logger.critical("Aborted by user")
        raise
    except:  # @IgnorePep8 pylint: disable=broad-except
        # log the (last) exception traceback and raise
        noexc_occurred = False
        # https://stackoverflow.com/questions/5191830/best-way-to-log-a-python-exception:
        logger.critical("Download aborted", exc_info=True)
        raise
    finally:
        if session is not None:
            closesession(session)
            # write log to db if default handlers are provided:
            if log2file and loghandlers is not None and download_id is not None:
                # remove file if no exceptions occurred:
                loghandlers[0].finalize(session, download_id, removefile=noexc_occurred)
                # the method above closes the logger, let's remove it manually
                # before calling closelogger below to avoid closing loghandlers[0] twice:
                logger.removeHandler(loghandlers[0])

        closelogger(logger)

    return ret


def _to_pretty_str(yaml_dict, unparsed_yaml_dict):
    """Return a pretty printed string from yaml_dict

    :param yaml_dict: the PARSED yaml as dict. It might contain variables, such as
        `session`, not in the original yaml file (these variables are not returned
        in this function)
    :param unparsed_yaml_dict: the UNPARSED yaml dict.
    """

    # the idea is: get the param value from yaml_dict, if not present get it from
    # unparsed_yaml_dict. Use this list of params so we can control order
    # (this works only in PyYaml>=5.1 and Python 3.6+, otherwise yaml_dict
    # keys order can not be known):
    params = ['dburl', 'starttime', 'endtime', 'eventws',
              # These variables are merged into eventws_params in yaml_dict,
              # so do not show them:
              # 'minlatitude', 'maxlatitude', 'minlongitude', 'maxlongitude',
              # 'mindepth', 'maxdepth', 'minmagnitude', 'maxmagnitude',
              'eventws_params', 'channel', 'network', 'station', 'location',
              'min_sample_rate', 'update_metadata', 'inventory', 'search_radius',
              'dataws', 'traveltimes_model', 'timespan', 'restricted_data',
              'retry_seg_not_found', 'retry_url_err', 'retry_mseed_err',
              'retry_client_err', 'retry_server_err', 'retry_timespan_err',
              'advanced_settings']

    newdic = {}
    for k in params:  # add yaml_dic[k] or unparsed_yaml_dict[k]:
        if k in yaml_dict:
            newdic[k] = yaml_dict[k]
        elif k in unparsed_yaml_dict:
            newdic[k] = unparsed_yaml_dict[k]
    newdic['dburl'] = secure_dburl(newdic['dburl'])  # don't show passowrd, if present
    ret = [
        "Parsed input parameters",
        "-----------------------",
        yaml_safe_dump(newdic)
    ]
    return "%s\n" % ('\n'.join(ret)).strip()


def process(dburl, pyfile, funcname=None, config=None, outfile=None, log2file=False,
            verbose=False, append=False, **param_overrides):
    """Process the segment saved in the database at the given URL and optionally
    saves the results into `outfile`. See docstrings in stream2segment templates
    (command `s2s init`) for implementing a process module and configuration.

    :param pyfile: string (path to the processing module)
    :param funcname: str or None: the function name in `pyfile` to be used
        (None, the default means: use default name, currently "main")
    :param config: str path of the configuration file in YAML syntax
    :param outfile: str or None. The destination file where to write the processing
        output, either ".csv" or ".hdf". If not given, the returned values of
        `funcname` in `pyfile` will be ignored, if given.
    :param log2file: bool or str. If str, it is the log file path (whose directory
        must exist). If True (the default), the log file path will be built as
        `outfile` + ".[now].log" or (if no output file is given) as
        `pyfile` + ".[now].log" ([now] = current date and time in ISO format).
        If False, logging is disabled.
    :param verbose: if True (False by default) print some log information also on the
        screen (messages of level info and critical), as well as a progress-bar showing
        also the estimated remaining time will be printed on screen. This option is set
        to True when this function is invoked from the command line interface (`cli.py`)
    :param append: bool (default False) ignored if the output file is not given or non
        existing, otherwise: if False, overwrite the existing output file. If True,
        process unprocessed segments only (checking the segment id), and append to the
        given file, without replacing existing data.
    :param param_overrides: parameter that will override the YAML config. Nested dict will be
        merged, not replaced
    """
    # implementation details: this function returns 0 on success and raises otherwise.
    # First, it can raise ValueError for a bad parameter (checked before starting db session and
    # logger),
    # Then, during processing, each segment error which is not (ImportError, NameError,
    # AttributeError, SyntaxError, TypeError) is logged as warning and the program continues.
    # Other exceptions are raised, caught here and logged with level CRITICAL, with the stack trace:
    # this allows to help users to discovers possible bugs in pyfile, without waiting for
    # the whole process to finish

    # checks dic values (modify in place) and returns dic value(s) needed here:
    session, pyfunc, funcname, config_dict = \
        load_config_for_process(dburl, pyfile, funcname, config, outfile, **param_overrides)

    if log2file is True:
        log2file = logfilepath(outfile or pyfile)  # auto create log file
    else:
        log2file = log2file or ''  # assure we have a string
    loghandlers = configlog4processing(logger, log2file, verbose)
    try:
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
        writer_options = config_dict.get('advanced_settings', {}).get('writer_options', {})
        run_process(session, pyfunc, get_writer(outfile, append, writer_options),
                    config_dict, verbose)
        logger.info("Completed in %s", str(totimedelta(stime)))
        return 0  # contrarily to download, an exception should always raise and log as error
        # with the stack trace
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


def totimedelta(t0_sec, t1_sec=None):
    '''time elapsed from `t0_sec` until `t1_sec`, as `timedelta` object rounded to
    seconds.
    If `t1_sec` is None, it will default to `time.time()` (the current time since the epoch,
    in seconds)

    :param t0_sec: (float) the start time in seconds. Usually it is the result of a
        previous call to `time.time()`, before starting a process that had to be monitored
    :param t1_sec: (float) the end time in seconds. If None, it defaults to `time.time()`
        (current time since the epoch, in seconds)

    :return: a timedelta object, rounded to seconds
    '''
    return timedelta(seconds=round((time.time() if t1_sec is None else t1_sec) - t0_sec))


def closesession(session):
    '''closes the session,
    This method simply calls `session.close()`, passing all exceptions, if any.
    Useful for unit testing and mock
    '''
    try:
        session.close()
    except:
        pass


def show(dburl, pyfile, configfile):
    '''show downloaded data plots in a system browser dynamic web page'''
    run_in_browser(create_s2s_show_app(dburl, pyfile, configfile))
    return 0


def init(outpath, prompt=True, *filenames):
    '''initilizes an output directory writing therein the given template files

    :param prompt: boolean telling if a prompt message (python `input` function)
        should be issued to warn the user when overwriting files. Default: True.
        The user should return a string or integer where '1' means 'overwrite all files',
        '2' means 'overwrite only non-existing', and any other value will return without copying.
    '''
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


def helpmathiter(type, filter):  # @ReservedAssignment pylint: disable=redefined-outer-name
    '''iterator yielding the doc-string of :module:`stream2segment.process.math.ndarrays` or
    :module:`stream2segment.process.math.traces`

    :param type: select the module: 'numpy' for doc of
        :module:`stream2segment.process.math.ndarrays`,
        'obspy' for the doc of :module:`stream2segment.process.math.traces`, 'all' for both

    :param filter: a filter (with wildcard expressions allowed) to filter by function name

    :return: doc-string for all matching functions and classes
    '''
    itr = [s2s_math.ndarrays] if type == 'numpy' else [s2s_math.traces] if type == 'obspy' else \
        [s2s_math.ndarrays, s2s_math.traces]
    reg = re.compile(strconvert.wild2re(filter))
    _indent = "   "

    def render(string, indent_num=0):
        '''renders a string stripping newlines at beginning and end and with the intended indent
        number'''
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


def dreport(dburl, download_ids=None, config=True, log=True, html=False, outfile=None):
    '''Creates a diagnostic html page (or text string) showing the status of the download.
    Note that html is not supported for the moment and will raise an Exception.
    (leaving the same signatire as dstats for compatibility and easing future implementations
    of the html page if needed)

    :param config: boolean (True by default)
    :param log: boolean (True by default)
    '''
    _get_download_info(DReport(config, log), dburl, download_ids, html, outfile)


def dstats(dburl, download_ids=None, maxgap_threshold=0.5, html=False, outfile=None):
    '''Creates a diagnostic html page (or text string) showing the status of the download

    :param maxgap_threshold: the max gap threshold (float)
    '''
    _get_download_info(DStats(maxgap_threshold), dburl, download_ids, html, outfile)


def _get_download_info(info_generator, dburl, download_ids=None, html=False, outfile=None):
    '''processes dinfo ro dstats'''
    session = load_session_for_dinfo(dburl)
    if html:
        openbrowser = False
        if not outfile:
            openbrowser = True
            outfile = os.path.join(gettempdir(),
                                   "s2s_%s.html" % info_generator.__class__.__name__.lower())
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
            # itr is an iterator of strings in py2, and str in py3, so open must be input
            # differently (see utils module):
            with open2writetext(outfile, encoding='utf8', errors='replace') as opn:
                for line in itr:
                    line += '\n'
                    opn.write(line)
        else:
            for line in itr:
                print(line)


def ddrop(dburl, download_ids, prompt=True):
    '''Drops data from the database by download id(s). Drops also all segments

    :return: None if prompt is True and the user decided not to drop via user input,
        otherwise a dict of deleted download ids mapped to either:
        -  an int (the number of segments deleted)
        - an exception (if the download id could not be deleted)
    '''
    ret = {}
    session = load_session_for_dinfo(dburl)
    try:
        ids = [_[0] for _ in session.query(Download.id).filter(Download.id.in_(download_ids))]
        if not ids:
            return ret
        if prompt:
            segs = \
                session.query(func.count(Segment.id)).filter(Segment.download_id.in_(ids)).scalar()
            val = input('Do you want to delete %d download execution(s) (id=%s) and the associated '
                        '%d segment(s) from the database [y|n]?' % (len(ids), str(ids), segs))
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
