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
from datetime import timedelta
from webbrowser import open as open_in_browser
import threading
from tempfile import gettempdir

from future.utils import PY2

import yaml
import jinja2

from stream2segment.utils.inputargs import load_config_for_process, load_config_for_download,\
    load_session_for_dinfo
from stream2segment.utils.log import configlog4download, configlog4processing
from stream2segment.io.db.models import Download
from stream2segment.process.main import run as run_process
from stream2segment.download.main import run as run_download, new_db_download
from stream2segment.utils import secure_dburl, strconvert, iterfuncs, open2writetext, ascii_decorate
from stream2segment.utils.resources import get_templates_dirpath
from stream2segment.gui.main import create_main_app, run_in_browser
from stream2segment.process import math as s2s_math
from stream2segment.download.utils import QuitDownload
from stream2segment.gui.dinfo import get_dstats_html, get_dstats_str_iter
from stream2segment.resources.templates import DOCVARS
from stream2segment.process.writers import get_writer


if PY2:
    # https://stackoverflow.com/questions/45946051/signature-method-in-inspect-module-for-python-2
    import funcsigs
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
    """
        Downloads the given segment providing a set of keyword arguments to match those of the
        config file `config`

        :param config: a valid path to a file in yaml format, or a dict of parameters reflecting
            a download config file
        :param log2file: if True (the default) configures a logger handler which redirects to
            a file named: `config.<now>.log`, where now is the current date-time in iso format.
            The file will be used to log all warning, error and critical messages and will write
            its content in the Donwload table. If the program
            does not exit for unexpected exception, the file will be deleted
        :param verbose: if True (False by default) all info and critical messages, as well as
            a progress-bar showing also the estimated remaining time will be printed on screen
    """
    # implementation details: this function can return 0 on success and 1 on failure.
    # First, it can raise ValueError for a bad parameter (checked before starting db session and
    # logger),
    # Then, during download, if the process completed 0 is returned. This includes the case
    # when according to our config, there are no segments to download
    # For any other case where we cannot proceed (we do not have data, e.g. no stations,
    # for whatever reason it is), 1 is returned. We should actually check better if there
    # might be some of these cases where 0 should be returned instead of 1.
    # When 1 is returned, a QuitDownload is raised and logged to error.
    # Other exceptions are caught, logged with the stack trace as critical, and raised

    # check and parse config values (modify in place):
    yaml_dict = load_config_for_download(config, True, **param_overrides)
    # get the session object and the tt_table object (needed separately, see below):
    session = yaml_dict['session']
    # print yaml_dict to terminal if needed. Do not use input_yaml_dict as
    # params needs to be shown as expanded/converted so the user can check their correctness
    # Do no use loggers yet:
    if verbose:
        # print to terminal an informative config. First objects with custom string outputs:
        sessstr = "<db session object, dburl='%s'>" % secure_dburl(str(session.bind.engine.url))
        tttable = yaml_dict['tt_table']
        tttablestr = "<%s object, model=%s, phases=%s>" % (tttable.__class__.__name__,
                                                           str(tttable.model),
                                                           str(tttable.phases))
        # replace dburl hiding passowrd for printing to terminal, tt_table with a short repr str,
        # and restore traveltimes_model because we popped from yaml_dict it out in load_tt_table
        yaml_safe = dict(yaml_dict, session=sessstr, tt_table=tttablestr)
        # replace authorizer class with the original 'restricted_data' config param and a
        # meaningful message
        authorizer = yaml_safe.pop('authorizer')
        if authorizer.token:
            str_ = "enabled, with token"
        elif authorizer.userpass:
            str_ = "enabled, with user+password"
        else:
            str_ = "disabled, download open data only"
        yaml_safe['restricted_data'] = str_
        # create a yaml string from the yaml_safe and print/log the string:
        ip_params = yaml.safe_dump(yaml_safe, default_flow_style=False)
        ip_title = "Input parameters"
        print("%s\n%s\n%s\n" % (ip_title, "-" * len(ip_title), ip_params))

    # create download row with unprocessed config (yaml_load function)
    # Note that we call again load_config with parseargs=False:
    download_id = new_db_download(session,
                                  load_config_for_download(config, False, **param_overrides))
    loghandlers = configlog4download(logger, config if log2file else None, verbose)
    ret = 0
    noexc_occurred = True
    log_elapsedtime_errs_warns = True
    try:
        if log2file and verbose:  # (=> loghandlers not empty)
            print("Log file:\n'%s'\n"
                  "(if the program does not quit for unexpected exceptions,\n"
                  "the file will be deleted before exiting and its content will be written\n"
                  "to the table '%s', column '%s')" % (str(loghandlers[0].baseFilename),
                                                       Download.__tablename__,
                                                       Download.log.key))

        stime = time.time()
        try:
            run_download(download_id=download_id, isterminal=verbose, **yaml_dict)
        except QuitDownload as quitdownloadexc:
            ret = 1
            log_elapsedtime_errs_warns = not quitdownloadexc.iscritical

        if log_elapsedtime_errs_warns:
            # print "completed in ..." only if we do not have a critical quitdownload
            # as in the latter case the message below might obfuscate more important
            # messages, and the time completion
            logger.info("Completed in %s", str(totimedelta(stime)))
            if log2file:
                errs, warns = loghandlers[0].errors, loghandlers[0].warnings

                def frmt(n, text):
                    '''stupid function to format 'No error', '1 error' , '2 errors', ...'''
                    return "%s %s%s" % ("No" if n == 0 else str(n), text, '' if n == 1 else 's')
                logger.info("%s, %s", frmt(errs, 'error'), frmt(warns, 'warning'))
    except KeyboardInterrupt:
        # https://stackoverflow.com/questions/5191830/best-way-to-log-a-python-exception:
        logger.critical("Aborted by user")
        raise
    except:  # @IgnorePep8 pylint: disable=broad-except
        # log the exception traceback (only last) and raise,
        # so that in principle the full traceback is printed on terminal (or caught by the caller)
        noexc_occurred = False
        # https://stackoverflow.com/questions/5191830/best-way-to-log-a-python-exception:
        logger.critical("Download aborted", exc_info=True)
        raise
    finally:
        # write log to db if default handlers are provided:
        if log2file:
            # remove file if no exceptions occurred:
            loghandlers[0].finalize(session, download_id, removefile=noexc_occurred)
        closesession(session)

    return ret


def process(dburl, pyfile, funcname=None, config=None, outfile=None, log2file=False,
            verbose=False, append=False, **param_overrides):
    """
        Process the segment saved in the db and optionally saves the results into `outfile`
        in .csv format. Calles F the function named `funcname` defined in `pyfile`
        If `outfile` is given , then F should return lists/dicts to be written as
            csv row.
        If `outfile` is not given, then the returned values of F will be ignored
            (F is supposed to process data without returning a value, e.g. save processed
            miniSeed to the FileSystem)

        :param log2file: if True, all messages with level >= logging.INFO will be printed to
            a log file named  <outfile>.<now>.log  (where now is the current date and time ins iso
            format) or <pyfile>.<now>.log, if <outfile> is None

        :param verbose: if True, all messages with level logging.INFO, logging.ERROR and
            logging.CRITICAL will be printed to the screen, as well as a progress-bar showing the
            eta (estimated time available).

        :param param_overrides: paramter that will override the yaml config. Nested dict will be
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

    loghandlers = configlog4processing(logger, (outfile or pyfile) if log2file else None, verbose)
    try:

        info = ["Processing function: %s:%s" % (pyfile, funcname),
                "Input database:      %s" % secure_dburl(dburl),
                "Config. file:        %s" % str('n/a' if not config else config),
                "Log file:            %s" % str(loghandlers[0].baseFilename if log2file else 'n/a'),
                "Output file:         %s" % (outfile or 'n/a')]

        logger.info(ascii_decorate("\n".join(info)))

        stime = time.time()
        run_process(session, pyfunc, get_writer(outfile, append), config_dict, verbose)
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
    run_in_browser(create_main_app(dburl, pyfile, configfile))
    return 0


def init(outpath, promptfunc=True, *filenames):
    '''initilizes an output directory writing therein the given template files

    :param promptfunc: True or function. This argument is used only
    in case some files are already present in `outpath` to decide the action to be taken.
    If function, it must accept a single string argument
    (the prompt message) and must return a string or integer where '1' means 'overwrite all files',
    '2' means 'overwrite only non-existing', and any other value will return without copying.
    If this argument is True (the default), the builtin `input` function is used to interactively
    to prompt the user. If this argument is nor True neither a function, all files will be
    overridden without prompting.
    '''
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        if not os.path.isdir(outpath):
            raise Exception("Unable to create '%s'" % outpath)
    templates_dir = get_templates_dirpath()
    if promptfunc is True:
        promptfunc = input
    if not hasattr(promptfunc, '__call__'):
        promptfunc = None
    if promptfunc is not None:
        existing_files = [f for f in filenames
                          if os.path.isfile(os.path.join(outpath, f))]
        non_existing_files = [f for f in filenames if f not in existing_files]
        if existing_files:
            suffix = ("Type:\n1: overwrite all files\n2: write only non-existing\n"
                      "0 or any other value: do nothing (exit)")
            msg = ("The following file(s) "
                   "already exist on '%s':\n%s"
                   "\n\n%s") % (outpath, "\n".join([_ for _ in existing_files]), suffix)
            val = promptfunc(msg)
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
        env.get_template(filename).stream(DOCVARS).dump(outfilepath)
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


def dinfo(dburl, download_ids=None, maxgap_threshold=0.5, html=False, outfile=None):
    '''Creates a diagnostic html page (or text string) showing the status of the download
    Note that due to current implementation, segments re-downloaded have the download id of the
    last download attempt, even if the download code is the same as previous run'''
    session = load_session_for_dinfo(dburl)
    if html:
        openbrowser = False
        if not outfile:
            openbrowser = True
            outfile = os.path.join(gettempdir(), "s2s_dinfo.html")
        # get_dstats_html returns unicode characters in py2, str in py3,
        # so it is safe to use open like this (cf below):
        with open(outfile, 'w', encoding='utf8', errors='replace') as opn:
            opn.write(get_dstats_html(session, download_ids, maxgap_threshold))
        if openbrowser:
            open_in_browser('file://' + outfile)
        threading.Timer(1, lambda: sys.exit(0)).start()
    else:
        itr = get_dstats_str_iter(session, download_ids, maxgap_threshold)
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
