# -*- coding: utf-8 -*-
"""
Main module with all root functions (download, process, ...)

:date: Feb 2, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import range, round, open

import time
import logging
import re
import sys
import os
import shutil
import inspect
from datetime import timedelta
from webbrowser import open as open_in_browser
import threading
from tempfile import gettempdir

from future.utils import string_types, text_type, PY2

import yaml
import click


from stream2segment.utils.inputargs import load_config_for_process, load_config_for_download,\
    load_session_for_dinfo
from stream2segment.utils.log import configlog4download, configlog4processing
from stream2segment.io.db.models import Download
from stream2segment.process.main import run as run_process
from stream2segment.download.main import run as run_download, new_db_download
from stream2segment.utils import secure_dburl, strconvert, iterfuncs, open2writetext
from stream2segment.utils.resources import get_templates_fpaths
from stream2segment.gui.main import create_main_app, run_in_browser
from stream2segment.process import math as s2s_math
from stream2segment.download.utils import QuitDownload
from stream2segment.gui.dinfo import get_dstats_html, get_dstats_str_iter

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
logger = logging.getLogger("stream2segment")


def download(config, verbosity=2, **param_overrides):
    """
        Downloads the given segment providing a set of keyword arguments to match those of the
        config file (see confi.example.yaml for details)

        :param config: a valid path to a file in yaml format, or a dict of parameters reflecting
            a download config file
        :param verbosity: integer: 0 means: no logger configured, no print to standard output.
            Use this option if you want to have maximum flexibility (e.g. configure your logger)
            and - in principle - shorter execution time (although the real benefits have not been
            measured): this means however that no log information will be saved to the database
            (including the execution time) unless a logger has been explicitly set by the user
            beforehand (see :clas:`DbHandler` in case)
            1 means: configure default logger, no print to standard output. Use this option if you
            are not calling this function from the command line but you want to have all log
            information stored to the database (including execution time)
            2 (the default) means: configure default logger, and print to standard output. Use this
            option if calling this program from the command line. This is the same as verbosity=1
            but in addition some informations are printed to the standard output, including
            progresss bars for displaying the estimated remaining time of each sub-task

        :raise: ValueError if some parameter is invalid in configfile (yaml format)
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
    session, tttable = yaml_dict['session'], yaml_dict['tt_table']

    # print yaml_dict to terminal if needed. Do not use input_yaml_dict as
    # params needs to be shown as expanded/converted so the user can check their correctness
    # Do no use loggers yet:
    is_from_terminal = verbosity >= 2
    if is_from_terminal:
        # print to terminal an informative config. First objects with custom string outputs:
        sessstr = "<session object, dburl='%s'>" % secure_dburl(str(session.bind.engine.url))
        tttablestr = "<%s object, model=%s, phases=%s>" % (tttable.__class__.__name__,
                                                           str(tttable.model),
                                                           str(tttable.phases))
        # replace dburl hiding passowrd for printing to terminal, tt_table with a short repr str,
        # and restore traveltimes_model because we popped from yaml_dict it out in load_tt_table
        yaml_safe = dict(yaml_dict, session=sessstr, tt_table=tttablestr)
        ip_params = yaml.safe_dump(yaml_safe, default_flow_style=False)
        ip_title = "Input parameters"
        print("%s\n%s\n%s\n" % (ip_title, "-" * len(ip_title), ip_params))

    # create download row with unprocessed config (yaml_load function)
    # Note that we call again load_config with parseargs=False:
    download_id = new_db_download(session,
                                  load_config_for_download(config, False, **param_overrides))
    loghandlers = configlog4download(logger, is_from_terminal) if verbosity > 0 else []
    ret = 0
    noexc_occurred = True
    log_elapsedtime_errs_warns = True
    try:
        if is_from_terminal:  # (=> loghandlers not empty)
            print("Log file:\n'%s'\n"
                  "(if the program does not quit for unexpected exceptions or external causes, "
                  "e.g., memory overflow,\n"
                  "the file will be deleted before exiting and its content will be written\n"
                  "to the table '%s', column '%s')" % (str(loghandlers[0].baseFilename),
                                                       Download.__tablename__,
                                                       Download.log.key))

        stime = time.time()
        try:
            run_download(download_id=download_id, isterminal=is_from_terminal, **yaml_dict)
        except QuitDownload as quitdownloadexc:
            log_elapsedtime_errs_warns = not quitdownloadexc.iscritical

        if log_elapsedtime_errs_warns:
            logger.info("Completed in %s", str(totimedelta(stime)))
            if loghandlers:
                errs, warns = loghandlers[0].errors, loghandlers[0].warnings
                def frmt(n, text):
                    '''stupid function to format 'No error', '1 error' , '2 errors', ...'''
                    return "%s %s%s" % ("No" if n == 0 else str(n), text, '' if n == 1 else 's')
                logger.info("%s, %s", frmt(errs, 'error'), frmt(warns, 'warning'))

    except:  # @IgnorePep8 pylint: disable=broad-except
        # log the exception traceback (only last) and raise,
        # so that in principle the full traceback is printed on terminal (or caught by the caller)
        noexc_occurred = False
        # https://stackoverflow.com/questions/5191830/best-way-to-log-a-python-exception:
        logger.critical("Download aborted", exc_info=True)
        raise
    finally:
        # write log to db if default handlers are provided:
        if loghandlers:
            # remove file if we do not print to terminal (as it would be impossible to
            # know which file we logged into), or no exceptions occurred
            loghandlers[0].finalize(session, download_id,
                                    removefile=not is_from_terminal or noexc_occurred)
        closesession(session)

    return ret


def process(dburl, pyfile, funcname=None, config=None, outfile=None, verbose=False,
            **param_overrides):
    """
        Process the segment saved in the db and optionally saves the results into `outfile`
        in .csv format
        If `outfile` is given, `pyfile` should return lists/dicts to be written as
            csv row, and a handler will redirect all logged messages to a the file `[outfile].log`
        If `outfile` is not given, then the returned values of `pyfile` will be ignored
            (`pyfile` is supposed to process data without returning a value, e.g. save processed
            miniSeed to the FileSystem), and a handler will redirect all logged messages to
            `stderr`.
        In both cases, if `verbose` is True, a handler will redirect all informations, errors
            and critical logged messages to the standard output (also, and a progressbar will be
            printed to standard output)

        :param param_overrides: paramter that will override the yaml config. Nested dict will be
            merged, not replaced
    """
    # implementation details: this function returns 0 on success and raises otherwise.
    # First, it can raise ValueError for a bad parameter (checked before starting db session and
    # logger),
    # Then, during processing, each segment error which is not (ImportError, NameError,
    # AttributeError, SyntaxError, TypeError) is logged as warning and the program continues.
    # Other exceptions are raised, caught here and logged as error, with the stack trace:
    # this allows to help users to discovers possible bugs in pyfile, without waiting for
    # the whole process to finish. Note that this does not distinguish the case where
    # we have any other exception (e.g., keyboard interrupt), but that's not a requirement

    # checks dic values (modify in place) and returns dic value(s) needed here:
    session, pyfunc, funcname, config_dict = \
        load_config_for_process(dburl, pyfile, funcname, config, outfile, **param_overrides)

    configlog4processing(logger, outfile, verbose)
    try:

        if outfile:
            logger.info('Output file: %s', outfile)
        logger.info("Executing '%s' in '%s'", funcname, pyfile)
        logger.info("Input database: '%s", secure_dburl(dburl))
        if config and isinstance(config, string_types):
            logger.info("Config. file: %s", str(config))

        stime = time.time()
        run_process(session, pyfunc, config_dict, outfile, verbose)
        logger.info("Completed in %s", str(totimedelta(stime)))
        return 0  # contrarily to download, an exception should always raise and log as error
        # with the stack trace
        # (this includes pymodule exceptions e.g. TypeError)
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
    run_in_browser(create_main_app(dburl, pyfile, configfile))
    return 0


def init(outpath, prompt=True, *filenames):
    # get the template files. Use all files except those with more than one dot
    # This might be better implemented
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        if not os.path.isdir(outpath):
            raise Exception("Unable to create '%s'" % outpath)
    template_files = get_templates_fpaths(*filenames)
    if prompt:
        existing_files = [t for t in template_files
                          if os.path.isfile(os.path.join(outpath, os.path.basename(t)))]
        non_existing_files = [t for t in template_files if t not in existing_files]
        if existing_files:
            suffix = ("Type:\n1: overwrite all files\n2: write only non-existing\n"
                      "0 or any other value: do nothing (exit)")
            msg = ("The following file(s) "
                   "already exist on '%s':\n%s"
                   "\n\n%s") % (outpath, "\n".join([os.path.basename(_)
                                                    for _ in existing_files]), suffix)
            val = click.prompt(msg)
            try:
                val = int(val)
                if val == 2:
                    if not non_existing_files:
                        raise ValueError()
                    else:
                        template_files = non_existing_files
                elif val != 1:
                    raise ValueError()
            except ValueError:
                return []
    copied_files = []
    for tfile in template_files:
        shutil.copy2(tfile, outpath)
        copied_files.append(os.path.join(outpath, os.path.basename(tfile)))
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
    INDENT = "   "

    def render(string, indent_num=0):
        '''renders a string stripping newlines at beginning and end and with the intended indent
        number'''
        if not indent_num:
            return string
        indent = INDENT.join('' for _ in range(indent_num+1))
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
                            yield "%s%s%s:" % (INDENT, funcname, SIGNATURE(func_))
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
