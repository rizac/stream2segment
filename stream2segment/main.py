# -*- coding: utf-8 -*-
"""
Main module with all root functions (download, process, ...)

:date: Feb 2, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip, object)

import logging
import re
import sys
import os
from contextlib import contextmanager
import shutil
import inspect
from datetime import datetime, timedelta 

# iterate over dictionary keys without list allocation in both py 2 and 3:
# from future.utils import viewitems

import yaml
import click

from stream2segment.utils.log import configlog4download, configlog4processing,\
    elapsedtime2logger_when_finished
from stream2segment.io.db.models import Download
from stream2segment.process.main import to_csv
from stream2segment.download.main import run as run_download, new_db_download
from stream2segment.utils import get_session, indent, secure_dburl, strptime, strconvert, iterfuncs
from stream2segment.utils.resources import get_templates_fpaths, yaml_load, get_ttable_fpath
from stream2segment.gui.main import create_p_app, run_in_browser, create_d_app
from stream2segment.process import math as s2s_math
from stream2segment.download.utils import nslc_param_value_aslist
from stream2segment.traveltimes.ttloader import TTTable


# set root logger if we are executing this module as script, otherwise as module name following
# logger conventions. Discussion here:
# http://stackoverflow.com/questions/30824981/do-i-need-to-explicitly-check-for-name-main-before-calling-getlogge
# howver, based on how we configured entry points in config, the name is (as november 2016)
# 'stream2segment.main', which messes up all hineritances. So basically setup a main logger
# with the package name
logger = logging.getLogger("stream2segment")


def download(config_yaml_path, isterminal=False, **overrides):
    """
        Downloads the given segment providing a set of keyword arguments to match those of the
        config file (see confi.example.yaml for details)
    """
    yaml_dict = yaml_load(config_yaml_path, **overrides)
    dburl = yaml_dict['dburl']
    with closing(dburl) as session:
        # check for networks, stations, locations and channels and harmonize them:
        adjust_nslc_params(yaml_dict)
        adjust_times(yaml_dict)
        tt_table = load_tt_table(yaml_dict['traveltimes_model'])
        download_id = new_db_download(session, yaml_dict)

        if isterminal:
            print("Arguments:")
            # replace dbrul passowrd for printing to terminal
            # Note that we remove dburl from yaml_dict cause query_main gets its session object
            # (which we just built)
            yaml_safe = dict(yaml_dict, dburl=secure_dburl(yaml_dict.pop('dburl')))
            print(indent(yaml.safe_dump(yaml_safe, default_flow_style=False), 2))

        loghandler = configlog4download(logger, session, download_id, isterminal)

        if isterminal:
            print("Log messages will be temporarily written to:\n'%s'" %
                  str(loghandler.baseFilename))
            print("(if the program does not quit for external causes, e.g., memory overflow,\n"
                  "the file will be deleted before exiting and its content will be written\n"
                  "to the table '%s', column '%s')" % (Download.__tablename__,
                                                       Download.log.key))

        with elapsedtime2logger_when_finished(logger):
            # replace arg and set tt_table now it's fine (no error sofar):
            yaml_dict.pop('traveltimes_model')
            yaml_dict['tt_table'] = tt_table
            run_download(session=session, download_id=download_id, isterminal=isterminal,
                         **yaml_dict)
            logger.info("\n%d total error(s), %d total warning(s)", loghandler.errors,
                        loghandler.warnings)

    return 0


def load_tt_table(file_or_name):
    try:
        filepath = get_ttable_fpath(file_or_name)
        if not os.path.isfile(filepath):
            filepath = file_or_name
        return TTTable(filepath)
    except Exception as exc:
        msg = "Error loading travel-times table from file '%s': %s" % (filepath, str(exc))
        raise ValueError(msg)

def adjust_times(yaml_dict):
    try:
        for pname in ['start', 'end']:
            yaml_dict[pname] = valid_date(yaml_dict[pname])
    except ValueError as exc:
        raise ValueError("Invalid parameter '%s': %s" % (pname, str(exc)))


def valid_date(obj):
    try:
        return strptime(obj)  # if obj is datetime, returns obj
    except ValueError as _:
        try:
            days = int(obj)
            NOW = datetime.utcnow()
            endt = datetime(NOW.year, NOW.month, NOW.day, 0, 0, 0, 0)
            return endt - timedelta(days=days)
        except Exception:
            pass
    raise ValueError("please supply a date-time or an integer")


def adjust_nslc_params(yaml_dic):
    '''Scans `dic` keys and returtns the tuple
        ```
        (N, S, L, C)
        ```
    where each element is a list of networks (N), stations (S), locations (L) or channels (C)
    composed by strings in valid ASCII characters with three special characters:
    the 2 FDSN-compliant wildcards '*' and '?', and '!' which means NOT (when placed as first
    character only).

    This function basically returns `",".join(dic[key])` where `key` is any of the following: 
        'net', 'network' or 'networks'
        'sta', 'stations' or 'stations'
        'loc', 'location' or 'locations'
        'cha', 'channel' or 'channels'
    In case of keys conflicts (e.g. 'net' and 'network' are both in `dict`) a ValueError is raised.
    In case a key not found, None or '*', the corresponding element will be the empty list.
    A returned empty list has to be interpreted as "accept all" (i.e. no filter for that key).
    All string elements are stripped, meaning that leading and trailing spaces are removed.

    This function doe salso some preliminary check on each string, so that e.g.
    strings like "!*", or both "A?" and !A?"specified will raise a ValueError in case

    :return: a 4-element tuple net, sta, loc, cha. All elements are lists of strings. Returned
        empty lists mean: no filter for that key (accept all)
    '''
    
    params = [('net', 'network', 'networks'), ('sta', 'station', 'stations'),
              ('loc', 'location', 'locations'), ('cha', 'channel', 'channels')]
    
    for i, pars in enumerate(params):
        
        arg = None
        parconflicts = []
        for p in pars:
            if p in yaml_dic:
                parconflicts.append(p)
                arg = yaml_dic.pop(p)
            if len(parconflicts) > 1:
                raise ValueError("Parameter name conflict: cannot handle both %s" %
                                 (" and ".join('%s' % _ for _ in parconflicts)))
            
        s2s_name = pars[-1]
        val = []
        if len(parconflicts) and arg is not None and arg not in ([], ()):
            val = nslc_param_value_aslist(i, arg)
        
        yaml_dic[s2s_name] = val
        

def process(dburl, pyfile, configfile, outcsvfile, isterminal=False):
    """
        Process the segment saved in the db and saves the results into a csv file
        :param processing: a dict as load from the config
    """
    with closing(dburl) as session:
        if isterminal:
            print("Processing, please wait")
        logger.info('Output file: %s', outcsvfile)

        configlog4processing(logger, outcsvfile, isterminal)
        with elapsedtime2logger_when_finished(logger):
            to_csv(outcsvfile, session, pyfile, configfile, isterminal)

    return 0


def show(dburl, pyfile, configfile):
    run_in_browser(create_p_app(dburl, pyfile, configfile))
    return 0


def show_download_report(dburl):
    run_in_browser(create_d_app(dburl))
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
                    if not len(non_existing_files):
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


@contextmanager
def closing(dburl, scoped=False, close_logger=True, close_session=True):
    """Opens a sqlalchemy session and closes it. Also closes and removes all logger handlers if
    close_logger is True (the default)
    :example:
        # configure logger ...
        with closing(dburl) as session:
            # ... do stuff, print to logger etcetera ...
        # session is closed and also the logger handlers
    """
    try:
        session = get_session(dburl, scoped=scoped)
        yield session
    except:
        logger.critical(sys.exc_info()[1])
        raise
    finally:
        if close_logger:
            handlers = logger.handlers[:]  # make a copy
            for handler in handlers:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except (AttributeError, TypeError, IOError, ValueError):
                    pass
        if close_session:
            # close the session at the **real** end! we might need it above when closing loggers!!!
            try:
                session.close()
                session.bind.dispose()
            except NameError:
                pass


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
                yield "%s%s:" % (func.__name__, inspect.signature(func))
                yield render(func.__doc__ or '(No documentation found)', indent_num=1)
                if inspect.isclass(func):
                    for funcname, func in inspect.getmembers(func):
                        if funcname != "__class__" and not funcname.startswith("_"):
                            # Consider anything that starts with _ private
                            # and don't document it
                            yield "\n"
                            yield "%s%s%s:" % (INDENT, funcname, inspect.signature(func))
                            yield render(func.__doc__, indent_num=2)

                yield "\n"
