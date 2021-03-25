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

from stream2segment.io.inputvalidation import valid_session, load_config_for_visualization, \
    validate_param
import stream2segment.download.db as dbd
from stream2segment.io import open2writetext
from stream2segment.resources import get_templates_fpaths
from stream2segment.gui.main import create_s2s_show_app, run_in_browser
from stream2segment.gui.dinfo import DReport, DStats
from stream2segment.resources.templates import DOCVARS

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


# def show(dburl, pyfile, configfile):
#     """Show downloaded data plots in a system browser dynamic web page"""
#     session, pymodule, config_dict, segments_selection = \
#         load_config_for_visualization(dburl, pyfile, configfile)
#     run_in_browser(create_s2s_show_app(session, pymodule, config_dict,
#                                        segments_selection))
#     return 0


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
