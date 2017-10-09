'''
Module documentation to be implemented

:date: Oct 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# provide some imports to let python3 syntax work also in python 2.7+ effortless. 
# Any of the defaults import below can be safely removed if python2+
# compatibility is not needed

# standard python imports (must be the first import)
from __future__ import absolute_import, division, print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

# future aliased imports(needs future package installed, otherwise remove):
# You want to import and use safely, e.g. collections.UserDict, collections.UserList, collections.UserString
# urllib.parse, urllib.request, urllib.response, urllib.robotparser, urllib.error
# itertools.filterfalse, itertools.zip_longest
# subprocess.getoutput, subprocess.getstatusoutput, sys.intern
# (a full list available on http://python-future.org/imports.html#aliased-imports)
# If none of the above is needed, you can safely remove the next two lines
# from future.standard_library import install_aliases
# install_aliases()

import re
import logging
import sys
import os
from contextlib import contextmanager
import shutil

import yaml

from stream2segment.utils.log import configlog4download, configlog4processing,\
    elapsedtime2logger_when_finished, configlog4stdout
# from stream2segment.download.utils import run_instance
from stream2segment.utils.resources import version
from stream2segment.io.db.models import Segment, Download
from stream2segment.process.main import run as run_process, to_csv, default_funcname
from stream2segment.download.main import run as run_download
from stream2segment.utils import tounicode, get_session, strptime,\
    indent, secure_dburl, iterfuncs, strconvert
from stream2segment.utils.resources import get_templates_fpath, yaml_load, yaml_load_doc,\
    get_templates_fpaths
from stream2segment import analysis
from stream2segment.analysis import mseeds

# set root logger if we are executing this module as script, otherwise as module name following
# logger conventions. Discussion here:
# http://stackoverflow.com/questions/30824981/do-i-need-to-explicitly-check-for-name-main-before-calling-getlogge
# howver, based on how we configured entry points in config, the name is (as november 2016)
# 'stream2segment.main', which messes up all hineritances. So basically setup a main logger
# with the package name
logger = logging.getLogger("stream2segment")


def download(isterminal=False, **yaml_dict):
    """
        Downloads the given segment providing a set of keyword arguments to match those of the
        config file (see confi.example.yaml for details)
    """
    dburl = yaml_dict['dburl']
    with closing(dburl) as session:
        # print local vars: use safe_dump to avoid python types. See:
        # http://stackoverflow.com/questions/1950306/pyyaml-dumping-without-tags
        download_inst = Download(config=tounicode(yaml.safe_dump(yaml_dict,
                                                                 default_flow_style=False)),
                                 # log by default shows error. If everything works fine, we replace
                                 # the content later
                                 log=('Content N/A: this is probably due to an unexpected'
                                      'and out-of-control interruption of the download process '
                                      '(e.g., memory error)'), program_version=version())

        session.add(download_inst)
        session.commit()
        download_id = download_inst.id
        session.close()  # frees memory?

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
            run_download(session=session, download_id=download_id, isterminal=isterminal,
                         **yaml_dict)
            logger.info("\n%d total error(s), %d total warning(s)", loghandler.errors,
                        loghandler.warnings)

    return 0


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


def visualize(dburl, pyfile, configfile):
    from stream2segment.gui import main as main_gui
    main_gui.run_in_browser(dburl, pyfile, configfile)
    return 0


# def data_aval(dburl, outfile, max_gap_ovlap_ratio=0.5):
#     from stream2segment.gui.da_report.main import create_da_html
#     # errors are printed to terminal:
#     configlog4stdout(logger)
#     with closing(dburl) as session:
#         create_da_html(session, outfile, max_gap_ovlap_ratio, True)
#     if os.path.isfile(outfile):
#         import webbrowser
#         webbrowser.open_new_tab('file://' + os.path.realpath(outfile))

_TEMPLATE_FILES = {"download.yaml": "download's configuration file ('s2s d' -c option)",
                   "processing.py": "processing/gui python file "
                   "('s2s p' and 's2s v' -p option)",
                   "processing.yaml": ("processing/gui configuration file "
                                       "('s2s p' and 's2s v' -c option)")}


def create_templates(outpath, force_overwrite_existing=True):
    '''Creates template files in the specified directory output path
    :param force_overwrite_existing: boolean (default: True) if True, all files will be copied, and
    existing files will be overridden. If False **only on-existing files will be copied**
    '''
    # get the template files. Use all files except those with more than one dot
    # This might be better implemented
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        if not os.path.isdir(outpath):
            raise Exception("Unable to create '%s'" % outpath)
    template_files = get_templates_fpaths(*_TEMPLATE_FILES)
    copied_files = []
    for tfile in template_files:
        if not os.path.isfile(os.path.join(outpath, os.path.basename(tfile))) or \
                force_overwrite_existing:
            shutil.copy2(tfile, outpath)
            copied_files.append(os.path.join(outpath, os.path.basename(tfile)))
    return copied_files


def helpmathiter(type, filter):  # @ReservedAssignment
    # print("%s\n\n%s" % (analysis.__doc__, mseeds.__doc__))  # @UndefinedVariable
    itr = [analysis] if type == 'numpy' else [mseeds] if 'type' == 'obspy' else [analysis, mseeds]
    reg = re.compile(strconvert.wild2re(filter))
    for pymodule in itr:
        module_doc_printed = False
        for func in iterfuncs(pymodule):
            if reg.search(func.__name__):
                if not module_doc_printed:
                    modname = pymodule.__name__
                    yield "=" * len(modname)
                    yield modname
                    yield "=" * len(modname)
                    yield pymodule.__doc__
                    module_doc_printed = True
                    yield "-" * len(modname) + "\n"
                # get func name
                fname = func.__name__
                # get arguments:
                args = func.__code__.co_varnames[:func.__code__.co_argcount]
                # get defaults:
                funcdefaults = func.__defaults__ or []
                # get positional arguments:
                posargslen = len(args) - len(funcdefaults)
                # posargs 2 string:
                strargs = [str(args[i]) for i in range(posargslen)]
                for i, defval in enumerate(funcdefaults, posargslen):
                    if isinstance(defval, str):
                        frmat = "%s='%s'"
                    else:
                        frmat = "%s=%s"
                    strargs.append(frmat % (args[i], str(defval)))
                yield "%s(%s):" % (fname, ", ".join(strargs))
                yield "    %s" % func.__doc__.strip()
                yield "\n"


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
