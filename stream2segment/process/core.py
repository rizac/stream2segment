'''
Main module for the segment processing and .csv output

Created on Feb 2, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, chr, dict, filter, hex, input, # int, 
                      map, next, oct, open, pow, range, round,
                      super, zip)

# iterating over dictionary keys with the same set-like behaviour on Py2.7 as on Py3:
# from future.utils import viewkeys

import os
import sys
import logging
from contextlib import contextmanager
import warnings
import inspect
from multiprocessing import Pool, cpu_count
import signal

from future.utils import itervalues

import numpy as np

from sqlalchemy import func
# from sqlalchemy.orm import load_only

from stream2segment.process.utils import enhancesegmentclass, set_classes, get_slices
from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.utils import get_progressbar, StringIO, get_session
from stream2segment.io.db.models import Segment, Station, Event, Channel
from stream2segment.utils.inputargs import load_pyfunc


logger = logging.getLogger(__name__)


def run(session, pyfunc, ondone=None, config=None, show_progress=False):
    if config is None:
        config = {}

    # suppress obspy warnings
    # # https://docs.python.org/2/library/warnings.html#the-warnings-filter
#     warnings.filterwarnings("default")
#     warn_string_io = StringIO()
#     logger_handler = logging.StreamHandler(warn_string_io)
#     logger_handler.setLevel(logging.WARNING)
#     logging.captureWarnings(True)
#     warnings_logger = logging.getLogger("py.warnings")
#     warnings_logger.addHandler(logger_handler)

    # multiprocess with sessions is a mess. Among other problems, we should not share any
    # session-related operation with the same session object across different multiprocesses.
    # Is it worth to create a new session each time? not for the moment

    # NOT USED ANYMORE: maybe in the future if we exprience memory problems:
    # clear the session and expunge all every clear_session_step iterations:
    # (set a multiple of three might be in sync with other orientations, which is 3):
    # clear_session_step = 60

    logger.info("Fetching segments to process, please wait...")

    done_skipped_errors = [0, 0, 0]

    adv_settings = config.get('advanced_settings', {})
    multi_process = adv_settings.get('multi_process', False)
    num_processes = adv_settings.get('num_processes', cpu_count())
    chunksize = adv_settings.get('segments_chunk', 1200)

    # Note on chunksize above:
    # When loading segments, we have two strategies:
    # A) load only a Segment with its id (and defer loading of other
    # attributes upon access) or B) load a Segment with all attributes
    # (columns) or. From experiments on a 16 GB memory Mac:
    # Querying A) and then accessing (=loading) later two likely used attributes
    # (data and arrival_time) we take:
    # ~= 0.043 secs/segment, Peak memory (Kb): 111792 (0.650716 %)
    # Querying B) and then accessing the already loaded data and arrival_time attributes,
    # we get:
    # 0.024 secs/segment, Peak memory (Kb): 409194 (2.381825 %).
    # With millions of segments, the latter
    # approach can save up to 9 hours with almost no memory perf issue (1.5% more).
    # So we define a chunk size whereby we load all segments:

    # the query is always loaded in memory, see:
    # https://stackoverflow.com/questions/11769366/why-is-sqlalchemy-insert-with-sqlite-25-times-slower-than-using-sqlite3-directly/11769768#11769768
    # thus we load it as np.array to save memory.
    # We might want to avoid this by querying chunks of segments using limit and order keywords
    # or query yielding:
    # http://docs.sqlalchemy.org/en/latest/orm/query.html#sqlalchemy.orm.query.Query.yield_per
    # but that makes code too complex
    # Thus, load first the
    # id attributes of each segment and station, sorted by station.
    # Note that querying attributes instead of instances does not cache the results
    # (i.e., after the line below we do not need to issue session.expunge_all()
    seg_sta_ids = np.array(query4process(session, config.get('segment_select', {})).all(),
                           dtype=int)
    # get total segment length (in numpy it is equivalent to len(seg_sta_ids)):
    seg_len = seg_sta_ids.shape[0]

    def stationssaved():
        '''returns how many station inventories are saved on the db (int)'''
        return session.query(func.count(Station.id)).filter(Station.has_inventory).scalar()

    stasaved = stationssaved()

    logger.info("%d segments found to process", seg_len)
    # set/update classes, if written in the config, so that we can set instance classes in the
    # processing, if we want:
    set_classes(session, config)

    session.close()  # expunge all, clear all states

    with create_processing_env(seg_len, config=None if multi_process else config,
                               redirect_stderr=False if multi_process else True,
                               warnings_filter=None if multi_process else 'ignore') as pbar:
        if multi_process:
            # little hacky (1): the db engine has to be disposed now
            # if we want to use multi-processing:
            # https://stackoverflow.com/questions/41279157/connection-problems-with-sqlalchemy-and-multiple-processes
            dburl = session.bind.engine.url
            session.bind.engine.dispose()
            # little hacky (2): We need to pass pickable stuff to each child sub-process,
            # therefore no imported functions.
            pyfile = inspect.getsourcefile(pyfunc)
            funcname = pyfunc.__name__

            def mp_initializer():
                '''set up the worker processes to ignore SIGINT altogether,
                and confine all the cleanup code to the parent process (e.g. Ctrl+C pressed)'''
                # https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
                signal.signal(signal.SIGINT, signal.SIG_IGN)

            pool = Pool(processes=num_processes, initializer=mp_initializer)
            mp_map = pool.imap_unordered
            try:
                for results in mp_map(process_segments_mp,
                                      ((seg_sta_chunk, dburl, config, pyfile, funcname)
                                       for seg_sta_chunk in get_slices(seg_sta_ids, chunksize))):
                    for output, is_ok, segment_id in results:
                        process_output(output, is_ok, segment_id, ondone, done_skipped_errors)
                    pbar.update(len(results))
            # https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
            # (2nd answer) and links therein:
            # https://noswap.com/blog/python-multiprocessing-keyboardinterrupt
            # https://github.com/jreese/multiprocessing-keyboardinterrupt
            except:
                pool.terminate()
                pool.join()
                raise
            else:
                pool.close()
                pool.join()
        else:
            sta_query = session.query(Station)  # for getting station (cache)
            inventory = None  # currently processed inventory (Inventory object, Exception or None)
            station_id = None  # currently processed station id (integer)

            # load all segments at once. The number 1200 seems to be a reasonable choice
            for seg_sta_chunk in get_slices(seg_sta_ids, chunksize):
                # clear identity map, i.e. the cache-like sqlalchemy object, to free memory.
                # do it before querying otherwise all queried stuff is detached from the session
                # Keep the station in case, as relationships like segment.station will
                # use the cached value in case and we avoid re-loading data:
                # http://docs.sqlalchemy.org/en/latest/orm/query.html#sqlalchemy.orm.query.Query.yield_per
                station = sta_query.get(int(station_id)) \
                    if inventory is not None and station_id == seg_sta_chunk[0, 1] else None
                session.expunge_all()  # clear session identity map (sort of cache)
                if station is not None:  # re-assign station and related stuff (datacenter),
                    # we could use merge but it's useless as we do not have other instances
                    # http://docs.sqlalchemy.org/en/latest/orm/session_state_management.html#merging
                    # Note that also related objects are added (e.g., station's datacenter)
                    # The instance should be persistent after adding it, for info see:
                    # http://docs.sqlalchemy.org/en/latest/orm/session_state_management.html
                    session.add(station)
                # the query with "in" operator might NOT return segments in the same order
                # as the ids provided in the "in" argument. Store them into a dict:

#                 segments = query_segments(seg_query, seg_sta_chunk[:, 0])
#                 station_ids = seg_sta_chunk[:, 1]  # does not allocate new memory
#                 for segment, sta_id in zip(segments, station_ids):
#                     if station_id == sta_id:  # set cached inventory object (might be None)
#                         segment._inventory = inventory  # pylint: disable=protected-access
#                     output, is_ok = process_segment(segment, config, pyfunc)
#                     process_output(output, is_ok, segment.id, ondone, done_skipped_errors)
#                     # set cached inventory using private-like field:
#                     inventory = getattr(segment, "_inventory", None)
#                     station_id = sta_id
#                 pbar.update(len(station_ids))

                # station_id and inventory are updated and used only in the outer loop (see above)
                for output, is_ok, segment_id, station_id, inventory in \
                        process_segments(session, seg_sta_chunk, config, pyfunc, station_id,
                                         inventory):
                    process_output(output, is_ok, segment_id, ondone, done_skipped_errors)
                pbar.update(int(seg_sta_chunk.shape[0]))

#     captured_warnings = warn_string_io.getvalue()
#     if captured_warnings:
#         logger.info("(external warnings captured, if provided, see log file for details)")
#         logger.info("")
#         logger.warning("Captured external warnings:")
#         logger.warning("%s", captured_warnings)
#         logger.warning("(only the first occurrence of an external warning for each location "
#                        "where the warning is issued is reported. Because of maintainability "
#                        "and performance potential issues, the segment id which originated "
#                        "these warnings cannot be shown. However, in most cases the process "
#                        "completed successfully, and if you want to check the correctness of "
#                        "the data please check the results)")
# 
#     logging.captureWarnings(False)  # form the docs the redirection of warnings to the logging
#     # system will stop, and warnings will be redirected to their original destinations
#     # (i.e. those in effect before captureWarnings(True) was called).

    done, skipped, errors = done_skipped_errors
    # get stations with data and inform the user if any new has been saved:
    stasaved2 = stationssaved()
    if stasaved2 > stasaved:
        logger.info("station inventories saved: %d", (stasaved2-stasaved))

    logger.info("%d of %d segments successfully processed\n", done, seg_len)
    if skipped:  # this is the case when ondone is provided AND pyfunc returned None
        logger.info("%d of %d segments skipped without messages\n", skipped, seg_len)
    logger.info("%d of %d segments skipped with error message "
                "(check log or details)\n", errors, seg_len)


def process_segments_mp(args):
    seg_sta_chunk, dburl, config, pyfile, funcname = args
    pyfunc = load_pyfunc(pyfile, funcname)
    session = get_session(dburl)
    ret = []
#     with create_processing_env(0, config, redirect_stderr=True, warnings_filter='ignore'):
#         segments = query_segments(get_seg_query(session), segment_ids)
#         try:
#             for segment in segments:
#                 output, is_ok = process_segment(segment, config, pyfunc)
#                 ret.append(segment.id, output, is_ok)
#             return ret
#         finally:
#             session.close()
#             session.bind.engine.dispose()
    with create_processing_env(0, config, redirect_stderr=True, warnings_filter='ignore'):
        try:
            for output, is_ok, segment_id, station_id, inventory in \
                        process_segments(session, seg_sta_chunk, config, pyfunc):
                ret.append((output, is_ok, segment_id))
            return ret
        finally:
            session.close()
            session.bind.engine.dispose()


def process_segments(session, seg_sta_chunk, config, pyfunc, station_id=None, inventory=None):
    segments = joinandorder(session.query(Segment)).\
        filter(Segment.id.in_(seg_sta_chunk[:, 0].tolist()))
    station_ids = seg_sta_chunk[:, 1]  # does not allocate new memory
    for segment, sta_id in zip(segments, station_ids):
        if station_id == sta_id:  # set cached inventory object (might be None)
            segment._inventory = inventory  # pylint: disable=protected-access
        output, is_ok = process_segment(segment, config, pyfunc)
        # set cached inventory using private-like field:
        inventory = getattr(segment, "_inventory", None)
        station_id = sta_id
        yield output, is_ok, segment.id, station_id, inventory


def process_segment(segment, config, pyfunc):
    try:
        return pyfunc(segment, config), True
    except (ImportError, NameError, AttributeError, SyntaxError,
            TypeError) as _:
        raise  # sys.exc_info()
    except Exception as generr:
        return generr, False


def process_output(output, is_ok, segment_id, ondone, done_skipped_errors):
    '''to be executed in the main process'''
    if is_ok:
        if ondone:
            if output is not None:
                ondone(segment_id, output)
                done_skipped_errors[0] += 1
            else:
                done_skipped_errors[1] += 1
        else:
            done_skipped_errors[0] += 1
    else:
        logger.warning("segment (id=%d): %s", segment_id, str(output))
        done_skipped_errors[2] += 1


def query4process(session, conditions=None):
    '''Returns a query yielding the the segments ids (and their stations ids) for the processing.
    The returned tuples are sorted by station id, event id, channel location and channel's channel

    :param session: the sql-alchemy session
    :param condition: a dict of segment attribute names mapped to a select expression, each
    identifying a filter (sql WHERE clause). See `:ref:sqlevalexpr.py`. Can be empty (no filter)

    :return: a query yielding the tuples: ```(Segment.id, Segment.station.id)```
    '''
    # Note: without the join below, rows would be duplicated
    qry = joinandorder(session.query(Segment.id, Station.id))
    # Now parse selection:
    if conditions:
        # parse user defined conditions (as dict of key:value <=> "column": "expr")
        qry = exprquery(qry, conditions=conditions, orderby=None, distinct=True)
    return qry


def joinandorder(query):
    return query.join(Segment.station, Segment.channel).\
        order_by(Station.id, Segment.event_id, Channel.location, Channel.channel)


# def get_seg_query(session):
#     return order(session.query(Segment))


@contextmanager
def create_processing_env(length=0, config=None, redirect_stderr=False, warnings_filter=None):
    '''Context manager to be used in a with statement, returns the progress bar
    which can be called with pbar.update(int). The latter is no-op if length ==0

    Typical usage without multi-processing from the main function (activate all 'with' statements):
    ```
        with create_proc_env(10, config, redirect_stderr=True, 'ignore') as pbar:
            ...
            pbar.update(1)
    ```
    Typical usage with multi-processing from the main function (activate only progressbar 'with'
    statement):
    ```
        with create_proc_env(10, None, redirect_stderr=False, None) as pbar:
            ...
            pbar.update(1)
    ```
    Typical usage with multi-processing from a child process  (activate all but progressbar 'with'
    statement):
    ```
        with create_proc_env(0, config, redirect_stderr=True, 'ignore') as pbar:
            ...
            pbar.update(1)
    ```

    :param length: the number of tasks to be done. If zero, the returned progressbar will
    be no-op. Otherwise, it is an object which updates a progressbar on terminal
    :param config: if None, it does not enhance the Segment class. Otherwise, it adds to it
    methods for processing (e.g., segment.stream())
    :param redirect_stderr: if True, captures the output of all C external functions and does not
    print them to the screen
    :param warnings_filter: if None, it does not capture python warnings. Otherwise it denotes the
    python filter. E.g. 'ignore'. For info see FIXME: add link
    '''
    with get_progressbar(length > 0, length=length) as pbar:  # no-op if length not > 0
        with redirect(sys.stderr if redirect_stderr else None):   # no-op if redirect_stderr is None
            if config is not None and warnings_filter:
                with warnings.catch_warnings():
                    warnings.simplefilter(warnings_filter)
                    with enhancesegmentclass(config):
                        yield pbar
            elif config is not None:
                with enhancesegmentclass(config):
                    yield pbar
            elif warnings_filter:
                with warnings.catch_warnings():
                    warnings.simplefilter(warnings_filter)
                    yield pbar
            else:
                yield pbar


# THE FUNCTION BELOW REDIRECTS STANDARD ERROR/OUTPUT FROM EXTERNAL PROGRAM
# http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
# there's a second one easier to understand but does not restore old std/err stdout
# Added comments from
# http://stackoverflow.com/questions/8804893/redirect-stdout-from-python-for-c-calls
@contextmanager
def redirect(src=None, dst=os.devnull):
    '''
    FIXME: write doc
    import os

    with stdout_redirected(src=sys.stderr):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''

    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything
    # Moreover, passing None as first argument means no redirection
    try:
        file_desc = src.fileno()
    except (AttributeError, OSError) as _:
        yield
        return

    # # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == file_desc == 1

    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), file_desc)  # file_desc writes to 'to' file
        sys.stderr = os.fdopen(file_desc, 'w')  # Python writes to file_desc

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), file_desc)  # file_desc writes to 'to' file
        sys.stdout = os.fdopen(file_desc, 'w')  # Python writes to file_desc

    _redirect_ = _redirect_stderr if src is sys.stderr else _redirect_stdout

    with os.fdopen(os.dup(file_desc), 'w') as old_:
        with open(dst, 'w') as fopen:
            _redirect_(to=fopen)
        try:
            yield  # allow code to be run with the redirected stdout/err
        finally:
            # restore stdout. buffering and flags such as CLOEXEC may be different:
            _redirect_(to=old_)
