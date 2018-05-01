'''
Main module for the segment processing and .csv output

Created on Feb 2, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      super, zip)

# iterating over dictionary keys with the same set-like behaviour on Py2.7 as on Py3:
# from future.utils import viewkeys

import os
import sys
import logging
from contextlib import contextmanager
import warnings

import numpy as np

from sqlalchemy import func, inspect
from sqlalchemy.orm import load_only

from stream2segment.process.utils import enhancesegmentclass, set_classes, get_slices
from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.utils import get_progressbar, StringIO
from stream2segment.io.db.models import Segment, Station, Event, Channel


logger = logging.getLogger(__name__)


def run(session, pyfunc, ondone=None, config=None, show_progress=False):
    if config is None:
        config = {}

    # suppress obspy warnings
    # # https://docs.python.org/2/library/warnings.html#the-warnings-filter
    warnings.filterwarnings("default")
    warn_string_io = StringIO()
    logger_handler = logging.StreamHandler(warn_string_io)
    logger_handler.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(logger_handler)

    # multiprocess with sessions is a mess. Among other problems, we should not share any
    # session-related operation with the same session object across different multiprocesses.
    # Is it worth to create a new session each time? not for the moment

    # NOT USED ANYMORE: maybe in the future if we exprience memory problems:
    # clear the session and expunge all every clear_session_step iterations:
    # (set a multiple of three might be in sync with other orientations, which is 3):
    # clear_session_step = 60

    logger.info("Fetching segments to process, please wait...")

    inventory = None  # currently processed inventory (Inventory object, Exception or None)
    station_id = None  # currently processed station id (integer)
    done, skipped, skipped_error = 0, 0, 0

    with enhancesegmentclass(config):

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
        seg_query = session.query(Segment)  # allocate once (speeds up a bit)
        sta_query = session.query(Station)  # for getting station (cache)

        # Now we have to process each segment:
        # Two strategies: A) load only a Segment with its id (and defer loading of other
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
        chunksize = config.get('advanced_settings', {}).get('segments_chunk', 1200)

        with redirect(sys.stderr):
            with get_progressbar(show_progress, length=seg_len) as pbar:

                # load all segments at once. The number 1200 seems to be a reasonable choice
                for seg_sta_chunk in get_slices(seg_sta_ids, chunksize):
                    # clear identity map, i.e. the cache-like sqlalchemy object, to free memory.
                    # do it before querying otherwise all queried stuff is detached from the session
                    # Keep the station in case, as relationships like segment.station will
                    # use the cached value in case and we avoid re-loading data:
                    # http://docs.sqlalchemy.org/en/latest/orm/query.html#sqlalchemy.orm.query.Query.yield_per
                    station = sta_query.get(station_id) \
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
                    segments = {s.id: s for s in
                                seg_query.filter(Segment.id.in_(seg_sta_chunk[:, 0].tolist()))}
                    for seg_id, sta_id in seg_sta_chunk:
                        segment = segments[seg_id]
                        if station_id == sta_id:  # set cached inventory object (might be None)
                            segment._inventory = inventory  # pylint: disable=protected-access
                        try:
                            array_or_dic = pyfunc(segment, config)
                            if ondone:
                                if array_or_dic is not None:
                                    ondone(segment, array_or_dic)
                                    done += 1
                                else:
                                    skipped += 1
                            else:
                                done += 1
                        except (ImportError, NameError, AttributeError, SyntaxError,
                                TypeError) as _:
                            raise  # sys.exc_info()
                        except Exception as generr:
                            logger.warning("segment (id=%d): %s", segment.id, str(generr))
                            skipped_error += 1
                        # set cached inventory using private-like field:
                        inventory = getattr(segment, "_inventory", None)
                        station_id = sta_id

                        pbar.update(1)

        captured_warnings = warn_string_io.getvalue()
        if captured_warnings:
            logger.info("(external warnings captured, if provided, see log file for details)")
            logger.info("")
            logger.warning("Captured external warnings:")
            logger.warning("%s", captured_warnings)
            logger.warning("(only the first occurrence of an external warning for each location "
                           "where the warning is issued is reported. Because of maintainability "
                           "and performance potential issues, the segment id which originated "
                           "these warnings cannot be shown. However, in most cases the process "
                           "completed successfully, and if you want to check the correctness of "
                           "the data please check the results)")

        logging.captureWarnings(False)  # form the docs the redirection of warnings to the logging
        # system will stop, and warnings will be redirected to their original destinations
        # (i.e. those in effect before captureWarnings(True) was called).

        # get stations with data and inform the user if any new has been saved:
        stasaved2 = stationssaved()
        if stasaved2 > stasaved:
            logger.info("station inventories saved: %d", (stasaved2-stasaved))

        logger.info("%d of %d segments successfully processed\n", done, seg_len)
        if skipped:  # this is the case when ondone is provided AND pyfunc returned None
            logger.info("%d of %d segments skipped without messages\n", skipped, seg_len)
        logger.info("%d of %d segments skipped with error message "
                    "(check log or details)\n", skipped_error, seg_len)


# THE FUNCTION BELOW REDIRECTS STANDARD ERROR/OUTPUT FROM EXTERNAL PROGRAM
# http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
# there's a second one easier to understand but does not restore old std/err stdout
# Added comments from
# http://stackoverflow.com/questions/8804893/redirect-stdout-from-python-for-c-calls
@contextmanager
def redirect(src=sys.stdout, dst=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''

    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything:
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


def query4process(session, conditions=None):
    '''Returns a query yielding the the segments ids (and their stations ids) for the processing.
    The returned tuples are sorted by station id, event id, channel location and channel's channel

    :param session: the sql-alchemy session
    :param condition: a dict of segment attribute names mapped to a select expression, each
    identifying a filter (sql WHERE clause). See `:ref:sqlevalexpr.py`. Can be empty (no filter)

    :return: a query yielding the tuples: ```(Segment.id, Segment.station.id)```
    '''
    # Note: without the join below, rows would be duplicated
    qry = session.query(Segment.id, Station.id).\
        join(Segment.station, Segment.event, Segment.channel).\
        order_by(Station.id, Channel.location, Channel.channel)
    # Now parse selection:
    if conditions:
        # parse user defined conditions (as dict of key:value <=> "column": "expr")
        qry = exprquery(qry, conditions=conditions, orderby=None, distinct=True)
    return qry
