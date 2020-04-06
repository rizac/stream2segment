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
import time
import sys
import logging
from contextlib import contextmanager
import warnings
import inspect
from multiprocessing import Pool, cpu_count
import signal
from itertools import chain, repeat

# from future.utils import itervalues

import numpy as np

from sqlalchemy import func
# from sqlalchemy.orm import load_only

from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.utils import get_progressbar, StringIO
from stream2segment.process.db import get_session, configure_classes
from stream2segment.io.db.models import Segment, Station, Event, Channel, DataCenter
from stream2segment.utils.inputargs import load_pyfunc


logger = logging.getLogger(__name__)


def run(session, pyfunc, writer, config=None, show_progress=False):
    '''Runs the processing routine

    :param session: the sql-alchemy db session
    :pram pyfunc: a python function to be executed on all selected segments
    :param writer: a Writer handling the processed output to filesystem.
        use :class:`writers.BaseWriter` for a no-op class. The writer handles the append
        feature for existing output files and is a callable with signature:
        ```
        def __call__(segment_id, result):  # result is surely not None
        ```
    where result is an iterable of values/objects, and segment_id is ... the segment id.
    :param config: dict of configuration parameters, ususally the result of an associated YAML
    configuration file
    :param skip_ids: an iterable of integers denoting segments to be skipped among the
    selected ones. None (the default) is equivalent to pass the empty list []
    :param show_progress: (boolean) whether or not to show progress status and remaining time
    on the terminal
    '''
    if config is None:
        config = {}
    done_skipped_errors = [0, 0, 0]

    seg_ids = fetch_segments_ids(session, config, writer)
    seg_len = len(seg_ids)
    # get total segment length (in numpy it is equivalent to len(seg_sta_ids)):
    chunksize, multi_process, num_processes = \
        get_advanced_settings(config, seg_len, show_progress)
    # Note on chunksize above:
    # When loading segments, we have two strategies:
    # A) load only a Segment with its id (and defer loading of other
    # attributes upon access) or B) load a Segment with all attributes
    # (columns). From experiments on a 16 GB memory Mac:
    # Querying A) and then accessing (=loading) later two likely used attributes
    # (data and arrival_time) we take:
    # ~= 0.043 secs/segment, Peak memory (Kb): 111792 (0.650716 %)
    # Querying B) and then accessing the already loaded data and arrival_time attributes,
    # we get:
    # 0.024 secs/segment, Peak memory (Kb): 409194 (2.381825 %).
    # With millions of segments, the latter
    # approach can save up to 9 hours with almost no memory perf issue (2.4% vs 0.7%).
    # So we define a chunk size whereby we load all segments:

    # set/update classes, if written in the config, so that we can set instance classes in the
    # processing, if we want:
    configure_classes(session, config.get('class_labels', []))
    session.close()  # expunge all, clear all states

    with writer:
        # redirection of stderr prevents Python BUT ALSO external
        # libraries to print unwanted stuff on screen which might mess
        # up the progressbar. Note that this should capture also warnings
        # because they are normally directed to stderr. This is why warnings_filter is None,
        # (also, this method is usually called inside a with statements ignoring warnings)
        # Sub-processes will call (see below) create_processing_env with redirect_stderr=False
        # (we do not want to mess up too much with leaking file descriptors, a complex topic)
        # and warnings_filter='ignore', because we would have too many redundant messages
        with create_processing_env(seg_len if show_progress else 0,
                                   redirect_stderr=True,
                                   warnings_filter=None) as pbar:
            if show_progress and seg_len:
                # if we are here we want to show the progressbar. Problem is, we can not
                # know in how much time the progress bar will be rendered on the terminal
                # (for heavy user-defined processing routines, this might happen in several
                # minutes). Showing the progress bar immediately is therefore of help for
                # the user (even if the bar has obviously no progress done yet). Is there a
                # 'flush' method? Looking at click, it's 'render_progress()'. But the latter
                # does not work if called within 'pbar.short_limit' seconds after the progress
                # bar has been created. Therefore, let's force a progressabr flush. First wait:
                time.sleep(pbar.short_limit)
                # now render_progress will render:
                pbar.render_progress()

            if multi_process:
                process_mp(session, pyfunc, config, get_slices(seg_ids, chunksize), writer,
                           done_skipped_errors, pbar, num_processes)
            else:
                process_simple(session, pyfunc, config, get_slices(seg_ids, chunksize),
                               writer, done_skipped_errors, pbar)

    logger.info('')
    done, skipped, errors = done_skipped_errors

    logger.info("%d of %d segment(s) successfully processed", done, seg_len)
    if skipped:  # this is the case when ondone is provided AND pyfunc returned None
        logger.info("%d of %d segment(s) skipped without messages", skipped, seg_len)
    logger.info("%d of %d segment(s) skipped with error message "
                "reported in the log file", errors, seg_len)
    logger.info('')


def fetch_segments_ids(session, config, writer):
    '''Returns the numpy array of segments ids to process

    :return: the numpy array of integers denoting the ids of the segments to process
        according to `config` and `writer` settings
    '''

    # (Over commenting this module to keep track of all choices done)

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
    qry = query4process(session, config.get('segment_select', {}))

    skip_already_processed = False
    if writer.append:
        if not writer.outputfileexists:
            logger.info('Ignoring `append` functionality: output file does not exist '
                        'or not provided')
        else:
            logger.info('Appending results to existing file')
            skip_already_processed = True
    elif writer.outputfileexists:
        logger.info('Overwriting existing output file')

    logger.info("Fetching segments to process (please wait)")
    seg_ids = np.array(qry.all(), dtype=int).flatten()
    # we flatten the array because the qry.all() returns 1element tuples,
    # so we want to convert e.g. [[1], [5], [6]] to [1, 5, 6]

    if skip_already_processed:
        # it might be more elegant to issue a query with a NOT IS IN ...
        # and the already processed ids. But for large files, the query might be huge
        # not necessarily faster (we need to build a string from a huge numpy array)
        # but more importantly the database might simply not support such a long query string,
        # and raise Exceptions.
        # It is therefore way more efficient to do it in numpy. The drawback is that
        # we will query more segments than needed and we might waste time in the query above
        # but this is outweighted by the efficiency here
        logger.info("Fetching already processed segment(s) (please wait)")
        skip_ids = writer.already_processed_segments()
        logger.info("Skipping %d already processed segment(s)", len(skip_ids))
        seg_ids = np.copy(seg_ids[np.isin(seg_ids, skip_ids, assume_unique=True, invert=True)])

    logger.info("%d segment(s) found to process", len(seg_ids))
    logger.info('')
    return seg_ids


def get_advanced_settings(config, segments_count, show_progress):
    '''Extracts the advanced settings from the given config, returning their defaults
    if not given.

    :return: the tuple chunksize (int, default:depends on segments_count and show_progress,
                                  however it stems from a default_chunk_size of 1200),
        multi_process (boolean, default:False), num_processes (int, default: cpu_count())
    '''
    adv_settings = config.get('advanced_settings', {})  # dict
    multi_process = adv_settings.get('multi_process', False)
    num_processes = adv_settings.get('num_processes', None)
    if num_processes is None:
        num_processes = cpu_count()
    chunksize = adv_settings.get('segments_chunksize', None)
    if chunksize is None:
        default_chuknksize, min_pbar_iterations = _get_chunksize_defaults()
        if not show_progress or segments_count >= 2 * default_chuknksize:
            chunksize = default_chuknksize
        else:
            # determine the chunlsize in order to have `min_pbar_iterations` iterations
            # use np.true_divide so that py2/3 division is not a problem:
            chunksize = max(1, int(np.true_divide(segments_count, min_pbar_iterations).item()))
    return chunksize, multi_process, num_processes


def _get_chunksize_defaults():
    '''returns a 2 element tuple with
    the default segment chunksize (1200) and the minimum progressbar iterations (10)
    This function is implemented mainly for mocking in tests
    '''
    return 1200, 10


def process_mp(session, pyfunc, config, seg_ids_chunks, writer, done_skipped_errors, pbar,
               num_processes):
    '''Executes `pyfunc` using the multiprocessing Python module

    :param seg_ids_chunks: iterable yielding numpy arrays of segment ids
    :param done_skipped_errors: list of three int elements denoting segments done, skipped, and
        discarded (due to errors), respectively
    '''
    # The two actions here below are a little hacky in that we might simply pass
    # strings to this functions (dburl and python module path), but this would require
    # checking for well formed dburl and paths here, which is what we do BEFORE, and
    # also, the non pythin-multiprocessing case benefits of having already db session
    # object and python function loaded.

    # 1. The db engine has to be disposed now if we want to use multi-processing:
    # https://stackoverflow.com/questions/41279157/connection-problems-with-sqlalchemy-and-multiple-processes
    dburl = session.bind.engine.url
    session.bind.engine.dispose()
    # 2. We need to pass pickable stuff to each child sub-process,
    # therefore no imported functions, get the file instead:
    pyfile = inspect.getsourcefile(pyfunc)

    pool = Pool(processes=num_processes, initializer=_mp_initializer)
    try:
        for results in \
                pool.imap_unordered(process_segments_mp,
                                    ((seg_ids_chunk, dburl, config, pyfile, pyfunc.__name__)
                                     for seg_ids_chunk in seg_ids_chunks)):
            for output, is_ok, segment_id in results:
                process_output(output, is_ok, segment_id, writer, done_skipped_errors)
            pbar.update(len(results))
    except:  # @IgnorePep8 pylint: disable=bare-except
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()


def _mp_initializer():
    '''set up the worker processes to ignore SIGINT altogether,
    and confine all the cleanup code to the parent process (e.g. Ctrl+C pressed)'''
    # See https://stackoverflow.com/a/6191991 and links therein:
    # https://noswap.com/blog/python-multiprocessing-keyboardinterrupt
    # https://github.com/jreese/multiprocessing-keyboardinterrupt
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_simple(session, pyfunc, config, seg_ids_chunks, writer, done_skipped_errors, pbar):
    '''Executes `pyfunc` in a single system process

    :param seg_ids_chunks: iterable yielding numpy arrays of segment ids
    :param done_skipped_errors: list of three int elements denoting segments done, skipped, and
        discarded (due to errors), respectively
    '''
    segment = None  # currently processed segment id (will be used later)
    _clear_session(session)

    for seg_ids_chunk in seg_ids_chunks:
        for output, is_ok, segment in \
                process_segments(session, seg_ids_chunk, config, pyfunc):
            # `process_segment` uses internally an inventory cache mechanism, i.e.
            # it avoids loading an inventory again, if we have it already calculated.
            # (this is why, to make the mechanism easier, we query to the db segments
            # in a particular order according to their station id first, see
            # :func:`joinandorder`).
            # `process_segment` yields also station_id and inventory with the only
            # purpose to perform the same inventory cache mechanism also on the first
            # segment of the next `seg_sta_chunk` loop (see above)
            process_output(output, is_ok, segment.id, writer, done_skipped_errors)

        _clear_session(session, segment)
        pbar.update(len(seg_ids_chunk))


def _clear_session(session, segment=None):
    '''Clears the session and re-assigns `segment` to it

    :param segment: A Segment object denoting the last processed segment, or None
    '''
    # clear session identity map (sort of cache), freeing memory:
    session.expunge_all()

    if segment is None:
        return

    # re-assign the segment. This will add also to session.identity_map
    # all the segment's already loaded related objects (e.g., Station, with
    # relative inventory). This will not garbage collect these objects (keeping
    # them in memory) but if in the next loop we work
    # with segments sharing the related objects (which might happen because
    # segments are sorted by station id and event id, see `query4process'), accessing
    # those objects (e.g. Segment.inventory()) is faster because no database query
    # is issued. For info see:
    # https://docs.sqlalchemy.org/en/13/orm/query.html#sqlalchemy.orm.query.Query.get
    # Note: session.merge it's useless as we don't have other instances in the identity map:
    # http://docs.sqlalchemy.org/en/latest/orm/session_state_management.html#merging
    session.add(segment)


def process_segments_mp(args):
    '''Function to be used INSIDE A CHILD python-process to process a chunk of segments
    Takes care of releasing db session and other stuff.

    :param args: the tuple (seg_ids_chunk, dburl, config, pyfile, funcname)
        where seg_ids_chunk is a numpy array of segment ids, dburl is the database
        url (string), config is the config dict, pyfile is the path to the processing
        module (string) and funcname is the processing function name to be called

    :return: a list of (output, is_ok, segment_id) tuples, where output is the
    output of the processing funcion name (either iterable or Exception), is_ok (boolean)
    tells if `output` is NOT an Exception, and segment_id is the id of the segment processed
    '''
    seg_ids_chunk, dburl, config, pyfile, funcname = args
    pyfunc = load_pyfunc(pyfile, funcname)
    session = get_session(dburl)
    ret = []

    with create_processing_env(0, redirect_stderr=False, warnings_filter='ignore'):
        try:
            for output, is_ok, segment in \
                        process_segments(session, seg_ids_chunk, config, pyfunc):
                # `process_segment` uses internally an inventory cache mechanism, i.e.
                # it avoids loading an inventory again, if we have it already calculated.
                # (this is why, to make the mechanism easier, we query to the db segments in a
                # particular order according to their station id first, see :func:`joinandorder`).
                # `process_segment` yields also station_id and inventory, but they are useless
                # here: they are used only when `process_segments` is called inside an outer loop,
                # to perform the same inventory cache mechanism also on the first segment of the
                # next loop
                ret.append((output, is_ok, segment.id))
            return ret
        finally:
            session.close()
            session.bind.engine.dispose()


def process_segments(session, seg_ids_chunk, config, pyfunc):
    '''Function that proceses a chunk of segments and yield the resulted output.
    Yields a list of (output, is_ok, segment_id, station_id, inventory) tuples, where output is the
    output of the processing funcion name (either iterable or Exception), is_ok (boolean)
    tells if `output` is NOT an Exception, segment_id is the id of the segment processed,
    station_id the segment station id, inventory is the station inventory, which might be None
    if an inventory was not requested in `pyfunc` implementation, or an Exception if the
    operation of reading the inventory raised.

    :param session: the db session (sql-alcheemy session)
    :param seg_ids_chunk: a numpy array of segment ids
    :param config: the config dict
    :param pyfunc: a python function to be invoked on each segment

    if this function is called in a loop on the same python process. Otherwise ignore (pass None)
    '''
    # To make the query with "in" operator return segments in the same order
    # (station id, then event id etceters) use joinandorder:
    segments = joinandorder(session.query(Segment)).\
        filter(Segment.id.in_(seg_ids_chunk.tolist()))
    for segment in segments:
        output, is_ok = process_segment(segment, config, pyfunc)
        yield output, is_ok, segment


def process_segment(segment, config, pyfunc):
    '''processes a signle segment and return the output of
    `pyfunc(segment, config)

    :return: the tuple (output, is_ok), where output is either an iterable or a `ValueError`.
        in the former case, `is_ok` is True, otherwise False (the variable is returned
        as a faster alias of `isinstance(output, Exception)`)

    Note that any exception other than `ValueError` will raise and thus interrupt the program
    `'''
    try:
        return pyfunc(segment, config), True
    except ValueError as valueerr:
        return valueerr, False
    except Exception:
        raise


def process_output(output, is_ok, segment_id, writer, done_skipped_errors):
    '''Function processing the output of `:func:process_segment`
    This function MUST be executed in the main python-process, and not from within sub-processes'''
    if is_ok:
        if writer.isbasewriter:
            done_skipped_errors[0] += 1
        else:
            if output is not None:
                writer.write(segment_id, output)
                done_skipped_errors[0] += 1
            else:
                done_skipped_errors[1] += 1
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
    qry = joinandorder(session.query(Segment.id))
    # Now parse selection:
    if conditions:
        # parse user defined conditions (as dict of key:value <=> "column": "expr")
        qry = exprquery(qry, conditions=conditions, orderby=None)
    return qry


def joinandorder(query):
    '''Given an already built query for processing, adds joins and order_by to the
    query in order to return segments ordered by stationid, then event, and then
    channel's location and channel's channel.

    :param query: an sql-alchemy query object
    :return: a new query identical to the passed `auery` argument with joined table and
    oredr_by applied
    '''

    return query.join(Segment.station, Segment.channel).\
        order_by(Station.id, Segment.event_id, Channel.location, Channel.channel)


@contextmanager
def create_processing_env(length=0, redirect_stderr=False, warnings_filter=None):
    '''Context manager to be used in a with statement, returns the progress bar
    which can be called with pbar.update(int). The latter is no-op if length ==0

    Typical usage without multi-processing from the main function (activate all 'with' statements):
    ```
        with create_proc_env(10, redirect_stderr=True, 'ignore') as pbar:
            ...
            pbar.update(1)
    ```
    Typical usage with multi-processing from the main function (activate only progressbar 's
    'with' statement):
    ```
        with create_proc_env(10, redirect_stderr=True, None) as pbar:
            ...
            pbar.update(1)
    ```
    Typical usage with multi-processing from a child process  (activate only ignrore warnings):
    ```
        with create_proc_env(0, redirect_stderr=False, 'ignore') as pbar:
            ...
            pbar.update(1)
    ```

    :param length: the number of tasks to be done. If zero, the returned progressbar will
    be no-op. Otherwise, it is an object which updates a progressbar on terminal
    :param redirect_stderr: if True, captures the output of all C external functions and does not
    print them to the screen, as it might be the case with some obspy C-imported libraries
    :param warnings_filter: if None, it does not capture python warnings. Otherwise it denotes the
    python filter. E.g. 'ignore'. (FIXME: add link)
    '''
    with get_progressbar(length > 0, length=length) as pbar:  # no-op if length not > 0
        with redirect(sys.stderr if redirect_stderr else None):   # no-op if redirect_stderr=None
            if warnings_filter:
                with warnings.catch_warnings():
                    warnings.simplefilter(warnings_filter)
                    yield pbar
            else:
                yield pbar


@contextmanager
def redirect(src=None, dst=os.devnull):
    '''
    This method prevents Python AND external C shared library to print to stdout/stderr in python,
    preventing also leaking file descriptors.
    If the first argument is None or any object not having a fileno() argument, this
    context manager is simply no-op and will yield and then return

    See (in this order):
    https://stackoverflow.com/a/14797594
    and (final solution modified here):

    Example

    with redirect(sys.stdout):
        print("from Python")
        os.system("echo non-Python applications are also supported")

    :param src: file-like object with a fileno() method. Usually is either `sys.stdout` or
        `sys.stderr`.
    '''

    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything
    # Moreover, passing None as first argument means no redirection
    try:
        file_desc = src.fileno()
    except (AttributeError, OSError) as _:
        yield
        return

    # if you want to assert that Python and C stdio write using the same file descriptor:
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == file_desc == 1

    def _redirect_stderr_to(fileobject):
        sys.stderr.close()  # + implicit flush()
        # make `file_desc` point to the same file as `fileobject`.
        # First closes file_desc if necessary:
        os.dup2(fileobject.fileno(), file_desc)
        # Make Python write to file_desc
        sys.stderr = os.fdopen(file_desc, 'w')

    def _redirect_stdout_to(fileobject):
        sys.stdout.close()  # + implicit flush()
        # make `file_desc` point to the same file as `fileobject`.
        # First closes file_desc if necessary:
        os.dup2(fileobject.fileno(), file_desc)
        # Make Python write to file_desc
        sys.stdout = os.fdopen(file_desc, 'w')

    _redirect_to = _redirect_stderr_to if src is sys.stderr else _redirect_stdout_to

    with os.fdopen(os.dup(file_desc), 'w') as src_fileobject:
        with open(dst, 'w') as dst_fileobject:
            _redirect_to(dst_fileobject)
        try:
            yield  # allow code to be run with the redirected stdout/err
        finally:
            # restore stdout/err. buffering and flags such as CLOEXEC may be different:
            _redirect_to(src_fileobject)


def get_slices(array, chunksize):
    '''Divides len(array)
    by `chunksize` yielding the array slices until exaustion.
    If `array` is an integer, it denotes the length of the array and the tuples (start, end)
    will be yielded.
    This method intelligently re-arranges the (start, end) indices in order to minimize the
    number of iterations yielded. ``
    '''
    if hasattr(array, '__len__'):
        total = len(array)  # == array.shape[0] in case of numpy arrays
    else:
        total = array
        array = None
    rem = total % chunksize
    quot = int(np.true_divide(total, chunksize))
    if rem == 0:
        # eg: total=6, chunksize=2:  rem=0, quot=3
        # repeat 2 three times:
        iterable = repeat(chunksize, quot)
    elif quot > rem:
        # eg: total=7, chunksize=2: rem=1, quot=3
        # a) repeat 2 two times
        # b) repeat 3 one time
        # start with a) so that the 1st progressbar update might be slightly faster:
        iterable = chain(repeat(chunksize, quot-rem), repeat(chunksize+1, rem))
    else:
        # eg: total=7, chunksize=5: rem=2, quot=1
        # a) yield 2 (one time)
        # b) repeat 5 one time
        # start with a) so that the 1st progressbar update might be slightly faster:
        iterable = chain([rem], repeat(chunksize, quot))
    start = end = 0
    for chunk in iterable:
        start = end
        end = start + chunk
        yield array[start:end] if array is not None else (start, end)
