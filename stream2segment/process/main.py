"""
Main module for the segment processing and .csv output

Created on Feb 2, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (  # int,
    open)

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

import numpy as np

from stream2segment.io.db import secure_dburl, close_session
from stream2segment.process.db.sqlevalexpr import exprquery
from stream2segment.io.log import logfilepath, close_logger, elapsed_time
from stream2segment.io.cli import get_progressbar, ascii_decorate
from stream2segment.io.inputvalidation import validate_param, valid_session
from stream2segment.process.inputvalidation import load_config_for_process, valid_pyfunc
from stream2segment.process import SkipSegment
from stream2segment.process.db.models import get_session, Segment, Station
from stream2segment.process.log import configlog4processing
from stream2segment.process.writers import get_writer


# make the logger refer to the parent of this package (`rfind` below. For info:
# https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__[:__name__.rfind('.')])


# Disclaimer: this module is over-documented to keep track of all implementation
# details addressing the following issues when handling big data and a RDBMS:
# 1. Memory leaks (too many objects in the RDBMS session)
# 2. Slowdowns (long RDBMS queries)
# 3. Undesired printouts (external ObsPy C libraries that have to be caught)
# 4. Python multiprocessing with RDBMS queries


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
    # checks dic values (modify in place) and returns dic value(s) needed here:
    # Outside the try catch below as BadParam might be raised and need to
    # be caught by the caller (see `cli.py`)
    session, pyfunc, funcname, config_dict, segments_selection, multi_process, \
        chunksize = load_config_for_process(dburl, pyfile, funcname,
                                                             config, outfile,
                                                             **param_overrides)

    if log2file is True:
        log2file = logfilepath(outfile or pyfile)  # auto create log file
    else:
        log2file = log2file or ''  # assure we have a string

    try:
        configlog4processing(logger, log2file, verbose)
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
        _run(session, pyfunc, get_writer(outfile, append, writer_options),
             config_dict, segments_selection, verbose, multi_process,
             chunksize, None)
        logger.info("Completed in %s", str(elapsed_time(stime)))
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
        close_session(session, True)
        close_logger(logger)


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
        configlog4processing(logger, logfile, show_progress)
        stime = time.time()
        yield from run_and_yield(session,
                                 fetch_segments_ids(session, segments_selection),
                                 pyfunc, config, show_progress, multi_process,
                                 chunksize, skip_exceptions)
        logger.info("Completed in %s", str(elapsed_time(stime)))
    except KeyboardInterrupt:
        logger.critical("Aborted by user")  # see comment above
        raise
    except:  # @IgnorePep8 pylint: disable=broad-except
        logger.critical("Process aborted", exc_info=True)  # see comment above
        raise
    finally:
        close_session(session, True)
        close_logger(logger)


def _run(session, pyfunc, writer, config=None, segments_selection=None,
         show_progress=False, multi_process=False, chunksize=None,
         skip_exceptions=None):
    """Run `pyfunc` according to the given `config`, outputting result to `writer`

    :param session: the SQLAlchemy database session
    :param pyfunc: a Python function accepting as arguments a given segment object and
        a `dict` of optional user-defined parameters. The function will be called with
        any given segment and the provided `config` argument
    :param writer: a Writer handling the processed output to filesystem.
        use :class:`writers.BaseWriter` for a no-op class.
        See :module:`stream2segment.process.writers`
    :param config: dict of configuration parameters, usually the result of an associated
        YAML configuration file, to be passed as second argument of `pyfunc`
    :param segments_selection: dict denoting the segments to be selected in form of
        segment attribute mapped to a selection expression in string format
         (if not given, all segments will be processed)
    :param show_progress: (boolean, default False) whether or not to show progress bar
        and other info (e.g. remaining time, successfully processed segments) on the
        standard output (usually, the terminal window)
    """
    if config is None:
        config = {}
    if segments_selection is None:
        segments_selection = {}

    seg_ids = fetch_segments_ids(session, segments_selection, writer)
    written = 0

    with writer:
        for output, segment_id in \
                run_and_yield(session, seg_ids, pyfunc, config, show_progress,
                              multi_process, chunksize, skip_exceptions):
            if not writer.isbasewriter and output is not None:
                writer.write(segment_id, output)
                written += 1

    logger.info("%d of %d segment(s) successfully written to the provided output",
                written, len(seg_ids))


def run_and_yield(session, seg_ids, pyfunc, config, show_progress=False,
                  multi_process=False, chunksize=None, skip_excetpions=None):
    """Run `pyfunc(segment, config)` on each given segment and yields its output
    as the tuple
    ```(output, segment_id)```

    :param session: the SQLAlchemy database session
    :param pyfunc: a Python function accepting as arguments a given segment object and
        a `dict` of optional user-defined parameters. The function will be called with
        any given segment and the provided `config` argument
    :param yield_exceptions: boolean (default False). If True, output can be also
        an Exception, and will denote unsuccessfully processed segments. If False, output
        is yielded only upon successful execution of `pyfunc`
    :param config: dict of configuration parameters, usually the result of an associated
        YAML configuration file, to be passed as second argument of `pyfunc`
    :param show_progress: (boolean, default False) whether or not to show progress bar
        and other info (e.g. remaining time, successfully processed segments) on the
        standard output (usually, the terminal window)
    :param multi_process: (bool, or numeric) Use multiprocessing.Pool. A numeric value
        will be equal to true, using a Pool with the specified number of processes (only
        for advanced users: true is fine and sufficient in most cases)
    """
    done, errors = 0, 0
    seg_len = len(seg_ids)

    if multi_process is True:
        num_processes = cpu_count()  # or None (let's set it directly here though)
    elif multi_process not in (0, False):
        num_processes = int(multi_process)
    else:
        multi_process = False

    if chunksize is None:
        chunksize = get_default_chunksize(seg_len, show_progress)

    if skip_excetpions is None:
        skip_excetpions = [SkipSegment]
    safe_excetpions = tuple(skip_excetpions)  # for safety, in case list

    session.close()  # expunge all, clear all states

    # `create_processing_env` redirects Python BUT ALSO external libraries errors which
    # might mess up the terminal printout (e.g. progressbar). Python warnings should be
    # redirected as well because normally printed to `stderr`, so avoid capturing them
    # (`warnings_filter=None`). `create_processing_env` is also called in Python
    # subprocesses, if present. For info see :func:`process_segments_mp`
    with create_processing_env(seg_len if show_progress else 0,
                               redirect_stderr=True,
                               warnings_filter=None) as pbar:
        if show_progress and seg_len:
            # Show the progressbar now, because the 1st chunk might be ready in minutes,
            # and an empty screen might give the impression of a program hang:
            time.sleep(pbar.short_limit)
            pbar.render_progress()

        if multi_process:
            itr = process_mp(session, pyfunc, config, get_slices(seg_ids, chunksize),
                             pbar, num_processes, safe_excetpions)
        else:
            itr = process_simple(session, pyfunc, config, get_slices(seg_ids, chunksize),
                                 pbar, safe_excetpions)

        for output, is_ok, segment_id in itr:
            if is_ok:
                done += 1
            else:
                logger.warning("segment (id=%d): %s", segment_id, str(output))
                errors += 1
            if is_ok:
                yield output, segment_id

    logger.info('')
    logger.info("%d of %d segment(s) successfully processed", done, seg_len)
    logger.info("%d of %d segment(s) skipped with error message "
                "reported in the log file", errors, seg_len)


def fetch_segments_ids(session, segments_selection, writer=None):
    """Return the numpy array of segments ids to process

    :param segments_selection: dict[str, str] denoting a segment selection
    :param writer: A Writer or None. See :module:`stream2segment.process.writers`.
        If not None, the writer is used to fetch the already processed segments
        and return only segments to process
    :return: the numpy array of integers denoting the ids of the segments to process
        according to `config` and `writer` settings
    """
    # The query is always loaded in memory (https://stackoverflow.com/a/11769768), thus
    # we load ids only into a numpy array for efficiency. Querying with offset
    # and limit is not necessarily faster (if offset is close to "the end", the db might
    # build anyway the full list of segments, we experienced it in the GUI), other
    # solutions as query and yielding make the code too complex
    # (http://docs.sqlalchemy.org/en/latest/orm/query.html#sqlalchemy.orm.query.Query.yield_per).
    # `query4process` below is thus called with "ids_only" (see doc for details).
    # Note that querying attributes instead of the full instances does not cache the
    # results. I.e., after the line below we do not need to issue `session.expunge_all()`
    qry = query4process(session, segments_selection, ids_only=True)

    skip_already_processed = False
    if writer is not None:
        if writer.append:
            if not writer.outputfileexists:
                logger.info('Ignoring `append` functionality: output file does not '
                            'exist or not provided')
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
        # but more importantly the database might simply not support such a long query
        # string, and raise Exceptions.
        # It is therefore way more efficient to do it in numpy. The drawback is that we
        # will query more segments than needed and we might waste time in the query above
        # but this is outweighed by the efficiency here
        logger.info("Fetching already processed segment(s) (please wait)")
        skip_ids = writer.already_processed_segments()
        logger.info("Skipping %d already processed segment(s)", len(skip_ids))
        seg_ids = np.copy(seg_ids[np.isin(seg_ids, skip_ids, assume_unique=True,
                                          invert=True)])

    logger.info("%d segment(s) found to process", len(seg_ids))
    logger.info('')
    return seg_ids


def get_default_chunksize(segments_count, show_progress):
    """Get the segments chunksize according to the segments to
    process (`segments_count`) and whether the progress bar is being shown

    :return: the tuple: `multi_process, num_processes, chunksize`
        (int, bool, int)
    """
    default_chuknksize, min_pbar_iterations = _get_chunksize_defaults()
    if not show_progress or segments_count >= 2 * default_chuknksize:
        chunksize = default_chuknksize
    else:
        # determine the chunksize in order to have `min_pbar_iterations` iterations
        # use np.true_divide so that py2/3 division is not a problem:
        chunksize = max(1, int(np.true_divide(segments_count,
                                              min_pbar_iterations).item()))
    return chunksize


def _get_chunksize_defaults():
    """Return a 2 element tuple with the default segment chunksize (600) and the minimum
    progressbar iterations (10). This function is implemented mainly for mocking in tests
    """
    return 600, 10


def process_mp(session, pyfunc, config, seg_ids_chunks, pbar, num_processes,
               safe_exceptions_tuple):
    """Execute `pyfunc` using the multiprocessing Python module

    :param seg_ids_chunks: iterable yielding numpy arrays of segment ids
    """
    # The two actions here below are a little hacky in that we might simply pass
    # strings to this functions (dburl and python module path), but this would require
    # checking for well formed dburl and paths here, which is what we do BEFORE, and
    # also, the non Python multiprocessing case benefits of having already db session
    # object and python function loaded.

    # 1. The db engine has to be disposed now if we want to use multi-processing:
    # https://stackoverflow.com/a/41722129
    dburl = session.bind.engine.url
    session.bind.engine.dispose()
    # 2. We need to pass pickable stuff to each child sub-process,
    # therefore no imported functions, get the file instead:
    pyfile = inspect.getsourcefile(pyfunc)

    pool = Pool(processes=num_processes, initializer=_mp_initializer)
    try:
        for results in \
                pool.imap_unordered(process_segments_mp,
                                    ((seg_ids_chunk, dburl, config, pyfile,
                                      pyfunc.__name__, safe_exceptions_tuple)
                                     for seg_ids_chunk in seg_ids_chunks)):
            for output, is_ok, segment_id in results:
                yield output, is_ok, segment_id
            pbar.update(len(results))
    except:  # @IgnorePep8 pylint: disable=bare-except
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()


def _mp_initializer():
    """Set up the worker processes to ignore SIGINT altogether,
    and confine all the cleanup code to the parent process (e.g. Ctrl+C pressed)
    """
    # See https://stackoverflow.com/a/6191991 and links therein:
    # https://noswap.com/blog/python-multiprocessing-keyboardinterrupt
    # https://github.com/jreese/multiprocessing-keyboardinterrupt
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_simple(session, pyfunc, config, seg_ids_chunks, pbar, safe_exceptions_tuple):
    """Execute `pyfunc` in a single system process

    :param seg_ids_chunks: iterable yielding numpy arrays of segment ids
    """
    segment = None  # currently processed segment id (will be used later)
    _clear_session(session)

    for seg_ids_chunk in seg_ids_chunks:
        for output, is_ok, segment in \
                process_segments(session, seg_ids_chunk, config, pyfunc,
                                 safe_exceptions_tuple):
            yield output, is_ok, segment.id

        _clear_session(session, segment)
        pbar.update(len(seg_ids_chunk))


def _clear_session(session, segment=None):
    """Clear the session and re-assigns `segment` to it

    :param segment: A Segment object denoting the last processed segment, or None
    """
    # clear session identity map (sort of cache), freeing memory:
    session.expunge_all()

    if segment is None:
        return

    # re-assign the segment to the session. This will add also to all the segment's
    # already loaded related objects (e.g., Station, with relative inventory) which will
    # thus not be garbage collected (i.e., they will be kept in memory). The advantage is
    # to (potentially) avoid database queries in the next loop by reusing already loaded
    # data (see also the query order given in `query4process'). For info see:
    # https://docs.sqlalchemy.org/en/13/orm/query.html#sqlalchemy.orm.query.Query.get
    # http://docs.sqlalchemy.org/en/latest/orm/session_state_management.html#merging
    session.add(segment)


def process_segments_mp(args):
    """Function to be used INSIDE A CHILD python-process to process a chunk of segments
    Takes care of releasing db session and other stuff.

    :param args: the tuple (seg_ids_chunk, dburl, config, pyfile, funcname)
        where seg_ids_chunk is a numpy array of segment ids, dburl is the database
        url (string), config is the config dict, pyfile is the path to the processing
        module (string) and funcname is the processing function name to be called

    :return: a list of (output, is_ok, segment_id) tuples, where output is the
        output of the processing function name (either iterable or Exception), is_ok
        (boolean) tells if `output` is NOT an Exception, and segment_id is the id of the
        segment processed
    """
    seg_ids_chunk, dburl, config, pyfile, funcname, safe_exceptions_tuple = args
    pyfunc = valid_pyfunc(pyfile, funcname)
    session = get_session(dburl)
    ret = []

    # Do not capture external C libraries errors (we do it already in the parent process,
    # and we do not want to mess up too much with leaking file descriptors, a complex
    # topic) but use warnings_filter='ignore', to avoid redundant messages
    with create_processing_env(0, redirect_stderr=False, warnings_filter='ignore'):
        try:
            for output, is_ok, segment in \
                        process_segments(session, seg_ids_chunk, config, pyfunc,
                                         safe_exceptions_tuple):
                ret.append((output, is_ok, segment.id))
            return ret
        finally:
            session.close()
            session.bind.engine.dispose()


def process_segments(session, seg_ids_chunk, config, pyfunc, safe_exceptions_tuple):
    """Process a chunk of segments and yield the resulted output. Yields
    tuples of the form `(output, is_ok, segment_id)`, where output is the output of the
    `pyfunc`, is_ok (boolean) tells if `pyfunc` run successfully, segment_id is the id
    of the segment processed.

    Note: `is_ok` indicates if `pyfunc` run successfully, and it is a shorthand for:
    `is_ok == not isinstance(output, ValueError)`
    By conventions only `ValueError`s are caught and passed as output, any other
    exception raised by `pyfunc` will raise and thus interrupt the whole program

    :param session: the db session (SQLAlchemy session)
    :param seg_ids_chunk: a numpy array of segment ids
    :param config: the config dict
    :param pyfunc: a Python function to be invoked on each segment
    :param safe_exceptions_tuple: a tuple of Exceptions to be caught and returned
        instead of raising
    """
    # We reuse `query4process` for simplicity. The query will sort segments returning
    # them in the same order as the given indices (this is not a strict requirement but
    # removing the sort does not improves significantly performances)
    segments = query4process(session, conditions=None, ids_only=False).\
        filter(Segment.id.in_(seg_ids_chunk.tolist()))

    # Note that we could have loaded only the segment ids, deferring the load of all
    # other Segment attribute upon access (see SQLAlchemy `load_only`). Performance tests
    # accessing attributes on several segments reported:
    # defer load: 0.043 secs/segment, Peak memory (Kb): 111792 (0.650716 %)
    # full load:  0.024 secs/segment, Peak memory (Kb): 409194 (2.381825 %).
    # So we go for the full load
    for segment in segments:
        output, is_ok = process_segment(segment, config, pyfunc, safe_exceptions_tuple)
        yield output, is_ok, segment


def process_segment(segment, config, pyfunc, safe_exceptions_tuple):
    """Process a single segment and return the output of `pyfunc(segment, config)`

    :return: the tuple (output, is_ok), where output is either an iterable or a
        `ValueError`. in the former case, `is_ok` is True, otherwise False (the variable
        is returned as a faster alias of `isinstance(output, Exception)`)

    Note that any exception other than `ValueError` will raise and thus interrupt the
    program
    """
    try:
        return pyfunc(segment, config), True
    except safe_exceptions_tuple as exc:
        return exc, False
    except Exception:
        raise


def query4process(session, conditions=None, ids_only=True):
    """Return a query yielding the the segments ids (and their stations ids) for the
    processing. The returned tuples are sorted by station id, event id, channel location
    and channel's channel

    :param session: the sql-alchemy session
    :param conditions: a dict of segment attribute names mapped to a select expression,
        each identifying a filter (sql WHERE clause). See :module:`sqlevalexpr.py`. It
        can be empty (no filter)

    :return: a query yielding the tuples: ```(Segment.id, Segment.station.id)```
    """

    if ids_only:
        query = session.query(Segment.id)
    else:
        query = session.query(Segment)

    # Querying segments for processing should be sorted by station id first (so that a
    # segment station inventory is likely cached from a previously processed segment)
    # and segment id then (to return consistent ordering across queries):
    query = query.join(Segment.station).order_by(Station.id, Segment.id)

    # Performance hints based on remote Postgres tests: the number of joined tables and
    # number of arguments to `order_by` might affect performances, but the latter depends
    # heavily on what is being queried, and its size. So, jsust as a rule of thumb, few
    # arguments, and preferably primary keys (or any indexed column, I suspect) might be
    # better

    if conditions:
        query = exprquery(query, conditions=conditions, orderby=None)

    return query


@contextmanager
def create_processing_env(length=0, redirect_stderr=False, warnings_filter=None):
    """Context manager to be used in a with statement, returns the progress bar
    which can be called with pbar.update(int). The latter is no-op if length ==0

    Typical usage without multi-processing from the main function (activate all 'with'
    statements):
    ```
        with create_proc_env(10, redirect_stderr=True, 'ignore') as pbar:
            ...
            pbar.update(1)
    ```
    Typical usage with multi-processing from the main function (activate only
    progressbar 's 'with' statement):
    ```
        with create_proc_env(10, redirect_stderr=True, None) as pbar:
            ...
            pbar.update(1)
    ```
    Typical usage with multi-processing from a child process  (activate only ignore
    warnings):
    ```
        with create_proc_env(0, redirect_stderr=False, 'ignore') as pbar:
            ...
            pbar.update(1)
    ```

    :param length: the number of tasks to be done. If zero, the returned progressbar will
        be no-op. Otherwise, it is an object which updates a progressbar on terminal
    :param redirect_stderr: if True, captures the output of all C external functions and
        does not print them to the screen, as it might be the case with some ObsPy
        C-imported libraries
    :param warnings_filter: if None, it does not capture Python warnings. Otherwise it
        denotes the Python filter. E.g. 'ignore'. (FIXME: add link)
    """
    with get_progressbar(length > 0, length=length) as pbar:  # no-op if length not > 0
        with redirect(sys.stderr if redirect_stderr else None):
            # redirect is no-op if redirect_stderr=None
            if warnings_filter:
                with warnings.catch_warnings():
                    warnings.simplefilter(warnings_filter)
                    yield pbar
            else:
                yield pbar


@contextmanager
def redirect(src=None, dst=os.devnull):
    """Prevent Python AND external C shared library to print to stdout/stderr in Python,
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

    :param src: file-like object with a fileno() method. Usually is either `sys.stdout`
        or `sys.stderr`.
    """

    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything
    # Moreover, passing None as first argument means no redirection
    try:
        file_desc = src.fileno()
    except (AttributeError, OSError, ValueError) as _:
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
    """Divide `len(array)` by `chunksize` yielding the array slices until exhaustion.
    If `array` is an integer, it denotes the length of the array and the tuples
    (start, end) will be yielded.
    This method intelligently re-arranges the (start, end) indices in order to minimize
    the number of iterations yielded. ``
    """
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
