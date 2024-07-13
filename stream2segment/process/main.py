"""
Main module for the segment processing and .csv output

Created on Feb 2, 2017

.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
import os
import time
import sys
import logging
from contextlib import contextmanager
import warnings
from multiprocessing import Pool, cpu_count
import signal
from itertools import chain, repeat
import inspect

import numpy as np

from stream2segment.io import yaml_load
from stream2segment.io.db import secure_dburl, close_session
from stream2segment.process.db.sqlevalexpr import exprquery
from stream2segment.io.log import logfilepath, close_logger, elapsed_time
from stream2segment.io.cli import get_progressbar, ascii_decorate
from stream2segment.io.inputvalidation import validate_param
from stream2segment.process.db import get_session
from stream2segment.process.db.models import Segment, Station, SkipSegment
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


def process(pyfunc, dburl, segments_selection=None, config=None, outfile=None,
            append=False, writer_options=None, logfile=False, verbose=False,
            multi_process=False, chunksize=None, skip_exceptions=None):
    """Iteratively applies a function (`pyfunc`) to every selected segment
    (`segments_selection`) found on the database (`dburl`), optionally saving the
    result as a row of a tabular file (`outfile`) in CSV or
    HDF format.

    :param pyfunc: the processing function, i.e. a Python function with signature
        (arguments) `(segment, config)`
    :param dburl: str. The URL of the database where data has been previously downloaded
    :param segments_selection: The segments to be processed. It can be a sequence of
        integers (tuple, list, numpy array) denoting the segment IDs, or a dict[str, str]
        of Segments attributes mapped to a given selection expression, e.g.:
        ```
        {
            'event.magnitude': '<=6',
            'channel.channel': 'HHZ',
            'maxgap_numsamples': '(-0.5, 0.5)',
            'has_valid_data': 'true'
        }
        ```
        If None or missing, it defaults to a dict with the last two keys listed above
    :param config: dict or str. The `dict` with user-defined parameters to tune your
        processing function (passed as 2nd argument to `pyfunc`).
        If `str`, it is supposed to be a path to a YAML file that will be read as
        Python `dict`. If missing or None, it will default to `{}` (empty dict, i.e. no
        config)
    :param outfile: str or None. The destination file where to write the
        processing output, either ".csv" or ".hdf". The type of output will be inferred
        from the file extension, and defaults to CSV when missing or unknown.
        If no output file is given, the returned values of the processing function will
        be ignored. Otherwise, `pyfunc` must either return a `dict`, `list` or
        - for hdf output - pandas `DataFrame`. The returned value will be written as
        row of the tabular output.
    :param append: bool (default False) ignored if the output file is not given
        or non-existing, otherwise: if False, overwrite the existing output
        file. If True, process unprocessed segments only (checking the segment
        id), and append to the given file, without replacing existing data.
    :param writer_options: dict of options for the writer. When None or missing, it
        defaults to the empty dict (no options). As option, you can pass any keyword
        argument of the functions linked below.
        For CSV output:
        https://docs.python.org/3/library/csv.html#csv.writer     (any CSV)
        https://docs.python.org/3/library/csv.html#csv.DictWriter (CSV with header, e.g.
        your processing function returns dicts). Note that some arguments are ignored
        ('f', 'fieldnames', 'csvfile') and other set by default if missing:
        `delimiter` (","), `quotechar` ('"') and `quoting` (`csv.QUOTE_MINIMAL`).
        For HDF output:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.append.html
        with the exception of the arguments `value` and `append`, which are not
        configurable and will be overwritten (also note that `format` and `key`
        will be set by default and do not need to be input). Example of such a dict:
        ```
        {
            'chunksize': 1200,
            'min_itemsize': {
                'network': 2
                'station': 5
                'location': 2
                'channel: 3
            }
        }
        ```
    :param logfile: str or boolean (default: False). The path of the file to log
        :class:`stream2segmetn.process.SkipExeption`, when raised.
        If True, it will be inferred from the output file or the Python file of the
        processing function, if given, by appending ".[now].log" to the file path ([now]
        denotes the current UTC timestamp). If False, empty string or in any case where
        no log file path can be inferred, then logging is disabled
    :param verbose: bool, default False. Print progress bar and estimated remaining time
        to standard output (usually, the terminal window)
    :param multi_process: enable multiprocessing to speed up execution.
        When not boolean, this parameter can be an integer denoting the
        exact number of subprocesses to be allocated (only for advanced users. True is
        fine in most cases). If multiprocessing is enabled `pyfunc` must be pickable
        (https://docs.python.org/3/library/pickle.html#pickle-picklable)
    :param chunksize: the size, in number of segments, of each chunk of data that will
        be loaded from the database. Increasing this number speeds up the load but also
        increases memory consumption. None (the default) will set the size automatically
    :param skip_exceptions: tuple of Python exceptions that will not interrupt the whole
        execution but will be logged to file, with the relative segment id. When missing
        or None, it defaults to :class:`stream2segmetn.process.SkipExeption`

    :return: the number of successfully processed segments. If a file is provided, it
        is the number of written rows
    """
    cfg_file = config if isinstance(config, str) else None
    pyfile = getattr(pyfunc, "__name__", str(pyfunc))

    if logfile is True:
        logfile = '' if not outfile else logfilepath(outfile)  # auto create log file

    with _setup_logging(logfile, verbose):
        abp = os.path.abspath
        info = [
            "Input database:      %s" % secure_dburl(dburl),
            "Processing function: %s" % pyfile or 'n/a',
            "Log file:            %s" % (abp(logfile) if logfile else 'n/a'),
            "Output file:         %s" % (abp(outfile) if outfile else 'n/a')
        ]
        if cfg_file:
            info.append("Config. file:        %s" % abp(cfg_file))
        logger.info(ascii_decorate("\n".join(info)))

        return _run_and_write(pyfunc, dburl, segments_selection, config, outfile, append,
                              writer_options, verbose, multi_process, chunksize, None)


def _run_and_write(pyfunc, dburl, segments_selection=None, config=None, outfile=None,
                   append=False, writer_options=None, show_progress=False,
                   multi_process=False, chunksize=None, skip_exceptions=None):
    if outfile is not None:
        validate_param('outfile', outfile, _valid_filewritable)

    if config is None:
        config = {}

    writer = get_writer(outfile, append, writer_options)
    seg_ids = fetch_segments_ids(dburl, segments_selection, writer)

    num_ok = 0
    write2file = not writer.isbasewriter
    with writer:
        for output, segment_id in \
                run_and_yield(dburl, seg_ids, pyfunc, config, show_progress,
                              multi_process, chunksize, skip_exceptions):
            if output is not None:
                num_ok += 1
                if write2file:
                    writer.write(segment_id, output)

    if write2file:
        logger.info("%d of %d segment(s) successfully written to the provided output",
                    num_ok, len(seg_ids))

    return num_ok


def _valid_filewritable(filepath):
    """Check that the file is writable, i.e. that is a string and its
    directory exists"""
    if not isinstance(filepath, str):
        raise TypeError('string required, found %s' % str(type(filepath)))

    if not os.path.isdir(os.path.dirname(filepath)):
        raise ValueError('cannot write file: parent directory does not exist')

    return filepath


def imap(pyfunc, dburl, segments_selection=None, config=None,
         logfile='', verbose=False, multi_process=False, chunksize=None,
         skip_exceptions=None):
    """Return an iterator that applies a function (`pyfunc`) to every selected segment
    (`segments_selection`) of a SQL database (`dburl`), yielding the function results.


    :param pyfunc: the processing function, i.e. a Python function with signature
        (arguments) `(segment, config)`
    :param dburl: str. The URL of the database where data has been previously downloaded
    :param segments_selection: The segments to be processed. It can be a sequence of
        integers (tuple, list, numpy array) denoting the segment IDs, or a dict[str, str]
        of Segments attributes mapped to a given selection expression, e.g.:
        ```
        {
            'event.magnitude': '<=6',
            'channel.channel': 'HHZ',
            'maxgap_numsamples': '(-0.5, 0.5)',
            'has_valid_data': 'true'
        }
        ```
        If None or missing, it defaults to a dict with the last two keys listed above
    :param config: dict or str. The `dict` with user-defined parameters to tune your
        processing function (passed as 2nd argument to `pyfunc`).
        If `str`, it is supposed to be a path to a YAML file that will be read as
        Python `dict`. If missing or None, it will default to `{}` (empty dict, i.e. no
        config)
    :param logfile: string. the path of the file to log
        :class:`stream2segmetn.process.SkipExeption`, when raised.
        Empty string (the default) disables logging
    :param verbose: bool, default False. Print progress bar and estimated remaining time
        to standard output (usually, the terminal window)
    :param multi_process: enable multiprocessing to speed up execution.
        When not boolean, this parameter can be an integer denoting the
        exact number of subprocesses to be allocated (only for advanced users. True is
        fine in most cases). If multiprocessing is enabled `pyfunc` must be pickable
        (https://docs.python.org/3/library/pickle.html#pickle-picklable)
    :param chunksize: the size, in number of segments, of each chunk of data that will
        be loaded from the database. Increasing this number speeds up the load but also
        increases memory consumption. None (the default) will set the size automatically
    :param skip_exceptions: tuple of Python exceptions that will not interrupt the whole
        execution but will be logged to file, with the relative segment id. When missing
        or None, it defaults to :class:`stream2segmetn.process.SkipExeption`
    """
    seg_ids = fetch_segments_ids(dburl, segments_selection)
    with _setup_logging(logfile, verbose):
        for result, seg_id in run_and_yield(dburl, seg_ids, pyfunc, config,
                                            verbose, multi_process, chunksize,
                                            skip_exceptions):
            yield result


@contextmanager
def _setup_logging(logfile, verbose):
    """Contextmanager handling log stuff and closing session at the end"""
    try:
        configlog4processing(logger, logfile, verbose)
        stime = time.time()
        yield
        logger.info("Completed in %s", str(elapsed_time(stime)))
    except KeyboardInterrupt:
        logger.critical("Aborted by user")  # see comment above
        raise
    except:  # noqa
        logger.critical("Process aborted", exc_info=True)  # see comment above
        raise
    finally:
        close_logger(logger)


def run_and_yield(dburl, seg_ids, pyfunc, config, show_progress=False,
                  multi_process=False, chunksize=None, skip_exceptions=None):
    """Run `pyfunc(segment, config)` on each given segment and yields its output
    as the tuple
    ```(output, segment_id)```

    :param dburl: string database URL
    :param pyfunc: a Python function accepting as arguments a given segment object and
        a `dict` of optional user-defined parameters. The function will be called with
        any given segment and the provided `config` argument. The argument can also be
        a path to a Python module, with optional double semicolon separating the name
        of the function to search therein. Use this latter option when multi_process is
        on to avoid pickable problems, e.g. if the function has to be loaded from custom
        file.
    :param config: dict of configuration parameters, usually the result of an associated
        YAML configuration file, to be passed as second argument of `pyfunc`
    :param show_progress: (boolean, default False) whether to show progress bar
        and other info (e.g. remaining time, successfully processed segments) on the
        standard output (usually, the terminal window)
    :param multi_process: (bool, or numeric) Use multiprocessing.Pool. A numeric value
        will be equal to true, using a Pool with the specified number of processes (only
        for advanced users: true is fine and sufficient in most cases)
    :param skip_exceptions: tuple of Python exceptions that will not interrupt the whole
        execution but will be logged to file, with the relative segment id. When missing
        or None, it defaults to :class:`stream2segmetn.process.SkipExeption`
    """
    # check params:
    if isinstance(config, str):
        config = validate_param("config", config or {}, yaml_load)
    elif not config:
        config = {}

    validate_param('pyfunc', pyfunc, _valid_pyfunc)

    done, errors = 0, 0
    seg_len = len(seg_ids)
    num_processes = 0
    if multi_process is True:
        num_processes = cpu_count()  # or None (let's set it directly here though)
    elif multi_process not in (0, False):
        num_processes = max(0, int(multi_process))

    if chunksize is None:
        chunksize = get_default_chunksize(seg_len, show_progress)

    if skip_exceptions is None:
        skip_exceptions = [SkipSegment]
    skip_exceptions = tuple(skip_exceptions)  # for safety, in case list

    # `create_processing_env` redirects Python BUT ALSO external libraries errors which
    # might mess up the terminal printout (e.g. progressbar). Python warnings should be
    # redirected as well because normally printed to `stderr`, so avoid capturing them
    # (`warnings_filter=None`). `create_processing_env` is also called in Python
    # subprocesses, if present. For info see :func:`process_segments_mp`
    with create_processing_env(seg_len if show_progress else 0,
                               redirect_stderr=sys.stderr.isatty(),
                               warnings_filter=None) as pbar:
        if show_progress and seg_len:
            # Show the progressbar now, because the 1st chunk might be ready in minutes,
            # and an empty screen might give the impression of a program hang:
            time.sleep(0.5)
            pbar.render_progress()

        if num_processes:
            itr = process_mp(dburl, pyfunc, config, get_slices(seg_ids, chunksize),
                             pbar, num_processes, skip_exceptions)
        else:
            itr = process_simple(dburl, pyfunc, config, get_slices(seg_ids, chunksize),
                                 pbar, skip_exceptions)

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
                "reported in the log file, if provided", errors, seg_len)


def _valid_pyfunc(pyfunc):
    """Checks if the argument is a valid processing Python function by inspecting its
    signature"""
    params = inspect.signature(pyfunc).parameters  # dict[str, inspect.Parameter]
    # less than two arguments? then function invalid:
    if len(params) < 2:
        raise ValueError('Python function should have 2 arguments '
                         '`(segment, config)`, %d found' % len(params))
    # more than 2 args? then we need to have them with a default set:
    for pname, param in list(params.items())[2:]:
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # it is not *args or **kwargs, does it have a default?
            if not param.default == param.empty:
                raise ValueError('Python function argument "%s" should have a '
                                 'default, or be removed' % pname)


def fetch_segments_ids(dburl, segments_selection, writer=None):
    """Return the numpy array of segments ids to process

    :param dburl: database URL (str)
    :param segments_selection: dict[str, str] denoting a segment selection, or an
        iterable of integers denoting the segment ids. None will get a default
        segment selection (see `get_default_segment_selection`)
    :param writer: A Writer or None. See :module:`stream2segment.process.writers`.
        If not None, the writer is used to fetch the already processed segments
        and return only segments to process
    :return: the numpy array of integers denoting the ids of the segments to process
        according to `config` and `writer` settings
    """
    if segments_selection is None:
        segments_selection = get_default_segments_selection()

    session = validate_param('dburl', dburl, get_session)

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

    if not isinstance(segments_selection, dict):
        try:
            seg_ids = np.asarray(segments_selection, dtype=int)
        except Exception as exc:
            raise ValueError('Unable to convert the segments selection to an array of '
                             'integer IDs: %s' % str(exc))
    else:
        # The query is always loaded in memory (https://stackoverflow.com/a/11769768), so
        # we load ids only into a numpy array for efficiency. Querying with offset and
        # limit is not necessarily faster (if offset is close to "the end", the db might
        # build anyway the full list of segments, as seen in the GUI), other solutions as
        # query and yielding are not a full solution and make the code too complex
        # (http://docs.sqlalchemy.org/en/latest/orm/query.html#sqlalchemy.orm.query.Query.yield_per).
        # `query4process` below is thus called with "ids_only". Note that querying
        # attributes instead of the full instances does not cache the results. I.e.,
        # after the line below we do not need to issue `session.expunge_all()`
        qry = query4process(session, segments_selection, ids_only=True)
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
    close_session(session)
    return seg_ids


def get_default_segments_selection():
    """Return a dict with a default segments selection for processing"""
    return {
        'has_valid_data': 'true',
        'maxgap_numsamples': '(-0.5, 0.5)'
    }


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
        chunksize = max(1, int(np.true_divide(segments_count,
                                              min_pbar_iterations).item()))
    return chunksize


def _get_chunksize_defaults():
    """Return a 2 element tuple with the default segment chunksize (600) and the minimum
    progressbar iterations (10). This function is implemented mainly for mocking in tests
    """
    return 600, 10


def process_mp(dburl, pyfunc, config, seg_ids_chunks, pbar, num_processes,
               safe_exceptions_tuple):
    """Execute `pyfunc` using the multiprocessing Python module

    :param seg_ids_chunks: iterable yielding numpy arrays of segment ids
    """
    pool = Pool(processes=num_processes, initializer=_mp_initializer)
    try:
        for results in \
                pool.imap_unordered(process_segments_mp,
                                    ((seg_ids_chunk, dburl, config, pyfunc,
                                      safe_exceptions_tuple)
                                     for seg_ids_chunk in seg_ids_chunks)):
            for output, is_ok, segment_id in results:
                yield output, is_ok, segment_id
            pbar.update(len(results))
    except:  # noqa
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


def process_simple(dburl, pyfunc, config, seg_ids_chunks, pbar, safe_exceptions_tuple):
    """Execute `pyfunc` in a single system process

    :param seg_ids_chunks: iterable yielding numpy arrays of segment ids
    """
    segment = None  # currently processed segment id (will be used later)
    session = get_session(dburl, scoped=False, check_db_existence=False)
    try:
        for seg_ids_chunk in seg_ids_chunks:
            for output, is_ok, segment in \
                    process_segments(session, seg_ids_chunk, config, pyfunc,
                                     safe_exceptions_tuple):
                yield output, is_ok, segment.id

            _clear_session(session, segment)
            pbar.update(len(seg_ids_chunk))
    finally:
        close_session(session)


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

    :param args: the tuple (seg_ids_chunk, dburl, config, func_or_pyfile, funcname)
        where seg_ids_chunk is a numpy array of segment ids, dburl is the database
        url (string), config is the config dict, pyfile is the path to the processing
        module (string) and funcname is the processing function name to be called

    :return: a list of (output, is_ok, segment_id) tuples, where output is the
        output of the processing function name (either iterable or Exception), is_ok
        (boolean) tells if `output` is NOT an Exception, and segment_id is the id of the
        segment processed
    """
    seg_ids_chunk, dburl, config, pyfunc, safe_exceptions_tuple = args
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
            close_session(session)


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
    # removing the sort does not improve significantly performances)
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
    with get_progressbar(length) as pbar:  # no-op if length not > 0
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

    Example:

    with redirect(sys.stdout):
        print("from Python")
        os.system("echo non-Python applications are also supported")

    :param src: file-like object with a fileno() method. Usually is either `sys.stdout`
        or `sys.stderr`.
    """
    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything
    # Moreover, passing None as first argument means no redirection
    if src is None:
        yield
        return

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
