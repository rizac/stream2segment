"""
Processing functions
"""
# Feb 2, 2017
from __future__ import annotations
import os
from pathlib import Path
import time
import sys
import logging
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta, datetime, UTC
import warnings
from io import BytesIO
from multiprocessing import Pool, cpu_count
import signal
import inspect

from typing import Any

from sqlalchemy import select, tuple_, Engine
import yaml
from obspy.core.event import Event as ObspyEvent
from obspy import Stream, Inventory, read, read_events, read_inventory
from obspy.geodetics import locations2degrees, degrees2kilometers

from stream2segment.io.db import secure_dburl, create_engine
from stream2segment.process.segments_selection import get_segments_count, build_select, \
    get_orderby_columns
from stream2segment.io.utils import (
    get_progressbar, ascii_decorate, start_logging, BadParam, estimate_buffer_size
)
from stream2segment.process.writers import get_writer


# make the logger refer to the parent of this package (`rfind` below. For info:
# https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__[:__name__.rfind('.')])


class SkipSegment(Exception):
    """Stream2segment exception indicating a segment processing error that should
    resume to the next segment without interrupting the whole routine
    """
    pass


def process(
    pyfunc: Callable,
    dburl: str,
    segments_selection: dict | None = None,
    group_components: bool = False,
    config: str | Path | dict = None,
    outfile: str | Path = None,
    append=False,
    writer_options=None,
    logfile: str | Path | bool = '',
    verbose=False,
    multi_process=False,
    chunksize: int | None = None,
    skip_exceptions=None
):
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
            'event_magnitude': '<=6',
            'station_code': 'AB',
            'gap_score_percent': '[-0.5, 0.5]',
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
    :param writer_options: dict of options for the writer, i.e. optional keyword
        arguments that will be passed to pandas Dataframe `to_hdf` or `to_csv` (
        depending on the desired file format). See pandas doc for details (note that
        some options might be overwritten). Example (for HDF output):
        ```
        writer_options = {
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
        :class:`stream2segment.process.SkipException`, when raised.
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
        or None, it defaults to :class:`stream2segment.process.SkipException`

    :return: the number of successfully processed segments. If a file is provided, it
        is the number of written rows
    """
    cfg_file = Path(config).resolve() if isinstance(config, str) else None
    pyfile = getattr(pyfunc, "__name__", str(pyfunc))

    if logfile is True:
        if not outfile:
            logfile = ''
        else:
            _now = datetime.now(UTC).replace(microsecond=0, tzinfo=None)
            _now_iso = _now.isoformat('_').replace(':', '')
            logfile = f'{outfile}.{_now_iso}.log'

    if logfile:
        logfile = Path(logfile).resolve()

    if outfile is not None:
        if not isinstance(outfile, (str, Path)) or not Path(outfile).parent.is_dir():
            raise BadParam('invalid output file path (check parent dir)')
        outfile = Path(outfile).resolve()

    # print summary (if verbose is on):
    if verbose:

        out_file_str = f"{str(outfile) if outfile else 'n/a'}"
        log_file_str = f"{str(logfile) if logfile else 'n/a'}"
        cfg_file_str = f"{str(cfg_file) if cfg_file else 'n/a'}"
        py_file_str = f"{pyfile or 'n/a'}"

        if outfile is not None:
            if outfile.exists():
                if append:
                    out_file_str = f"(write mode: append) {outfile}"
                else:
                    out_file_str = f"(write mode: overwrite) {outfile}"
            else:
                if append:
                    out_file_str = f"(new file, 'append' ignored) {outfile}"
                else:
                    out_file_str = f"(new file) {outfile}"

        print(
            ascii_decorate("\n".join([
                f"Input database:      {secure_dburl(dburl)}",
                f"Output file:         {out_file_str}",
                f"Processing function: {py_file_str}",
                f"Log file:            {log_file_str}",
                f"Config. file:        {cfg_file_str}"
            ]))
        )

    writer = get_writer(outfile, append, writer_options)
    num_ok = 0
    with writer:

        for output in imap(
            pyfunc,
            dburl,
            segments_selection=segments_selection,
            group_components=group_components,
            config=config,
            logfile=logfile,
            verbose=verbose,
            multi_process=multi_process,
            chunksize=chunksize,
            skip_exceptions=skip_exceptions
        ):
            if output is None:
                continue
            num_ok += 1
            writer.write(output)

    return num_ok


def imap(
    pyfunc: Callable,
    dburl: str,
    segments_selection: dict | None = None,
    group_components: bool = False,
    config: str | Path | dict | None = None,
    logfile: str='',
    verbose=False,
    multi_process=False,
    chunksize=None,
    skip_exceptions=None
):
    """Return an iterator that applies a function (`pyfunc`) to every selected segment
    (`segments_selection`) of a SQL database (`dburl`), yielding the function results.


    :param pyfunc: the processing function, i.e. a Python function with signature
        (arguments) `(segment, config)`
    :param dburl: str. The URL of the database where data has been previously downloaded
    :param segments_selection: The segments to be processed. It can be a sequence of
        integers (tuple, list, numpy array) denoting the segment IDs, or a
        dict[str, str] of Segments attributes mapped to a selection expression, e.g.:
        ```
        {
            'event_magnitude': '<=6',
            'station_code': 'AB',
            'gap_score_percent': '(-0.5, 0.5)'
        }
        ```
        If None or missing, it defaults to a dict with the last two keys listed above
    :param config: dict or str. The `dict` with user-defined parameters to tune your
        processing function (passed as 2nd argument to `pyfunc`).
        If `str`, it is supposed to be a path to a YAML file that will be read as
        Python `dict`. If missing or None, it will default to `{}` (empty dict, i.e. no
        config)
    :param logfile: string. the path of the file to log
        :class:`stream2segment.process.SkipException`, when raised.
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
        or None, it defaults to :class:`stream2segment.process.SkipException`
    """
    num_processes = 0
    if multi_process is True:
        num_processes = cpu_count()  # or None (let's set it directly here though)
    elif multi_process not in (0, False):
        num_processes = max(0, int(multi_process))

    if skip_exceptions is None:
        skip_exceptions = [SkipSegment]
    skip_exceptions = tuple(skip_exceptions)  # for safety, in case list

    oks = 0
    errors = 0

    if num_processes > 1:
        capture = suppress_printouts
    else:
        capture = nullcontext


    # `create_processing_env` redirects Python BUT ALSO external libraries errors which
    # might mess up the terminal printout (e.g. progressbar). Python warnings should be
    # redirected as well because normally printed to `stderr`, so avoid capturing them
    # (`warnings_filter=None`). `create_processing_env` is also called in Python
    # subprocesses, if present. For info see :func:`process_segments_mp`
    with (
        start_logging(logger, logfile, verbose),
        capture(),
    ):
        try:

            # check params:
            n_args = 4
            if isinstance(config, (str, Path)):
                try:
                    with open(config) as _config:
                        config = yaml.safe_load(_config)
                except yaml.YAMLError as exc:
                    raise BadParam(f"invalid config: {exc}") from exc
            elif not config:
                n_args = 3
                config = {}

            params = inspect.signature(pyfunc).parameters
            # params is a sort of # dict[str, inspect.Parameter]. Check params:
            if len(params) != n_args:
                raise TypeError(
                    f'"{pyfunc.__name__}" expects {n_args} parameters, '
                    f'but {len(params)} were given (are you maybe using a '
                    f'legacy s2s function, version <= 4)?'
                )

            total = 0
            engine = get_engine(dburl)
            if verbose:
                total = get_segments_count(dburl, segments_selection)

            with get_progressbar(total) as pbar:  # no-op if length not > 0
                if verbose and total:
                    logger.info(f"{total:,} segment(s) found to process")
                    # Show the progress bar immediately to avoid an empty screen being
                    # mistaken for a program hang if 1st chunk takes time:
                    time.sleep(0.25)
                    pbar.render_progress()

                exec_processing_func_args = (
                    (pyfunc, args, config, skip_exceptions, num_processes > 1)
                    for args in get_segments(
                        engine,
                        segments_selection,
                        group_components,
                        False,
                        chunksize
                    )
                )

                stime = time.time()

                if num_processes <= 1:
                    for (output, is_ok, ids) in map(
                        execute_processing_function, exec_processing_func_args
                    ):
                        pbar.update(len(ids))
                        if is_ok:
                            oks += len(ids)
                            yield output
                        else:
                            errors += len(ids)
                            logger.warning(
                                f"segment id(s)="
                                f"{' ,'.join(str(i) for i in ids)}): "
                                f"{output}"
                            )
                else:

                    with Pool(
                        processes=num_processes, initializer=_mp_initializer
                    ) as pool:

                        try:

                            for (output, is_ok, ids) in pool.imap_unordered(
                                execute_processing_function, exec_processing_func_args
                            ):
                                pbar.update(len(ids))
                                if is_ok:
                                    oks += len(ids)
                                    yield output
                                else:
                                    errors += len(ids)
                                    logger.warning(
                                        f"segment id(s)="
                                        f"{' '.join(str(i) for i in ids)}): "
                                        f"{output}"
                                    )

                            pool.close()  # iterable fully exhausted, normal completion
                        except Exception:
                            # explicit terminate because we are yielding (generator)
                            pool.terminate()
                            raise
                        finally:
                            pool.join()

            logger.info(
                f"Completed in {timedelta(seconds=round((time.time()) - stime))}"
            )

            logger.info('')
            logger.info(f"{oks} of {oks+errors} segment(s) successfully processed")
            logger.info(
                f"{errors} of {oks+errors} segment(s) skipped with error message "
                f"reported in the log file, if provided"
            )

        except KeyboardInterrupt:
            logger.critical("Aborted by user")  # see comment above
            raise

        except:  # noqa
            logger.critical("Process aborted", exc_info=True)  # see comment above
            raise


def _mp_initializer():
    """
    Set up the worker processes to ignore SIGINT altogether,
    and confine all the cleanup code to the parent process (e.g. Ctrl+C pressed)
    """
    # For info see https://stackoverflow.com/a/6191991 and links therein
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, check_db_existence=True)


def get_segments(
    db: str | Engine,
    segments_selection: dict,
    group_components: bool = False,
    segments_only: bool = False,
    chunksize: int | None = None
) -> Iterable[tuple[Stream, Inventory | None, ObspyEvent | None]]:

    if isinstance(db, str):
        engine = get_engine(db)
    else:
        engine = db

    orderby_columns = get_orderby_columns(db, group_components)

    buffer = []
    last_key = None
    stmt_base = build_select(
        db, segments_selection, group_components
    ).order_by(*orderby_columns)

    if chunksize is None:
        chunksize = estimate_buffer_size(5)  # 5 Mb per row

    while True:
        stmt = stmt_base
        if last_key is not None:
            stmt = stmt.where(tuple_(*orderby_columns) > last_key)

        stmt = stmt.limit(chunksize)

        with engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        if not rows:
            break

        last_key = tuple(getattr(rows[-1], c.key) for c in orderby_columns)
        # (c.key resolves to c.label, if a label is set, otherwise column name)

        if not group_components:
            for r in rows:
                yield db_to_obspy(engine, segments_only, r)

        else:
            buffer.extend(rows)

            # split_idx = len(buffer) - 1
            # while split_idx > 0 and same_group(buffer[split_idx], buffer[split_idx-1]):
            #     split_idx -= 1
            #
            # if split_idx == 0:
            #     continue
            #
            # to_yield = buffer[:split_idx]
            # buffer = buffer[split_idx:]

            for (start, end) in split_in_same_group_chunks(buffer):
                if end == len(buffer):
                    buffer = buffer[start:]
                    break
                yield db_to_obspy(engine, segments_only, *buffer[start: end])

    if buffer:
        # if buffer => we surely grouped components:
        yield db_to_obspy(engine, segments_only, *buffer)


def same_group(row1, row2):
    return (
        row1.data_webservice_id == row2.data_webservice_id and
        row1.network_code == row2.network_code and
        row1.station_code == row2.station_code and
        row1.location_code == row2.location_code and
        row1.band_code == row2.band_code and
        row1.instrument_code == row2.instrument_code and
        row1.event_id == row2.event_id
    )


def split_in_same_group_chunks(rows):
    start = 0
    end = 0
    while end < len(rows):
        if not same_group(rows[start], rows[end]):
            yield start, end
            start = end
        end += 1
    if end < len(rows):
        yield end, len(rows)


_inventory_cache = {}
_inventory_cache_maxsize = estimate_buffer_size('stationxml')
_event_cache = {}
_event_cache_maxsize = estimate_buffer_size('quakeml')


def db_to_obspy(
    engine: Engine, segments_only: bool, *db_rows
) -> tuple[Stream, Inventory | None, ObspyEvent | None]:
    """
    return the tuple
        (segment: Stream, station: Inventory, event: Event)
    from the given db_rows
    """
    first_row = db_rows[0]

    inventory=None
    if not segments_only:
        stationxml_id = first_row.stationxml_id
        if stationxml_id is not None:
            inventory = _inventory_cache.get(stationxml_id)
            if inventory is None:
                while len(_inventory_cache) >= _inventory_cache_maxsize:
                    _inventory_cache.pop(next(iter(_inventory_cache)))
                try:
                    with engine.connect() as conn:
                        data = conn.execute(select(StationXML.data).where(
                            StationXML.id == stationxml_id)
                        ).scalar_one_or_none()
                    if data is not None:
                        inventory = read_inventory(BytesIO(data), format='STATIONXML')
                    else:
                        inventory = None
                    _inventory_cache[stationxml_id] = inventory
                except Exception as e:
                    logger.warning(e)  # FIXME automatize

    event=None
    if not segments_only:
        quakeml_id = first_row.event_id
        if quakeml_id is not None:
            event = _event_cache.get(quakeml_id)
            if event is None:
                while len(_event_cache) >= _event_cache_maxsize:
                    _event_cache.pop(next(iter(_event_cache)))
                try:
                    with engine.connect() as conn:
                        data = conn.execute(select(QuakeML.data).where(
                            QuakeML.id == quakeml_id)
                        ).scalar_one_or_none()
                    if data is not None:
                        event = read_events(BytesIO(data), format='QUAKEML')
                    else:
                        event = None
                    _event_cache[quakeml_id] = event
                except Exception as e:
                    logger.warning(e)  # FIXME automatize else:

    stream = Stream()
    for db_row in db_rows:
        _stream = read(db_row.data, format='MSEED')
        start_time = _stream[0].stats.starttime.datetime
        end_time = _stream[0].stats.endtime.datetime
        arr_time1 = start_time + timedelta(seconds=db_row.noise_window_s)
        arr_time2 = end_time - timedelta(seconds=db_row.signal_window_s)
        min_time = min(arr_time1, arr_time2)
        max_time = max(arr_time1, arr_time2)
        arrival_time = min_time + ((max_time - min_time)/ 2)
        try:
            net = _stream[0].stats.network
            sta = _stream[0].stats.station
            loc = _stream[0].stats.location
            cha = _stream[0].stats.channel
        except AttributeError:  # legacy Obspy? FIXME check!
            net, sta, loc, cha = _stream[0].stats.id.split('.')
        s_meta = SegmentMetadata(
            id=db_row.id,
            network_code=net,
            station_code=sta,
            location_code=loc,
            channel_code=cha,
            latitude=db_row.latitude,
            longitude=db_row.longitude,
            depth=db_row.depth,
            dip=db_row.dip,
            azimuth=db_row.azimuth,
            data_webservice_id=db_row.data_webservice_id,
            elevation=db_row.elevation,
            arrival_time=arrival_time,
            channel_id=db_row.channel_id,
            event_id=db_row.event_id,
            event_latitude=db_row.event_latitude,
            event_longitude=db_row.event_longitude,
            event_webservice_id=db_row.event_webservice_id,
            event_depth_km=db_row.event_depth_km,
            event_time=db_row.event_time,
            event_magnitude=db_row.event_magnitude,
            event_magnitude_type=db_row.event_magnitude_type,
            noise_window_s=(arrival_time - start_time).total_seconds(),
            signal_window_s=(end_time-arrival_time).total_seconds()
        )
        for t in _stream:
            t.stats.segment_metadata = s_meta

        stream += _stream

    return (
        stream,
        inventory,
        event
    )


def get_default_segments_selection():
    """Return a dict with a default segments selection for processing"""
    return {
        'gap_score_percent': '<=50'
    }


def execute_processing_function(args: tuple[
    Callable[[Stream, Inventory | None, Event | None, dict], Any],
    tuple[Stream, Inventory | None, Event | None],
    dict,
    tuple[Exception],
    bool
]) -> tuple[Any, bool, set[int]]:

    pyfunc: Callable[[Stream, Inventory | None, Event | None, dict], Any] = args[0]
    pyfunc_args: tuple[Stream, Inventory | None, Event | None] = args[1]
    config: dict = args[2]
    safe_exceptions_tuple: tuple[Exception] = args[3]
    suppress_stout_err: bool = args[4]
    ids = set(t.stats.segment_metadata.id for t in pyfunc_args[0])

    if suppress_stout_err:
        capture = suppress_printouts
    else:
        capture = nullcontext

    with capture():
        try:
            return pyfunc(*pyfunc_args, config), True, ids
        except safe_exceptions_tuple as exc:
            return exc, False, ids


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentMetadata:
    id: int
    latitude: float
    longitude: float
    depth: float
    dip: float
    azimuth: float
    elevation: float
    network_code: str
    station_code: str
    location_code: str
    channel_code: str
    arrival_time: datetime
    data_webservice_id: int
    event_webservice_id: int
    channel_id: int
    event_id: int
    event_latitude: float
    event_longitude: float
    event_depth_km: float
    event_time: datetime
    event_magnitude: float
    event_magnitude_type: str
    noise_window_s: float
    signal_window_s: float

    @property
    def event_distance_deg(self) -> float:
        return locations2degrees(
            lat1=self.latitude,
            long1=self.longitude,
            lat2=self.event_latitude,
            long2=self.event_longitude
        )

    @property
    def event_distance_km(self) -> float:
        return degrees2kilometers(self.event_distance_deg)

    @property
    def band_code(self):
        return self.channel_code[0:1]

    @property
    def instrument_code(self):
        return self.channel_code[1:2]

    @property
    def orientation_code(self):
        return self.channel_code[2:3]


@contextmanager
def suppress_printouts(*, suppress_warnings=False):
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with redirect(sys.stderr):
                with redirect(sys.stdout):
                    yield
    else:
        with redirect(sys.stderr):
            with redirect(sys.stdout):
                yield


@contextmanager
def redirect(src=None, dst: str | Path = os.devnull):
    """Redirect the OS-level file descriptor of `src` (sys.stdout or sys.stderr)
    to `dst` for the duration of the block. This silences C shared libraries that
    write directly to the underlying fd, while Python's own sys.stdout/sys.stderr
    objects are preserved and restored unchanged.

    No-op when:
      - src is None
      - src has no real fileno() (e.g. pytest's StringIO replacement for sys.stderr)

    :param src: sys.stdout or sys.stderr
    :param dst: destination path, default os.devnull
    """
    if src is None:
        yield
        return

    try:
        file_desc = src.fileno()
    except (AttributeError, OSError, ValueError):
        # pytest and similar tools replace sys.stderr/stdout with objects that
        # have no real file descriptor; treat as no-op
        yield
        return

    # Save the current Python wrapper so we can restore it exactly (same object,
    # no new wrapper created, no GC leak).
    # ORIGINAL BUG: created new wrappers via os.fdopen() on both redirect and restore
    old_stream = sys.stderr if src is sys.stderr else sys.stdout

    # Flush before touching the fd so no buffered Python output goes to dst.
    # Use the current stream object (old_stream), not src, which may be stale
    # if sys.stderr was already replaced (though here they are the same).
    old_stream.flush()

    # Save a duplicate of the original fd so we can restore it later.
    saved_fd = os.dup(file_desc)
    try:
        # Open dst and dup2 it onto file_desc, then close the temporary dst_fd.
        # ORIGINAL BUG: used `with open(dst) as dst_fileobject` which closed dst
        # before yield, leaving file_desc pointing to a closed fd during the block.
        dst_fd = os.open(dst, os.O_WRONLY)
        os.dup2(dst_fd, file_desc)
        os.close(dst_fd)
        # At this point file_desc points to dst at the OS level.
        # The existing Python wrapper (old_stream) still holds the same fd number
        # and will now write to dst — no new wrapper needed.

        try:
            yield
        finally:
            # Flush whatever is currently assigned to the stream (which is
            # old_stream, now writing to dst) before restoring.
            # ORIGINAL BUG: called src.flush() where src was the stale reference
            # captured at function entry, not the currently assigned stream.
            if src is sys.stderr:
                sys.stderr.flush()
            else:
                sys.stdout.flush()

            # Restore the original fd at OS level.
            os.dup2(saved_fd, file_desc)

            # Restore the original Python wrapper (same object, no new allocation).
            if src is sys.stderr:
                sys.stderr = old_stream
            else:
                sys.stdout = old_stream
    finally:
        os.close(saved_fd)


# def get_slices(array, chunksize):
#     """Divide `len(array)` by `chunksize` yielding the array slices until exhaustion.
#     If `array` is an integer, it denotes the length of the array and the tuples
#     (start, end) will be yielded.
#     This method intelligently re-arranges the (start, end) indices in order to minimize
#     the number of iterations yielded. ``
#     """
#     if hasattr(array, '__len__'):
#         total = len(array)  # == array.shape[0] in case of numpy arrays
#     else:
#         total = array
#         array = None
#     rem = total % chunksize
#     quot = int(np.true_divide(total, chunksize))
#     if rem == 0:
#         # eg: total=6, chunksize=2:  rem=0, quot=3
#         # repeat 2 three times:
#         iterable = repeat(chunksize, quot)
#     elif quot > rem:
#         # eg: total=7, chunksize=2: rem=1, quot=3
#         # a) repeat 2 two times
#         # b) repeat 3 one time
#         # start with a) so that the 1st progressbar update might be slightly faster:
#         iterable = chain(repeat(chunksize, quot-rem), repeat(chunksize+1, rem))
#     else:
#         # eg: total=7, chunksize=5: rem=2, quot=1
#         # a) yield 2 (one time)
#         # b) repeat 5 one time
#         # start with a) so that the 1st progressbar update might be slightly faster:
#         iterable = chain([rem], repeat(chunksize, quot))
#     start = end = 0
#     for chunk in iterable:
#         start = end
#         end = start + chunk
#         yield array[start:end] if array is not None else (start, end)
#
#
