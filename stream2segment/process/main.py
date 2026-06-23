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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta, datetime, UTC
import warnings
from io import BytesIO
from multiprocessing import Pool, cpu_count
import signal
import inspect

from typing import Any

from sqlalchemy import select, func, tuple_, Engine
import yaml
from obspy.core.event import Event as ObspyEvent
from obspy import Stream, Inventory, read, read_events, read_inventory
from obspy.geodetics import locations2degrees, degrees2kilometers

from stream2segment.io.db import secure_dburl, create_engine
from stream2segment.io.db.models import (
    StationXML, Channel, Segment, Event, QuakeML, MiniSeed
)

from stream2segment.process.segments_selection import build_where_clause
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
    pass  # (we can also pass an exception in the __init__, superclass converts it)


# Disclaimer: this module is over-documented to keep track of all implementation
# details addressing the following issues when handling big data and a RDBMS:
# 1. Memory leaks (too many objects in the RDBMS session)
# 2. Slowdowns (long RDBMS queries)
# 3. Undesired printouts (external ObsPy C libraries that have to be caught)
# 4. Python multiprocessing with RDBMS queries


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

    writer = get_writer(outfile, append, writer_options)

    num_ok = 0
    with writer:

        if verbose:
            print(
                ascii_decorate("\n".join([
                    f"Input database:      {secure_dburl(dburl)}",
                    f"Output file:         {str(outfile) if outfile else 'n/a'}",
                    f"Processing function: {pyfile or 'n/a'}",
                    f"Log file:            {str(logfile) if logfile else 'n/a'}",
                    f"Config. file:        {str(cfg_file) if cfg_file else 'n/a'}"
                ]))
            )

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

    # check params:
    if isinstance(config, (str, Path)):
        try:
            with open(config) as _config:
                config = yaml.safe_load(_config)
        except yaml.YAMLError as exc:
            raise BadParam(f"invalid config: {exc}") from exc

    elif not config:
        config = {}

    _valid_pyfunc(pyfunc)

    stmt = build_select(segments_selection)

    total = 0
    engine = get_engine(dburl)
    if verbose:
        stmt_count = select(func.count()).select_from(stmt.subquery())
        with engine.connect() as conn:
            total = conn.execute(stmt_count).scalar_one()

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
    # `create_processing_env` redirects Python BUT ALSO external libraries errors which
    # might mess up the terminal printout (e.g. progressbar). Python warnings should be
    # redirected as well because normally printed to `stderr`, so avoid capturing them
    # (`warnings_filter=None`). `create_processing_env` is also called in Python
    # subprocesses, if present. For info see :func:`process_segments_mp`
    with (
        start_logging(logger, logfile, verbose),
        create_processing_env(
            total,
            redirect_stderr=sys.stderr.isatty(),
            warnings_filter=None
        ) as pbar
    ):
        try:
            if verbose and total:
                logger.info(f"{total:,} segment(s) to process found")
                # Show the progressbar now, because the 1st chunk might be ready in min,
                # and an empty screen might give the impression of a program hang:
                time.sleep(0.5)
                pbar.render_progress()

            exec_process_func_args = (
                (pyfunc, args, config, skip_exceptions) for args in get_segments(
                    engine,
                    segments_selection,
                    group_components,
                    False,
                    chunksize
                )
            )

            stime = time.time()

            if not num_processes:
                for (output, is_ok, ids) in map(
                    execute_process_function, exec_process_func_args
                ):
                    pbar.update(len(ids))
                    if is_ok:
                        oks += len(ids)
                        yield output
                    else:
                        errors += len(ids)
                        logger.warning(
                            f"segment id(s)={' ,'.join(str(i) for i in ids)}): {output}"
                        )
            else:

                with Pool(
                    processes=num_processes,
                    initializer=_mp_initializer
                ) as pool:

                    try:

                        for (output, is_ok, ids) in pool.imap_unordered(
                            execute_process_function, exec_process_func_args
                        ):
                            pbar.update(len(ids))
                            if is_ok:
                                oks += len(ids)
                                yield output
                            else:
                                errors += len(ids)
                                logger.warning(
                                    f"segment id(s)={' ,'.join(str(i) for i in ids)}): "
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


def _valid_pyfunc(pyfunc):
    """
    Check if the argument is a valid processing Python function by inspecting its
    signature
    """
    params = inspect.signature(pyfunc).parameters  # dict[str, inspect.Parameter]
    # less than two arguments? then function invalid:
    if len(params) < 3:
        raise BadParam(
            f'Python function should have at least 3 arguments '
            f'`(segment, station, event)`, {len(params)} found'
        )
    # more than 2 args? then we need to have them with a default set:
    for pname, param in list(params.items())[2:]:
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # it is not *args or **kwargs, does it have a default?
            if not param.default == param.empty:
                raise BadParam(
                    f'Python function argument "{pname}" should have a default, '
                    f'or be removed'
                )


def _mp_initializer():
    """
    Set up the worker processes to ignore SIGINT altogether,
    and confine all the cleanup code to the parent process (e.g. Ctrl+C pressed)
    """
    # For info see https://stackoverflow.com/a/6191991 and links therein
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, check_db_existence=True)


def build_select(where_conditions: dict | None = None):
    """
    build select statement with provided where_conditions
    """
    stmt = (
        select(
            Segment.id,
            Segment.noise_window_s,
            Segment.signal_window_s,
            MiniSeed.data,
            Channel.id.label('channel_id'),
            Channel.data_webservice_id,
            Channel.stationxml_id ,
            # Channel.network_code,
            # Channel.station_code,
            Channel.latitude,
            Channel.longitude,
            Channel.elevation,
            # Channel.start_time,
            # Channel.location_code,
            # Channel.band_code,
            # Channel.instrument_code ,
            # Channel.orientation_code,
            Channel.depth,
            Channel.azimuth,
            Channel.dip,
            # Channel.sample_rate.label('channel_sample_rate'),
            Event.id.label('event_id'),
            # Event.webservice_id,
            # Event.eventid,
            Event.time.label('event_time'),
            Event.latitude.label('event_latitude'),
            Event.longitude.label('event_longitude'),
            Event.depth_km.label('event_depth_km'),
            Event.mag_type.label('event_magnitude_type'),
            Event.magnitude.label('event_magnitude'),
        )
        .select_from(Segment)
        .join(Channel, Channel.id == Segment.channel_id)
        .join(Event, Event.id == Segment.event_id)
        .join(MiniSeed, MiniSeed.id == Segment.id)
    )

    if where_conditions:
        stmt = stmt.where(build_where_clause(where_conditions))

    return stmt


def get_segments(
    db: str | Engine,
    where_condition,
    group_components: bool = False,
    segments_only: bool = False,
    chunksize: int | None = None
) -> Iterable[tuple[Stream, Inventory | None, ObspyEvent | None]]:

    if isinstance(db, str):
        engine = get_engine(db)
    else:
        engine = db

    orderby_columns = (Channel.stationxml_id, Segment.event_id, Segment.id)

    if group_components:
        orderby_columns = (
            Channel.network_code,
            Channel.station_code,
            Channel.location_code,
            Channel.instrument_code,
            Channel.band_code,
            Event.id,
            Segment.id
        )

    buffer = []
    last_key = None
    stmt_base = build_select(where_condition).order_by(*orderby_columns)

    if chunksize is None:
        chunksize = estimate_buffer_size(50)

    while True:
        stmt = stmt_base
        if last_key is not None:
            stmt = stmt.where(tuple_(*orderby_columns) > last_key)

        stmt = stmt.limit(chunksize)

        with engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        if not rows:
            break

        if not group_components:
            last_key = (rows[-1].stationxml_id, rows[-1].event_id, rows[-1].id)
            for r in rows:
                yield db_to_obspy(engine, segments_only, r)
            continue

        buffer.extend(rows)

        split_idx = len(buffer) - 1
        while split_idx > 0 and same_group(rows[split_idx], rows[split_idx - 1]):
            split_idx -= 1

        if split_idx == 0:
            continue

        to_yield = buffer[:split_idx]
        buffer = buffer[split_idx:]

        last_key = (
            rows[-1].Channel.network_code,
            rows[-1].Channel.station_code,
            rows[-1].Channel.location_code,
            rows[-1].Channel.instrument_code,
            rows[-1].Channel.band_code,
            rows[-1].event_id,
            rows[-1].id,
        )
        for r in split_in_same_group_chunks(to_yield):
            yield db_to_obspy(engine, segments_only, *r)

    if buffer:
        # surely group by orientation:
        for b in split_in_same_group_chunks(buffer):
            yield db_to_obspy(engine, segments_only, *b)


def same_group(row1, row2):
    return (
        row1.Channel.network_code == row2.Channel.network_code and
        row1.Channel.station_code == row2.Channel.station_code and
        row1.Channel.location_code == row2.Channel.location_code and
        row1.Channel.band_code == row2.Channel.band_code and
        row1.Channel.instrument_code == row2.Channel.instrument_code and
        row1.event_id == row2.event_id
    )


def split_in_same_group_chunks(rows):
    start = 0
    end = 0
    while end < len(rows):
        if same_group(rows[start], rows[end]):
            end += 1
            continue
        yield rows[start:end]
        start = end
    if start < len(rows):
        yield rows[start:]


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


def execute_process_function(args: tuple[
    Callable[[Stream, Inventory | None, Event | None, dict], Any],
    tuple[Stream, Inventory | None, Event | None],
    dict,
    tuple[Exception]
]) -> tuple[Any, bool, set[int]]:

    pyfunc: Callable[[Stream, Inventory | None, Event | None, dict], Any] = args[0]
    pyfunc_args: tuple[Stream, Inventory | None, Event | None] = args[1]
    config: dict = args[2]
    safe_exceptions_tuple: tuple[Exception] = args[3]
    ids = set(t.stats.segment_metadata.id for t in pyfunc_args[0])
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
