"""
Segments download functions
"""
# :date: Dec 3, 2017
from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta, datetime
import logging
from enum import IntEnum
from io import BytesIO
from math import log
import time
from urllib.request import Request

import pandas as pd
import psutil
from sqlalchemy import Engine

from stream2segment.download import url
from stream2segment.download.modules.mseedlite import MSeedError, Input
from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import (
    sync_pkey, get_row_count, get_col_max, create_insert_statement, execute_sql
)
from stream2segment.io.db.models import (
    WebService, Segment, Channel, MiniSeed, SkippedSegment
)
from stream2segment.download.modules.utils import (
    fdsn_url_qs, IdOnceLogFilter, fdsn_url, FailedDownload
)
from stream2segment.download.url import (
    get_host, read_async, Response, urlread
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)

def prepare_for_download(
    *,
    engine: Engine,
    segments: pd.DataFrame,
    restricted_download: bool
):
    if get_row_count(engine, Segment) > 0:
        # remove already downloaded segments (with data):
        segments = sync_pkey(
            segments,
            engine,
            Segment,
            Segment.id.key,
        [Segment.event_id.key, Segment.channel_id.key],
            chunksize=min(1000, len(segments))
        )
        # max_id = segments_with_pkeys.attrs.pop(f'{Segment.id.key}_max')
        already_saved = segments[Segment.id.key].motna()
        if already_saved.any():
            logger.info(
                f"Discarding {already_saved.sum():, } already downloaded segments"
            )
            segments = segments[~already_saved]
            segments.pop(Segment.id.key)

    if get_row_count(engine, SkippedSegment) > 0:
        # Put ids for segments to be saved in a tmp column:
        # segments = segments.rename(columns={Segment.id.key: 'segment.id'})

        # find segments to retry from NoDataSegment table
        where_clause = (SkippedSegment.download_code != 204)
        if not restricted_download:
            where_clause = None
        segments = sync_pkey(
            segments,
            engine,
            SkippedSegment,
            SkippedSegment.id.key,
            [SkippedSegment.event_id.key, SkippedSegment.channel_id.key],
            where_clause,
            chunksize=min(1000, len(segments))
        )
        segments["_.already_skipped._"] =  segments[SkippedSegment.id.key].motna()
        segments.pop(SkippedSegment.id.key)
    # segments['_.new._'] = segments[NoDataSegment.id.key.isna()].astype(bool)
    # segments.pop(NoDataSegment.id.key)
    # segments = segments.rename(columns={SkippedSegment.id.key: "skipped_segment.id"})
    return segments


def download_and_save(
    *,
    engine: Engine,
    segments: pd.DataFrame,
    time_window,
    credentials: tuple[str, str] | bytes | None,
    max_download_concurrency,
    download_timeout,
    download_blocksize,
    show_progress=False
):
    """Download and saves the segments. segments_df MUST not be empty (this is
    not checked for)

    :param segments: the dataframe resulting from `prepare_for_download`.
        The Dataframe might or might not have the column 'download_code'. If it
        has, it will skip writing to db segments whose code did not change: in
        this case, nans stored under 'download_code' in segments_df indicate
        new segments, or segments for which the update has to be forced,
        whatever code is obtained (e.g., queryauth when previously a simple
        query was used)
    """

    stats = DownloadStats(
        get_host(u) for u in segments[WebService.url.key].cat.categories
    )

    user_passwords: dict[str, tuple[str,str]] = None
    if credentials is not None:
        user_passwords = {}
        for url in segments[WebService.url.key].cat.categories:
            if isinstance(credentials, bytes):
                req = Request(
                    fdsn_url(url, new_service='dataselect', new_method='auth'),
                    data=credentials
                )
                response = urlread(req)
                if not response.is_ok or not response.data:
                    if 'queryauth' in url:
                        segments = segments[segments[WebService.url.key] != url]
                    logger.warning(
                        f'Could not get user password via token from {req.full_url}')
                    continue
                data = response.data
                if ':' not in data:
                    raise ValueError('Invalid user and password returned. '
                                     'This could be a data-center bug')
                else:
                    user_pass = tuple(data.split(':'))
            else:
                user_pass = tuple(credentials)

            user_passwords[get_host(url)] = user_pass

        if segments.empty:
            raise FailedDownload(
                'Could not get user password from eida token for any URLs'
            )

    retry_decreasing_concurrency = False
    if max_download_concurrency is None:
        retry_decreasing_concurrency = True
        max_download_concurrency = 8

    # report seg. errors only once per error type and data center:
    id_once_filter: IdOnceLogFilter | None = None

    db_bufsize = compute_db_buf_size(1)  # avg size of 10 minutes MiniSeed in Mb

    processed_indices = []
    # this is the maximum id (primary key) of NoDataSegments.
    # On download error (no data), it will be used to get if we need to save the segment
    # download info:


    already_skipped_col = "_.already_skipped._"
    if already_skipped_col not in segments.columns:
        def not_already_skipped(*a, **kw):
            return True
    else:
        def not_already_skipped(idx, segs):
            return not segs.at[idx, already_skipped_col]

    skipped_segment_codes = set(MiniSeedErrorCode) | {204}

    segments_current_id = get_col_max(engine, Segment.id)

    sql_insert_ok = [
        create_insert_statement(Segment),
        create_insert_statement(MiniSeed)
    ]
    rows_ok = []
    sql_insert_skip = [create_insert_statement(SkippedSegment)]
    rows_skip = []

    written_ok = 0
    written_skipped = 0

    try:
        with get_progressbar(len(segments) if show_progress else 0) as pbar:

            while not segments.empty:
                for idx, response in download(
                    segments,
                    time_window,
                    user_passwords,
                    max_download_concurrency,
                    download_timeout,
                    download_blocksize
                ):
                    url_domain = get_host(response.request)
                    processed_indices.append(idx)

                    if response.status_code in skipped_segment_codes:
                        if not_already_skipped(idx, segments):
                            rows_skip.append(
                                prepare_skipped_segment_to_insert(
                                    segments,
                                    idx,
                                    MiniSeedErrorCode(response.status_code)
                                )
                            )
                            if len(rows_skip) >= db_bufsize:
                                written_skipped += sum(
                                    1 for _ in execute_sql(engine, sql_insert_skip, rows_skip)
                                )
                                rows_skip.clear()

                    elif response.is_ok:  # status code in [200, 300[, not 204

                        segments_current_id += 1
                        rows_ok.append(
                            prepare_segment_to_insert(
                                segments_current_id, segments, idx, *response.data  # noqa
                            )
                        )
                        if len(rows_ok) >= db_bufsize:
                            written_ok += sum(
                                1 for _ in execute_sql(engine, sql_insert_ok, rows_ok)
                            )
                            rows_ok.clear()


                    else:
                        if id_once_filter is None:
                            logger.warning('Detailed segment download errors '
                                           '(showing only first of each type per '
                                           'URL domain):')
                            id_once_filter = IdOnceLogFilter()
                            logger.addFilter(id_once_filter)

                        logger.warning(
                            f"Error class (code): {response.data} "
                            f"({response.status_code}), "
                            f"URL: {response.request}",
                            extra={'ID': (url_domain, response.status_code)}
                        )

                    stats.increment(url_domain, response.status_code)
                    pbar.update(1)

                if max_download_concurrency <= 1 or not retry_decreasing_concurrency:
                    # add segments not processed to the stats
                    counts = segments[WebService.url.key].value_counts()
                    for url_, count in counts.items():
                        stats.increment(get_host(url_),None, count)
                        pbar.update(count)
                    # close loop: set empty dataframe (will break the loop)
                    segments = pd.DataFrame()
                    logger.info(
                        f'Download not performed for {counts.sum():,} segments '
                        f'due to consistently repeated failures from their '
                        f'URL domain'
                    )  # FIXME BETTER (consistently?)
                else:
                    if processed_indices:
                        segments = segments.loc[
                            segments.index.difference(processed_indices)
                        ]
                    max_download_concurrency //= 2
                    # stop for a while to avoid stressing URL domains
                    time.sleep(30)
    finally:
        if len(rows_ok):
            written_ok += sum(
                1 for _ in execute_sql(engine, sql_insert_ok, rows_ok)
            )
        if len(rows_skip):
            written_skipped += sum(
                1 for _ in execute_sql(engine, sql_insert_skip, rows_skip)
            )

    if id_once_filter is not None:
        logger.removeFilter(id_once_filter)

    return stats


def download(
    segments: pd.DataFrame,
    time_window: tuple[float, float],
    user_passwords: dict[str, tuple[str, str]],
    max_download_concurrency: int,
    download_timeout,
    download_blocksize
):
    """Download segments and yields results

    :param dataframes: iterable of dataframes, one dataframe per request, one row
        per requested waveform. Moreover, each dataframe row is assumed to refer to the
        same data center (base URL) and have the same time span (start end)
    """

    # FIXME REMOVE
    # def openerfunc(dframe):
    #     """Return a Opener (or None) from the given dataframe. An Opener is the
    #     object needed to download restricted data"""
    #     return dc_dataselect_manager.opener(dframe[SEG.DCID].iloc[0])

    grp_cols = [
        Segment.event_id.key,
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.band_code.key,
        Channel.instrument_code.key
    ]
    dataframes = segments.groupby(grp_cols, sort=False, observed=True)

    requests_cache: dict[str, dict[str, int | datetime]] = {}
    noise_w = timedelta(minutes=time_window[0])
    signal_w = timedelta(minutes=time_window[1])

    def get_request(ev_id, net, sta, loc, band, inst, dfr:pd.DataFrame) -> str:
        dc_url = dfr[WebService.utl.key].iloc[0]
        a_time = dfr[Segment.arrival_time.key].iloc[0].to_pydatetime()
        # start and end (round down and round up to nearest second):
        req_start = (a_time + noise_w).replace(microsecond=0),
        req_end = (a_time + signal_w + timedelta(seconds=1)).replace(microsecond=0),
        params = {
            'start': req_start,
            'end': req_end,
            'net': net or None,
            'sta': sta or None,
            'loc': loc or None,
            'cha': ",".join(f'{band}{inst}{o}' for o in dfr[Channel.orientation_code.key]),
        }
        _url = fdsn_url_qs(dc_url, **params)
        requests_cache[_url] = {
            f'{net}.{sta}.{loc}.{band}{inst}{o}': i
            for i, o in zip(dfr.index, dfr[Channel.orientation_code])
        } | {'request_start': req_start, 'request_end': req_end}
        return _url

    for response in read_async(
        (get_request(*params, dfr) for (params, dfr) in dataframes),  # noqa
        max_concurrency=max_download_concurrency,
        #max_global_concurrency=max_thread_workers,
        #max_workers_d=max_workers_d,
        #max_concurrency=max_workers_d,
        timeout=download_timeout,
        blocksize=download_blocksize,
        credentials=user_passwords
    ):
        req_cache: dict = requests_cache.pop(response.request)
        req_start = req_cache.pop('start')
        req_end = req_cache.pop('end')

        if not response.is_ok or response.status_code == 204:
            for idx in req_cache.values():
                yield idx, response
            continue

        for seed_id, data in unpack_miniseed(response.data, set(req_cache.keys())):
            idx = req_cache[seed_id]  # dataframe index value

            if data is None:
                yield idx, Response(
                    None,
                    SkippedSegment.BAD_DATA,
                    response.request
                )
                continue

            (mseed_data, fsamp, start, end, maxgap) = data

            # check time bounds:
            # start = start.replace(microsecond=0)
            # end = (end + timedelta(seconds=1)).replace(microsecond=0)

            # we want at least something before and after the arrival time:
            if start >= req_end or end <= req_start:
                yield idx, Response(
                    None,
                    SkippedSegment.OUT_OF_TIME_BOUNDS,
                    response.request
                )
                continue

            yield idx, Response(
                data, response.status_code, response.request
            )


def unpack_miniseed(
    data: bytes, expected_seed_ids: set[str]
) -> Iterable[tuple[str, tuple | None]]:
    """
    Unpack data into its "traces" (time series). Returns an iterable of MiniSeedInfo
    """
    unpacked_records = {_:[] for _ in expected_seed_ids}
    stream = BytesIO(data)
    try:
        for rec in Input(stream):
            try:
                seed_id = (
                    b"%s.%s.%s.%s" %
                    (rec.net.strip(), rec.sta.strip(), rec.loc.strip(), rec.cha.strip())
                ).decode('utf8')
                if seed_id in unpacked_records:
                    unpacked_records[seed_id].append(rec)
            except UnicodeDecodeError as exc:
                # invalidate all miniseed. though harsh, it allows us to track problems
                unpacked_records = {_:[] for _ in expected_seed_ids}
                break
    finally:
        stream.close()


    for seed_id, records in unpacked_records.items():
        bytesio = BytesIO()

        try:
            if not len(records):
                raise MSeedError()

            # get records and sort ascending by time
            records.sort(key=lambda elm: elm.begin_time)
            fsamp = records[0].fsamp
            max_gap_ratios: list[float] = []

            for i, record in enumerate(records):

                if record.fsamp != fsamp:
                    raise MSeedError()

                try:
                    record.write(bytesio, int(log(record.size) / log(2)))
                except Exception:
                    raise MSeedError()

                if i > 0:
                    gap_ratio = (
                        (record.begin_time - records[i-1].end_time).total_seconds()
                        * fsamp - 1
                    )
                    max_gap_ratios.append(gap_ratio)
                # if abs(curr_max_gap_ratio) > abs(max_gap_overlap_ratio):
                #     max_gap_overlap_ratio = curr_max_gap_ratio

            yield (
                seed_id, (
                    bytesio.getvalue(),
                    fsamp,
                    records[0].begin_time,
                    records[-1].end_time,
                    max(max_gap_ratios)
                )
            )

        except MSeedError as _:
            yield seed_id, None

        finally:
            bytesio.close()


def prepare_skipped_segment_to_insert(
    segments: pd.DataFrame, idx: int, code: MiniSeedErrorCode
) -> dict:
    return {
        SkippedSegment.id.key: int(
            segments.at[idx, SkippedSegment.id.key]
        ),
        SkippedSegment.channel_id.key: int(
            segments.at[idx, SkippedSegment.channelid.key]
        ),
        SkippedSegment.event_id.key: int(
            segments.at[idx, SkippedSegment.event_id.key]
        ),
        SkippedSegment.download_code.key: int(code.value)
    }


def prepare_segment_to_insert(
    db_id: int, segments: pd.DataFrame, idx: int, mseed_data, fsamp, start, end, maxgap
) -> dict:

    arrival_time = segments.at[idx, "arrival_time"]
    return{
        Segment.id.key: db_id,
        Segment.event_distance_km.key: int(
            segments.at[idx, Segment.event_distance_km.key]
        ),
        Segment.webservice_id.key: int(segments.at[idx, Segment.webservice_id.key]),
        Segment.event_id.key: int(segments.at[idx, Segment.event_id.key]),
        Segment.channel_id.key: int(segments.at[idx, Segment.channel_id.key]),
        Segment.noise_window_sec.key: int(round(
            (start - arrival_time).total_seconds()
        )),
        Segment.signal_window_sec.key: int(round(
            (end - arrival_time).total_seconds()
        )),
        Segment.gap_score_percent.key: int(min(
            100 * maxgap, 10000
            # 100000 because we want smallint. Also, high values provide no relevant info
        )),
        MiniSeed.data.key: mseed_data,
    }


class MiniSeedErrorCode(IntEnum):
    BAD_DATA = -201
    OUT_OF_TIME_BOUNDS = -202

# responses[CustomResponseCode.BAD_DATA] = \
#     "MiniSeed data is corrupted"
# responses[CustomResponseCode.OUT_OF_TIME_BOUNDS] = \
#     "MiniSeed time window is outside the requested time window"
# responses[200] += '. Data successfully downloaded'
# responses[CustomResponseCode.NOT_DOWNLOADED] = \
#     ('Data not downloaded (e.g., download suspended after '
#      'repeated failures from the same domain)')


class DownloadStats:

    def __init__(self, url_domains: Iterable[str]):
        self._stats = {}
        self._codes = {_.value: url.responses[_] for _ in url.responses}
        self._unknown_code = max(self._codes) + 1
        self._codes[self._unknown_code] = 'Download not performed'
        self._codes[MiniSeedErrorCode.BAD_DATA.value] = 'Corrupted MiniSeed'
        self._codes[
            MiniSeedErrorCode.OUT_OF_TIME_BOUNDS.value
        ] = 'MiniSeed time range mismatch'

        for u in url_domains:
            self._stats[u] = {}

    def increment(self, url_domain, status_code, count=1):
        try:
            if status_code is None:
                code = self._unknown_code
            else:
                code = int(status_code)
                if code not in self._codes:
                    return
        except ValueError:
            return

        row = self._stats.get(url_domain)
        if row is None:
            return

        row[code] = row.get(code, 0) + count

    @property
    def all_url_domains(self):
        return self._stats.keys()

    @property
    def all_codes(self) -> list:
        statuses = {s for statuses in self._stats.values() for s in statuses}
        ok_statuses = {s for s in statuses if 200 <= s < 300}
        err_statuses = statuses - ok_statuses
        return sorted(ok_statuses) + sorted(err_statuses)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self._stats, orient="index")
        df = df.reindex(
            index=sorted(self._stats.keys()),
            columns=[self._codes[c] for c in self.all_codes],
            fill_value=0
        ).astype(int)
        df["Total"] = df.sum(axis=1)
        df.loc["Total"] = df.sum(axis=0)
        df.loc["Total", "Total"] = df.iloc[:-1, :-1].values.sum()
        df.index.name='URL:'
        df.columns.name='Download message:'
        return df

    def to_dict(self):
        """"""
        return self.to_dataframe().to_dict(orient='index')

    def __str__(self):
        """Print a nicely formatted table with the statistics of the download.
        Return the empty string if this object is empty
        """
        df = self.to_dataframe()
        return df.to_string(
            index=True,
            header=True,
            na_rep="",
            formatters={col: "{:,}".format for col in df.columns}
        )


def compute_db_buf_size(
    item_avg_size_mb: float,
    max_mem_fraction: float = 0.2,
    hard_cap_mb: int = 4096
) -> int:
    mem = psutil.virtual_memory()

    available_mb = mem.available / (1024 ** 2)
    usable_mb = min(available_mb * max_mem_fraction, hard_cap_mb)

    return max(1, int(usable_mb // item_avg_size_mb))



# def check_suspiciously_duplicated_segment(segments_df):
#     """Check for suspiciously duplicated segments, i.e. different ids
#     but same (channel_id, request_start, request_end). These segments stem from distinct
#     events with very close spatio-temporal coordinates.
#     This function simply logs a message if any such duplicated segment is found,
#     it does NOT modify segments_df
#     """
#     seg_dupes_mask = segments_df.duplicated(subset=[SEG.CHAID, SEG.REQSTART,
#                                                     SEG.REQEND],
#                                             keep=False)
#     if seg_dupes_mask.any():
#         seg_dupes = segments_df[seg_dupes_mask]
#         msg = ("%d suspiciously duplicated segments found: this is most likely\n"
#                "due to events with different ids\n"
#                "but same (or very close) latitude, longitude, depth and time.")
#         logger.info(msg, len(seg_dupes))
#         seg_dupes_sorted = seg_dupes.sort_values(by=[SEG.CHAID, SEG.REQSTART,
#                                                      SEG.REQEND])
#         logwarn_dataframe(seg_dupes_sorted, "Suspicious duplicated segments",
#                           [SEG.CHAID, SEG.REQSTART, SEG.REQEND, SEG.EVID],
#                           max_row_count=100)
#
#
