"""
Segments download functions
"""
# :date: Dec 3, 2017
from __future__ import annotations

from collections import namedtuple
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
from stream2segment.download.channels import (
    url_col, orient_col, net_col, sta_col, loc_col, band_col, inst_col
)
from stream2segment.download.mseedlite import MSeedError, Input
from stream2segment.download.stationsearch import (
    atime_col, dist_col, ev_id_col, ch_id_col
)
from stream2segment.io.utils import get_progressbar, estimate_buffer_size
from stream2segment.io.db.pdsql import (
    sync_pkey, get_row_count, get_col_max, insert, executemany, fetch_df, select
)
from stream2segment.io.db.models import Segment, MiniSeed, SkippedSegment
from stream2segment.download.utils import (
    fdsn_url_qs, IdOnceLogFilter, fdsn_url, FailedDownload, NoSegmentsToDownload
)
from stream2segment.download.url import (
    get_host, read_urls, Response, read_url, responses
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)

already_skipped_col = "_.already_skipped._"


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
        [ev_id_col, ch_id_col],
            chunksize=min(1000, len(segments))
        )
        already_saved = segments[Segment.id.key].notna()
        if already_saved.any():
            logger.info(
                f"Discarding {already_saved.sum():,} already downloaded segments"
            )
            segments = segments[~already_saved]
            segments.pop(Segment.id.key)

    if get_row_count(engine, SkippedSegment) > 0 and not segments.empty:

        # segments = sync_pkey(
        #     segments,
        #     engine,
        #     SkippedSegment,
        #     [SkippedSegment.event_id.key, SkippedSegment.channel_id.key],
        #     chunksize=min(1000, len(segments))
        # )
        _suf = '_.db._'
        select_stmt = select(
            SkippedSegment.id,
            SkippedSegment.download_code,
            SkippedSegment.event_id,
            SkippedSegment.channel_id
        )
        cmp_cols = [SkippedSegment.channel_id.key, SkippedSegment.event_id.key]
        id_col = SkippedSegment.id.key
        dl_col  = SkippedSegment.download_code.key
        segments[id_col] = pd.Series(pd.NA, index=segments.index, dtype='Int64')
        segments[dl_col] = pd.Series(pd.NA, index=segments.index, dtype='Int64')
        for skipped_segments in fetch_df(engine, select_stmt):
            segments = segments.merge(
                skipped_segments, how='left', on=cmp_cols, suffixes=('', _suf)
            )
            segments[id_col] = segments[id_col].fillna(segments[id_col + _suf])
            segments[dl_col] = segments[dl_col].fillna(segments[dl_col + _suf])

        mask = segments[id_col].notna()
        if restricted_download:
            # with restricted download (credentials), retry also 204, as sometimes that
            # is the code returned when we request restricted data with no credentials
            mask &= (segments[dl_col] != 204)
        segments = segments[~mask]
        segments.pop(SkippedSegment.id.key)

    if segments.empty:
        raise NoSegmentsToDownload(_nothing_to_download_msg)

    return segments

# used for testing
_nothing_to_download_msg = 'all segments already downloaded'


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
        get_host(u) for u in segments[url_col].cat.categories
    )

    user_passwords: dict[str, tuple[str,str]] = None
    if credentials is not None:
        user_passwords = {}
        for url in segments[url_col].cat.categories:
            if isinstance(credentials, bytes):
                req = Request(
                    fdsn_url(url, new_service='dataselect', new_method='auth'),
                    data=credentials
                )
                response = read_url(req)
                if not response.is_ok:
                    logger.warning(str(response))
                    if 'queryauth' in url:
                        rem = segments[url_col] == url
                        if rem.any():
                            logger.warning(
                                f'Discarding {rem.sum():,} segment(s) '
                                f'(error acquiring credentials from their URL domain)'
                            )
                            segments = segments[segments[url_col] != url]
                    continue
                data = response.data
                if ':' not in data:
                    raise ValueError(
                        f'Invalid user and password from {req.full_url}. '
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

    db_bufsize = estimate_buffer_size('miniseed')  # avg size of MiniSeed as benchmark

    processed_indices = []
    # this is the maximum id (primary key) of NoDataSegments.
    # On download error (no data), it will be used to get if we need to save the segment
    # download info:

    if already_skipped_col not in segments.columns:
        def not_already_skipped(*a, **kw):
            return True
    else:
        def not_already_skipped(idx, segs):
            return not segs.at[idx, already_skipped_col]

    skipped_segment_codes = set(m.value for m in MiniSeedErrorCode) | {204}

    skipped_segments_current_id = get_col_max(engine, SkippedSegment.id)
    segments_current_id = get_col_max(engine, Segment.id)

    sql_insert_ok = [insert(Segment), insert(MiniSeed)]
    rows_ok = []
    sql_insert_skip = [insert(SkippedSegment)]
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
                            skipped_segments_current_id += 1
                            rows_skip.append(
                                prepare_skipped_segment_to_insert(
                                    skipped_segments_current_id,
                                    segments,
                                    idx,
                                    response.status_code
                                )
                            )
                            if len(rows_skip) >= db_bufsize:
                                written_skipped += sum(
                                    1 for _ in executemany(
                                        engine, sql_insert_skip, rows_skip
                                    )
                                )
                                rows_skip.clear()

                    elif response.is_ok:

                        segments_current_id += 1
                        rows_ok.append(
                            prepare_segment_to_insert(
                                segments_current_id, segments, idx, response.data  # noqa
                            )
                        )
                        if len(rows_ok) >= db_bufsize:
                            written_ok += sum(
                                1 for _ in executemany(engine, sql_insert_ok, rows_ok)
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
                            str(response),
                            extra={'ID': (url_domain, response.status_code)}
                        )

                    stats.increment(url_domain, response.status_code)
                    pbar.update(1)

                if max_download_concurrency <= 1 or not retry_decreasing_concurrency:
                    # add segments not processed to the stats
                    counts = segments[url_col].value_counts()
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
                    # if not segments.empty:
                    #     time.sleep(30)
    finally:
        if len(rows_ok):
            written_ok += sum(1 for _ in executemany(engine, sql_insert_ok, rows_ok))
        if len(rows_skip):
            written_skipped += sum(
                1 for _ in executemany(engine, sql_insert_skip, rows_skip)
            )

    if id_once_filter is not None:
        logger.removeFilter(id_once_filter)

    return stats


class MiniSeedErrorCode(IntEnum):
    BAD_DATA = -201
    OUT_OF_TIME_BOUNDS = -202


def download(
    segments: pd.DataFrame,
    time_window: tuple[float, float],
    user_passwords: dict[str, tuple[str, str]],
    max_download_concurrency: int,
    download_timeout,
    download_blocksize
) -> Iterable[tuple[int, Response]]:
    """
    Download segments and yields results
    """
    grp_cols = [
        ev_id_col, net_col, sta_col, loc_col, band_col, inst_col
    ]
    dataframes = segments.groupby(grp_cols, sort=False, observed=True)

    requests_cache: dict[str, dict[str, int | datetime]] = {}
    noise_w = timedelta(minutes=time_window[0])
    signal_w = timedelta(minutes=time_window[1])

    def get_request(ev_id, net, sta, loc, band, inst, dfr:pd.DataFrame) -> str:
        dc_url = dfr[url_col].iloc[0]
        a_time = dfr[atime_col].iloc[0].to_pydatetime()
        # start and end (round down and round up to nearest second):
        req_start = (a_time + noise_w).replace(microsecond=0)
        req_end = (a_time + signal_w + timedelta(seconds=1)).replace(microsecond=0)
        params = {
            'start': req_start,
            'end': req_end,
            'net': net or None,
            'sta': sta or None,
            'loc': loc or None,
            'cha': ",".join(f'{band}{inst}{o}' for o in dfr[orient_col]),
        }

        url_with_query = fdsn_url_qs(dc_url, **params)
        url_req_cache = {
            f'{net}.{sta}.{loc}.{band}{inst}{o}': i
            for i, o in zip(dfr.index, dfr[orient_col])
        }
        url_req_cache |= {'_.request_start': req_start, '_.request_end': req_end}
        requests_cache[url_with_query] = url_req_cache
        return url_with_query

    for response in read_urls(
        (get_request(*params, dfr) for (params, dfr) in dataframes),  # noqa
        max_concurrency=max_download_concurrency,
        timeout=download_timeout,
        blocksize=download_blocksize,
        credentials=user_passwords
    ):
        req_cache: dict = requests_cache.pop(response.request)
        req_start = req_cache.pop('_.request_start')
        req_end = req_cache.pop('_.request_end')

        if not response.is_ok:
            for idx in req_cache.values():
                yield idx, response
            continue

        for m_seed in unpack_miniseed(response.data, set(req_cache.keys())):
            idx = req_cache[m_seed.seed_id]  # dataframe index value

            if m_seed.data is None:
                yield idx, Response(
                    None,
                    SkippedSegment.BAD_DATA,
                    response.request
                )
                continue

            # we want at least something before and after the arrival time:
            if m_seed.start >= req_end or m_seed.end <= req_start:
                yield idx, Response(
                    None,
                    SkippedSegment.OUT_OF_TIME_BOUNDS,
                    response.request
                )
                continue

            yield idx, Response(
                m_seed, response.status_code, response.request
            )

unpacked_miniseed = namedtuple(
    'unpacked_miniseed',
    ['seed_id', 'data', 'start', 'end', 'fsamp', 'maxgap']
)

def unpack_miniseed(
    data: bytes, expected_seed_ids: set[str]
) -> Iterable[unpacked_miniseed]:
    """
    Unpack data into its "traces" (time series). Returns an iterable of MiniSeedInfo
    """
    unpacked_records = {_:[] for _ in expected_seed_ids}
    stream = BytesIO(data)
    try:
        for rec in Input(stream):
            seed_id = (
                f'{rec.net.strip()}.'
                f'{rec.sta.strip()}.'
                f'{rec.loc.strip()}.'
                f'{rec.cha.strip()}'
            )
            if seed_id in unpacked_records:
                unpacked_records[seed_id].append(rec)
    except UnicodeDecodeError as exc:
        # invalidate all miniseed. though harsh, it allows us to track problems
        unpacked_records = {_: [] for _ in expected_seed_ids}

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
                        * fsamp # - 1
                    )
                    max_gap_ratios.append(gap_ratio)
                # if abs(curr_max_gap_ratio) > abs(max_gap_overlap_ratio):
                #     max_gap_overlap_ratio = curr_max_gap_ratio

            yield unpacked_miniseed(
                seed_id=seed_id,
                data=bytesio.getvalue(),
                start=records[0].begin_time,
                end=records[-1].end_time,
                fsamp=fsamp,
                maxgap=max(max_gap_ratios)
            )

        except MSeedError as _:
            yield unpacked_miniseed(
                seed_id=seed_id,
                data=None,
                start=None,
                end=None,
                fsamp=None,
                maxgap=None
            )

        finally:
            bytesio.close()


def prepare_skipped_segment_to_insert(
    db_id: int, segments: pd.DataFrame, idx: int, code: int
) -> dict:
    return {
        SkippedSegment.id.key: db_id,
        SkippedSegment.channel_id.key: int(
            segments.at[idx, SkippedSegment.channel_id.key]
        ),
        SkippedSegment.event_id.key: int(
            segments.at[idx, SkippedSegment.event_id.key]
        ),
        SkippedSegment.download_code.key: code
    }


def prepare_segment_to_insert(
    db_id: int, segments: pd.DataFrame, idx: int, m_seed: unpacked_miniseed
) -> dict:

    arrival_time = segments.at[idx, atime_col]
    return{
        Segment.id.key: db_id,
        dist_col: int(
            segments.at[idx, dist_col]
        ),
        # Segment.webservice_id.key: int(segments.at[idx, Segment.webservice_id.key]),
        ev_id_col: int(segments.at[idx, ev_id_col]),
        ch_id_col: int(segments.at[idx, ch_id_col]),
        Segment.noise_window_s.key: int(round(
            (m_seed.start - arrival_time).total_seconds()
        )),
        Segment.signal_window_s.key: int(round(
            (m_seed.end - arrival_time).total_seconds()
        )),
        Segment.gap_score_percent.key: int(min(
            100 * m_seed.maxgap, 10000
            # 100000 because we want smallint. Also, high values provide no relevant
            # info
        )),
        MiniSeed.data.key: m_seed.data,
    }


class DownloadStats:

    def __init__(self, url_domains: Iterable[str]):
        self._stats = {}
        self._status_msg = {_.value: url.responses[_] for _ in url.responses}
        self._status_msg[MiniSeedErrorCode.BAD_DATA.value] = 'Corrupted MiniSeed'
        self._status_msg[
            MiniSeedErrorCode.OUT_OF_TIME_BOUNDS.value
        ] = 'MiniSeed time range mismatch'
        self._unknown_code = max(self._status_msg) + 1
        self._status_msg[self._unknown_code] = 'No available info'

        for u in url_domains:
            self._stats[u] = {}

    def increment(self, url_domain, status_code, count=1):

        row = self._stats.get(url_domain)
        if row is None:
            return

        if status_code is None:
            real_code = self._unknown_code
        else:
            real_code = status_code
            if real_code not in self._status_msg:
                try:
                    _ = self._status_msg[int(real_code)]
                except (TypeError, ValueError, KeyError):
                    real_code = self._unknown_code

        row[real_code] = row.get(real_code, 0) + count

    @property
    def _status_codes(self) -> list:
        statuses = {s for statuses in self._stats.values() for s in statuses}
        ok_statuses = {s for s in statuses if 200 <= s < 300}
        err_statuses = statuses - ok_statuses
        return sorted(ok_statuses) + sorted(err_statuses)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self._stats, orient="index").fillna(0).astype(int)
        df = df.reindex(
            index=sorted(self._stats.keys()), columns=self._status_codes
        ).rename(
            columns={c: self._status_msg[c] for c in self._status_codes},
        )
        df["Total"] = df.sum(axis=1)
        df.loc["Total"] = df.sum(axis=0)
        df.loc["Total", "Total"] = df.iloc[:-1, :-1].values.sum()
        df.index.name='URL:'
        df.columns.name='Download message:'
        return df

    def to_dict(self) -> dict:
        """"""
        return self.to_dataframe().to_dict(orient='index')

    def __str__(self):
        """
        Print a nicely formatted table with the statistics of the download.
        Return the empty string if this object is empty
        """
        df = self.to_dataframe()
        return df.to_string(
            index=True,
            header=True,
            na_rep="",
            formatters={col: "{:,}".format for col in df.columns}
        )
