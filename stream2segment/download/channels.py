"""
Stations/Channels download functions
"""
# :date: Dec 3, 2017
import re
import logging
from collections.abc import Iterable, Callable
import json
from datetime import datetime, timedelta, UTC
from io import BytesIO
from multiprocessing.pool import ThreadPool
from urllib.request import urlopen

import pandas as pd
from sqlalchemy import and_, Select, insert
from sqlalchemy.exc import IntegrityError

from stream2segment.download.events import (
    sync_webservice_urls_and_assign_ids, insert_id_col_na_values_to_db
)
from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import apply_table_dtypes, fetch_df, select, Engine, \
    get_col_max
from stream2segment.io.db.models import Channel, WebService, StationXML
from stream2segment.download.url import read_url, build_and_read_urls
from stream2segment.download.utils import (
    fdsn_url,
    fdsn_url_qs,
    fdsn_response_text_to_df,
    FailedDownload,
    NoSegmentsToDownload
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


start_col = Channel.start_time.key
end_col = "end"
cha_col = "channel_code"
net_col = Channel.network_code.key
sta_col = Channel.station_code.key
loc_col = Channel.location_code.key
band_col = Channel.band_code.key
inst_col = Channel.instrument_code.key
orient_col = Channel.orientation_code.key
url_col = WebService.url.key
lat_col = Channel.latitude.key
lon_col = Channel.longitude.key
sta_id_col = Channel.station_id.key
end_time_replacement = (
    datetime.now(UTC).replace(microsecond=0, tzinfo=None) + timedelta(days=30)
)  # some random margin in the future


def get_channels(
    engine : Engine,
    datacenter_urls,
    network: list[str],
    station: list[str],
    location: list[str],
    channel: list[str],
    start,
    end,
    min_sample_rate,
    eida_rs_urls,
    download_timeout: int | None = None,
    show_progress=False
) -> pd.DataFrame:
    cha_urls = get_channel_urls(
        datacenter_urls,
        eida_rs_urls,
        [n for n in network if not n.startswith('!')],
        [s for s in station if not s.startswith('!')],
        [l for l in location if not l.startswith('!')],
        [c for c in channel if not c.startswith('!')],
        start,
        end
    )

    filter_funcs = {}
    for key, vals in {
        net_col: network, sta_col: station, loc_col: location, cha_col: channel
    }.items():
        val = [v[1:] for v in vals if v.startswith('!')]
        if not val:
            continue
        reg = re.compile("|".join(f"^(?:{wild2regex(v)})$" for v in val))
        filter_funcs[key] = lambda s: ~s.str.match(reg)
    if min_sample_rate is not None and min_sample_rate > 0:
        filter_funcs[Channel.sample_rate.key] = lambda s: s >= min_sample_rate

    channels = download_channels(
        cha_urls,
        filter_funcs=filter_funcs,
        timeout=download_timeout,
        show_progress=show_progress
    )
    if channels.empty:
        raise FailedDownload(
            f'no channels downloaded; check your configuration, '
            'web services status, your internet connection. See log for details'
        )
    else:
        logger.info(
            f"{len(channels):,} channel(s) downloaded; "
            f"checking duplicates, conflicts, and saving"
        )

    categorical_columns = [
        net_col, sta_col, loc_col, band_col, inst_col, orient_col, url_col
    ]
    for c in categorical_columns:
        if not pd.api.types.is_categorical_dtype(channels[c]):
            channels[c] = channels[c].astype('str').astype('category')

    channels = resolve_inter_conflicts(channels, eida_rs_urls)
    if channels.empty:
        raise FailedDownload(
            'All channels discarded while dropping conflicts; See log for details'
        )

    ws_id_col = Channel.data_webservice_id.key
    channels = sync_webservice_urls_and_assign_ids(
        channels, engine, ids_column_name=ws_id_col
    )
    rem = channels[ws_id_col].isna() | channels[url_col].isna()
    if rem.any():
        channels = channels[~rem]
        if channels.empty:
            raise FailedDownload(
                f"All channels ({rem.sum()}) discarded (associated URL not saved to DB)"
            )
        else:
            logger.warning(
                f"Discarding {rem.sum()} channel(s) (associated URL not saved to DB)"
            )

    channels[ws_id_col] = channels[ws_id_col].astype(int)
    channels = save_channels(engine, channels)
    if channels.empty:
        raise FailedDownload(
            'No channels saved; this is likely due to a Database I/O error. '
            'See log for details'
        )

    # restore channel column by concatenating inst band orient:
    channels[cha_col] = (
        channels[band_col].str.cat(channels[inst_col]).str.cat(channels[orient_col])
    ).astype('category')
    channels.drop(columns=[band_col, inst_col, orient_col], inplace=True)

    # for safety, re-check categorical columns:
    for c in categorical_columns + [cha_col]:
        # need to check that column exists now:
        if c in channels.columns and not pd.api.types.is_categorical_dtype(channels[c]):
            channels[c] = channels[c].astype('str').astype('category')

    channels[sta_id_col] = channels[sta_id_col].astype('category')

    # return a copy of relevant columns only:
    return channels[[
        Channel.id.key,
        net_col,
        sta_col,
        loc_col,
        cha_col,
        lat_col,
        lon_col,
        start_col,
        end_col,
        url_col,
        sta_id_col
    ]]


def wild2regex(text: str):
    return (
        text.replace('.', r'\.').replace('?', '.').replace('*', '.*')
    )


def get_channel_urls(
    webservice_url,
    routing_service_url,
    network: list[str] | None = None,
    station: list[str] | None = None,
    location: list[str] | None = None,
    channel: list[str] | None= None,
    starttime: datetime | None = None,
    endtime: datetime | None = None
) -> Iterable[str]:
    """
    Return an iterator of FDSN station urls from the given arguments
    """
    if isinstance(webservice_url, str):
        webservice_url = [webservice_url]
    params = {
        'net': ','.join(n for n in network or []) or '*',
        'sta': ','.join(s for s in station or []) or '*',
        'loc': ','.join(l for l in location or []) or '*',
        'cha': ','.join(c for c in channel or []) or '*',
        'start': starttime,
        'end': endtime
    }

    urls_done = set()
    parsed_urls = []
    for service_url in webservice_url:
        service_url = service_url.lower().strip()

        if service_url in urls_done:
            continue
        urls_done.add(service_url)

        if service_url == 'eida':
            priority_skipped = 0
            eida_rs: dict = get_eida_rs_response(routing_service_url, **params)

            if not eida_rs:
                raise FailedDownload(
                    "No EIDA routing services returned valid data. Check "
                    "internet connection or configure the URLs in advanced settings"
                )

            # sort (for easier grouping here below):
            for eida_dc_dict in eida_rs:
                eida_dc_dict['params'] = sorted(
                    eida_dc_dict['params'],
                    key=lambda p: (p['net'], p['sta'], p['loc'], p['cha'])
                )
            for line in eida_rs:
                # overwrite start end:
                url = line['url']
                for _params in line['params']:
                    if _params.pop('priority', 2) > 1:
                        priority_skipped += 1
                        continue
                    # _params.pop('priority')
                    _params['start'] = starttime
                    _params['end'] = endtime
                    merged_into_previous = False
                    if len(parsed_urls) and parsed_urls[-1][0] == url:
                        _last_params = parsed_urls[-1][1]
                        merged_into_previous = sum([
                            _params['net'] == _last_params['net'],
                            _params['sta'] == _last_params['sta'],
                            _params['loc'] == _last_params['loc'],
                            _params['cha'] == _last_params['cha']
                        ]) == 3
                        if merged_into_previous:
                            for prm in ['net', 'sta', 'loc', 'cha']:
                                if _params[prm] != _last_params[prm]:
                                    _last_params[prm] += f',{_params[prm]}'
                                    break
                    if not merged_into_previous:
                        parsed_urls.append((url, _params))

        else:
            if service_url == 'iris':
                service_url = 'https://service.iris.edu/fdsnws/station/1/query'
            parsed_urls.append((service_url, params))

    override_params = dict(level='channel', format='text')
    for url, params in parsed_urls:
        try:
            fdsn_station_url = fdsn_url(url, new_service='station', new_method='query')
            if params.get('net', None) not in {'*', None}:
                # response likely no too big, proceed
                yield fdsn_url_qs(
                    fdsn_station_url, **{**params, **override_params}
                )
            else:
                # response likely too big, split by network size:
                for new_url, new_params in split_url_by_network_quantiles(
                    fdsn_station_url, params
                ):
                    yield fdsn_url_qs(
                        new_url, **{**new_params, **override_params}
                    )
        except ValueError as v_err:
            logger.warning(f"{url}: {v_err}")


def get_eida_rs_response(
    routing_service_url: list[str],
    net: str | None = None,
    sta: str | None = None,
    loc: str | None = None,
    cha: str | None= None,
    start: datetime | None = None,
    end: datetime | None = None,
    service='dataselect'
) -> dict:
    """Return the EIDA Routing Service response, as json dict"""
    for eida_rs_url in routing_service_url:
        url = fdsn_url_qs(
            eida_rs_url,
            net=net,
            sta=sta,
            loc=loc,
            cha=cha,
            start=start,
            end=end,
            service=service,
            format='json'
        )
        response = read_url(url, decode='utf8')
        if response.is_ok:
            try:
                return json.loads(response.data)
            except json.decoder.JSONDecodeError:
                logger.warning(
                    f"Skipping malformed JSON data from {response.request_url}"
                )
                pass
    return {}


def split_times(start: datetime, end: datetime, interval_years=5):
    cur = start
    while cur < end:
        nxt = min(cur.replace(year=cur.year + interval_years), end)
        yield cur, nxt
        cur = nxt


def split_url_by_network_quantiles(fdsn_station_url, params):
    """
    """
    # no network specified, query might take long (even for short time bounds).
    # get all networks and perform n subsets queries
    rows = []
    try:
        with urlopen(
            fdsn_url_qs(fdsn_station_url, **params, level='network', format='text')
        ) as r:
            _ = r.readline().decode().strip().split('|')  # header (just ignore)
            for line in r:
                split_line = line.decode().strip().split('|')
                rows.append((split_line[0].strip(), split_line[-1].strip()))
        dfr = pd.DataFrame(rows, columns=['net', 'count'])
        # convert count to numeric (should be int, but we set NaN as #station mean):
        dfr['count'] = pd.to_numeric(dfr['count'], errors='coerce')
        dfr.loc[pd.isna(dfr['count']), 'count'] = int(dfr['count'].median().round())
        dfr['count'] = dfr['count'].astype(int)
        # group by net and sum counts
        dfr = dfr.groupby('net', as_index=False)['count'].sum()
        # cumulative sum of counts (running total)
        c = dfr['count'].cumsum()
        # total sum of counts
        t = dfr['count'].sum()
        stations_per_request = 900
        if t <= stations_per_request:
            yield fdsn_station_url, params
            return
        # max num of stations per download is 1000:
        n = int(t / stations_per_request)  # number of split requests
        # (i.e., how many I need to have ~ 1000 stations requested each time)
        # map cumulative proportion to chunk index [0, n-1]
        dfr['chunk'] = (c / t * n).astype(int).clip(upper=n - 1)
        for _, df_ in dfr.groupby('chunk'):
            _params = dict(params)
            _params['net'] = ",".join(sorted(set(df_['net'])))
            yield fdsn_station_url, _params
    except Exception as e:  # noqa
        yield fdsn_station_url, params


from typing import TypeAlias
BooleanSeries: TypeAlias = pd.Series  # just for hint clarity (see below)


def download_channels(
    fdsn_station_urls,
    filter_funcs: dict[str, Callable[[pd.Series], BooleanSeries]],
    timeout: int,
    show_progress=False
):
    """

    """
    rank_col = "_.rank._"

    # use enumerate to keep order based rank to resolve conflicts
    # (not implemented, but might be in the future). Use also list to have the len
    # (progressbar)
    urls = list(enumerate(fdsn_station_urls))

    # define url builder func:
    def url_builder(enum_item: tuple[int, str]):
        return fdsn_url(enum_item[1], new_service='station')

    ch_list = []
    # station_urls = set()
    with get_progressbar(len(urls) if show_progress else 0) as pbar:

        failed: list[tuple[int,str]] = []
        max_concurrency = 4  # per domain concurrency

        while len(urls):
            for response in build_and_read_urls(
                url_builder,
                urls,
                max_concurrency=max_concurrency,
                error_limit=None,
                same_error_limit=None,
                timeout=timeout,
                blocksize=-1
            ):
                pbar.update(1)
                if not response.is_ok:
                    if 200 <= response.status_code < 400:
                        logger.warning(str(response))
                    else:
                        failed.append(response.meta)
                        if max_concurrency < 2:
                            # no further attempt, log:
                            logger.warning(str(response))
                    continue

                try:
                    dframe = fdsn_channel_response_text_to_df(
                        BytesIO(response.data), filter_funcs
                    )
                except Exception as e:
                    logger.warning(
                        f"Unable to read data downloaded from {response.request}: {e}"
                    )
                    continue
                dframe[rank_col] = response.meta[0]  # enumerate int value

                # replace full url with the future dataselect url

                dframe[url_col] = fdsn_url_qs(  # <- basically, remove query from final URL
                    fdsn_url(
                        response.request,
                        new_service='dataselect',
                        new_method='query'
                    )
                )
                # station_urls.add(station_url)
                ch_list.append(resolve_intra_conflicts(dframe, response.request))


            urls = []  # will exit the loop. unless:
            if max_concurrency >= 2 and failed:
                # retry failed with lower concurrency:
                max_concurrency //= 2
                urls = failed

    # build two dataframes which we will concatenate afterward
    channels = pd.DataFrame()
    if ch_list:  # pd.concat complains about empty list
        # save urls and set them as categorical
        channels = pd.concat(ch_list, axis=0, ignore_index=True, copy=False)

        if not channels.empty:
            # sort by rank col and reset index so that we can see with the index which
            # urls have priority in case of conflicts:
            channels.index.name = '._index._'
            channels = (
                channels.sort_values([rank_col, channels.index.name], ascending=True).
                drop(columns=rank_col).reset_index(drop=True)
            )

    return channels


def fdsn_channel_response_text_to_df(
    response: BytesIO, filter_func: dict[str, Callable[[pd.Series], BooleanSeries]],
):
    """
    Convert a response content obtained from a FDSN station webservice with
    level=channel and  format=text into a pandas DataFrame
    with proper dtypes associated to the SQL mapped class
    """

    # Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|
    # Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|
    # StartTime|EndTime`
    columns = {
        0: net_col,
        1: sta_col,
        2: loc_col,
        3: cha_col,
        4: lat_col,
        5: lon_col,
        6: Channel.elevation.key,
        7: Channel.depth.key,
        8: Channel.azimuth.key,
        9: Channel.dip.key,
        # skip sensor_description and instrument attrs below:
        # 11: "scale",
        # 12: "scale_freq",
        # 13: "scale_units",
        14: Channel.sample_rate.key,
        15: Channel.start_time.key,
        16: end_col
    }

    dfr = fdsn_response_text_to_df(
        response, usecols=list(columns.keys()), names=list(columns.values())
    )

    if dfr.empty:
        return dfr

    # cast and work with non-sql columns:
    dfr[end_col] = pd.to_datetime(dfr[end_col], errors='coerce').fillna(
        end_time_replacement
    )
    dfr[[band_col, inst_col, orient_col]] = dfr.pop(cha_col).str.extract(r"(.)(.)(.)")

    # apply data types now (e.g., we might filter sample_rate it needs to be float)
    dfr = apply_table_dtypes(Channel, dfr, drop_non_nullable=True)
    if dfr.empty:
        raise Exception('no channel with valid data')

    # filter out
    for key, func in filter_func.items():
        dfr = dfr[func(dfr[key])]
        if dfr.empty:
            break
    if dfr.empty:
        raise Exception('all channels filtered out')

    # round unique constraint values to remove noise whilst preserving integrity:
    dfr[start_col] = dfr[start_col].dt.round('s')
    dfr[end_col] = dfr[end_col].dt.round('s')
    dfr[lat_col] = dfr[lat_col].round(6)
    dfr[lon_col] = dfr[lon_col].round(6)

    return dfr


def resolve_intra_conflicts(fdsn_df: pd.DataFrame, url:str) -> pd.DataFrame:
    """
    Resolve intra conflicts (same URL, different lat lon, overlapping time ranges)
    """
    # dframe here has single ws_id
    uc_cols = [net_col, sta_col, loc_col, band_col, inst_col, orient_col, start_col]

    # if same start time, only one lat lon, otherwise remove (conflict)
    fdsn_df = fdsn_df.groupby(uc_cols).filter(
        lambda g:
          g[lat_col].nunique(dropna=False) == 1 and
          g[lon_col].nunique(dropna=False) == 1
    )

    # if same channels share same start time, take the largest end_col:
    idx = fdsn_df.groupby(uc_cols)[end_col].idxmax()
    fdsn_df = fdsn_df.loc[idx]

    # make end_time not overlapping previous start_time in case:
    for _, dfr in fdsn_df.groupby(uc_cols[:-1]):
        new_end_times = clip_overlapping_end_times(dfr)
        if not new_end_times.empty:
            fdsn_df.loc[new_end_times.index, end_col] = new_end_times

    return fdsn_df


def clip_overlapping_end_times(dfr: pd.DataFrame) -> pd.Series:
    dfr = dfr.sort_values(by=[start_col], ascending=True)
    next_start = dfr[start_col].shift(-1)
    mask = dfr[end_col] > next_start
    mask.iloc[-1] = False  # for safety
    return next_start[mask]


def resolve_inter_conflicts(
    channels: pd.DataFrame, eida_rs_urls: list[str] | None = None
) -> pd.DataFrame:
    """
    Resolve inter conflicts (same channel, different URLs, overlapping time ranges
    """
    channels.reset_index(drop=True, inplace=True)
    grp_cols = [net_col, sta_col, loc_col, band_col, inst_col, start_col]
    # for safety:
    channels = channels.drop_duplicates(subset=grp_cols + [orient_col], keep='first')
    # move on:
    drop_indices = set()
    new_channels = []

    for (net, sta, loc, band, inst, start), chs in channels.groupby(grp_cols):

        if chs[url_col].nunique(dropna=False) == 1:
            continue

        for o in pd.unique(chs[orient_col]):
            if not clip_overlapping_end_times(chs[chs[orient_col] == o]).empty:
                break  # overlapping time ranges: break and handle it below
        else:
            # no overlapping time ranges: next loop
            continue

        # handle overlapping times. Regardless of the outcome, drop current df:
        drop_indices.update(chs.index)
        # take only dataframes resolved buy eida routing service, if any:
        for dfr in resolve_via_eida_rs(chs, eida_rs_urls, net, sta, loc, band, inst):
            new_channels.append(dfr)

    channels = pd.concat(
        [channels.drop(list(drop_indices), errors='ignore')] + new_channels,
        ignore_index=True
    )

    return channels


def resolve_via_eida_rs(
    channels: pd.DataFrame,
    eida_rs_urls: list[str],
    net: str,
    sta: str,
    loc: str,
    band: str,
    inst: str
) -> Iterable[pd.DataFrame]:

    min_start = channels[start_col].min()
    max_end = channels[end_col].max()

    eida_rs_json = get_eida_rs_response(
        eida_rs_urls,
        net=net,
        sta=sta,
        loc=loc or None,
        cha=band + inst + "?",
        start=min_start,
        end=max_end,
    )
    # eida_rs_json is an Array of Objects of type:
    # [
    #   "url": url
    #   "params": [
    #       ["net,:..., "sta":... "loc": ..., "cha" ..., "start": ..., "end: ... "priority": ...]  # noqa
    #   ]
    # ]
    for item in eida_rs_json:
        if channels.empty:
            break
        url = item['url']
        for params in item['params']:
            start = datetime.fromisoformat(params["start"])
            end = datetime.fromisoformat(params["end"])
            time_range_outside = (
                (channels[start_col] >= end) | (channels[end_col] <= start)
            )
            if params['priority'] == 1:
                # trust only start time (we got matching start time and end time
                # differing for just 12h...)
                mask = (
                    channels[url_col].str.startswith(url) &
                    (channels[start_col] == start)
                )
                if mask.any():
                    yield channels[mask]
                channels = channels[time_range_outside]
            else:
                time_range_intersects = ~time_range_outside
                # drop rows of the current url (priority != 1) that intersect the
                # interval:
                channels = channels[
                    ~(channels[url_col].str.startswith(url) & time_range_intersects)
                ]

    if channels[url_col].nunique(dropna=True) == 1:
        yield channels


def save_channels(engine: Engine, channels: pd.DataFrame):
    """
    Save to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param engine: the sql alchemy Engine
    :param channels: pandas DataFrame
    """
    # if update is True, don't update inventories HERE (handled later)
    # depth_col = Channel.depth.key
    id_col = Channel.id.key
    ws_id_col = Channel.data_webservice_id.key

    uc_cols = [
        net_col,
        sta_col,
        loc_col,
        band_col,
        inst_col,
        orient_col,
        start_col
        # ws_id_col,
        # lat_col,
        # lon_col
    ]

    # for safety, let's drop duplicates that might still be there
    channels = channels.drop_duplicates(subset=uc_cols, keep='first')

    select_stmt = get_db_select_statement(channels)

    channels[id_col] = pd.Series(pd.NA, index = channels.index, dtype = "Int64")
    channels[sta_id_col] = pd.Series(pd.NA, index=channels.index, dtype="Int64")
    _suf = '_.db._'
    for saved_channels in fetch_df(engine, select_stmt):
        channels = channels.merge(
            saved_channels, how='left', on=uc_cols, suffixes = ('', _suf)
        )
        channels[sta_id_col] = channels[sta_id_col].fillna(channels[sta_id_col + _suf])
        on_db = channels[id_col + _suf].notna()
        ws_id_mismatch =  (channels[ws_id_col] != channels[ws_id_col + _suf])
        mismatch = on_db & (
            (channels[lat_col] != channels[lat_col + _suf]) |
            (channels[lon_col] != channels[lon_col + _suf]) |
            ws_id_mismatch
        )
        if mismatch.any():
            logger.warning(
                f'Replacing the following channels with matching database records '
                f'(url, lat or lon might differ):\n'
                f'{to_urls(channels[mismatch])}'
            )
            channels.loc[mismatch, lat_col] = channels.loc[mismatch, lat_col + _suf]
            channels.loc[mismatch, lon_col] = channels.loc[mismatch, lon_col + _suf]
            channels.loc[mismatch, ws_id_col] = channels.loc[mismatch, ws_id_col + _suf]

        channels.loc[on_db & ws_id_mismatch, url_col] = None  # flag for later

        channels[id_col] = channels[id_col].fillna(channels[id_col + _suf])
        channels.drop(
            columns=[c for c in channels.columns if c.endswith(_suf)], inplace=True
        )

    # if I changed webservice id, I must reset also urls, which has categorical dtype:
    url_na = channels[url_col].isna()
    if url_na.any():
        ws_ids = channels[url_na][ws_id_col].unique().tolist()
        with engine.connect() as conn:
            result = conn.execute(select(WebService.id, WebService.url).where(
                WebService.id.in_(ws_ids)
            ))
        mapping = {row[0]: row[1] for row in result}
        new_values = set(mapping.values()) - set(channels[url_col].cat.categories)
        if new_values:
            channels[url_col] = channels[url_col].cat.add_categories(list(new_values))
        channels.loc[url_na, url_col] = channels.loc[url_na, ws_id_col].map(mapping)
        # for safety:
        channels.dropna(subset=[url_col], inplace=True)

    # sync NULL stationxml_ids:
    sta_id_na = channels[sta_id_col].isna()
    if sta_id_na.any():
        sta_id = get_col_max(engine, StationXML.id) + 1
        for (net, sta, ws_id), cha in channels[sta_id_na].groupby(
            [net_col, sta_col, ws_id_col]
        ):
            try:
                with engine.begin() as conn:   # noqa
                    conn.execute(insert(StationXML), {
                        StationXML.id.key: sta_id,
                        StationXML.data.key: None,
                        StationXML.last_updated.key: None
                    })
            except IntegrityError:
                pass
            channels.loc[cha.index, sta_id_col] = sta_id
            sta_id += 1
    # for safety:
    channels.dropna(subset=[sta_id_col], inplace=True)

    # finally, insert new elements (= missing id_col):
    channels = insert_id_col_na_values_to_db(engine, Channel, channels, id_col)

    return channels


def get_db_select_statement(channels) -> Select:
    conditions = []
    if pd.notna(channels[start_col].max()):
        conditions.append(
            Channel.start_time <= channels[start_col].max().to_pydatetime()
        )

    for name, col in {
        net_col: Channel.network_code,
        sta_col: Channel.station_code,
        loc_col: Channel.location_code,
        band_col: Channel.band_code,
        inst_col: Channel.instrument_code,
        orient_col: Channel.orientation_code,
    }.items():
        if pd.api.types.is_categorical_dtype(channels[name]):
            categories = channels[name].cat.categories
            if 0 < len(categories) <= 250:
                conditions.append(col.in_(categories.tolist()))

    select_stmt = select(
        Channel.id,
        Channel.network_code,
        Channel.station_code,
        Channel.location_code,
        Channel.band_code,
        Channel.instrument_code,
        Channel.orientation_code,
        Channel.latitude,
        Channel.longitude,
        Channel.start_time,
        Channel.data_webservice_id,
        Channel.station_id
    )
    if conditions:
        select_stmt = select_stmt.where(and_(*conditions))
    return select_stmt


def to_urls(dfr:pd.DataFrame, max_rows=5):
    ret = []
    for (url, net, sta, start), df in dfr.groupby([
        url_col, net_col, sta_col, start_col
    ]):
        loc = ",".join(set(df[loc_col]))
        chas = df[band_col] + df[inst_col] + df[orient_col]
        cha = ",".join(set(chas))
        ret.append(fdsn_url_qs(
            url, net=net, sta=sta, start=start, loc=loc, cha=cha, format='text'
        ))
        if max_rows is not None and len(ret) > max_rows:
            break

    if max_rows is not None:
        ret.append(f'(showing first {max_rows:,} of {len(dfr):,})')

    return "\n".join(ret)