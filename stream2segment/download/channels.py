"""
Stations/Channels download functions
"""
# :date: Dec 3, 2017
import re
import logging
from collections.abc import Iterable, Callable
import json
from datetime import datetime, timedelta, UTC
from multiprocessing.pool import ThreadPool
from urllib.request import urlopen

import pandas as pd
from sqlalchemy import and_, Select

from stream2segment.download.events import sync_webservice_urls_and_assign_ids
from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    insert_df, apply_table_dtypes, fetch_df, select, Engine, set_pkeys
)
from stream2segment.io.db.models import Channel, WebService
from stream2segment.download.url import read_url
from stream2segment.download.utils import (
    fdsn_url, fdsn_url_qs, fdsn_response_text_to_df, FailedDownload, NothingToDownload
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


start_col = Channel.start_time.key
end_col = "end"
net_col = Channel.network_code.key
sta_col = Channel.station_code.key
loc_col = Channel.location_code.key
band_col = Channel.band_code.key
inst_col = Channel.instrument_code.key
orient_col = Channel.orientation_code.key
url_col = WebService.url.key
lat_col = Channel.latitude.key
lon_col = Channel.longitude.key
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
    restricted_download: bool,
    download_timeout: int | None = None,
    show_progress=False
):
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
        net_col: network, sta_col: station, loc_col: location, "channel_code": channel
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
        show_progress=show_progress,
        restricted_download=restricted_download,
    )
    if channels.empty:
        raise NothingToDownload(
            'No channels downloaded. Possible reasons: '
            'no channels found according to your config., web service down, '
            'no internet connection. See log for details'
        )
    logger.info(f"{len(channels)} channel(s) downloaded; "
                f"checking duplicates, conflicts, and saving")

    categorical_columns = [
        net_col, sta_col, loc_col, band_col, inst_col, orient_col, url_col
    ]
    for c in categorical_columns:
        channels[c] = channels[c].astype('str').astype('category')

    num_downloaded_channels = len(channels)

    channels = resolve_inter_conflicts(channels, eida_rs_urls)
    if channels.empty:
        raise FailedDownload('No channels left after dropping conflicts')

    ws_id_col = Channel.data_webservice_id.key
    rows = len(channels)
    channels = sync_webservice_urls_and_assign_ids(
        channels, engine, ids_column_name=ws_id_col
    )
    channels.dropna(subset=[ws_id_col], inplace=True)
    if channels.empty:
        raise FailedDownload("No channels left after failed DB URLs insertion")
    elif rows > len(channels):
        logger.warning(
            f"Discarding {rows-len(channels)} channel(s) "
            f"(associated URL not saved to DB)"
        )
    channels[ws_id_col] = channels[ws_id_col].astype(int)

    channels = save_channels(engine, channels)
    if channels.empty:
        raise FailedDownload(
            'No channels left after failed DB insertion'
        )

    for c in categorical_columns:  # for safety
        if not pd.api.types.is_categorical_dtype(channels[c]):
            channels[c] = channels[c].astype('str').astype('category')
    # channels[ws_id_col] = channels[ws_id_col].astype(int).astype('category')

    # return a copy of relevant columns only:
    return channels[[
        Channel.id.key,
        net_col,
        sta_col,
        loc_col,
        band_col,
        inst_col,
        orient_col,
        lat_col,
        lon_col,
        # Channel.depth.key,
        start_col,
        end_col,
        # ws_id_col,
        url_col,
        # Channel.data_webservice_id.key
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
            eida_rs_url, net=net, sta=sta, loc=loc,
            cha=cha, start=start, end=end,
            service=service, format='json'
        )
        response = read_url(url, decode='utf8')
        if response.is_ok:
            return json.loads(response.data)
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
            header = r.readline().decode().strip().split('|')
            for line in r:
                split_line = line.decode().strip().split('|')
                rows.append((split_line[0].strip(), split_line[-1].strip()))
        df = pd.DataFrame(rows, columns=['net', 'count'])
        # convert count to numeric (should be int, but we set NaN as #station mean):
        df['count'] = pd.to_numeric(df['count'], errors='coerce')
        df.loc[pd.isna(df['count']), 'count'] = int(df['count'].median().round())
        df['count'] = df['count'].astype(int)
        # group by net and sum counts
        df = df.groupby('net', as_index=False)['count'].sum()
        # cumulative sum of counts (running total)
        c = df['count'].cumsum()
        # total sum of counts
        t = df['count'].sum()
        stations_per_request = 1000
        if t <= stations_per_request:
            yield fdsn_station_url, params
            return
        # max num of stations per download is 1000:
        n = int(t / stations_per_request)  # number of split requests
        # (i.e., how many I need to have ~ 1000 stations requested each time)
        # map cumulative proportion to chunk index [0, n-1]
        df['chunk'] = (c / t * n).astype(int).clip(upper=n - 1)
        for _, df_ in df.groupby('chunk'):
            _params = dict(params)
            _params['net'] = ",".join(sorted(set(df_['net'])))
            yield fdsn_station_url, _params
    except Exception as e:
        yield fdsn_station_url, params


def download_channels(
    fdsn_station_urls,
    filter_funcs: dict[str, Callable],  # column string -> callable on series
    restricted_download: bool,
    timeout: int,
    show_progress=False
):
    """

    """
    rank_col = "_.rank._"

    t_pool = ThreadPool(4)
    def _urlread(_):
        return _[0], read_url(
            fdsn_url(_[1], new_service='station'), timeout=timeout, blocksize=-1
        )

    urls = list(fdsn_station_urls)

    channels_dfs = []
    # station_urls = set()
    with get_progressbar(len(urls) if show_progress else 0) as pbar:
        for idx, response in t_pool.imap_unordered(_urlread, enumerate(urls)):
            pbar.update(1)
            if not response.is_ok:
                logger.warning(
                    f"Unable to download data from {response.request}: "
                    f"{response.data}"
                )
                continue

            try:
                if response.status_code == 204:
                    raise Exception('No data (HTTP code 204)')
                dframe = fdsn_channel_response_text_to_df(
                    response.data, filter_funcs
                )
                if dframe.empty:
                    raise Exception('No rows left after type conversion and filtering')
            except Exception as e:
                logger.warning(
                    f"Unable to read data downloaded from {response.request}: {e}"
                )
                continue
            dframe[rank_col] = idx

            # replace full url with the future dataselect url
            if restricted_download:
                datasel_url = fdsn_url_qs(  # <- basically, remove query from final URL
                    fdsn_url(
                        response.request,
                        new_service='dataselect',
                        new_method='queryauth'
                    )
                )
            else:
                datasel_url = fdsn_url_qs(  # <- basically, remove query from final URL
                    fdsn_url(
                        response.request,
                        new_service='dataselect',
                        new_method='query'
                    )
                )
            dframe[url_col] = datasel_url
            # station_urls.add(station_url)
            channels_dfs.append(resolve_intra_conflicts(dframe, response.request))

    # build two dataframes which we will concatenate afterward
    cha_df = pd.DataFrame()
    if channels_dfs:  # pd.concat complains about empty list
        # save urls and set them as categorical
        cha_df = pd.concat(channels_dfs, axis=0, ignore_index=True, copy=False)

        if not cha_df.empty:
            # sort by rank col and reset index so that we can see with the index which
            # urls have priority in case of conflicts:
            cha_df.index.name = '._index._'
            cha_df = (
                cha_df.sort_values(by=[rank_col, cha_df.index.name], ascending=True).
                drop(columns=rank_col).reset_index(drop=True)
            )

    return cha_df


def fdsn_channel_response_text_to_df(
    response: str, filter_func: dict[str, Callable],  # col string -> callable on series
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
        3: "channel_code",
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

    dframe = fdsn_response_text_to_df(
        response, usecols=list(columns.keys()), names=list(columns.values())
    )

    if dframe.empty:
        return dframe

    # cast and work with non-sql columns:
    dframe[end_col] = pd.to_datetime(dframe[end_col], errors='coerce').fillna(
        end_time_replacement
    )
    dframe[
        [band_col, inst_col, orient_col]
    ] = dframe.pop("channel_code").str.extract(r"(.)(.)(.)")

    # apply data types now (e.g., we might filter sample_rate it needs to be float)
    dframe = apply_table_dtypes(Channel, dframe, drop_non_nullable=True)
    if dframe.empty:
        return dframe

    # filter out
    for key, func in filter_func.items():
        dframe = dframe[func(dframe[key])]
        if dframe.empty:
            return dframe

    dframe[start_col] = dframe[start_col].dt.round('s')
    dframe[end_col] = dframe[end_col].dt.round('s')
    dframe[lat_col] = dframe[lat_col].round(6)
    dframe[lon_col] = dframe[lon_col].round(6)

    return dframe


# def to_datetime(series: pd.Series, fillna=None):
#     if fillna is not None:
#         series.fillna(fillna, inplace=True)
#     return series.dt.round('s')
#
#
# def to_latlon(series: pd.Series):
#     return series.round(6)


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
    #       ["net,:..., "sta":... "loc": ..., "cha" ..., "start": ..., "end: ... "priority": ...] #noqa
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
    _suf = '_.db._'
    for saved_channels in fetch_df(engine, select_stmt):
        channels = channels.merge(
            saved_channels, how='left', on=uc_cols, suffixes = ('', _suf)
        )
        on_db = channels[id_col + _suf].notna()
        ws_id_mismatch =  (channels[ws_id_col] != channels[ws_id_col + _suf])
        mismatches = on_db & (
            (channels[lat_col] != channels[lat_col + _suf]) |
            (channels[lon_col] != channels[lon_col + _suf]) |
            ws_id_mismatch
        )
        channels.loc[on_db & ws_id_mismatch, url_col] = None  # flag for later

        if mismatches.any():
            # write to dataframe and log FIXME log!
            channels.loc[mismatches, lat_col] = channels.loc[mismatches, lat_col + _suf]
            channels.loc[mismatches, lon_col] = channels.loc[mismatches, lon_col + _suf]
            channels.loc[mismatches, ws_id_col] = (
                channels.loc[mismatches, ws_id_col + _suf]
            )
        channels[id_col] = channels[id_col].fillna(
            channels[id_col + _suf]
        )
        channels.drop(
            columns=[c for c in channels.columns if c.endswith(_suf)], inplace=True
        )

    # if I changed webservice id, I must reset also urls, which has categorical dtype:
    url_na = channels[url_col].isna()
    if url_na.any():
        ws_ids = channels[url_na][ws_id_col].unique()
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

    to_insert = set_pkeys(channels[channels[id_col].isna()], engine, Channel)
    inserted = insert_df(to_insert, engine, Channel)
    if not inserted.empty:
        channels[id_col] = inserted[id_col]  # assignment is index aligned

    id_na = channels[id_col].isna()
    if id_na.any():
        logger.warning(
            f"{id_na.sum():,} "
            f"channel(s) discarded (likely error while inserting to DB)\n" +
            channels[id_na].to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
        channels.dropna(subset=[id_col], inplace=True)

    # for safety:
    if not channels.empty:
        channels[id_col] = channels[id_col].astype(int)

    return channels


def get_db_select_statement(channels) -> Select:
    conditions = []
    if pd.notna(channels[start_col].max()):
        conditions.append(Channel.start_time <= channels[start_col].max())
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
        # Channel.depth,
        Channel.data_webservice_id
    )
    if conditions:
        select_stmt = select_stmt.where(and_(*conditions))
    return select_stmt
