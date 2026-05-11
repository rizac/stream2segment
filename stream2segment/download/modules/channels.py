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

import numpy as np
import pandas as pd
from obspy.signal.evrespwrapper import Channel
from pandas.core.dtypes.common import is_categorical_dtype

from stream2segment.download.modules.events import sync_webservice_ids_with_db
from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    insert_df, apply_table_dtypes, fetch_df, get_col_max, select, Engine, set_pkeys
)
from stream2segment.io.db.models import Channel, WebService, Segment
from stream2segment.download.url import read_url
from stream2segment.download.modules.utils import (
    fdsn_url, fdsn_url_qs, fdsn_response_text_to_df, FailedDownload, NothingToDownload
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


start_col = "start"
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
        filter_funcs[Channel.sample_rate.key] = lambda s: s > min_sample_rate

    cha_df = download_channels(
        cha_urls,
        filter_funcs=filter_funcs,
        timeout=download_timeout,
        show_progress=show_progress,
        restricted_download=restricted_download,
    )
    if cha_df.empty:
        raise NothingToDownload(
            'No channel downloaded. Possible reasons: network error, '
            'all channels filtered out'
        )

    num_downloaded_channels = len(cha_df)

    cha_df = resolve_inter_conflicts(cha_df, eida_rs_urls)
    if cha_df.empty:
        raise FailedDownload('No channel to work with after conflicts dropping')

    ws_id_col = Channel.data_webservice_id.key
    cha_df = sync_webservice_ids_with_db(cha_df, engine, merge_on=ws_id_col)
    wsid_na = pd.isna(cha_df[ws_id_col])
    if wsid_na.any():
        logger.warning(
            f"{wsid_na.sum()} channel(s) discarded (associated URL not saved to DB)\n" +
            cha_df[wsid_na].to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
        cha_df = cha_df[~wsid_na].copy()

    if cha_df.empty:
        raise FailedDownload(
            'No channel to work with after failed attempt to save channel webservices'
        )
    cha_df = save_channels(engine, cha_df)
    if cha_df.empty:
        raise FailedDownload(
            'No channel to work with after failed attempt to save channels'
        )

    logger.info(
        f'Working with {len(cha_df):,} station channels '
        f'(downloaded {num_downloaded_channels:,})'
    )

    # convert to categorical type (for safety):
    for c in (
        net_col, sta_col, loc_col, band_col, inst_col, orient_col, url_col
    ):
        if not is_categorical_dtype(cha_df[c]):
            cha_df[c] = cha_df[c].astype('str').astype('category')

    cha_df.rename(columns={Channel.id.key: Segment.channel_id.key}, inplace=True)

    # return a copy of relevant columns only:
    return cha_df[[
        Segment.channel_id.key,
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
                    "None of the EIDA routing services returned valid data. Check "
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
            # # restrict the search if too big:
            # if params['net'] == '*' or params['sta'] == '*':
            #     for start_, end_ in split_times(params['start'], params['end'], 5):
            #         _params = dict(params)
            #         _params['start'] = start_
            #         _params['end'] = end_
            #         parsed_urls.append((service_url, _params))

    for url, params in parsed_urls:
        try:
            for url, params in check_and_yield_fdsn_urls(url, params):
                yield fdsn_url_qs(url, **params, level='channel', format='text')
        except ValueError as v_err:
            logger.warning(f"Error from {url}: {v_err}")


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


def check_and_yield_fdsn_urls(url, params):
    try:
        fdsn_station_url = fdsn_url(url, new_service='station', new_method='query')
    except ValueError as e:
        raise ValueError("Invalid FDSN URL")

    if params['net'] != '*' and params['net'] is not None:
        yield fdsn_station_url, params
        return
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
        df.loc[pd.isna(df['count']), 'count'] = df['count'].mean()
        df['count'] = df['count'].astype(int)
        n = 5 # number of split requests
        # cumulative sum of counts (running total)
        c = df['count'].cumsum()
        # total sum of counts
        t = df['count'].sum()
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
    """Return a Dataframe representing a query to the station service of each
    URL in :func:`stream2segment.download.modules.datacenters_df` with the
    given arguments.

    :param datacenters_df: (DataFrame) the first item resulting from
        `get_datacenters_df`
    :param min_sample_rate: minimum sampling rate, set to negative value
        for no-filtering (all channels)
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
                    f"Unable to fetch stations from {response.request}: {response.data}"
                )
                continue

            try:
                dframe = fdsn_channel_response_text_to_df(
                    response.data.decode('utf8'), filter_funcs
                )
                dframe[rank_col] = idx
                discarded = dframe.attrs.pop('discarded', 0)
                if discarded > 0:
                    logger.warning(
                        f"{discarded} malformed row(s) discarded from{response.request}"
                    )
            except ValueError as verr:
                logger.warning(
                    f"Discarding malformed response from {response.request}"
                )
                continue

            if dframe.empty:
                continue
            # replace full url with the future dataselect url
            if restricted_download:
                datasel_url = fdsn_url(
                    response.request,
                    new_service='dataselect',
                    new_method='queryauth',
                    new_query_string=""
                )
            else:
                datasel_url = fdsn_url(
                    response.request,
                    new_service='dataselect',
                    new_method='query',
                    new_query_string=""
                )
            dframe[url_col] = datasel_url
            # station_urls.add(station_url)
            channels_dfs.extend(resolve_intra_conflicts(dframe, response.request))

    # build two dataframes which we will concatenate afterward
    cha_df = pd.DataFrame()
    if channels_dfs:  # pd.concat complains about empty list
        # save urls and set them as categorical
        cha_df = pd.concat(channels_dfs, axis=0, ignore_index=True, copy=False)
        cha_df[url_col] = cha_df[url_col].astype('category')

    if cha_df.empty:
        raise FailedDownload('No channel found, please retry later')

    # sort by rank col and reset index so that we can see with the index which urls
    # have priority in case of conflicts:
    cha_df.index.name = '._index._'
    return (
        cha_df.sort_values(by=[rank_col, cha_df.index.name], ascending=True).
        drop(columns=rank_col).reset_index(drop=True)
    )


def fdsn_channel_response_text_to_df(
    response: str, filter_func: dict[str, Callable],  # col string -> callable on series
):
    """
    Convert a response content obtained from a FDSN station webservice with
    level=channel and  format=text into a pandas DataFrame
    with proper dtypes associated to the SQL mapped class
    """
    dframe = fdsn_response_text_to_df(response)

    if not dframe.empty:
        # Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|
        # Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|
        # StartTime|EndTime`
        columns = {
            dframe.columns[0]: net_col,
            dframe.columns[1]: sta_col,
            dframe.columns[2]: loc_col,
            dframe.columns[3]: "channel_code",
            dframe.columns[4]: lat_col,
            dframe.columns[5]: lon_col,
            dframe.columns[6]: Channel.elevation.key,
            dframe.columns[7]: Channel.depth.key,
            dframe.columns[8]: Channel.azimuth.key,
            dframe.columns[9]: Channel.dip.key,
            # skip sensor_description and instrument attrs below:
            # dframe.columns[11]: "scale",
            # dframe.columns[12]: "scale_freq",
            # dframe.columns[13]: "scale_units",
            dframe.columns[14]: Channel.sample_rate.key,
            dframe.columns[15]: start_col,
            dframe.columns[16]: end_col
        }

        # rename round and set order:
        dframe = dframe.rename(columns=columns)[list(columns.values())]
        dframe[start_col] = to_datetime(dframe[start_col])
        dframe[end_col] = to_datetime(dframe[end_col], end_time_replacement)
        dframe[lat_col] = to_latlon(dframe[lat_col])
        dframe[lon_col] = to_latlon(dframe[lon_col])
        # fir safety:
        dframe[Channel.sample_rate.key] = dframe[Channel.sample_rate.key].astype(float)

        for key, func in filter_func.items():
            dframe = dframe[func(dframe[key])]
            if dframe.empty:
                # log filtered out
                continue

        # dframe = dframe[dframe[[start_col, end_col]].notna().all(axis=1)]
        dframe[
            [band_col, inst_col, orient_col]
        ] = dframe.pop("channel_code").str.extract(r"(.)(.)(.)")
        dframe = apply_table_dtypes(Channel, dframe, drop_non_nullable=True)

    if dframe.empty:
        raise ValueError("Malformed data (e.g., no data, type mismatch, NaN)")

    return dframe


def to_datetime(series: pd.Series, fillna=None):
    ret = pd.to_datetime(series, errors='coerce')
    if fillna is not None:
        ret.fillna(fillna, inplace=True)
    return ret.dt.round('s')


def to_latlon(series: pd.Series):
    return pd.to_numeric(series, errors='coerce').round(6)


def resolve_intra_conflicts(fdsn_df: pd.DataFrame, url:str):
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

    # if same channels share same start time, take the biggest time window:
    fdsn_df = fdsn_df.groupby(uc_cols, group_keys=False).apply(
        lambda g: g if len(g) == 1 else g.loc[g[end_col] == g[end_col].max()].head(1)
    )

    # make end_time not overlapping previous start_time in case:
    for _, dfr in fdsn_df.groupby(uc_cols[:-1]):
        new_end_times = clip_overlapping_end_times(dfr)
        if not new_end_times.empty:
            fdsn_df.loc[new_end_times.index, end_col] = new_end_times

    return fdsn_url


def clip_overlapping_end_times(dfr: pd.DataFrame) -> pd.Series:
    dfr = dfr.sort_values(by=[start_col], ascending=True)
    next_start = dfr[start_col].shift(-1)
    mask = dfr[end_col] > next_start
    mask.iloc[-1] = False  # for safety
    return next_start[mask]


def resolve_inter_conflicts(
    channels: pd.DataFrame, eida_rs_urls: list[str] | None = None
):
    """
    Resolve inter conflicts (same channel, different URLs, overlapping time ranges
    """
    channels.reset_index(drop=True, inplace=True)

    # 1) CHANNELS WITH SAME CODE AND START TIME MUST COME FROM A SINGLE URL:
    grp_cols = [net_col, sta_col, loc_col, band_col, inst_col, start_col]

    # table_empty = get_row_count(engine, Channel) < 1
    indices2discard = set()

    new_cha_df = []

    for (net, sta, loc, band, inst, start), cha_df in channels.groupby(grp_cols):

        for o in pd.unique(cha_df[orient_col]):
            if not clip_overlapping_end_times(cha_df[cha_df[orient_col] == o]).empty:
                break
        else:
            continue

        indices2discard.update(cha_df.index)
        if cha_df[url_col].nunique(dropna=False) <= 1:
            # log FIXME
            continue

        for dfr in resolve_via_eida_rs(cha_df, eida_rs_urls, net, sta, loc, band, inst):
            new_cha_df.append(dfr)

    channels = pd.concat(
        [channels[~channels.index.isin(indices2discard)]] + new_cha_df,
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

    min_start=channels[start_col].min()
    max_end=channels[end_col].max()

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
                time_range_inside = (
                    (channels[start_col] >= start) & (channels[end_col] <= end)
                )
                tmp = (
                    channels[channels[url_col].str.startswith(url) & time_range_inside]
                )
                if not tmp.empty:
                    yield tmp
                channels = channels[time_range_outside]
            else:
                time_range_intersects = ~time_range_outside
                channels = channels[
                    ~(channels[url_col].str.startswith(url) & time_range_intersects)
                ]

    if channels[url_col].nunique(dropna=False) == 1:
        yield channels


def save_channels(engine: Engine, channels: pd.DataFrame):
    """Saves to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param channels: pandas DataFrame
    """
    # if update is True, don't update inventories HERE (handled later)
    # depth_col = Channel.depth.key
    id_col = Channel.id.key
    ws_id_col = Channel.data_webservice_id.key
    depth_col = Channel.depth.key
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
    ).where(
        (Channel.latitude >= channels[lat_col].min()) &
        (Channel.latitude <= channels[lat_col].max()) &
        (Channel.longitude <= channels[lon_col].min()) &
        (Channel.longitude >= channels[lon_col].mmax()) &
        (Channel.data_webservice_id.isin_(pd.unique(channels[ws_id_col]))) &
        (Channel.band_code.isin_(channels[band_col].cat.categories)) &
        (Channel.instrument_code_col.isin_(channels[inst_col].cat.categories)) &
        (Channel.orientation_code.isin_(channels[orient_col].cat.categories)) &
        (Channel.start_time <= get_col_max(engine, Channel.start_time))
    )

    channels[id_col] = pd.Series(pd.NA, index = channels.index, dtype = "Int64")
    _suf = '_.db._'
    for saved_channels in fetch_df(engine, select_stmt):
        channels = channels.merge(
            saved_channels, how='left', on=uc_cols, suffixes = ('', _suf)
        )
        on_db = channels[id_col + _suf].notna()
        mismatches = on_db & (
            (channels[lat_col] != channels[lat_col + _suf]) |
            (channels[lon_col] != channels[lon_col + _suf]) |
            (channels[ws_id_col] != channels[ws_id_col + _suf]) # |
            # (channels[depth_col] != channels[depth_col + _suf])
        )
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

    to_insert = set_pkeys(channels[channels[id_col].isna()], engine, Channel)
    inserted, failed = insert_df(to_insert, engine, Channel)
    if not failed.empty:
        logger.warning(
            f"{len(failed)} channel(s) discarded (error while inserting to DB)\n" +
            failed.to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
    if not inserted.empty:
        channels[id_col] = inserted[id_col]  # assignment is index aligned
    # for safety:
    channels[id_col] = channels[id_col].astype(int)

    return channels


def to_urls(dfr:pd.DataFrame, max_rows=5):
    ret = []
    _i = 0
    if max_rows is None:
        max_rows = np.inf
    group_by = dfr.groupby([
        net_col, sta_col, loc_col, band_col, inst_col, start_col, end_col, url_col
    ], sort=False)
    total = group_by.ngroups
    for (n, s, l, b, i, st, et, u), _ in group_by:
        ret.append(fdsn_url_qs(u, net=n, sta=s, loc=l, cha=b+i+"?", start=st, end=et))
        _i += 1
        if _i >= max_rows:
            ret.append(f'(showing first {max_rows:,} of {total:,})')
            break
    return "\n".join(ret)