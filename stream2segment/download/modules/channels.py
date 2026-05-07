"""
Stations/Channels download functions
"""
# :date: Dec 3, 2017
import re
import logging
from collections.abc import Iterable
import json
from datetime import datetime, timedelta, UTC
from io import BytesIO
from multiprocessing.pool import ThreadPool
from urllib.parse import urlunparse, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
from obspy.signal.evrespwrapper import Channel
from pandas.core.dtypes.common import is_categorical_dtype
from sqlalchemy import select, Engine

from stream2segment.download.modules.events import sync_webservice_ids_with_db
from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import insert_df, apply_table_dtypes, select_df
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

    cha_df = download_channels(
        cha_urls,
        timeout=download_timeout,
        show_progress=show_progress,
        restricted_download=restricted_download,
    )
    if cha_df.empty:
        raise NothingToDownload('No channel downloaded')
    num_downloaded_channels = len(cha_df)

    # post filter (negation "!", sample rate) which raises FailedDownload if no rows:
    cha_df = filter_out_channels_df(
        cha_df,
        [n for n in network if n.startswith('!')],
        [s for s in station if s.startswith('!')],
        [l for l in location if l.startswith('!')],
        [c for c in channel if c.startswith('!')],
        # network, station, location, channel, starttime, endtime,
        min_sample_rate
    )
    if cha_df.empty:
        raise NothingToDownload('All channels filtering out according to settings')

    # set ranking based on the order of urls
    cha_df = drop_conflicts(cha_df, eida_rs_urls)
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
        Channel.depth.key,
        start_col,
        end_col,
        # ws_id_col,
        url_col,
        Channel.data_webservice_id.key
    ]]


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
    fdsn_station_urls, restricted_download: bool, timeout: int, show_progress=False
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
                dframe = fdsn_channel_response_text_to_df(response.data.decode('utf8'))
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
            base_fdsn_url = str(urlunparse(
                urlparse(response.request)._replace(query='', fragment='')
            ))
            if restricted_download:
                datasel_url = fdsn_url(
                    base_fdsn_url, new_service='dataselect', new_method='queryauth'
                )
            else:
                datasel_url = fdsn_url(
                    base_fdsn_url, new_service='dataselect', new_method='query'
                )
            dframe[url_col] = datasel_url
            # station_urls.add(station_url)
            channels_dfs.append(dframe)

    # build two dataframes which we will concatenate afterward
    cha_df = pd.DataFrame()
    if channels_dfs:  # pd.concat complains about empty list
        # save urls and set them as categorical
        cha_df = pd.concat(channels_dfs, axis=0, ignore_index=True, copy=False)
        cha_df[url_col] = cha_df[url_col].astype('category')

    # sort by rank col and reset index so that we can see with the index which urls
    # have priority in case of conflicts:
    cha_df.index.name = '._index._'
    return (
        cha_df.sort_values(by=[rank_col, cha_df.index.name], ascending=True).
        drop(columns=rank_col).reset_index(drop=True)
    )


def fdsn_channel_response_text_to_df(response: str):
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

        # rename and set order:
        dframe = dframe.rename(columns=columns)[list(columns.values())]
        dframe[start_col] = pd.to_datetime(dframe[start_col], errors='coerce')
        dframe[end_col] = pd.to_datetime(dframe[end_col], errors='coerce')
        dframe = dframe[dframe[[start_col, end_col]].notna().all(axis=1)]
        dframe[
            [band_col, inst_col, orient_col]
        ] = dframe.pop("channel_code").str.extract(r"(.)(.)(.)")
        dframe = apply_table_dtypes(Channel, dframe, drop_non_nullable=True)

    if dframe.empty:
        raise ValueError("Malformed data (e.g., no data, type mismatch, NaN)")
    return dframe


def filter_out_channels_df(
    channels: pd.DataFrame,
    net: list[str],
    sta: list[str],
    loc: list[str],
    cha: list[str],
    min_sample_rate
):
    """Filter out `channels_df` according to the given parameters. Raise
    `FailedDownload` if the returned filtered data frame woul be empty

    Note that `net, sta, loc, cha` filters will be considered only if negations
    (i.e., with leading exclamation mark: "!A*") because the 'positive' filters
    are FDSN stantard and are supposed to be already used in producing
    `channels_df`. Example:
    ```
        filter_channels_df(d, [], ['ABC'], [''], ['!A*', 'HH?', 'HN?'])
    ```
    basically takes the dataframe `d`, finds the column related to the
    `channels` key and removes all rows whose channel starts with 'A',
    returning the new filtered data frame.

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    :param min_sample_rate: numeric, minimum sample rate. If negative or zero,
        this parameter is ignored
    """
    cols = (net_col, sta_col, loc_col)
    expressions = []

    for lst, col in zip((net, sta, loc), cols):
        if not lst:
            continue
        condition = "|".join(f"^(?:{wild2regex(x[1:])})$" for x in lst)
        flt = channels[col].str.match(re.compile(condition))
        expressions.append(flt)

    for c in cha or []:
        expressions.append(
            channels[band_col].str.macth(wild2regex(c[1]))  and
            channels[inst_col].str.macth(wild2regex(c[2])) and
            channels[orient_col].str.macth(wild2regex(c[3]))
        )

    if min_sample_rate is not None and min_sample_rate > 0:
        # None should evaluate to False, thus negate the predicate below:
        flt = channels[Channel.sample_rate.key] < min_sample_rate
        expressions.append(flt)

    ret = channels
    if expressions:
        expr = expressions[0]
        for e in expressions[1:]:
            expr |= e
        ret = channels[~expr].copy()

    discarded_sr = len(channels) - len(ret)
    if discarded_sr:
        logger.warning(f"{discarded_sr:,} channel(s) discarded according to "
                       f"current configuration filters (network, channel, sample rate, "
                       "...)")

    return ret


def drop_conflicts(
    channels: pd.DataFrame, eida_rs_urls: list[str] | None = None
):
    """
    Drop from channels_df conflict between, i.e., network.station codes
    returned by several URLs. Duplicated rows will be resolved against the
    database or, if keep_first is True, by taking the first row

    :param channels: pandas DataFrame
    :param eida_rs_urls: an optional list of source station FDSN urls that
        where used to build the passed dataframe. Order matters as conflicts
        will be resolved by taking
        the first matching url. If None, conflicts will cause all channels
        involved to be dropped
    :return: a new dataframe with duplicated rows removed
    """
    channels.reset_index(drop=True, inplace=True)
    channels.loc[channels['end'].isna(), 'end'] = (
        datetime.now(UTC).replace(microsecond=0, tzinfo=None) + timedelta(days=30)  # some random margin in the future
    )

    channels.dropna(subset=[lat_col, lon_col, url_col], inplace=True)


    # 1) CHANNELS WITH SAME CODE AND START TIME MUST COME FROM A SINGLE URL:
    grp_cols = [net_col, sta_col, loc_col, band_col, inst_col]

    # will this speed up grouping?
    for c in grp_cols + [url_col, orient_col]:
        if not pd.api.types.is_categorical_dtype(channels[c]):
            channels[c] = channels[c].astype('category')

    # table_empty = get_row_count(engine, Channel) < 1
    indices2keep = set()

    new_cha_df = []

    for (net, sta, loc, band, inst), cha_df in (
        channels.groupby(grp_cols)
    ):

        if (
            cha_df[lat_col].nunique(dropna=False) ==
            cha_df[lon_col].nunique(dropna=False) ==
            cha_df[url_col].nunique(dropna=False) ==
            cha_df[start_col].nunique(dropna=False) ==
            cha_df[end_col].nunique(dropna=False) == 1
        ):
            indices2keep.update(cha_df.drop_duplicates(orient_col, keep='first').index)
            continue

        lat_lons: dict[str, list[tuple[datetime, datetime, float, float]]] = {}

        times = pd.concat(
            [cha_df[start_col], cha_df[end_col]]
        ).drop_duplicates().sort_values().to_numpy()  # noqa

        for start, end in zip(times[:-1], times[1:]):

            tmp_df = cha_df[~((cha_df[start_col] >= end) | (cha_df[end_col] <= start))]

            if tmp_df.empty:
                continue  # for safety

            if tmp_df[url_col].nunique() != 1:  # surely at least 2 (dfr non empty)
                tmp_df2 = resolve_via_eida_rs(
                    tmp_df, eida_rs_urls, net, sta, loc, band, inst, start, end
                )
                if tmp_df2[url_col].nunique() > 1:
                    # take first url (it is sorted by rank, i.e. index):
                    for url in cha_df[url_col].values:
                        tmp_df2 = tmp_df2[tmp_df2[url_col] == url]
                        if tmp_df2.empty:
                            continue
                        break
                if tmp_df2[url_col].nunique() != 1:
                    logger.warning(
                        f'Discarding channels (unable to resolve non-unique URLs):\n'
                        f'{to_urls(tmp_df2)}'
                    )
                    continue
                tmp_df = tmp_df2

            lat_lon_unique = (
                tmp_df[lat_col].nunique() == 1 and tmp_df[lon_col].nunique() == 1
            )
            time_bounds_coincide = (
                (tmp_df[start_col] == start).all() and (tmp_df[end_col] == end).all()
            )
            if lat_lon_unique and time_bounds_coincide:
                indices2keep.update(
                    tmp_df.drop_duplicates(orient_col, keep='first').index
                )
                continue

            tmp_df = tmp_df.copy()
            tmp_df[start_col] = start
            tmp_df[end_col] = end
            url = tmp_df[url_col].values[0]
            if not lat_lon_unique:
                if url not in lat_lons:
                    lat_lons[url] = resolve_station_lat_lon(
                        url,
                        net,
                        sta,
                        loc,
                        band + inst + '?',
                        times[0],
                        times[-1]
                    )
                lat, lon = None, None
                for start_, end_, lat_, lon_ in lat_lons[url]:
                    if start_ <= start and end_ >= end:
                        lat = lat_
                        lon = lon_
                        break
                if lat is None or lon is None:
                    logger.warning(
                        f'Discarding channels (unable to resolve non-unique lat/lon):\n'
                        f'{to_urls(tmp_df)}'
                    )
                    continue
                tmp_df[lat_col] = lat
                tmp_df[lon_col] = lon
            new_cha_df.append(tmp_df.drop_duplicates(subset=[orient_col], keep='first'))

    channels = pd.concat(
        [channels.loc[list(indices2keep)]] + new_cha_df, ignore_index=True
    )

    for c in grp_cols + [url_col, orient_col]:
        if not pd.api.types.is_categorical_dtype(channels[c]):
            channels[c] = channels[c].astype('category')
    return channels


def resolve_station_lat_lon(
    url, net, sta, loc, cha, start, end
) -> list[tuple[datetime, datetime, float, float]]:

    resp = read_url(
        fdsn_url_qs(
            fdsn_url(url, new_service='station'),
            net=net,
            sta=sta,
            loc=loc or None,
            cha=cha,
            start=start,
            end=end,
            level='station',
            format='text'
        )
    )
    if not resp.is_ok:
        return []
    data = pd.read_csv(BytesIO(resp.data), sep='|', header=None, comment='#')

    data = pd.DataFrame({
        'lat': pd.to_numeric(data[data.columns[2]], errors='coerce'),
        'lon': pd.to_numeric(data[data.columns[2]], errors='coerce'),
        'start': pd.to_datetime(data[data.columns[6]], errors='coerce'),
        'end': pd.to_datetime(data[data.columns[7]], errors='coerce'),
    })
    data.loc[data['end'].isna(), 'end'] = datetime.now(UTC).replace(
        microsecond=0, tzinfo=None
    )
    data_starts = np.append(data['start'], data['end'].max())
    ret = []
    for start, end in zip(data_starts[:-1], data_starts[1:]):
        _ = data[data['start'] == start]
        ret.append((start, end, _['lat'].values[0], _['lon'].values[0]))

    return ret


def resolve_via_eida_rs(
    channels: pd.DataFrame,
    eida_rs_urls: list[str],
    net: str,
    sta: str,
    loc: str,
    band: str,
    inst: str,
    start: datetime,
    end: datetime
) -> pd.DataFrame:

    eida_rs_json = get_eida_rs_response(
        eida_rs_urls,
        net=net,
        sta=sta,
        loc=loc or None,
        cha=band + inst + "?",
        start=start,
        end=end
    )
    # eida_rs_json is an Array of Objects of type:
    # [
    #   "url": url
    #   "params": [
    #       ["net,:..., "sta":... "loc": ..., "cha" ..., "start": ..., "end: ... "priority": ...] #noqa
    #   ]
    # ]
    urls_no_priority = set()
    for item in eida_rs_json:
        url = item['url']
        for params in item['params']:
            if params['priority'] == 1:
                return channels[channels[url_col].str.startswith(url)]
            else:
                urls_no_priority.add(url)

    if urls_no_priority:
        return channels[
            ~channels[url_col].str.startswith(tuple(urls_no_priority), na=False)
        ]

    return channels.head(0)


def wild2regex(text: str):
    return (
        text
        .replace('.', r'\.')
        .replace('?', '.')
        .replace('*', '.*')
    )


def save_channels(engine: Engine, channels: pd.DataFrame):
    """Saves to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param channels: pandas DataFrame
    """
    # if update is True, don't update inventories HERE (handled later)
    depth_col = Channel.depth.key
    id_col = Channel.id.key
    ws_id_col = Channel.data_webservice_id.key
    uc_cols = [net_col, sta_col, loc_col, band_col, inst_col, orient_col, ws_id_col]

    channels[id_col] = pd.Series(pd.NA, index = channels.index, dtype = "Int64")
    _suf = '_db_'

    for df in select_df(engine, select(
        Channel.id,
        Channel.network_code,
        Channel.station_code,
        Channel.location_code,
        Channel.band_code,
        Channel.instrument_code,
        Channel.orientation_code,
        Channel.latitude,
        Channel.longitude,
        Channel.data_webservice_id,
        Channel.depth
    )):
        channels = channels.merge(
            df, how='left', on=uc_cols, suffixes = ('', _suf)
        )
        on_db = channels[id_col + _suf].notna()
        mismatches = (
            on_db & (
            (channels[lat_col] != channels[lat_col + _suf]) |
            (channels[lon_col] != channels[lon_col + _suf]) |
            (channels[depth_col] != channels[depth_col + _suf])
            )
        )
        if mismatches.any():
            # write to dataframe and log FIXME log!
            channels.loc[mismatches, lat_col] = channels.loc[mismatches, lat_col + _suf]
            channels.loc[mismatches, lon_col] = channels.loc[mismatches, lon_col + _suf]
            channels.loc[mismatches, depth_col] = (
                channels.loc[mismatches, depth_col + _suf]
            )
        channels[id_col] = channels[id_col].fillna(
            channels[id_col + _suf]
        )
        channels.drop(
            columns=[c for c in channels.columns if c.endswith(_suf)], inplace=True
        )

    # for safety:
    if not pd.api.types.is_integer_dtype(channels[id_col]):
        channels[id_col] = channels[id_col].astype("Int64")

    is_na = channels[id_col].isna()
    if is_na.any():
        already_saved = channels[~is_na]
        channels = channels[is_na].drop(columns=id_col)
        ws_id_col = Channel.data_webservice_id.key
        uc_cols = [
            net_col, sta_col, loc_col, band_col, inst_col, orient_col, ws_id_col
        ]
        inserted, failed = insert_df(
            channels.drop_duplicates(subset=uc_cols, keep='first'), engine, Channel
        )
        if not failed.empty:
            logger.warning(
                f"{len(failed)} channel(s) discarded (error while inserting to DB)\n" +
                failed.to_string(
                    max_rows=30, index=False, na_rep='', show_dimensions=True
                )
            )
        # put id_col into channels:
        channels = channels.merge(inserted[uc_cols + [id_col]], how='left', on=uc_cols)
        # for safety:
        channels.dropna(subset=[id_col], inplace=True)
        channels[id_col] = channels[id_col].astype(int)
        if not already_saved.empty:
            channels = pd.concat([channels, already_saved], ignore_index=True)
            # for safety:
            channels[id_col] = channels[id_col].astype(int)

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