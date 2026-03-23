"""
Data center(s) download functions

:date: Dec 3, 2017
"""
from collections.abc import Iterable
from datetime import datetime
import logging
# from itertools import chain
from typing import Optional
from urllib.request import urlopen

import pandas as pd

from stream2segment.io.db.models import WebService  # , Channel
from stream2segment.download.modules.utils import (formatmsg, fdsn_url_qs, fdsn_url)
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import urlread

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def get_stations_urls(
    session, webservice_url, routing_service_url,
    network: Optional[list[str]] = None,
    station: Optional[list[str]] = None,
    location: Optional[list[str]] = None,
    channel: Optional[list[str]] = None,
    starttime: Optional[datetime] = None,
    endtime: Optional[datetime] = None,
    db_bufsize=None
) -> Iterable[str]:
    """
    Return an iterator of FDSN station urls from the given arguments
    """
    # eida response text will be needed anyway to create an EidaValidator
    eidars_response_text = None  # lazy loaded
    if isinstance(webservice_url, str):
        webservice_url = [webservice_url]
    params = {
        'net': ','.join(n for n in network or [] if not n.startswith('!')) or '*',
        'sta': ','.join(s for s in station or [] if not s.startswith('!')) or '*',
        'loc': ','.join(l for l in location or [] if not l.startswith('!')) or '*',
        'cha': ','.join(c for c in channel or [] if not c.startswith('!')) or '*',
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
            if eidars_response_text is None:
                eidars_response_text = get_eidars_response_text(
                    routing_service_url, **params
                )
            for url, _params in (
                scan_eida_rs_station_response(eidars_response_text, True)
            ):
                # overwrite start end:
                _params['start'] = starttime
                _params['end'] = endtime
                parsed_urls.append((url, _params))

        else:
            if service_url == 'iris':
                service_url = 'https://service.iris.edu/fdsnws/station/1/query'

            # restrict the search if too big:
            if params['net'] == '*' or params['sta'] == '*':
                for start_, end_ in split_times(params['start'], params['end'], 5):
                    _params = dict(params)
                    _params['start'] = start_
                    _params['end'] = end_
                    parsed_urls.append((service_url, _params))


    for url, params in parsed_urls:
        try:
            for url, params in check_and_yield(url, params):
                yield fdsn_url_qs(url, **params, level='channel', format='text')
        except ValueError as v_err:
            logger.warning(formatmsg(str(v_err), '', url))  # FIXME CHECK


def check_and_yield(url, params):
    try:
        fdsn_station_url = fdsn_url(url, new_service='station')
    except ValueError as e:
        raise ValueError("Invalid FDSN URL")

    if params['net'] != '*':
        yield fdsn_station_url, params
        return
    # no network specified, querymight take long (even for short time bounds).
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
        # raise ValueError("Request too big, unable to "
        #                  "fetch network list to narrow it down")


        # harmonize urls and put them in the urls dict:
        # for url, params in items:
        #     try:
        #         station_url = fdsn_url(url, new_service='station')
        #         urls.setdefault(station_url, set()).add(params)
        #     except ValueError as verr:
        #         discarded += 1
        #         logger.warning(formatmsg("Discarding data center", (str(verr)), url))

    # if discarded > 0:
    #     logger.info(formatmsg("%d data center(s) discarded"), discarded)

    # write to db:
    # ws_df = pd.DataFrame([{'url': u} for u in urls])
    # if ws_df.empty:
    #     raise FailedDownload(Exception("No FDSN-compliant datacenter found"))
    #
    # ws_df = dbsyncdf(
    #     ws_df, session,
    #     [WebService.url],
    #     WebService.id,
    #     buf_size=db_bufsize or len(urls),
    #     keep_duplicates=False
    # )

    # url2id = dict(zip(ws_df['url'], ws_df['id']))
    # ws_id_col = Channel.webservice_id.key
    # datacenters_df = []
    # ws_url_col = WebService.url.key
    # param_names = ('net', 'sta', 'loc', 'cha', 'start', 'end')
    # for station_url, param_values_set in urls.items():
    #     for param_values in param_values_set:
    #         datacenters_df.append({
    #             ws_url_col: station_url,
    #             # ws_id_col: url2id[station_url],
    #             **dict(zip(param_names, param_values))
    #         })
    #
    # # convert to category the dtype of column more likely to have few distinct values:
    # datacenters_df = pd.DataFrame(datacenters_df).astype({
    #     ws_url_col: 'category',
    #     # ws_id_col: int,
    #     'net': 'category',
    #     'loc': 'category',
    #     'cha': 'category',
    #     'start': 'category',
    #     'end': 'category'
    # }).drop_duplicates(keep='last')
    # # note: We do not apply pd.to_datetime to 'start' and 'end' columns because pandas
    # # high resolution (ns) => limited range => troubles with some dates way in the future
    # # (check by supplying net=_ADARRAY. Although we replace start and end with our values
    # # the problem might persist. Note that columns values still stay datetime though)
    # return datacenters_df


def scan_eida_rs_station_response(
    eidars_response_text, aggregate_ignoring_time_bounds=True
):
    """
    Yield tuples of the form:
    (url, net, sta, loc, cha, start, end)
    from the given eida routing service post response. All elements are strings.

    :param aggregate_ignoring_time_bounds: if true, start and end time are ignored
        in aggregating the URLs, and the returned start and end will be taken from
        one of the first aggregated row (as such, users should not rely on them)
    """
    ws_url_col = WebService.url.key

    dfr = []
    for url, net, sta, loc, cha, start_, end_ in (
        _scan_eida_rs_station_response(eidars_response_text)
    ):
        dfr.append({
            ws_url_col: url,
            'net': net,
            'sta': sta,
            'loc': loc,
            'cha': cha,
            'start': start_,
            'end': end_,
        })

    # put in a dataframe and group urls to optimize queries:
    dfr = pd.DataFrame(dfr)
    all_cols = [ws_url_col, 'net', 'sta', 'loc', 'cha', 'start', 'end']
    for col in ['cha', 'loc', 'sta', 'net']:
        ret = []
        cols = all_cols.copy()
        cols.remove(col)
        if aggregate_ignoring_time_bounds:
            cols.remove('start')
            cols.remove('end')
        for _, sub_dfr in dfr.groupby(cols, sort=False, dropna=False):
            if len(sub_dfr) > 1:
                tmp_df = sub_dfr.iloc[:1]
                tmp_df[col] = ",".join(sorted(set(sub_dfr[col])))
                sub_dfr = tmp_df
            ret.append(sub_dfr)
        dfr = pd.concat(ret, axis=0, ignore_index=True, copy=False)

    # yield each url
    for url, net, sta, loc, cha, start, end in dfr[all_cols].itertuples(index=False):
        yield url, {
            'net': net,
            'sta': sta,
            'loc': loc,
            'cha': cha,
            'start': start,
            'end': end
        }


def get_eidars_response_text(
    routing_service_url: list[str],
    net: Optional[str] = None,
    sta: Optional[str] = None,
    loc: Optional[str] = None,
    cha: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
):
    """Return the EIDA Routing Service response text (str)"""
    for eida_rs_url in routing_service_url:
        url = fdsn_url_qs(
            eida_rs_url, net=net, sta=sta, loc=loc,
            cha=cha, start=start, end=end,
            service='dataselect', format='post'
        )
        response = urlread(url, decode='utf8')
        if response.is_ok:
            return response.data
    raise FailedDownload("None of the EIDA routing services returned valid data. "
                         "Check internet connection or configure the URLs in advanced "
                         "settings")


def _scan_eida_rs_station_response(response_text: str):
    """
    Simple scanner yielding
    (url, net, sta, loc, cha, start, end)
    from the given eida routing service post response.
    No preocess is done here: all elements are string (* indicates: match all)

    :param response_text: (str) the EIDA routing service response text
    """
    start = 0
    textlen = len(response_text)

    while start < textlen:
        # find the end of the url block (double newline):
        end = response_text.find("\n\n", start)
        # if not found, move to the end:
        if end < 0:
            end = textlen
        lines = response_text[start:end].strip().split("\n")
        start = end + 2
        if len(lines) < 2:
            continue
        url = lines[0].strip()
        if not url:
            continue
        for line in lines[1:]:
            params = line.strip().split(" ")
            if len(params) != 6 or not all(params):  # assure 6 non empty elements
                continue
            # validate date-times (sometime as date, in case later pandas complains):
            # try:
            #     if starttime is not None:
            #         params[-2] = starttime
            #     elif params[-2] == '*':
            #         params[-2] = None
            #     else:
            #         params[-2] = datetime.fromisoformat(params[-2])
            #     if endtime is not None:
            #         params[-1] = endtime
            #     elif params[-1] == '*':
            #         params[-1] = None
            #     else:
            #         params[-1] = datetime.fromisoformat(params[-1])
            # except ValueError:
            #     continue
            yield tuple([url] + params)
            # yield (fdsn_url(url, new_service='station'),) + tuple(params[:-2])
        # for line in sorted(lines[1:]):
        #     # sorting is slightly inefficient but helps packing similar urls (see below)
        #     params = line.strip().split(" ")
        #     if len(params) != 6 or not all(params):  # assure 6 non empty elements
        #         continue
        #     # validate date-times (sometime as date, in case later pandas complains):
        #     try:
        #         params[-1] = None if params[-1] == '*' else \
        #             datetime.fromisoformat(params[-1])
        #         params[-2] = None if params[-1] == '*' else \
        #             datetime.fromisoformat(params[-2])
        #     except ValueError:
        #         continue
        #     # try to pack together FDSN request urls if possible:
        #     arg_where_diff = [i for i in range(6) if params[i] != yield_params[i]]
        #     # only one index difference (and not in time ranges, i.e. < 4)?
        #     if len(arg_where_diff) == 1 and arg_where_diff[0] < 4:
        #         i = arg_where_diff[0]
        #         if yield_params[i] == '*' or params[i] == '*':
        #             yield_params[i] = '*'
        #         elif params[i] not in yield_params[i].split(','):
        #             # 2nd check is because sometimes items are returned twice
        #             yield_params[i] = f'{yield_params[i]},{params[i]}'
        #     else:
        #         if any(yield_params):
        #             yield url, tuple(yield_params)
        #         yield_params = params
        # if any(yield_params):
        #     yield url, tuple(yield_params)


def split_times(start: datetime, end: datetime, interval_years=5):
    cur = start
    while cur < end:
        nxt = min(cur.replace(year=cur.year + interval_years), end)
        yield cur, nxt
        cur = nxt
