"""
Data center(s) download functions

:date: Dec 3, 2017
"""
from datetime import datetime
import logging
from typing import Optional

import pandas as pd

from stream2segment.io import Fdsnws
from stream2segment.download.db.models import WebService
from stream2segment.download.modules.utils import dbsyncdf, formatmsg, fdsn_url
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import urlread

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def get_datacenters_df(
        session, service, routing_service_url,
        network: Optional[list[str]] = None,
        station: Optional[list[str]] = None,
        location: Optional[list[str]] = None,
        channel: Optional[list[str]] = None,
        starttime: Optional[datetime] = None,
        endtime: Optional[datetime] = None,
        db_bufsize=None
):
    """Returns a 2 elements tuple: the Dataframe of the datacenter(s) matching
    `service`, and an EidaValidator (built on the EIDA routing service response)
    for checking stations/channels duplicates after querying the datacenter(s)
    for stations / channels. If service != 'eida', this argument is None

    NOTE: The parameters
    network, station, location, channel, starttime`, endtime
    are NOT used and are here for legacy code, when we used them to filter
    results in the eida routing service. The filter is now handled in the code
    later

    :param service: (list[str] or str) the dataselect *or* station url(s) in
        FDSN format, or the shortcuts 'eida', or 'iris'
    """
    # eida response text will be needed anyway to create an EidaValidator
    eidars_response_text = None  # lazy loaded
    discarded = 0
    if isinstance(service, str):
        service = [service]
    params = (
        ','.join(n for n in network or [] if not n.startswith('!')) or '*',
        ','.join(s for s in station or [] if not s.startswith('!')) or '*',
        ','.join(l for l in location or [] if not l.startswith('!')) or '*',
        ','.join(c for c in channel or [] if not c.startswith('!')) or '*',
        starttime,
        endtime,
    )
    url2fdsn = {}  # url:str -> (station, dataselect) urls
    urls: dict[tuple[str, str], set[tuple]] = {}  # (station, dataselect) urls -> params  # noqa
    for service_url in service:
        organization = service_url.lower().strip()
        if organization == 'iris':
            items = [('https://service.iris.edu/fdsnws/dataselect/1/query', params)]
        elif organization == 'eida':
            if eidars_response_text is None:
                eidars_response_text = get_eidars_response_text(
                    routing_service_url, *params
                )
            items = eidarsiter(eidars_response_text)
        else:
            items = [(service_url, params)]

        # harmonize urls and put them in the urls dict:
        for url, params in items:
            if url not in url2fdsn:
                try:
                    fdsn = Fdsnws(url)
                    url2fdsn[url] = (fdsn.url(Fdsnws.STATION), fdsn.url(Fdsnws.DATASEL))
                except ValueError as verr:
                    url2fdsn[url] = None
                    discarded += 1
                    logger.warning(formatmsg("Discarding data center", (str(verr)), url))
                    continue
            key = url2fdsn[url]
            if key is None:  # previously discarded
                continue
            urls.setdefault(key, set()).add(params)

    if discarded > 0:
        logger.info(formatmsg("%d data center(s) discarded"), discarded)

    # write to db:
    ws_df = pd.DataFrame([{'url': u} for pair in urls for u in pair])
    if ws_df.empty:
        raise FailedDownload(Exception("No FDSN-compliant datacenter found"))

    ws_df = dbsyncdf(ws_df, session, [WebService.url],
                     WebService.id, buf_size=db_bufsize or len(urls),
                     keep_duplicates=False)

    datacenters_df = []
    url2id = dict(zip(ws_df['url'], ws_df['id']))
    param_names = ('net', 'sta', 'loc', 'cha', 'start', 'end')
    for (station_url, dataselect_url), param_values_set in urls.items():
        for param_values in param_values_set:
            datacenters_df.append({
                'dataselect_url': dataselect_url,
                'station_url': station_url,
                'station_ws_id': url2id[station_url],
                'dataselect_ws_id': url2id[station_url],
                **dict(zip(param_names, param_values))
            })
    # convert to category the dtype of column more likely to have few distinct values:
    datacenters_df = pd.DataFrame(datacenters_df).astype({
        'station_url': 'category',
        'dataselect_url': 'category',
        'net': 'category',
        'loc': 'category',
        'cha': 'category',
        'end': 'category'
    })
    # note: We do not apply pd.to_datetime to 'start' and 'end' columns because pandas
    # high resolution (ns) => limited range => troubles with some dates way in the future
    # (check by supplying net=_ADARRAY)
    return datacenters_df


def get_eidars_response_text(
        routing_service_url: list[str],
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[datetime] = None,
        endtime: Optional[datetime] = None
):
    """Return the EIDA Routing Service response text (str)"""
    for eida_rs_url in routing_service_url:
        url = fdsn_url(eida_rs_url, net=network, sta=station, loc=location,
                       cha=channel, start=starttime, end=endtime,
                       service='dataselect', format='post')
        response_text, error, code = urlread(url, decode='utf8')
        if not error:
            return response_text
    raise FailedDownload("None of the EIDA routing services returned valid data. "
                         "Check internet connection or configure the URLs in advanced "
                         "settings")


def eidarsiter(response_text):
    """Iterator yielding from the given eida routing service post response
    tuples of the form (url, params) where params is in turn a 6 element tuple
    of 6 query parameters [net, sta, loc, cha, start, end] (all str).
    url can be a station or dataselect FDSN url

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
        yield_params = [''] * 6
        for line in sorted(lines[1:]):
            # sorting is slightly inefficient but helps packing similar urls (see below)
            params = line.strip().split(" ")
            if len(params) != 6 or not all(params):  # assure 6 non empty elements
                continue
            # validate date-times (sometime as date, in case later pandas complains):
            try:
                params[-1] = None if params[-1] == '*' else \
                    datetime.fromisoformat(params[-1])
                params[-2] = None if params[-1] == '*' else \
                    datetime.fromisoformat(params[-2])
            except ValueError:
                continue
            if params[1] in ('A076A', 'A079A'):
                asd = 0
            # try to pack together FDSN request urls if possible:
            arg_where_diff = [i for i in range(6) if params[i] != yield_params[i]]
            # only one index difference (and not in time ranges, i.e. < 4)?
            if len(arg_where_diff) == 1 and arg_where_diff[0] < 4:
                i = arg_where_diff[0]
                if yield_params[i] == '*' or params[i] == '*':
                    yield_params[i] = '*'
                elif params[i] not in yield_params[i].split(','):
                    # 2nd check is because sometimes items are returned twice
                    yield_params[i] = f'{yield_params[i]},{params[i]}'
            else:
                if any(yield_params):
                    yield url, tuple(yield_params)
                yield_params = params
        if any(yield_params):
            yield url, tuple(yield_params)
