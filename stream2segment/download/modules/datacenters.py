'''
Download module for data-centers download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import logging
import re

from collections import defaultdict
import pandas as pd

from stream2segment.io.db.models import DataCenter, fdsn_urls
from stream2segment.download.utils import QuitDownload, dbsyncdf, empty, to_fdsn_arg
from stream2segment.utils import strconvert, urljoin
from stream2segment.utils.url import URLException, urlread
from stream2segment.utils.msgs import MSG
from stream2segment.io.db.pdsql import dbquery2df


# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


def get_datacenters_df(session, service, routing_service_url,
                       net, sta, loc, cha, starttime=None, endtime=None,
                       db_bufsize=None):
    """Returns a 2 elements tuple: the dataframe of the datacenter(s) matching `service`,
    and an EidaValidator (built on the eida routing service response)
    for checking stations/channels duplicates after querying the datacenter(s)
    for stations / channels. If service != 'eida', this argument is None
    Note that channels, starttime, endtime can be all None and
    are used only if service = 'eida'
    :param service: the string denoting the dataselect *or* station url in fdsn format, or
    'eida', or 'iris'. In case of 'eida', `routing_service_url` must denote an url for the
    edia routing service. If falsy (e.g., empty string or None), `service` defaults to 'eida'
    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    DC_SURL = DataCenter.station_url.key
    DC_DURL = DataCenter.dataselect_url.key
    DC_ORG = DataCenter.organization_name.key

    eidavalidator = None
    eidars_responsetext = ''

    if not service:
        service = 'eida'

    if service.lower() == 'iris':
        IRIS_NETLOC = 'https://service.iris.edu'
        dc_df = pd.DataFrame(data={DC_DURL: '%s/fdsnws/dataselect/1/query' % IRIS_NETLOC,
                                   DC_SURL: '%s/fdsnws/station/1/query' % IRIS_NETLOC,
                                   DC_ORG: 'iris'}, index=[0])
    elif service.lower() != 'eida':
        fdsn_normalized = fdsn_urls(service)
        if fdsn_normalized:
            station_ws = fdsn_normalized[0]
            dataselect_ws = fdsn_normalized[1]
            dc_df = pd.DataFrame(data={DC_DURL: dataselect_ws,
                                       DC_SURL: station_ws,
                                       DC_ORG: None}, index=[0])
        else:
            raise QuitDownload(Exception(MSG("Unable to use datacenter",
                                             "Url does not seem to be a valid fdsn url", service)))
    else:
        dc_df, eidars_responsetext = get_eida_datacenters_df(session, routing_service_url,
                                                             net, sta, loc, cha, starttime, endtime)

    # attempt saving to db only if we might have something to save:
    if service != 'eida' or eidars_responsetext:  # not eida, or eida succesfully queried: Sync db
        dc_df = dbsyncdf(dc_df, session, [DataCenter.station_url], DataCenter.id,
                         buf_size=len(dc_df) if db_bufsize is None else db_bufsize)
        if eidars_responsetext:
            eidavalidator = EidaValidator(dc_df, eidars_responsetext)

    return dc_df, eidavalidator


def get_eida_datacenters_df(session, routing_service_url, net, sta, loc, cha,
                            starttime=None, endtime=None):
    """Returns the tuple (datacenters_df, eidavalidator) from eidars or from the db (in this latter
    case eidavalidator is None)
    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    DC_SURL = DataCenter.station_url.key
    DC_DURL = DataCenter.dataselect_url.key
    DC_ORG = DataCenter.organization_name.key

    # do not return only new datacenters, return all of them
    query_args = {'service': 'dataselect', 'format': 'post'}
    if starttime:
        query_args['start'] = starttime.isoformat()
    if endtime:
        query_args['end'] = endtime.isoformat()

    for param, lst in zip(('net', 'sta', 'loc', 'cha'), (net, sta, loc, cha)):
        if lst:
            query_args[param] = to_fdsn_arg(lst)

    url = urljoin(routing_service_url, **query_args)

    dc_df = None
    dclist = []

    try:
        responsetext, status, msg = urlread(url, decode='utf8', raise_http_err=True)
        for url, postdata in eidarsiter(responsetext):  # @UnusedVariable
            urls = fdsn_urls(url)
            if urls:
                dclist.append({DC_SURL: urls[0], DC_DURL: urls[1], DC_ORG: 'eida'})
        if not dclist:
            raise URLException(Exception("No datacenters found in response text"))
        return pd.DataFrame(dclist), responsetext

    except URLException as urlexc:
        dc_df = dbquery2df(session.query(DataCenter.id, DataCenter.station_url,
                                         DataCenter.dataselect_url).
                           filter(DataCenter.organization_name == 'eida')).\
                                reset_index(drop=True)
        if empty(dc_df):
            msg = MSG("Eida routing service error, no eida data-center saved in database",
                      urlexc.exc, url)
            raise QuitDownload(Exception(msg))
        else:
            msg = MSG("Eida routing service error", urlexc.exc, url)
            logger.warning(msg)
            # logger.info(msg)
            return dc_df, None


class EidaValidator(object):
    '''Class for validating stations duplicates according to the eida routing service
    response text'''
    def __init__(self, datacenters_df, responsetext):
        """Initializes a validator. You can then call `isin` to check if a station is valid
        :param datacenters_df: a dataframe representing the datacenters read from the eida
        routing service
        :param responsetext: the plain response text from the eida routing service
        """
        self.dic = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
                                                                                           set()))))
        reg = re.compile("^(\\S+) (\\S+) (\\S+) (\\S+) .*$",
                         re.MULTILINE)  # @UndefinedVariable
        for url, postdata in eidarsiter(responsetext):
            _ = datacenters_df[datacenters_df[DataCenter.dataselect_url.key] == url]
            if _.empty:
                _ = datacenters_df[datacenters_df[DataCenter.station_url.key] == url]
            if len(_) != 1:
                continue
            dc_id = _[DataCenter.id.key].iloc[0]
            for match in reg.finditer(postdata):
                try:
                    net, sta, loc, cha = \
                        match.group(1), match.group(2), match.group(3), match.group(4)
                except IndexError:
                    continue
                self.add(dc_id, net, sta, loc, cha)

    @staticmethod
    def _tore(wild_str):
        if wild_str == '--':
            wild_str = ''
        return re.compile("^%s$" % strconvert.wild2re(wild_str))

    def add(self, dc_id, net, sta, loc, cha):
        """adds the tuple datacenter id, network station location channels to the internal dic
        :param dc_id: integer
        :param net: string, the network name
        :param sta: string, the station name. Special cases: '*' (match all), "--" (empty)
        :param sta: string, the location name. Special cases: '*' (match all), "--" (empty)
        :param cha: string, the channel (can contain wildcards like '*' or '?'). Special cases:
                    '*' (match all)
        """
        self.dic[dc_id][net][self._tore(sta)][self._tore(loc)].add(self._tore(cha))

    @staticmethod
    def _get(regexiterable, key, return_bool=False):
        for regex in regexiterable:
            if regex.match(key):
                return True if return_bool else regexiterable[regex]
        return False if return_bool else None

    def isin(self, dc_id, net, sta, loc, cha):
        """Returns a boolean (or a list of booleans) telling if the tuple arguments:
        ```(dc_id, net, sta, loc, cha)```
        match any of the eida response lines of text.
        Returns a list of boolean if the arguments are iterable (not including strings)
        Returns numpy.array if return_np = True
        """
        # dc_id - > {net_re -> //}
        stadic = self.dic.get(dc_id, {}).get(net, None)
        if stadic is None:
            return False
        # sta_re - > {loc_re -> //}
        locdic = self._get(stadic, sta)
        if locdic is None:
            return False
        # loc_re - > set(cha_re,..)
        chaset = self._get(locdic, loc)
        if chaset is None:
            return False
        return self._get(chaset, cha, return_bool=True)


def eidarsiter(responsetext):
    """iterator yielding the tuple (url, postdata) for each datacenter found in responsetext
    :param responsetext: the eida routing service response text
    """
    # not really pythonic code, but I enjoyed avoiding copying strings and creating lists
    # so this iterator is most likely really low memory consuming
    start = 0
    textlen = len(responsetext)
    while start < textlen:
        end = responsetext.find("\n\n", start)
        if end < 0:
            end = textlen
        mid = responsetext.find("\n", start, end)  # note: now we set a new value to idx
        if mid > -1:
            url, postdata = responsetext[start:mid].strip(), responsetext[mid:end].strip()
            if url and postdata:
                yield url, postdata
        start = end + 2


