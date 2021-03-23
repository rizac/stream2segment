"""
Data center(s) download functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# Make the following(s) behave like python3 counterparts if running from
# Python 2.7.x (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import os
from datetime import datetime, timedelta
import re

from collections import defaultdict
import pandas as pd
from future.utils import string_types

from stream2segment.io.utils import Fdsnws
from stream2segment.download.db import DataCenter
from stream2segment.download.utils import FailedDownload, dbsyncdf, formatmsg
from stream2segment.utils import strconvert, urljoin, strptime
from stream2segment.utils.url import URLException, urlread


# logger: do not use logging.getLogger(__name__) but point to
# stream2segment.download.logger: this way we preserve the logging namespace
# hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial)
# when calling logging functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8
from stream2segment.utils.resources import get_templates_fpath, get_resources_fpath


def get_datacenters_df(session, service, routing_service_url,
                       network, station, location, channel, starttime=None,
                       endtime=None, db_bufsize=None):
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

    # For convenience and readability, define once the mapped column names
    # representing the dataframe columns that we need:
    DC_SURL = DataCenter.station_url.key  # pylint: disable=invalid-name
    DC_DURL = DataCenter.dataselect_url.key  # pylint: disable=invalid-name
    DC_ORG = DataCenter.organization_name.key  # pylint: disable=invalid-name

    # eida response text will be needed anyway to create an EidaValidator

    eidars_response_text = get_eidars_response_text(routing_service_url)
    dclist = []
    discarded = 0
    if isinstance(service, string_types):
        service = [service]
    for service_url in service:
        organization = service_url.lower().strip()
        if organization == 'iris':
            urls = ['https://service.iris.edu/fdsnws/dataselect/1/query']
        elif organization == 'eida':
            urls = [url for url, _ in eidarsiter(eidars_response_text)]
        else:
            urls = [service_url]
            organization = None

        for _url in urls:
            try:
                fdsn = Fdsnws(_url)
                dclist.append({DC_SURL: fdsn.url(Fdsnws.STATION),
                               DC_DURL: fdsn.url(Fdsnws.DATASEL),
                               DC_ORG: organization})
            except ValueError as verr:
                discarded += 1
                logger.warning(formatmsg("Discarding data center",
                                         (str(verr)), _url))

    dc_df = pd.DataFrame()  # empty by default
    if dclist:
        # attempt saving to db only if we might have something to save:
        dc_df = dbsyncdf(pd.DataFrame(dclist), session, [DataCenter.station_url],
                         DataCenter.id, buf_size=db_bufsize or len(dclist),
                         keep_duplicates='first')

    if dc_df.empty:
        raise FailedDownload(Exception("No FDSN-compliant datacenter found"))

    if discarded:
        logger.info("%d data centers discarded", discarded)

    return dc_df, EidaValidator(dc_df, eidars_response_text)


def get_eidars_response_text(routing_service_url):
    """Return the EIDA Routing Service response text (str)"""
    # IMPORTANT NOTE:
    # We issue a "basic" query to the EIDA rs, with no params other than
    # 'service' and 'format'. The reasons are two:
    # 1) as of Jan 2019 the service is buggy if supplying some arguments (e.g.,
    # with long list of channels), adn this might happen again in the future.
    # 2) We can save a local file (independent from the custom query) and read
    # from it in case of request failure. The file should be updated from times
    # to times
    query_args = {'service': 'dataselect', 'format': 'post'}
    url = urljoin(routing_service_url, **query_args)

    try:
        responsetext, status, msg = urlread(url, decode='utf8',
                                            raise_http_err=True)
        if not responsetext:
            raise URLException(Exception("Empty data response"))  # fall below
    except URLException as urlexc:
        responsetext, last_mod_time_str = _get_local_routing_service()
        msg = ("Eida routing service error, reading routes from file "
               "(last updated: %s)" % last_mod_time_str)
        logger.info(formatmsg(msg, "eida routing service error"))
        logger.warning(formatmsg("Eida routing service error", urlexc.exc, url))

    return responsetext


def _get_local_routing_service():
    """Reads the routing service from local file where we stored a successful
    response from the EIDA routing service (format=post), returns the file
    content and last modified time (in string format)

    :return: the tuple of strings:
        (content, last_modified)
        where content is the file content which is a string in the same format
        expected from a successful server response, and last_modified is the
        local file last modification time
    """
    fpath = get_resources_fpath('eidars.txt')
    lastmod_dtime = datetime(1970, 1, 1) + timedelta(seconds=os.path.getmtime(fpath))
    # read from file
    with open(fpath, 'r') as opn_:
        responsetext = opn_.read()
    return responsetext, lastmod_dtime.strftime('%Y-%m-%d')


class RoutingService(object):
    """Class representing a Routing service, i.e. and object that
    returns datacenter ids (int) from given channels (given as the tuple
    `(net, sta, loc, cha, start_time, end_time)`). An object of this class is
    build as a dict of datacenter ids mapped to matcher objects via the
    method `add_matcher`"""

    def __init__(self):
        self.dic = defaultdict(set)

    def add_matcher(self, dc_id, net, sta, loc, cha, stime, etime):
        """Add an :class:`ItemMatcher` to the Routing service, and maps it
         to the given datacenter id `dc_id` (int). All arguments are strings
         and might contain wildcards
        """
        self.dic[dc_id].add(ItemMatcher(net, sta, loc, cha, stime, etime))

    def get_dc_ids(self, net, sta, loc, cha, stime, etime):
        """Return a set of unique integers denoting the data center id
        associated to the given channel identified by the function arguments
        (any argument which is None will be ignored)

        :param net: (str or None) the network. None means: ignore
        :param sta: (str or None) the station. None means: ignore
        :param loc: (str or None) the location. None means: ignore
        :param cha: (str or None) the channel. None means: ignore
        :param stime: (datetime or None) the start time. None means: ignore
        :param etime: (datetime or None) the end time. None means: ignore
        """
        ret = set()
        for dc_id, item_macthers in self.dic.items():
            if any(_.match(net, sta, loc, cha, stime, etime)
                   for _ in item_macthers):
                ret.add(dc_id)
        return ret


class ItemMatcher(object):
    """Class representing a matcher in a Routing service. E.g. a line of text
    of the form:
    ```
    XT * * * 2014-05-21T00:00:00 2015-10-22T08:43:00
    ```
    """
    def __init__(self, net, sta, loc, cha, stime, etime):
        """Initialize this Matcher with the components of a Routing
        service channel. All arguments are strings and might contain wildcards
        """
        self.regs = tuple(re.compile("^%s$" % _)
                          for _ in [strconvert.wild2re(net),
                                    strconvert.wild2re(sta),
                                    '' if loc == '--' else strconvert.wild2re(loc),
                                    strconvert.wild2re(cha)])

        self.stime = None if stime == '*' else strptime(stime)
        self.etime = None if etime == '*' else strptime(etime)

    def match(self, net, sta, loc, cha, stime, etime):
        """Return True if the given Matcher matches the channel identified by
        the function arguments (all strings except `stime` and `etime`, which
        must be both datetime). Any argument which is None will be ignored.
        """
        if stime is not None and self.etime is not None and stime >= self.etime:
            return False
        if etime is not None and self.stime is not None and etime <= self.stime:
            return False
        for reg, txt in zip(self.regs, [net, sta, loc, cha]):
            if txt is not None and not reg.match(txt):
                return False
        return True


class EidaValidator(RoutingService):
    """EIDA routing service for validating stations duplicates according to the
    EIDA routing service response text (see `get_dc_ids`)"""

    def __init__(self, datacenters_df, responsetext):
        """Initialize a validator. You can then call `get_dc_ids` to get the
        datacenter id from a channel parameters

        :param datacenters_df: a dataframe representing the datacenters read
            from the EIDA routing service
        :param responsetext: the plain response text from the EIDA routing
            service
        """
        super(EidaValidator, self).__init__()
        reg = re.compile("^(\\S+) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+)$",
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
                    net, sta, loc, cha, stime, etime = \
                        match.group(1), match.group(2), match.group(3), \
                        match.group(4), match.group(5), match.group(6)

                    self.add_matcher(dc_id, net, sta, loc, cha, stime, etime)
                except IndexError:
                    continue


def eidarsiter(responsetext):
    """Iterator yielding the tuple (url, postdata) for each datacenter found in
    `responsetext`

    :param responsetext: (str) the EIDA routing service response text
    """
    # Yielding strings consumes less memory than using `str.split`... although
    # the code below it's not super readable (maybe change in the future)
    start = 0
    textlen = len(responsetext)
    while start < textlen:
        end = responsetext.find("\n\n", start)
        if end < 0:
            end = textlen
        mid = responsetext.find("\n", start, end)  # note: now we set a new value to idx
        if mid > -1:
            url = responsetext[start:mid].strip()
            postdata = responsetext[mid:end].strip()
            if url and postdata:
                yield url, postdata
        start = end + 2
