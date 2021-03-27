"""
Module implementing the download info (print statistics and generate html page)

:date: Mar 15, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import json
import os
import sys
import threading
from collections import defaultdict
from io import StringIO
from tempfile import gettempdir
from webbrowser import open as open_in_browser

from future.utils import viewitems
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import func, or_

from stream2segment.download.db.models import Download, Segment, DataCenter, Station
from stream2segment.download.modules.utils import EVENTWS_SAFE_PARAMS, DownloadStats
from stream2segment.io import yaml_load, Fdsnws, open2writetext
from stream2segment.io.cli import ascii_decorate
from stream2segment.io.db.sqlconstructs import concat
from stream2segment.io.inputvalidation import validate_param, valid_session


def dreport(dburl, download_ids=None, config=True, log=True, html=False,
            outfile=None):
    """Create a diagnostic html page (or text string) showing the status of the
    download. Note that html is not supported for the moment and will raise an
    Exception. (leaving the same signatire as dstats for compatibility and
    easing future implementations of the html page if needed)

    :param config: boolean (True by default)
    :param log: boolean (True by default)
    """
    _get_download_info(DReport(config, log), dburl, download_ids, html, outfile)


def dstats(dburl, download_ids=None, maxgap_threshold=0.5, html=False,
           outfile=None):
    """Create a diagnostic html page (or text string) showing the status of the
    download

    :param maxgap_threshold: the max gap threshold (float)
    """
    _get_download_info(DStats(maxgap_threshold), dburl, download_ids, html,
                       outfile)


def _get_download_info(info_generator, dburl, download_ids=None, html=False,
                       outfile=None):
    """Process dinfo or dstats"""
    session = validate_param('dburl', dburl, valid_session)
    if html:
        openbrowser = False
        if not outfile:
            openbrowser = True
            outfile = os.path.join(gettempdir(), "s2s_%s.html" %
                                   info_generator.__class__.__name__.lower())
        # get_dstats_html returns unicode characters in py2, str in py3,
        # so it is safe to use open like this (cf below):
        with open(outfile, 'w', encoding='utf8', errors='replace') as opn:
            opn.write(info_generator.html(session, download_ids))
        if openbrowser:
            open_in_browser('file://' + outfile)
        threading.Timer(1, lambda: sys.exit(0)).start()
    else:
        itr = info_generator.str_iter(session, download_ids)
        if outfile is not None:
            # itr is an iterator of strings in py2, and str in py3, so open
            # must be input differently (see utils module):
            with open2writetext(outfile, encoding='utf8', errors='replace') as opn:
                for line in itr:
                    line += '\n'
                    opn.write(line)
        else:
            for line in itr:
                print(line)


class _InfoGenerator(object):
    '''Base class for any subclasses returning Download info in text and html
    content. Subclasses should overwrite `self.str_iter` and `self.html_template_arguments`
    '''

    def str_iter(self, session, download_ids=None):
        '''Returns an iterator yielding chunks of strings denoting the string
        representation of this object'''
        pass

    def html(self, session, download_ids=None):
        '''Returns a string with the html representation of this object'''
        args = self.html_template_arguments(session, download_ids)
        args.setdefault('title', self.__class__.__name__)
        return self.get_template().render(**args)

    def html_template_arguments(self, session, download_ids=None):
        '''Subclasses should return here a dict to be passed as arguments to
        the jinja2 template`
        '''
        return {}

    @classmethod
    def get_template(cls):
        '''Returns the jinja2 template for this object
        The html file must be an existing file a file with name
        `self.__class__.__name__.lower() + ',html'
        '''
        thisdir = os.path.dirname(__file__)
        templatespath = os.path.join(thisdir, 'templates')
        csspath = os.path.join(thisdir, 'static', 'css')
        jspath = os.path.join(thisdir, 'static', 'js')
        env = Environment(loader=FileSystemLoader([templatespath, jspath, csspath]))
        return env.get_template('%s.html' % cls.__name__.lower())


class DReport(_InfoGenerator):
    '''Class handling the generation of download reports in text format (no html supported
    for the moment)'''

    def __init__(self, config=True, log=True):
        self.config = config
        self.log = log

    def str_iter(self, session, download_ids=None):
        '''Returns an iterator yielding chunks of strings denoting the string
        representation of this object'''
        return get_dreport_str_iter(session, download_ids, self.config, self.log)

    def html_template_arguments(self, session, download_ids=None):
        '''Returns a dict to be passed as arguments to
        the jinja2 template'''
        raise Exception('html version not available')
        # return get_dreport_html_template_arguments(session, download_ids, self.config, self.log)


class DStats(_InfoGenerator):
    '''Class handling the generation of download statistics in text and html format'''

    def __init__(self, maxgap_threshold=0.5):
        self.maxgap_threshold = maxgap_threshold

    def str_iter(self, session, download_ids=None):
        '''Returns an iterator yielding chunks of strings denoting the string
        representation of this object'''
        return get_dstats_str_iter(session, download_ids, self.maxgap_threshold)

    def html_template_arguments(self, session, download_ids=None):
        '''Returns a dict to be passed as arguments to
        the jinja2 template'''
        return get_dstats_html_template_arguments(session, download_ids, self.maxgap_threshold)


def get_dreport_str_iter(session, download_ids=None, config=True, log=True):
    '''Returns an iterator yielding the download report (log and config) for the given
    download_ids

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics from all downloads run.
        Otherwise limit the output to the downloads whose ids are in the list
    :param config: boolean (default: True). Whether to show the download config
    :param log: boolean (default: True). Whether to show the download log messages
    '''
    data = infoquery(session, download_ids, config, log)
    for dwnl_id, dwnl_time, configtext, logtext in data:
        yield ''
        yield ascii_decorate('Download id: %d (%s)' % (dwnl_id, str(dwnl_time)))
        if config and log:
            yield ''
            yield 'Configuration:%s' % (' N/A' if not configtext else '')
        if configtext:
            yield ''
            yield configtext
        if config and log:
            yield ''
            yield 'Log messages:%s' % (' N/A' if not configtext else '')
        if logtext:
            yield ''
            yield logtext


def infoquery(session, download_ids=None, config=True, log=True):
    '''Returns a query for getting data for inspection (show_stats=False in the
    functions above)'''
    # IMPORTANT: If it happens to access backref relationships (e.g. Download.segments)
    # consider calling configure_mappers() first:
    # configure_mappers()  # https://stackoverflow.com/questions/14921777/backref-class-attribute
    attrs = [Download.id, Download.run_time]
    if config:
        attrs.append(Download.config)
    if log:
        attrs.append(Download.log)
    qry = session.query(*attrs)
    if download_ids is not None:
        qry = qry.filter(Download.id.in_(download_ids))
    for res in qry.order_by(Download.run_time.asc()):  # .group_by(Download.id):
        if not config and not log:  # False
            res = list(res) + ['', '']
        elif not log:
            res = list(res) + ['']
        elif not config:
            res = list(res)
            res.insert(-1, '')
        yield res


def tojson(obj):
    '''converts obj to json formatted string without whitespaces to minimize string size'''
    return json.dumps(obj, separators=(',', ':'))


def get_dstats_str_iter(session, download_ids=None, maxgap_threshold=0.5):
    '''Returns an iterator yielding the download statistics and information matching the
    given parameters.
    The returned string can be joined and printed to screen or file and is made of tables
    showing the segment data on the db per data-center and download run, plus some download
    information.

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics from all downloads run.
        Otherwise limit the output to the downloads whose ids are in the list. In any case, in
        case of more download runs to be considered, this function will
        yield also the statistics aggregating all downloads in a table at the end
    :param maxgap_threshold: (float, default 0.5).
        Sets the threshold whereby a segment is to be
        considered with gaps or overlaps. By default is 0.5, meaning that a segment whose
        'maxgap_numsamples' value is > 0.5 has gaps, and a segment whose 'maxgap_numsamples'
        value is < 0.5 has overlaps. Such segments will be marked with a special class
        'OK Gaps Overlaps' in the table columns.
    '''
    # Benchmark: the bare minimum (with postgres on external server) request takes around 12
    # sec and 14 seconds adding all necessary information. Therefore, we choose the latter
    maxgap_bexpr = get_maxgap_sql_expr(maxgap_threshold)
    data = session.query(func.count(Segment.id),
                         Segment.download_code,
                         Segment.datacenter_id,
                         Segment.download_id,
                         maxgap_bexpr)
    data = filterquery(data, download_ids).group_by(Segment.download_id, Segment.datacenter_id,
                                                    Segment.download_code, maxgap_bexpr)

    dwlids = get_downloads(session, download_ids)
    show_aggregate_stats = len(dwlids) > 1
    dcurl = get_datacenters(session)
    if show_aggregate_stats:
        agg_statz = DownloadStats2()
    stas = defaultdict(lambda: DownloadStats2())
    GAP_OVLAP_CODE = DownloadStats2.GAP_OVLAP_CODE  # pylint: disable=invalid-name
    for segcount, dwn_code, dc_id, dwn_id, has_go in data:
        statz = stas[dwn_id]

        if dwn_code == 200 and has_go is True:
            dwn_code = GAP_OVLAP_CODE

        statz[dcurl[dc_id]][dwn_code] += segcount
        if show_aggregate_stats:
            agg_statz[dcurl[dc_id]][dwn_code] += segcount

    evparamlen = None  # used for alignement of strings (calculated lazily in loop below)
    for did, (druntime, evtparams) in viewitems(dwlids):
        yield ''
        yield ''
        yield ascii_decorate('Download id: %d' % did)
        yield ''
        yield 'Executed: %s' % str(druntime)
        yield "Event query parameters:%s" % (' N/A' if not evtparams else '')
        if evparamlen is None and evtparams:  # calculate eventparamlen for string alignement
            evparamlen = max(len(_) for _ in evtparams)
        for param in sorted(evtparams):
            yield ("  %-{:d}s = %s".format(evparamlen)) % (param, str(evtparams[param]))
        yield ''
        statz = stas.get(did)
        if statz is None:
            yield "No segments downloaded"
        else:
            yield "Downlaoaded segments per data center url (row) and response type (column):"
            yield ""
            yield str(statz)

    if show_aggregate_stats:
        yield ''
        yield ''
        yield ascii_decorate('Aggregated stats (all downloads)')
        yield ''
        yield str(agg_statz)


def get_dstats_html_template_arguments(session, download_ids=None, maxgap_threshold=0.5):
    '''Returns an html page (string) yielding the download statistics and information matching the
    given parameters.

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics from all downloads run.
        Otherwise limit the output to the downloads whose ids are in the list. In any case, in
        case of more download runs to be considered, this function will
        yield also the statistics aggregating all downloads in a table at the end
    :param maxgap_threshold: (float, default 0.5).
        Sets the threshold whereby a segment is to be
        considered with gaps or overlaps. By default is 0.5, meaning that a segment whose
        'maxgap_numsamples' value is > 0.5 has gaps, and a segment whose 'maxgap_numsamples'
        value is < 0.5 has overlaps. Such segments will be marked with a special class
        'OK Gaps Overlaps' in the table columns.
    '''
    sta_data, codes, datacenters, downloads, networks = \
        get_dstats_html_data(session, download_ids, maxgap_threshold)
    # selected codes by default the Ok one. To know which position is
    # in codes is a little hacky:
    selcodes = [i for i, c in enumerate(codes) if list(c) == list(DownloadStats2.resp[200])[:2]]
    # downloads are all selected by default
    seldownloads = list(downloads.keys())
    seldatacenters = list(datacenters.keys())
    return dict(sta_data_json=tojson(sta_data),
                codes=codes,
                datacenters=datacenters,
                downloads=downloads,
                selcodes_set=set(selcodes),
                selcodes=selcodes,
                seldownloads=seldownloads,
                seldatacenters=seldatacenters,
                networks=networks)


def filterquery(query, download_ids=None):
    '''adds a filter to the given query if download_ids is not None, and returns a new
    query. Otherwise, if download_ids is None, it's no-op and returns query itself'''
    if download_ids is not None:
        query = query.filter(Segment.download_id.in_(download_ids))
    return query


def yaml_get(yaml_content):
    '''Returns the arguments used for the eventws query stored in the yaml,
    or an empty dict in case of errors

    :param yaml_content: yaml formatted string representing a download config'''
    try:
        dic = yaml_load(StringIO(yaml_content))
        ret = {k: dic[k] for k in EVENTWS_SAFE_PARAMS if k in dic}
        additional_eventws_params = dic.get('eventws_query_args', None) or {}
        ret.update(additional_eventws_params)
        return ret
    except Exception as _:  # pylint: disable=broad-except
        return {}


def get_downloads(sess, download_ids=None):
    '''Returns a dict of download ids mapped to the tuple
    (download_run_time, download_eventws_query_args)
    the first element is a string, the second a dict
    '''
    query = filterquery(sess.query(Download.id, Download.run_time, Download.config),
                        download_ids).order_by(Download.run_time.asc())
    return {did: (time.isoformat(), yaml_get(cfg))
            for (did, time, cfg) in query}


def get_datacenters(sess, dc_ids=None):
    '''returns a dict of datacenters id mapped to the network location of their url'''
    query = sess.query(DataCenter.id, DataCenter.dataselect_url)
    if dc_ids is not None:
        query = query.filter(DataCenter.id.in_(dc_ids))
    ret = {}
    for (datacenter_id, dataselect_url) in query:
        try:
            url = Fdsnws(dataselect_url).site
        except:  # @IgnorePep8
            url = dataselect_url
        ret[datacenter_id] = url
    return ret


def get_maxgap_sql_expr(maxgap_threshold=0.5):
    '''returns a sql-alchemy binary expression which matches segments with gaps/overlaps,
    according to the given threshold'''
    return or_(Segment.maxgap_numsamples < -abs(maxgap_threshold),
               Segment.maxgap_numsamples > abs(maxgap_threshold))


class DownloadStats2(DownloadStats):
    GAP_OVLAP_CODE = -2000
    resp = dict(DownloadStats.resp)
    resp[GAP_OVLAP_CODE] = ('OK Gaps Overlaps',  # title
                            'Data saved (download ok, '  # legend
                            'data has gaps or overlaps)',
                            0.1)  # sort order (just after 200 ok)


def get_dstats_html_data(session, download_ids=None, maxgap_threshold=0.5):
    '''Returns the tuple
        sta_list, codes, datacenters, downloads, networks

    where: sta_list is a list stations data and their download codes (togehter with the number
        of segments downloaded and matching the given code)
    codes is a list of tuples (title, legend) representing the titles and legends of all
        download codes found
    datacenters the output of `get_datacenters`
    downloads is the output of `get_downloads`
    networks is a list of strings denoting the networks found

    The returned data is used to build the html page showing the download info / statistics.
    All returned elements will be basically injected as json string in the html page and
    processed inthere by the browser with a js library also injected in the html page.

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics from all downloads run.
        Otherwise limit the output to the downloads whose ids are in the list. In any case, in
        case of more download runs to be considered, this function will
        yield also the statistics aggregating all downloads in a table at the end
    :param maxgap_threshold: (float, default 0.5) the threshold whereby a segment is to be
        considered with gaps or overlaps. By default is 0.5, meaning that a segment whose
        'maxgap_numsamples' value is > 0.5 has gaps, and a segment whose 'maxgap_numsamples'
        value is < 0.5 has overlaps. Such segments will be marked with a special class
        'OK Gaps Overlaps' in the table columns.
    '''
    # Benchmark: the bare minimum (with postgres on external server) request takes around 12
    # sec and 14 seconds adding all necessary information. Therefore, we choose the latter
    maxgap_bexpr = get_maxgap_sql_expr(maxgap_threshold)
    data = session.query(func.count(Segment.id),
                         Station.id,
                         concat(Station.network, '.', Station.station),
                         Station.latitude,
                         Station.longitude,
                         Station.datacenter_id,
                         Segment.download_id,
                         Segment.download_code,
                         maxgap_bexpr).join(Segment.station)
    data = filterquery(data, download_ids).group_by(Station.id, Segment.download_id,
                                                    Segment.download_code, maxgap_bexpr,
                                                    Segment.datacenter_id)

    codesfound = set()
    dcidsfound = set()
    # sta_data = {sta_name: [staid, stalat, stalon, sta_dcid,
    #                        {d_id: {code1: num_seg , codeN: num_seg}, ... }
    #                       ],
    #            ...,
    #            }
    sta_data = {}
    networks = {}
    _gap_ovlap_code = DownloadStats2.GAP_OVLAP_CODE
    for segcount, staid, staname, lat, lon, dc_id, dwn_id, dwn_code, has_go in data:
        network = staname.split('.')[0]
        netindex = networks.get(network, -1)
        if netindex == -1:
            networks[network] = netindex = len(networks)
        sta_list = sta_data.get(staname, [staid, round(lat, 2), round(lon, 2), dc_id, netindex,
                                          None])
        if sta_list[-1] is None:
            sta_list[-1] = defaultdict(lambda: defaultdict(int))
            sta_data[staname] = sta_list
        sta_dic = sta_list[-1][dwn_id]
        if dwn_code == 200 and has_go is True:
            dwn_code = _gap_ovlap_code
        sta_dic[dwn_code] += segcount
        codesfound.add(dwn_code)
        dcidsfound.add(dc_id)

    # In the html, we want to reduce all possible data, as the file might be huge
    # modify stas_data nested dicts, replacing codes with an incremental integer
    # and keep a separate list that maps uses codes to titles and legends
    # So, first sort codes and keep track of their index
    # Then, remove dicts for two reasons:
    # js objects converts int keys as string (it's a property of js objects), this makes:
    # 1. unnecessary quotes chars which take up space, and
    # 2. prevents to work with other objects, e.g., storing some int key in a js Set, makes
    #    set.has(same_key_as_string) return false
    # 3. We do not actually need object key search in the page, as we actully loop through elements
    #    arrays are thus fine
    # Thus sta_data should look like:
    # sta_data = [sta_name, [staid, stalat, stalon, sta_dcid, sta_net_index,
    #                        d_id1, [code1, num_seg1 , ..., codeN, num_seg],
    #                        d_id2, [code1, num_seg1 , ..., codeN, num_seg],
    #                       ],
    #            ...,
    #            ]
    sta_list = []
    sortedcodes = DownloadStats2.sortcodes(codesfound)
    codeint = {k: i for i, k in enumerate(sortedcodes)}
    for staname, values in viewitems(sta_data):
        staname = staname.split('.')[1]
        dwnlds = values.pop()  # remove last element
        for did, segs in viewitems(dwnlds):
            values.append(did)
            values.append([item for code in segs for item in (codeint[code], segs[code])])
        sta_list.append(staname)
        sta_list.append(values)

    codes = [DownloadStats2.titlelegend(code) for code in sortedcodes]
    networks = sorted(networks, key=lambda key: networks[key])
    return sta_list, codes, get_datacenters(session, list(dcidsfound) or None), \
        get_downloads(session), networks
