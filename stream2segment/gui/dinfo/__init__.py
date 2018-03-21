# -*- encoding: utf-8 -*-
'''
Module implementing the download info (print statistics and generate html page)

:date: Mar 15, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

from collections import defaultdict
from future.standard_library import install_aliases
from future.utils import viewitems, itervalues, viewvalues, viewkeys

from urllib.parse import urlparse  # @IgnorePep8

from jinja2 import Environment, FileSystemLoader
from sqlalchemy.sql.expression import func, or_

from stream2segment.utils.resources import yaml_load
from stream2segment.utils import get_session, get_progressbar, StringIO
from stream2segment.io.db.models import Segment, concat, Station, DataCenter, Download, substr
from stream2segment.download.utils import custom_download_codes, DownloadStats
import os
import json

install_aliases()


def filterquery(query, download_ids=None):
    '''adds a filter to the given query if download_ids is not None, and returns a new
    query. Otherwise, if download_ids is None, it's no-op and returns query itself'''
    if download_ids is not None:
        query = query.filter(Segment.download_id.in_(download_ids))
    return query


def yaml_get(yaml_content):
    '''Returns the eventws_query_args portion of the yaml identified by the given yaml_content,
    or an empty dict in case of errors'''
    try:
        return yaml_load(StringIO(yaml_content))['eventws_query_args']
    except Exception as exc:
        return {}


def get_downloads(sess, download_ids=None):
    '''Returns a dict of download ids mapped to the tuple
    (download_run_time, download_eventws_query_args)
    the first element is a string, the second a dict
    '''
    query = filterquery(sess.query(Download.id, Download.run_time, Download.config), download_ids)
    return {did: (time.isoformat(), yaml_get(cfg))
            for (did, time, cfg) in query}


def get_datacenters(sess, dc_ids=None):
    '''returns a dict of datacenters id mapped to the network location of their url'''
    query = sess.query(DataCenter.id, DataCenter.dataselect_url)
    if dc_ids is not None:
        query = query.filter(DataCenter.id.in_(dc_ids))
    return {did: urlparse(ds).netloc or ds for (did, ds) in query}


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
#     def __init__(self):
#         super(DownloadStats2, self).__init__()
#         self.GAP_OVLAP_CODE = 200.1  # place it right after 200 OK responses and before -204 (some
#         # chunks out of bounds)
#         self.resp[self.GAP_OVLAP_CODE] = ('OK Gaps Overlaps',
#                                           'Data saved (download ok, '
#                                           'data has gaps or overlaps)')


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
    GAP_OVLAP_CODE = DownloadStats2.GAP_OVLAP_CODE
    for segcount, dwn_code, dc_id, dwn_id, has_go in data:
        statz = stas[dwn_id]

        if dwn_code == 200 and has_go is True:
            dwn_code = GAP_OVLAP_CODE

        statz[dcurl[dc_id]][dwn_code] += segcount
        if show_aggregate_stats:
            agg_statz[dcurl[dc_id]][dwn_code] += segcount

    for did, dwl in viewitems(dwlids):
        yield ascii_decorate('Download id: %d' % did)
        yield 'executed: %s' % str(dwl[0])
        yield "even query (param 'eventws_query_args'):"
        for param in sorted(dwl[1]):
            yield " %s = %s" % (param, str(dwl[1][param]))
        yield ''
        yield str(stas.get(did, 'N/A'))
        yield ''

    if show_aggregate_stats:
        yield ascii_decorate('Aggregated stats (all downloads)')
        yield ''
        yield str(agg_statz)


def ascii_decorate(string):
    '''Decorates the string with a frame in unicode decoration characters, and returns
    the decorated string'''
    leng = (len(string) + 2)
    firstline = "╔" + "═" * leng + "╗"
    secondline = "║ " + string + " ║"
    thirdline = "╚" + "═" * leng + "╝"
    return "\n".join([firstline, secondline, thirdline])


def get_template():
    '''Returns the jinja2 template for the html page of the download statistics'''
    thisdir = os.path.dirname(__file__)
    templatespath = os.path.join(os.path.dirname(thisdir), 'webapp', 'templates')
    csspath = os.path.join(os.path.dirname(thisdir), 'webapp', 'static', 'css')
    env = Environment(loader=FileSystemLoader([thisdir, templatespath, csspath]))
    return env.get_template('dinfo.html')


def tojson(obj):
    '''converts obj to json formatted string without whitespaces to minimize string size'''
    return json.dumps(obj, separators=(',', ':'))


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
    GAP_OVLAP_CODE = DownloadStats2.GAP_OVLAP_CODE
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
            dwn_code = GAP_OVLAP_CODE
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


def get_dstats_html(session, download_ids=None, maxgap_threshold=0.5):
    '''Returns an html page (string) yielding the download statistics and information matching the
    given parameters.

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
    sta_data, codes, datacenters, downloads, networks = \
        get_dstats_html_data(session, download_ids, maxgap_threshold)
    # selected codes by default the Ok one. To know which position is in codes is a little hacky:
    selcodes = [i for i, c in enumerate(codes) if c == DownloadStats2.resp[200]]
    # downloads are all selected by default
    seldownloads = list(downloads.keys())
    seldatacenters = list(datacenters.keys())
    return get_template().render(title='Download info',
                                 sta_data_json=tojson(sta_data),
                                 codes=codes,
                                 datacenters=datacenters,
                                 downloads=downloads,
                                 selcodes_set=set(selcodes),
                                 selcodes=selcodes,
                                 seldownloads=seldownloads,
                                 seldatacenters=seldatacenters,
                                 networks=networks)
