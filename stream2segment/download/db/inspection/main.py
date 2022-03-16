"""
Module implementing the download info (print statistics and generate html page)

:date: Mar 15, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function
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

from stream2segment.io import yaml_load, Fdsnws, open2writetext
from stream2segment.io.cli import ascii_decorate
from stream2segment.io.db import close_session
from stream2segment.io.db.sqlconstructs import concat
from stream2segment.io.inputvalidation import validate_param, BadParam
from stream2segment.download.db import get_session
from stream2segment.download.db.models import Download, Segment, DataCenter, Station
from stream2segment.download.modules.utils import EVENTWS_SAFE_PARAMS, DownloadStats


def summary(dburl, download_indices=None, download_ids=None, outfile=None):
    """Show/print/return short download summary

    :param download_indices: download indices as slice, int, List[int] or str convertible
        to int or slice ("<start>:<stop>" or "<start>:<stop>:<step>"). None, the default
        when missing, will consider all indices
    :param download_ids: download ids (list of int), will be merged with
        `download_indices`. None, the default when missing, will consider all ids
    :param outfile: str to a local file, or any output supported, e.g. `sys.stderr`
        When missing or None it defaults to `sys.stdout`
    """
    _get_download_info(DSummary(), dburl, download_indices, download_ids,
                       False, outfile)


def log(dburl, download_indices=None, download_ids=None, outfile=None):
    """Show/print/return the config and/or log of the given download. If download ids
    and download indices are both Nones, shows stats for all downloads

    :param download_indices: download indices as slice, int, List[int] or str convertible
        to int or slice ("<start>:<stop>" or "<start>:<stop>:<step>"). None, the default
        when missing, will consider all indices
    :param download_ids: download ids (list of int), will be merged with
        `download_indices`. None, the default when missing, will consider all ids
    :param outfile: str to a local file, or any output supported, e.g. `sys.stderr`
        When missing or None it defaults to `sys.stdout`
    """
    _get_download_info(DLog(), dburl, download_indices, download_ids, False, outfile)


def config(dburl, download_indices=None, download_ids=None, outfile=None):
    """Show/print/return the config and/or log of the given download. If download ids
    and download indices are both Nones, shows stats for all downloads

    :param download_indices: download indices as slice, int, List[int] or str convertible
        to int or slice ("<start>:<stop>" or "<start>:<stop>:<step>"). None, the default
        when missing, will consider all indices
    :param download_ids: download ids (list of int), will be merged with
        `download_indices`. None, the default when missing, will consider all ids
    :param outfile: str to a local file, or any output supported, e.g. `sys.stderr`
        When missing or None it defaults to `sys.stdout`
    """
    _get_download_info(DConfig(), dburl, download_indices, download_ids, False, outfile)


def stats(dburl, download_indices=None, download_ids=None, maxgap_threshold=0.5,
          html=False, outfile=None):
    """Create a diagnostic html page (or text string) showing the status of the
    download. If download ids and download indices are both Nones, shows stats for all
    downloads

    :param download_indices: download indices as slice, int, List[int] or str convertible
        to int or slice ("<start>:<stop>" or "<start>:<stop>:<step>"). None, the default
        when missing, will consider all indices
    :param download_ids: download ids (list of int), will be merged with
        `download_indices`. None, the default when missing, will consider all ids
    :param maxgap_threshold: the max gap threshold (float)
    """
    _get_download_info(DStats(maxgap_threshold), dburl, download_indices, download_ids,
                       html, outfile)


def _get_download_info(info_generator, dburl, download_indices=None, download_ids=None,
                       html=False, outfile=None):
    """Process dreport or dstats"""
    # create the session by raising a BadParam (associated to the name 'dburl') in case:
    session = validate_param('dburl', dburl, get_session)
    try:
        download_ids = get_download_ids(session, download_indices, download_ids)

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
            # threading.Timer(1, lambda: sys.exit(0)).start()
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
                printed = False
                for line in itr:
                    printed = True
                    print(line, file=sys.stdout if not outfile else outfile)
                # if we are printing onn screen, show if nothing could be printed:
                if outfile in (None, sys.stdout) and not printed:
                    print('Nothing to show', file=sys.stderr)
    finally:
        close_session(session)


def get_download_ids(session, download_indices=None, download_ids=None):
    """Return a list of download ids from the given arguments, or None. Calling functions
    must interpret None as "all ids"

    :param download_indices: download indices as slice, int, List[int] or str convertible
        to int or slice ("<start>:<stop>" or "<start>:<stop>:<step>"). None, the default
        when missing, will consider all indices
    :param download_ids: download ids (list of int), will be merged with
        `download_indices`. None, the default when missing, will consider all ids
    """
    if not download_indices:
        return download_ids

    # now download indices is given, and will be merged into download_ids. Thus:
    download_ids = [] if not download_ids else list(download_ids)

    def raise_bad_param():
        raise BadParam("Invalid download indices", "", str(download_indices),
                       param_quote='')

    d_indices = download_indices
    if isinstance(d_indices, str) and ':' in d_indices:
        # parse as slice:
        try:
            start, stop, step = None, None, None
            _ = str(download_indices).split(':')
            if len(_) == 2:
                start = None if not _[0] else int(_[0])
                stop = None if not _[1] else int(_[1])
            elif len(_) == 3:
                start = None if not _[0] else int(_[0])
                stop = None if not _[1] else int(_[1])
                step = None if not _[2] else int(_[2])
            else:
                raise_bad_param()
        except Exception as exc:  # noqa
            raise_bad_param()
        d_indices = slice(start, stop, step)

    if not isinstance(d_indices, slice):
        try:
            d_indices = [int(download_indices)]
        except (ValueError, TypeError):
            try:
                d_indices = [int(_) for _ in download_indices]
            except Exception:  # noqa
                raise_bad_param()

    d_ids = [_[0] for _ in query_download_data(session, sort='asc')]
    if isinstance(d_indices, slice):
        download_ids.extend(_ for _ in d_ids[d_indices] if _ not in download_ids)
    # If d_indices is list, let's pass IndexError(s), as it happens for download ids not
    # in the db. So things are slightly more complex:
    for i in d_indices:
        try:
            did = d_ids[i]
            if did not in download_ids:
                download_ids.append(did)
        except IndexError:
            pass

    return download_ids or None


class _InfoGenerator(object):
    """Base class for any subclasses returning Download info in text and html
    content. Subclasses should overwrite `self.str_iter` and
    `self.html_template_arguments`
    """

    def str_iter(self, session, download_ids=None):
        """Return an iterator yielding chunks of strings denoting the string
        representation of this object"""
        pass

    def html(self, session, download_ids=None):
        """Return a string with the html representation of this object"""
        args = self.html_template_arguments(session, download_ids)
        args.setdefault('title', self.__class__.__name__)
        return self.get_template().render(**args)

    def html_template_arguments(self, session, download_ids=None):
        """Subclasses should return here a dict to be passed as arguments to
        the jinja2 template`
        """
        return {}

    @classmethod
    def get_template(cls):
        """Return the jinja2 template for this object
        The html file must be an existing file a file with name
        `self.__class__.__name__.lower() + ',html'
        """
        thisdir = os.path.dirname(__file__)
        templatespath = os.path.join(thisdir, 'templates')
        csspath = os.path.join(thisdir, 'static', 'css')
        jspath = os.path.join(thisdir, 'static', 'js')
        env = Environment(loader=FileSystemLoader([templatespath, jspath, csspath]))
        return env.get_template('%s.html' % cls.__name__.lower())


class DSummary(_InfoGenerator):
    """Class handling the generation of download reports in text format (no html
    supported for the moment)
    """

    def str_iter(self, session, download_ids=None):
        """Returns an iterator yielding chunks of strings denoting the string
        representation of this object
        """
        header = ('Download id', 'Execution time', 'Index')
        lengths = [len(header[0]), 19, len(header[2])]
        for i, (did, dtime) in enumerate(query_download_data(session,
                                                             attrs=(Download.id,
                                                                    Download.run_time),
                                                             sort='asc')):
            # We did not filter by download ids because we want to show the download
            # index. Thus query all download ids with `enumerate`, and filter now:
            if not download_ids or did in download_ids:
                if header:
                    yield '  '.join(_1.rjust(_2) for _1, _2 in zip(header, lengths))
                    header = None
                yield '  '.join([str(did).rjust(lengths[0]),
                                dtime.replace(microsecond=0).isoformat(),
                                str(i).rjust(lengths[2])])

    def html_template_arguments(self, session, download_ids=None):
        """Returns a dict to be passed as arguments to
        the jinja2 template"""
        raise Exception('html version not available')


class DLog(_InfoGenerator):
    """Class handling the generation of download config(s) in YAML format"""

    def str_iter(self, session, download_ids=None):
        """Returns an iterator yielding chunks of strings denoting the string
        representation of this object"""
        qry = query_download_data(session,
                                  (Download.id, Download.run_time, Download.log))
        if download_ids is not None:
            qry = qry.filter(Download.id.in_(download_ids))
        for dwnl_id, dwnl_time, log_text in qry:
            yield ascii_decorate('Download id: %d (%s)' % (dwnl_id, str(dwnl_time)))
            yield log_text or ''
            # when the log ends with an exception, on the terminal it looks like the
            # exception is raise, i.e. there is a program error. Provide an end tag to
            # make the distinction clear:
            yield "[Log file end]"
            yield ''

    def html_template_arguments(self, session, download_ids=None):
        """Returns a dict to be passed as arguments to
        the jinja2 template"""
        raise Exception('html version not available')


class DConfig(_InfoGenerator):
    """Class handling the generation of download config(s) in YAML format"""

    def str_iter(self, session, download_ids=None):
        """Returns an iterator yielding chunks of strings denoting the string
        representation of this object"""
        qry = query_download_data(session,
                                  (Download.id, Download.run_time, Download.config))
        if download_ids is not None:
            qry = qry.filter(Download.id.in_(download_ids))
        for dwnl_id, dwnl_time, text in qry:
            yield ascii_decorate('Download id: %d (%s)' % (dwnl_id, str(dwnl_time)), '#')
            yield text or ''
            yield ''

    def html_template_arguments(self, session, download_ids=None):
        """Returns a dict to be passed as arguments to
        the jinja2 template"""
        raise Exception('html version not available')


def query_download_data(session, attrs=(Download.id,), sort=None):
    qry = session.query(*attrs)
    if sort == 'desc':
        qry = qry.order_by(Download.run_time.desc())
    elif sort == 'asc':
        qry = qry.order_by(Download.run_time.asc())
    return qry


class DStats(_InfoGenerator):
    """Class handling the generation of download statistics in text and html format"""

    def __init__(self, maxgap_threshold=0.5):
        self.maxgap_threshold = maxgap_threshold

    def str_iter(self, session, download_ids=None):
        """Returns an iterator yielding chunks of strings denoting the string
        representation of this object"""
        return get_dstats_str_iter(session, download_ids, self.maxgap_threshold)

    def html_template_arguments(self, session, download_ids=None):
        """Returns a dict to be passed as arguments to
        the jinja2 template"""
        return get_dstats_html_template_arguments(session, download_ids,
                                                  self.maxgap_threshold)


def get_dstats_str_iter(session, download_ids=None, maxgap_threshold=0.5):
    """Return an iterator yielding the download statistics and information
    matching the given parameters.
    The returned string can be joined and printed to screen or file and is
    made of tables showing the segment data on the db per data-center and
    download run, plus some download information.

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics
        from all downloads run. Otherwise limit the output to the downloads
        whose ids are in the list. In any case, in case of more download runs
        to be considered, this function will yield also the statistics
        aggregating all downloads in a table at the end
    :param maxgap_threshold: (float, default 0.5). The threshold whereby a
        segment is to be considered with gaps or overlaps. By default is 0.5,
        meaning that a segment whose
        'maxgap_numsamples' value is > 0.5 has gaps, and a segment whose
        'maxgap_numsamples' value is < 0.5 has overlaps. Such segments will be
        marked with a special class 'OK Gaps Overlaps' in the table columns.
    """
    # Benchmark: the bare minimum (with postgres on external server) request
    # takes around 12 sec and 14 seconds adding all necessary information.
    # Therefore, we choose the latter
    maxgap_bexpr = get_maxgap_sql_expr(maxgap_threshold)
    qry = session.query(func.count(Segment.id),
                        Segment.download_code,
                        Segment.datacenter_id,
                        Segment.download_id,
                        maxgap_bexpr)

    data = filterquery(qry, download_ids).group_by(Segment.download_id,
                                                   Segment.datacenter_id,
                                                   Segment.download_code,
                                                   maxgap_bexpr)

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
        if evparamlen is None and evtparams:  # get evparamlen for str. align.
            evparamlen = max(len(_) for _ in evtparams)
        for param in sorted(evtparams):
            yield ("  %-{:d}s = %s".format(evparamlen)) % (param, str(evtparams[param]))
        yield ''
        statz = stas.get(did)
        if statz is None:
            yield "No segments downloaded"
        else:
            yield ("Downlaoaded segments per data center url (row) "
                   "and response type (column):")
            yield ""
            yield str(statz)

    if show_aggregate_stats:
        yield ''
        yield ''
        yield ascii_decorate('Aggregated stats (all downloads)')
        yield ''
        yield str(agg_statz)


def get_dstats_html_template_arguments(session, download_ids=None, maxgap_threshold=0.5):
    """Return an html page (string) yielding the download statistics and
    information matching the given parameters.

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics
        from all downloads run. Otherwise limit the output to the downloads
        whose ids are in the list. In any case, in case of more download runs
        to be considered, this function will yield also the statistics
        aggregating all downloads in a table at the end
    :param maxgap_threshold: (float, default 0.5). The threshold whereby a
        segment is to be considered with gaps or overlaps. By default is 0.5,
        meaning that a segment whose 'maxgap_numsamples' value is > 0.5 has
        gaps, and a segment whose 'maxgap_numsamples' value is < 0.5 has
        overlaps. Such segments will be marked with a special class
        'OK Gaps Overlaps' in the table columns.
    """
    sta_data, codes, datacenters, downloads, networks = \
        get_dstats_html_data(session, download_ids, maxgap_threshold)
    # selected codes by default the Ok one. To know which position is
    # in codes is a little hacky:
    selcodes = [i for i, c in enumerate(codes)
                if list(c) == list(DownloadStats2.resp[200])[:2]]
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


def tojson(obj):
    """Convert obj to json formatted string without whitespaces to minimize
    string size
    """
    return json.dumps(obj, separators=(',', ':'))


def yaml_get(yaml_content):
    """Return the arguments used for the eventws query stored in the yaml,
    or an empty dict in case of errors

    :param yaml_content: yaml formatted string representing a download config
    """
    try:
        dic = yaml_load(StringIO(yaml_content))
        ret = {k: dic[k] for k in EVENTWS_SAFE_PARAMS if k in dic}
        additional_eventws_params = dic.get('eventws_query_args', None) or {}
        ret.update(additional_eventws_params)
        return ret
    except Exception as _:  # pylint: disable=broad-except
        return {}


def get_downloads(sess, download_ids=None):
    """Returns a dict of download ids mapped to the tuple
    (download_run_time, download_eventws_query_args)
    the first element is a string, the second a dict
    """
    query = filterquery(sess.query(Download.id, Download.run_time, Download.config),
                        download_ids).order_by(Download.run_time.asc())
    return {did: (time.isoformat(), yaml_get(cfg))
            for (did, time, cfg) in query}


def filterquery(query, download_ids=None):
    """Add a filter to the given query if download_ids is not None, and return
    a new query. Otherwise, if download_ids is None, it's no-op and returns
    query itself
    """
    if download_ids is not None:
        query = query.filter(Segment.download_id.in_(download_ids))
    return query


def get_datacenters(sess, dc_ids=None):
    """Return a dict of datacenters id mapped to the network location of their
    url
    """
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
    """Return a SALAlchemy binary expression which matches segments with
    gaps/overlaps, according to the given threshold
    """
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
    """Return the tuple
        sta_list, codes, datacenters, downloads, networks

    where:
    - sta_list is a list stations data and their download codes (together with
      the number of segments downloaded and matching the given code)
    - codes is a list of tuples (title, legend) representing the titles and
      legends of all download codes found
    - datacenters the output of `get_datacenters`
    - downloads is the output of `get_downloads`
    - networks is a list of strings denoting the networks found

    The returned data is used to build the html page showing the download
    info / statistics. All returned elements will be basically injected as JSON
    string in the html page and processed therein by the browser with a js
    library also injected in the html page.

    :param session: an sql-alchemy session denoting a db session to a database
    :param download_ids: (list of ints or None) if None, collect statistics
        from all downloads run. Otherwise limit the output to the downloads
        whose ids are in the list. In any case, in case of more download runs
        to be considered, this function will yield also the statistics
        aggregating all downloads in a table at the end
    :param maxgap_threshold: (float, default 0.5) the threshold whereby a
        segment is to be considered with gaps or overlaps. By default is 0.5,
        meaning that a segment whose 'maxgap_numsamples' value is > 0.5 has
        gaps, and a segment whose 'maxgap_numsamples' value is < 0.5 has
        overlaps. Such segments will be marked with a special class
        'OK Gaps Overlaps' in the table columns.
    """
    # Benchmark: the bare minimum (with postgres on external server) request
    # takes around 12 sec and 14 seconds adding all necessary information.
    # Therefore, we choose the latter
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
        sta_list = sta_data.get(staname, [staid,
                                          round(lat, 2), round(lon, 2),
                                          dc_id,
                                          netindex,
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

    # In the html, we want to reduce all possible data, as the file might be
    # huge. Modify hereafter `sta_data` into a list `sta_list` of this form:
    # sta_list = [sta_name, [staid, stalat, stalon, sta_dcid, sta_net_index,
    #                        d_id1, [code1, num_seg1 , ..., codeN, num_seg],
    #                        d_id2, [code1, num_seg1 , ..., codeN, num_seg],
    #                       ],
    #            ...,
    #            ]
    # and keep a separate list `codes` that maps uses codes to titles and
    # legends (also note in case of size problems: JavaScript objects keys are
    # always strings, thus int keys will be rendered with unnecessary quotes
    # taking up space)
    sta_list = []
    sortedcodes = DownloadStats2.sortcodes(codesfound)
    codeint = {k: i for i, k in enumerate(sortedcodes)}
    for staname, values in viewitems(sta_data):
        staname = staname.split('.')[1]
        dwnlds = values.pop()  # remove last element
        for did, segs in viewitems(dwnlds):
            values.append(did)
            values.append([item for code in segs
                           for item in (codeint[code], segs[code])])
        sta_list.append(staname)
        sta_list.append(values)

    codes = [DownloadStats2.titlelegend(code) for code in sortedcodes]
    networks = sorted(networks, key=lambda key: networks[key])
    return sta_list, codes, get_datacenters(session, list(dcidsfound) or None), \
        get_downloads(session), networks
