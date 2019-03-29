'''
Core functionalities for the GUI web application (processing)

:date: Jul 31, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, zip

import re
from itertools import cycle
import json

from sqlalchemy import func

from stream2segment.io.db.models import Segment, Class, Station, ClassLabelling, Download
from stream2segment.io.db.sqlevalexpr import exprquery, Inspector
from stream2segment.gui.webapp.mainapp.plots.core import getseg
from stream2segment.gui.webapp.mainapp.plots.jsplot import jsontimestamp


NPTS_WIDE = 900  # FIXME: automatic retrieve by means of Segment class relationships?
NPTS_SHORT = 900  # FIXME: see above


def get_segments(session, conditions, orderby, metadata, classes):
    classes = get_classes(session) if classes else []
    _metadata = []
    if metadata:
        _metadata = [[n, t, conditions.get(n, '')] for n, t in get_metadata(session)]
    # parse the orderby if it has a minus at the end it's descending:
    oby = orderby if not orderby else \
        [(k, "asc") if not k[-1] == '-' else (k[:-1], "desc") for k in orderby]
    qry = query4gui(session, conditions=conditions, orderby=oby)
    return {'segment_ids': [seg[0] for seg in qry],
            'classes': classes,
            'metadata': _metadata}


def query4gui(session, conditions, orderby=None):
    '''Returns a query yielding the segments ids for the visualization in the GUI (processing)
    according to `conditions` and `orderby`, sorted by default (if orderby is None) by
    segment's event.time (descending) and then segment's event_distance_deg (ascending)

    :param session: the sql-alchemy session
    :param condition: a dict of segment attribute names mapped to a select expression, each
    identifying a filter (sql WHERE clause). See `:ref:sqlevalexpr.py`. Can be empty (no filter)
    :param orderby: if None, defaults to segment's event.time (descending) and then
    segment's event_distance_deg (ascending). Otherwise, a list of tuples, where the first
    tuple element is a segment attribute (in string format) and the second element is either 'asc'
    (ascending) or 'desc' (descending)
    :return: a query yielding the tuples: ```(Segment.id)```
    '''
    if orderby is None:
        orderby = [('event.time', 'desc'), ('event_distance_deg', 'asc')]
    return exprquery(session.query(Segment.id), conditions=conditions, orderby=orderby,
                     distinct=True)


def get_metadata(session, seg_id=None):
    '''Returns a list of tuples (column, column_type) if `seg_id` is None or
    (column, column_value) if segment is not None. In the first case, `column_type` is the
    string representation of the column python type (str, datetime,...), in the latter,
    it is the value of `segment` for that column'''
    excluded_colnames = set([Station.inventory_xml, Segment.data, Download.log,
                             Download.config, Download.errors, Download.warnings,
                             Download.program_version, Class.description])

    segment = None
    if seg_id is not None:
        # exclude all classes attributes (returned in get_classes):
        excluded_colnames |= {Class.id, Class.label}
        segment = getseg(session, seg_id)
        if not segment:
            return []
    
    insp = Inspector(segment or Segment)
    attnames = insp.attnames(Inspector.PKEY | Inspector.QATT | Inspector.REL | Inspector.COL,
                             sort=True, deep=True, exclude=excluded_colnames)
    if seg_id is not None:
        # return a list of (attribute name, attribute value)
        return [(_, insp.attval(_)) for _ in attnames]
    # return a list of (attribute name, str(attribute type))
    return [(_, getattr(insp.atttype(_), "__name__"))
            for _ in attnames if insp.atttype(_) is not None]


def set_class_id(session, segment_id, class_id, value):
    segment = getseg(session, segment_id)
    annotator = 'web app labeller'  # in the future we might use a session or computer username
    if value:
        segment.add_classes(class_id, annotator=annotator)
    else:
        segment.del_classes(class_id)
    return {}


def get_classes(session, seg_id=None):
    '''If seg_id is not None, returns a list of the segment class ids.
    Otherwise, a list of dicts where each dict is a db row in the form
    {table_column: row_value}. The dict will have also a "count" attribute
    denoting how many segments have that class set'''
    if seg_id is not None:
        segment = getseg(session, seg_id)
        return [] if not segment else sorted(c.id for c in segment.classes)

    colnames = [Class.id.key, Class.label.key, 'count']
    # Note isouter which produces a left outer join, important when we have no class labellings
    # (i.e. third column all zeros) otherwise with a normal join we would have no results
    data = session.query(Class.id, Class.label, func.count(ClassLabelling.id).label(colnames[-1])).\
        join(ClassLabelling, ClassLabelling.class_id == Class.id, isouter=True).group_by(Class.id).\
        order_by(Class.id)
    return [{name: val for name, val in zip(colnames, d)} for d in data]


def get_segment_data(session, seg_id, plotmanager, plot_indices, all_components, preprocessed,
                     zooms, metadata=False, classes=False, config=False):
    """Returns the segment data, depending on the arguments
    :param session: a flask sql-alchemy session object
    :param seg_id: integer denoting the segment id
    :param plotmanager: a PlotManager object, storing all plots data and sn/windows data
    :param plot_indices: a list of plots to be calculated from the given `plotmanager` (which caches
    its plot for performance speed)
    :param all_components: boolean, whether or not the `plotmanager` should give all components for
    the main plot (plot representing the given segment's data, whose plot index is currently 0).
    Ignored if 0 is not in `plot_indices`
    :param preprocessed: boolean, whether or not the `plotmanager` should calculate the plots on
    the pre-processing function defined in the config (if any), or on the raw obspy Stream
    :param zooms: a list of **all plots** defined in the plotmanager, or None.
    Each element is either None, or a tuple of [xmin, xmax] values (xmin and xmax can be both None,
    to conform python slicing behaviour). Thus, the length of `zooms` most likely differs from
    that of `plot_indices`. the zooms of interest are, roughly speaking,
    [zooms[i] for i in plot_indices] (if zoom is not None)
    :param metadata: boolean, whether or not to return a list of the segment metadata. The list
    is a list of tuples ('column', value). A list is used to preserve order for client-side
    javascript parsing
    :param classes: boolean, whether to return the integers classes ids (if any) of the given
    segment
    :param sn_wdws: boolean, whether to returns the sn windows calculated according to the
    config values. The returned list is a 2-element list, where each element is in turn a
    2-element numeric list: [noise_window_start, noise_window_end],
    [signal_window_start, signal_window_end]
    """
    plots = []
    zooms_ = parse_zooms(zooms, plot_indices)
    sn_windows = []
    if config:
        # set_sn_windows(self, session, a_time_shift, signal_window):
        plotmanager.update_config(**deflatten_dict(config, parsevals=True))

    if plot_indices:
        plots = plotmanager.get_plots(session, seg_id, plot_indices, preprocessed, all_components)
        try:
            # return always sn_windows, as we already calculated them. IT is better
            # to call this method AFTER get_plots_func defined above
            sn_windows = [sorted([jsontimestamp(x[0]), jsontimestamp(x[1])])
                          for x in plotmanager.get_data(seg_id, 'sn_windows',
                                                        preprocessed, [])]
        except Exception:
            sn_windows = []

    return {'plots': [p.tojson(z, NPTS_WIDE) for p, z in zip(plots, zooms_)],
            'plot_types': [p.is_timeseries for p in plots],
            'sn_windows': sn_windows,
            'metadata': [] if not metadata else get_metadata(session, seg_id),
            'classes': [] if not classes else get_classes(session, seg_id)}


def parse_zooms(zooms, plot_indices):
    '''parses the zoom received from the frontend. Basically, if any zoom is a string,
    tries to parse it to datetime
    :param zooms: a list of 2-element tuples, or None's. The elements of the tuple can be number,
    Nones or strings (in datetime format)
    :return: an iterator over zooms. Uses itertools cycle so that this method can be safely used
    with izip never estinguishing it
    '''
    if not zooms or not plot_indices:
        zooms = cycle([None, None])  # to be safe in iterations
    _zooms = []
    for plot_index in plot_indices:
        try:
            zoom = zooms[plot_index]
        except (IndexError, TypeError):
            zoom = [None, None]
        _zooms.append(zoom)
    return _zooms  # set zooms to None if length is not enough


def flatten_dict(config, prefix=''):
    '''flattens a dict, returning a one level dict where nested keys are joined with the dot
    Example:
    flatten_dict({'a': {'a1':5, 'a2': 6}, 'b': 7}) = {'a.a1':5, 'a.a2': 6, 'b': 7}
    '''
    ret = {}

    def concat(prefix, key):
        '''concat function for nested keys'''
        return key if not prefix else '%s.%s' % (prefix, key)

    for key, val in config.items():
        if isinstance(val, dict):
            ret.update(flatten_dict(val, concat(prefix, key)))
        else:
            ret[concat(prefix, key)] = val
    return ret


def deflatten_dict(dic, parsevals=True):
    '''de-flattens a dict, nesting dicts for those keys with dot, using the dot as separator.

    Example:
    flatten_dict({'a.a1':'5 6', 'a.a2': 6, 'b': 7}) = {'a': {'a1':[5, 6], 'a2': 6}, 'b': 7}

    :param parsevals: boolean (default True), parses each dict key using json and being flexible
    about how arrays can be input (i.e., wothout brakets, or with spaces as separators).
    Defaults to true as `dic` might be returned form a browser where currently arrays are
    rendered in the input box without brakets, and thus might be returned here as they are
    '''
    ret = {}
    for key, val in dic.items():
        if parsevals:
            val = parse_inputtag_value(val)
        keys = key.split('.')
        dic2add = ret
        for kkk in keys[:-1]:
            if kkk not in dic2add:
                dic2add[kkk] = {}
            dic2add = dic2add[kkk]
        dic2add[keys[-1]] = val
    return ret


def parse_inputtag_value(string):
    '''Tries to parse string into a python object guessing it and running
    some json loads functions. `string` is supposed to be returned from the browser
    where angular converts arrays to a list of elements separated by comma without enclosing
    brackets. This method first tries to load string as json. If it does not succeed, it checks
    for commas: if any present, and the string does not start nor ends with square brakets,
    it inserts the brakets and tries to run again json.loads. If it fails, splits the
    string using the comma ',' and returns an array of strings. This makes array of complex
    numbers and date-time returning the correct type (lists, it is then the caller responsible
    of parsing them), at least most likely as they were input in the yaml.
    If nothing succeeds, then string is returned
    '''
    string = string.strip()
    try:
        return json.loads(string)
    except:  #  @IgnorePep8 pylint: disable=bare-except
        if ',' in string:
            if string[:1] != '[' and string[-1:] != ']':
                try:
                    return json.loads('[%s]' % string)
                except:  # @IgnorePep8 pylint: disable=bare-except
                    pass
            return [str_.strip() for str_ in string.split(',')]
        return string
