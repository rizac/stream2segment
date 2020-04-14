'''
Core functionalities for the GUI web application (processing)

:date: Jul 31, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, zip

from itertools import cycle
from io import StringIO
import json

import numpy as np
from sqlalchemy import func

from stream2segment.io.db.models import Segment, Station, Download, Class,\
    ClassLabelling
from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load
from stream2segment.io.db.sqlevalexpr import exprquery, Inspector
from stream2segment.gui.webapp.mainapp.plots.core import getseg, PlotManager
from stream2segment.gui.webapp.mainapp.plots.jsplot import isoformat
import yaml


NPTS_WIDE = 900  # FIXME: automatic retrieve by means of Segment class relationships?
NPTS_SHORT = 900  # FIXME: see above
# LIMIT = 50


# The variables below are actually not a great idea for production. We should
# investigate some persistence storage (not during a request, but during all app
# lifecycle until it's kind of "closed"). This is a typical tpoic for which
# the web is full of untested / unexplained dogmas (do not use of globals,
# memcache or db for persistent cache, thread/process safety issues) and thus
# just keep in mind: if you want to use this app for production, you might need to
# get your hands dirty...
PLOT_MANAGER = None
SEG_IDS = []  # numpy array of segments ids (for better storage): filled with NaNs,
# populated on demand witht the block below:
SEG_QUERY_BLOCK = 50


def create_plot_manager(pyfile, configfile):
    pymodule = None if pyfile is None else load_source(pyfile)
    configdict = {} if configfile is None else yaml_load(configfile)
    global PLOT_MANAGER  # pylint: disable=global-statement
    PLOT_MANAGER = PlotManager(pymodule, configdict)
    return PLOT_MANAGER


def get_plot_manager():
    return PLOT_MANAGER


def init(session, orderby, metadata, classes):
    classes = get_classes(session) if classes else []
    _metadata = get_metadata(session) if metadata else []
    # qry = query4gui(session, conditions=conditions, orderby=None)
    return {'classes': classes,
            'metadata': _metadata}


def get_config(asstr=False):
    '''Returns the current config as YAML formatted string (if `asstr` is True)
    or as dict. In the former case, the parameter 'segment_select' is not
    included, becaue the configuration is itnended to be displayed in a
    browser editor (and the 'segment selection is handled separately in
    another form dialog)
    '''
    config_dict = get_plot_manager().config or {}
    if not asstr:
        return config_dict
    config_dict = dict(config_dict)
    config_dict.pop('segment_select', None)
    sio = StringIO()
    try:
        yaml.safe_dump(config_dict, sio, default_flow_style=False,
                       sort_keys=False)
    except TypeError:  # in case yaml version is not >= 5.1:
        yaml.safe_dump(config_dict, sio, default_flow_style=False)
    return sio.getvalue()


def validate_config_str(string_data):
    '''Validates the YAML formatted string and returns the corresponding
    Python dict. Raises ValueError if 'segment_select' is in the parsed config
    (there is a dedicated button in the page)
    '''
    sio = StringIO(string_data)
    ret = yaml.safe_load(sio.getvalue())
    if 'segment_select' in ret:
        raise ValueError('invalid segment_select parameter: use the dedicated button')
    return ret


def get_segments_count(session, conditions):
    num_segments = _query4gui(session.query(func.count(Segment.id)), conditions).scalar()
    if num_segments > 0:
        global SEG_IDS  # pylint: disable=global-statement
        SEG_IDS = np.full(num_segments, np.nan)
    return num_segments


def get_segment_id(session, seg_index):
    if np.isnan(SEG_IDS[seg_index]):
        # segment id not queryed yet: load chunks of segment ids:
        # Note that this is the best compromise between
        # 1) Querying by index, limiting by 1 and keeping track of the
        # offset: FAST at startup, TOO SLOW for each segment request
        # 2) Load all ids at once at the beginning: TOO SLOW at startup, FAST for each
        # segment request
        # (fast and slow refer to a remote db with 10millions row without config
        # and pyfile)
        limit = SEG_QUERY_BLOCK
        offset = int(seg_index / float(SEG_QUERY_BLOCK)) * SEG_QUERY_BLOCK
        limit = min(len(SEG_IDS) - offset, SEG_QUERY_BLOCK)
        segids = get_segment_ids(session,
                                 get_plot_manager().config['segment_select'] or {},
                                 offset=offset, limit=limit)
        SEG_IDS[offset:offset+limit] = segids
    return int(SEG_IDS[seg_index])


def get_segment_ids(session, conditions, limit=50, offset=0):
    # querying all segment ids is faster later when selecting a segment
    orderby = [('event.time', 'desc'), ('event_distance_deg', 'asc'),
               ('id', 'asc')]
    return [_[0] for _ in _query4gui(session.query(Segment.id),
                                     conditions, orderby).limit(limit).offset(offset)]


def _query4gui(what2query, conditions, orderby=None):
    return exprquery(what2query, conditions=conditions, orderby=orderby)


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
                     zooms, metadata=False, classes=False, config=None):
    """Returns the segment data, depending on the arguments
    :param session: a flask sql-alchemy session object
    :param seg_id: the segment id (int)
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
    :param config: a dict of new confiog values. Can be falsy to skip updating the config
    """
    plots = []
    zooms_ = parse_zooms(zooms, plot_indices)
    sn_windows = []
    if config:
        # set_sn_windows(self, session, a_time_shift, signal_window):
        plotmanager.update_config(**config)
    # segment = getseg(session, seg_id)

    if plot_indices:
        plots = plotmanager.get_plots(session, seg_id, plot_indices, preprocessed, all_components)
        try:
            # return always sn_windows, as we already calculated them. IT is better
            # to call this method AFTER get_plots_func defined above
            sn_windows = [sorted([isoformat(x[0]), isoformat(x[1])])
                          for x in plotmanager.get_data(seg_id, 'sn_windows',
                                                        preprocessed, [])]
        except Exception:
            sn_windows = []

    return {'plots': [p.tojson(z, NPTS_WIDE) for p, z in zip(plots, zooms_)],
            'seg_id': seg_id,
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
