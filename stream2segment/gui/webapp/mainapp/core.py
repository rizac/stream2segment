'''
Core functionalities for the GUI web application (processing)

:date: Jul 31, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
import os

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, zip

from itertools import cycle
from io import StringIO
import contextlib
import json

import numpy as np
import yaml
from sqlalchemy import func
from sqlalchemy.orm import load_only
from obspy import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime

from stream2segment.process import gui
# from stream2segment.io.db.models import Segment, Station, Download, Class,\
#     ClassLabelling
from stream2segment.utils import load_source, iterfuncs
from stream2segment.utils.resources import yaml_load
# from stream2segment.io.db.sqlevalexpr import Inspector
# from stream2segment.gui.webapp.mainapp.plots.core import getseg, PlotManager
from stream2segment.gui.webapp.mainapp.jsplot import Plot, isoformat
from stream2segment.gui.webapp.mainapp import db
from stream2segment.process.math.traces import sn_split

NPTS_WIDE = 900  # FIXME: automatic retrieve by means of Segment class relationships?
NPTS_SHORT = 900  # FIXME: see above
SEL_STR = 'segment_select'
# LIMIT = 50


def _escapedoc(string):
    if not string or not string.strip():
        return "No function doc found in GUI's Python file"
    for char in ('.\n', '. ', '\n\n'):
        if char in string:
            string = string[:string.index(char)]
            break
    string = string.strip()
    return string.replace('{', '&#123;').replace('}', '&#125;').replace("\"", "&quot;").\
        replace("'", '&amp;').replace("<", "&lt;").replace(">", "&gt;")

# The variables below are actually not a great idea for production. We should
# investigate some persistence storage (not during a request, but during all app
# lifecycle until it's kind of "closed"). This is a typical tpoic for which
# the web is full of untested / unexplained dogmas (do not use of globals,
# memcache or db for persistent cache, thread/process safety issues) and thus
# just keep in mind: if you want to use this app for production, you might need to
# get your hands dirty...
# PLOT_MANAGER = None
# SEG_IDS = []  # numpy array of segments ids (for better storage): filled with NaNs,
# # populated on demand witht the block below:
# SEG_QUERY_BLOCK = 50

# def create_plot_manager(pyfile, configfile):
#     pymodule = None if pyfile is None else load_source(pyfile)
#     configdict = {} if configfile is None else yaml_load(configfile)
#     global PLOT_MANAGER  # pylint: disable=global-statement
#     PLOT_MANAGER = PlotManager(pymodule, configdict)
#     return PLOT_MANAGER
# 
# 
# def get_plot_manager():
#     return PLOT_MANAGER

# Note that the use of global variables like this should be investigted
# in production (the web GUI is not intended to be used as web app in
# production for the moment):


g_config = {
    SEL_STR: {}
}


def _reset_global_config():
    g_config.clear()
    g_config[SEL_STR] = {}


def _default_preprocessfunc(*args, **kwargs):
    '''No function decorated with '@gui.preprocess'

    REAL DOC: (the string above is meaningless, it's just what we will be
    displayed on the browser if no custom process function is implemented):
    this is the default preprocess function - i.e. no-op - and might be changed
    dynamically by the user implemented one, see below)
    '''
    raise Exception("No function decorated with '@gui.preprocess'")


_preprocessfunc = _default_preprocessfunc

g_functions = [lambda segment, config: segment.stream()]

userdefined_plots = []

def _reset_global_functions():
    global _preprocessfunc
    _preprocessfunc = _default_preprocessfunc
    del g_functions[1:]
    del userdefined_plots[:]


def init(app, pyfile=None, configfile=None):
    
    if pyfile:
        _pymodule = load_source(pyfile)
        _reset_global_functions()
        for function in iterfuncs(_pymodule):
            att, pos, xaxis, yaxis = gui.get_func_attrs(function)
            if att == 'gui.preprocess':
                global _preprocessfunc
                _preprocessfunc = function
            elif att == 'gui.plot':
                userdefined_plots.append(
                    {
                        'name': function.__name__,
                        'index': len(g_functions),  # index >=1
                        'position': pos,
                        'xaxis': xaxis,
                        'yaxis': yaxis,
                        'doc': _escapedoc(function.__doc__)
                    }
                )
                g_functions.append(function)

    if configfile:
        with open(configfile) as _opn:
            newconfig = yaml.safe_load(_opn)
            _reset_global_config()
            g_config.update(newconfig)
#         if SEL_STR not in g_config:
#             g_config[SEL_STR] = {}


def has_preprocess_func():
    return _preprocessfunc is not _default_preprocessfunc


def get_func_doc(self, index=-1):
    '''Returns the documentation for the given custom function.
    :param index: if negative, returns the doc for the preprocess function, otherwise
    is the index of the i-th function (index 0 refers to the main function plotting the
    segment stream)
    '''
    if index < 0:
        return _escapedoc(getattr(_preprocessfunc, "__doc__", ''))
    return userdefined_plots[index]['doc']


def get_init_data(metadata=True, classes=True):
    classes = db.get_classes() if classes else []
    _metadata = db.get_metadata() if metadata else []
    # qry = query4gui(session, conditions=conditions, orderby=None)
    return {'classes': classes, 'metadata': _metadata}


def get_config(asstr=False):
    '''Returns the current config as YAML formatted string (if `asstr` is True)
    or as dict. In the former case, the parameter SEL_STR ('segment_select')
    is not included, becaue the configuration is itnended to be displayed in a
    browser editor (and the 'segment selection is handled separately in
    another form dialog)
    '''
    config_dict = dict(g_config)
    if not asstr:
        return config_dict
    config_dict.pop(SEL_STR, None)  # for safety
    if not config_dict:  # if dict is empty,
        # avoid returning: "{}\n", instead return emtpy string:
        return ''
    sio = StringIO()
    try:
        yaml.safe_dump(config_dict, sio, default_flow_style=False,
                       sort_keys=False)
    except TypeError:  # in case yaml version is not >= 5.1:
        yaml.safe_dump(config_dict, sio, default_flow_style=False)
    return sio.getvalue()


def validate_config_str(string_data):
    '''Validates the YAML formatted string and returns the corresponding
    Python dict. Raises ValueError if SEL_STR ('segment_select') is in the
    parsed config (there is a dedicated button in the page)
    '''
    sio = StringIO(string_data)
    ret = yaml.safe_load(sio.getvalue())
    if SEL_STR in ret:
        raise ValueError('invalid parameter %s: use the dedicated button' % 
                         SEL_STR)
    return ret

# def get_segments_count(session, conditions):
#     num_segments = _query4gui(session.query(func.count(Segment.id)), conditions).scalar()
#     if num_segments > 0:
#         global SEG_IDS  # pylint: disable=global-statement
#         SEG_IDS = np.full(num_segments, np.nan)
#     return num_segments


def get_select_conditions():
    return dict(g_config[SEL_STR])


def set_select_conditions(newdict):
    g_config[SEL_STR] = newdict

# def get_segment_id(session, seg_index):
#     if np.isnan(SEG_IDS[seg_index]):
#         # segment id not queryed yet: load chunks of segment ids:
#         # Note that this is the best compromise between
#         # 1) Querying by index, limiting by 1 and keeping track of the
#         # offset: FAST at startup, TOO SLOW for each segment request
#         # 2) Load all ids at once at the beginning: TOO SLOW at startup, FAST for each
#         # segment request
#         # (fast and slow refer to a remote db with 10millions row without config
#         # and pyfile)
#         limit = SEG_QUERY_BLOCK
#         offset = int(seg_index / float(SEG_QUERY_BLOCK)) * SEG_QUERY_BLOCK
#         limit = min(len(SEG_IDS) - offset, SEG_QUERY_BLOCK)
#         segids = get_segment_ids(session,
#                                  get_segment_select(),
#                                  offset=offset, limit=limit)
#         SEG_IDS[offset:offset+limit] = segids
#     return int(SEG_IDS[seg_index])
# 
# 
# def get_segment_ids(session, conditions, limit=50, offset=0):
#     # querying all segment ids is faster later when selecting a segment
#     orderby = [('event.time', 'desc'), ('event_distance_deg', 'asc'),
#                ('id', 'asc')]
#     return [_[0] for _ in _query4gui(session.query(Segment.id),
#                                      conditions, orderby).limit(limit).offset(offset)]

# def _query4gui(what2query, conditions, orderby=None):
#     return exprquery(what2query, conditions=conditions, orderby=orderby)


def get_segment(segment_id):
    return db.get_segment(segment_id)

# def get_metadata(seg_id=None):
#     '''Returns a list of tuples (column, column_type) if `seg_id` is None or
#     (column, column_value) if segment is not None. In the first case, `column_type` is the
#     string representation of the column python type (str, datetime,...), in the latter,
#     it is the value of `segment` for that column'''
#     excluded_colnames = set([Station.inventory_xml, Segment.data, Download.log,
#                              Download.config, Download.errors, Download.warnings,
#                              Download.program_version, Class.description])
# 
#     segment = None
#     if seg_id is not None:
#         # exclude all classes attributes (returned in get_classes):
#         excluded_colnames |= {Class.id, Class.label}
#         segment = get_segment(seg_id)
#         if not segment:
#             return []
# 
#     insp = Inspector(segment or Segment)
#     attnames = insp.attnames(Inspector.PKEY | Inspector.QATT | Inspector.REL | Inspector.COL,
#                              sort=True, deep=True, exclude=excluded_colnames)
#     if seg_id is not None:
#         # return a list of (attribute name, attribute value)
#         return [(_, insp.attval(_)) for _ in attnames]
#     # return a list of (attribute name, str(attribute type))
#     return [(_, getattr(insp.atttype(_), "__name__"))
#             for _ in attnames if insp.atttype(_) is not None]


def set_class_id(seg_id, class_id, value):
    segment = get_segment(seg_id)
    annotator = 'web app labeller'  # in the future we might use a session or computer username
    if value:
        segment.add_classes(class_id, annotator=annotator)
    else:
        segment.del_classes(class_id)
    return {}


def get_segment_id(segment_index):
    return db.get_segment_id(segment_index, get_select_conditions())


def get_segment_data(seg_id, plot_indices, all_components, preprocessed,
                     zooms, metadata=False, classes=False, config=None):
    """Returns the segment data, depending on the arguments

    :param seg_id: the segment id (int)
    :param plot_indices: a list of plots to be calculated from the given `plotmanager`
        (which caches its plot for performance speed)
    :param all_components: boolean, whether or not the `plotmanager` should give all
        components for the main plot (plot representing the given segment's data, whose
        plot index is currently 0). Ignored if 0 is not in `plot_indices`
    :param preprocessed: boolean, whether or not the `plotmanager` should calculate the
        plots on the pre-processing function defined in the config (if any), or on
        the raw obspy Stream
    :param zooms: a list of **all plots** defined in the plotmanager, or None.
        Each element is either None, or a tuple of [xmin, xmax] values (xmin and xmax can
        be both None, to conform python slicing behaviour). Thus, the length of `zooms`
        most likely differs from that of `plot_indices`. the zooms of interest are,
        roughly speaking, [zooms[i] for i in plot_indices] (if zoom is not None)
    :param metadata: boolean, whether or not to return a list of the segment metadata.
        The list is a list of tuples ('column', value). A list is used to preserve order
        for client-side javascript parsing
    :param classes: boolean, whether to return the integers classes ids (if any) of the
        given segment
    :param config: a dict of new confiog values. Can be falsy to skip updating the config
    """
    plots = []
    zooms_ = parse_zooms(zooms, plot_indices)
    sn_windows = []
    if config:
        g_config.update(**config)

    if plot_indices:
        # plots = plotmanager.get_plots(session, seg_id, plot_indices, preprocessed, all_components)
        plots = get_plots(seg_id, plot_indices, preprocessed, all_components)
        try:
            # return always sn_windows, as we already calculated them. IT is better
            # to call this method AFTER get_plots_func defined above
            sn_windows = [sorted([isoformat(x[0]), isoformat(x[1])])
                          for x in exec_func(get_segment(seg_id),
                                             preprocessed,
                                             get_sn_windows)]
        except Exception:  # pylint: disable=broad-except
            sn_windows = []

    return {
        'plots': [p.tojson(z, NPTS_WIDE) for p, z in zip(plots, zooms_)],
        'seg_id': seg_id,
        'plot_types': [p.is_timeseries for p in plots],
        'sn_windows': sn_windows,
        'metadata': [] if not metadata else db.get_metadata(seg_id),
        'classes': [] if not classes else db.get_classes(seg_id)
    }


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


def get_plots(seg_id, plot_indices, preprocessed, all_components):
    '''Returns the plots

    :param all_components: if 0 is in plot_indices, it is ingored. Otherwise
        returns in plot[I] all components (where I =
        argwhere(plot_indices == 0)
    '''
    segment = get_segment(seg_id)
    plots = [get_plot(segment, preprocessed, i) for i in plot_indices]
    if all_components and (0 in plot_indices):
        plot = plots[plot_indices.index(0)]
        for _ in segment.siblings(None, get_select_conditions(), 'id'):
            segment = get_segment(_[0])  # _[0] = segment id
            plt = get_plot(segment, preprocessed, 0)
            if plt.warnings:
                plot.warnings += list(plt.warnings)
            if plt.data:
                plot.data.extend(plt.data)
    return plots


def get_plot(segment, preprocessed, func_index):
    try:        
        plt = convert2plot(exec_func(segment, preprocessed,
                                     g_functions[func_index]))
        # set title:
        title = segment.seed_id
        func_name = str('' if func_index == 0 else
                        g_functions[func_index].__name__)
        sep = ' - ' if title and func_name else ''
        plt.title = '%s%s%s' % (title, sep, func_name)

    except Exception as exc:  # pylint: disable=broad-except
        # add dummy series (empty):
        plt = Plot('', warnings=str(exc)).add(0, 1, [])
    return plt


def exec_func(segment, preprocessed, function):
    '''
    Executes the given function, setting the internal stream
        to the preprocessed one of needed and restoring to its original
        before returning. `func` signature must be: func(segment, config)
        (config is the global g_config variable)
    '''
    with prepare_for_function(segment, preprocessed):
        return function(segment, g_config)


@contextlib.contextmanager
def prepare_for_function(segment, preprocessed=False):
    ''''''
    # side note: we might cache the stream, the preprocessed stream and so
    # on, so that each time we do not need to read por process the mseed
    # but this has no big impact. Caching ALL subplots is a pain (we already
    # tried ending up with unmaintainable code). Note however that
    # within the same web request, in case the same segment stream is needed,
    # it is cached inside the Segment object (same holds for the inventory
    # as response object as member of the Station object)
    tmpstream = None
    try:
        tmpstream = segment.stream().copy()
        if not preprocessed:
            yield
        else:
            stream = getattr(segment, '_p_p_stream', None)
            if isinstance(stream, Exception):
                raise stream
            elif stream is None:
                stream = \
                    _preprocessfunc(segment, g_config)
                if isinstance(stream, Trace):
                    stream = Stream([stream])
                elif not isinstance(stream, Stream):
                    raise Exception("The function decorated with "
                                    "'gui.preprocess' must return "
                                    "a Trace or Stream object")
                segment._p_p_stream = stream
            segment._stream = segment._p_p_stream.copy()
            yield
    except Exception as exc:
        segment._p_p_stream = exc
        raise exc
    finally:
        if tmpstream is not None:
            segment._stream = tmpstream

# @contextlib.contextmanager
# def prepare_for_function(segment, preprocessed=False):
#     tmpstream = None
#     try:
#         tmpstream = getattr(segment, '_stream', None)
#         if isinstance(tmpstream, Exception):
#             tmpstream = None
#             raise tmpstream  # pylint: disable=raising-bad-type
# 
#         if tmpstream is not None:
#             segment.stream(True)  # reload stream from bytes data, so that
#             # any function works on the unmodified source obspy Stream
#         if not preprocessed:
#             yield
#         else:
#             if not hasattr(segment, '_p_p_stream'):
#                 stream = \
#                     _preprocessfunc(segment, g_config)
#                 if isinstance(stream, Trace):
#                     stream = Stream([stream])
#                 elif not isinstance(stream, Stream):
#                     raise Exception("The function decorated with "
#                                     "'gui.preprocess' must return "
#                                     "a Trace or Stream object")
#                 segment._p_p_stream = stream
#             segment._stream = segment._p_p_stream
#             yield
#     except Exception as exc:
#         segment._p_p_stream = exc
#         raise exc
#     finally:
#         if tmpstream is not None:
#             segment._stream = tmpstream


def convert2plot(funcres):
    '''converts the result of a function to a plot. Raises if funcres is not
    in any valid format'''
    if isinstance(funcres, Plot):
            # this should be called internally when generating main plot:
        plt = funcres
    elif isinstance(funcres, Trace):
        plt = Plot.fromtrace(funcres)
    elif isinstance(funcres, Stream):
        plt = Plot.fromstream(funcres)
    else:
        labels = cycle([None])
        if isinstance(funcres, dict):
            labels = iter(funcres.keys())
            itr = iter(funcres.values())
        elif isinstance(funcres, tuple):
            itr = [funcres]  # (x0, dx, values, label_optional)
        else:
            itr = funcres  # list of mixed types above

        plt = Plot("", "")

        for label, obj in zip(labels, itr):
            if isinstance(obj, tuple):
                try:
                    x0, dx, y, label = obj
                except ValueError:
                    try:
                        x0, dx, y = obj
                    except ValueError:
                        raise ValueError(("Cannot create plot from tuple (length=%d): "
                                          "Expected (x0, dx, y) or (x0, dx, y, label)"
                                          "") % len(obj))
                plt.add(x0, dx, y, label)
            elif isinstance(obj, Trace):
                plt.addtrace(obj, label)
            elif isinstance(obj, Stream):
                for trace in obj:
                    plt.addtrace(trace, label)
            else:
                raise ValueError(("Cannot create plot from %s (length=%d): ") % 
                                 str(type(obj)))
    return plt


def get_sn_windows(segment, config):
    if len(segment.stream()) != 1:
        raise ValueError(("Unable to get sn-windows: %d traces in stream "
                          "(possible gaps/overlaps)") % len(segment.stream()))
    wndw = config['sn_windows']
    arrival_time = \
        UTCDateTime(segment.arrival_time) + wndw['arrival_time_shift']
    return sn_split(segment.stream()[0], arrival_time, wndw['signal_window'],
                    return_windows=True)

# def parse_inputtag_value(string):
#     '''Tries to parse string into a python object guessing it and running
#     some json loads functions. `string` is supposed to be returned from the browser
#     where angular converts arrays to a list of elements separated by comma without enclosing
#     brackets. This method first tries to load string as json. If it does not succeed, it checks
#     for commas: if any present, and the string does not start nor ends with square brakets,
#     it inserts the brakets and tries to run again json.loads. If it fails, splits the
#     string using the comma ',' and returns an array of strings. This makes array of complex
#     numbers and date-time returning the correct type (lists, it is then the caller responsible
#     of parsing them), at least most likely as they were input in the yaml.
#     If nothing succeeds, then string is returned
#     '''
#     string = string.strip()
#     try:
#         return json.loads(string)
#     except:  #  @IgnorePep8 pylint: disable=bare-except
#         if ',' in string:
#             if string[:1] != '[' and string[-1:] != ']':
#                 try:
#                     return json.loads('[%s]' % string)
#                 except:  # @IgnorePep8 pylint: disable=bare-except
#                     pass
#             return [str_.strip() for str_ in string.split(',')]
#         return string
