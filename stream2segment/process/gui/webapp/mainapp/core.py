"""
Core functionalities for the main GUI web application (show command)

:date: Jul 31, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import math
from itertools import cycle
import contextlib
from datetime import datetime, date, timedelta

from io import StringIO
import yaml
import numpy as np
from obspy import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime

from stream2segment.process import gui
from stream2segment.process.inspectimport import iterfuncs
from stream2segment.process.funclib.traces import sn_split
from stream2segment.io import yaml_safe_dump
from stream2segment.process.gui.webapp.mainapp import db


# Note that the use of global variables like this should be investigated
# in production (which is not the intended goal of the web GUI for the moment):

g_config = {}  # global config

g_selection = {}  # segments selection conditions

# `g_segment_ids` below is a numpy Array that caches the segments ids if `g_selection` is
# not empty in order to avoid modifying the matching segments from the GUI. E.g.,
# labelling segments while `g_selection` is configured to show unlabelled segments only
g_segment_ids = None


def _default_preprocessfunc(segment, config):
    """Default pre-process function: remove the instrumental response
    assuming output unit in m/s and water level for deconvolution = 60
    """
    s = Stream()
    inventory = segment.inventory()
    for t in segment.stream():
        t.remove_response(inventory)
        s.append(t)
    return s[0] if len(s) == 1 else s
    # raise Exception("No function decorated with '@gui.preprocess'")


# global variables (will be initialized in _reset_global_functions, see below):

_preprocessfunc = _default_preprocessfunc
g_functions = {}
userdefined_plots = {}


def _reset_global_functions():
    """mainly used for testing purposes and within the init method"""
    global _preprocessfunc
    _preprocessfunc = _default_preprocessfunc
    global g_functions
    g_functions = {"": lambda seg, cfg: seg.stream()}
    global userdefined_plots
    userdefined_plots = {}


_reset_global_functions()  # just initialize global vars


def init(app, session, pymodule=None, config=None, segments_selection=None):
    """Initialize global variables. This method must be called once
    after the Flask app has been created and before using it.

    :param session: a SQLAlchemy SCOPED session
    :param pymodule: Python module
    :param config: dict of the current configuration
    :param segments_selection: dict[str, str] of segment attributes mapped to a
        selection expression (str)
    """
    db.init(app, session)

    if pymodule:
        _reset_global_functions()
        for function in iterfuncs(pymodule):
            att, pos, xaxis, yaxis = gui.get_func_attrs(function)
            if att == 'gui.preprocess':
                global _preprocessfunc  # noqa
                _preprocessfunc = function
            elif att == 'gui.plot':
                func_name = function.__name__
                userdefined_plots[func_name] = (
                    {
                        'position': pos,
                        'layout': {  # layout object for the plotly library
                            'xaxis': xaxis,
                            'yaxis': yaxis
                        },
                        'doc': _escapedoc(function.__doc__)
                    }
                )
                g_functions[func_name] = function

    _reset_global_vars()

    if config:
        g_config.update(config)

    # if segments_selection:
        # g_selection.update(segments_selection)
    set_select_conditions(segments_selection)


def _reset_global_vars():
    """mainly used for testing purposes and within the init method"""
    g_config.clear()
    g_selection.clear()


def get_db_url(safe=True):
    return db.get_db_url(safe=safe)


def get_func_doc(function):
    """Return the documentation for the given custom function.

    :param function: a Ptyhon function. Usually, either the global variable
        `_preprocessfunc` or the values of the global fict `g_functions`
    """
    return _escapedoc(function.__doc__)


def _escapedoc(string):
    if not string or not string.strip():
        return "No function doc found in GUI's Python file"
    for char in ('.\n', '. ', '\n\n'):
        if char in string:
            string = string[:string.index(char)]
            break
    string = string.strip()
    return string.replace('{', '&#123;').replace('}', '&#125;').\
        replace("\"", "&quot;").replace("'", '&amp;').replace("<", "&lt;").\
        replace(">", "&gt;")


def get_init_data(metadata=True, classes=True):
    classes = db.get_classes() if classes else []
    _metadata = db.get_metadata() if metadata else []
    # add sel condition string to metadata:
    sel_conditions = get_select_conditions()
    metadata = [[m[0], m[1], sel_conditions.get(m[0], "")] for m in _metadata]
    # qry = query4gui(session, conditions=conditions, orderby=None)
    return {'classes': classes, 'metadata': metadata}


def get_config(as_str=False):
    """Returns the current config as YAML formatted string (if `asstr` is True)
    or as dict. The returned value does not include the segments selection,
    if given from the command line
    """
    config_dict = dict(g_config)
    if not as_str:
        return config_dict
    if not config_dict:  # if dict is empty,
        # avoid returning: "{}\n", instead return emtpy string:
        return ''
    return yaml_safe_dump(config_dict)


def validate_config_str(string_data):
    """Validates the YAML formatted string and returns the corresponding
    Python dict.
    """
    sio = StringIO(string_data)
    ret = yaml.safe_load(sio.getvalue())
    return ret


def get_select_conditions():
    """Return a dict representing the current select conditions (parameter
    'segments_selection' of the YAML file)
    """
    return dict(g_selection)


def set_select_conditions(sel_conditions=None):
    """Set a new a dict representing the current select conditions (parameter
    'segments_selection' of the YAML file)

    :param sel_conditions: a dict of new select expressions all in str format,
        or None to keep the dict as it is and just (re)compute the total number of
        segments to select. Note that if this parameter is None the internal array
        of segment ids might be updated anyway

    :return: the total number of segments to select
    """
    # Array caching the segment ids to select (or None if no selection condition is set):
    global g_segment_ids

    if sel_conditions is not None:
        g_selection.clear()
        g_selection.update(sel_conditions)
        update_segment_ids = True
    else:
        update_segment_ids = (not g_selection) != (g_segment_ids is None)

    if update_segment_ids:
        segments_count = db.get_segments_count(g_selection)
        if not g_selection:
            g_segment_ids = None
        else:
            # float32 max: np.finfo(np.float32).max
            g_segment_ids = np.full((segments_count,), np.nan, dtype=np.float32)
        return segments_count

    return len(g_segment_ids) if g_segment_ids is not None else \
        db.get_segments_count(g_selection)


def get_segment(segment_id):
    """Return  the Segment object of the given segment id"""
    return db.get_segment(segment_id)


def get_segment_id(segment_index, segment_count):
    """Return the segment id corresponding to the given segment index in
    the GUI

    :param segment_index: the segment index
    :param segment_count: the total number of segments, needed to make the db retrieval
        faster (see `db.get_segment_id`). Note that if some selection condition is set,
        then `g_segment_ids` is not None and `segment_count  equals `len(g_segment_ids)`
    """
    seg_id = np.nan if g_segment_ids is None else g_segment_ids[segment_index]  # noqa
    if np.isnan(seg_id):
        seg_id = db.get_segment_id(segment_index, segment_count, get_select_conditions())
        if g_segment_ids is not None:
            g_segment_ids[segment_index] = seg_id  # noqa
    return int(seg_id)


def set_class_id(seg_id, class_id, value):
    """Set the given class to the given segment (value=True), or removes it
    from the given segment (value=False)
    """
    segment = get_segment(seg_id)
    annotator = 'web app labeller'  # FIXME: use a session or computer username?
    if value:
        segment.add_classlabel(class_id, annotator=annotator)
    else:
        segment.del_classlabel(class_id)
    return {}


def get_segment_data(seg_id, plot_names, all_components, preprocessed,
                     zooms, attributes=False, classes=False, config=None):
    """Return the segment data, depending on the arguments

    :param seg_id: the segment id (int)
    :param plot_names: a list of plot names to be calculated. "" indicates the default
        plot
    :param all_components: boolean, whether or not the returned plots should
        include all segments components (channel orientations). Ignored if 0 is
        not in `plot_indices`
    :param preprocessed: boolean, whether or not the plot should be returned on
        the pre-processing function defined in the config (if any), or on the
        raw ObsPy Stream
    :param zooms: the plot bounds, list or None. NOT used.
        If list, each element is either None,  or a tuple of [xmin, xmax] values
        (xmin and xmax can be both None, to conform python slicing behaviour).
        If None, defaults
        to a list of [None, None] elements (one for each plot)
    :param attributes: boolean, whether or not to return a list of the segment
        metadata. The list is a list of tuples ('column', value). A list is
        used to preserve order for client-side javascript parsing
    :param classes: boolean, whether to return the integers classes ids (if
        any) of the given segment
    :param config: a dict of new config values. Can be falsy to skip updating
        the config
    """
    plots = []
    if zooms is None and plot_names:
        zooms = [(None, None) for _ in plot_names]
    sn_windows = []
    if config:
        g_config.update(**config)

    # if plot_indices:
    #     plots = get_plots(seg_id, plot_indices, preprocessed, all_components, zooms_)
    #     try:
    #         # return always sn_windows, as we already calculated them. It is
    #         # better to call this method AFTER get_plots_func defined above
    #         sn_windows = [sorted([_jsonify(x[0]), _jsonify(x[1])])
    #                       for x in exec_func(get_segment(seg_id),
    #                                          preprocessed,
    #                                          get_sn_windows)]
    #     except Exception:  # pylint: disable=broad-except
    #         sn_windows = []
    if plot_names:
        plots = get_plots(seg_id, plot_names, preprocessed, all_components, zooms)

    return {
        'plots': plots,
        # 'plots': [p.tojson(z, NPTS_WIDE) for p, z in zip(plots, zooms_)],
        'seg_id': seg_id,
        # 'plot_types': [p.is_timeseries for p in plots],
        'sn_windows': sn_windows,
        'attributes': [] if not attributes else db.get_metadata(seg_id),
        'classes': [] if not classes else db.get_classes(seg_id)
    }


def get_plots(seg_id, plot_names, preprocessed, all_components, zooms):
    """Return the plots

    :param all_components: if 0 is not in plot_indices, it is ignored.
        Otherwise returns in plot[I] all components
        (where I = argwhere(plot_indices == 0)
    :param zooms: list of x bounds to zoom or None, one for each plot (not used)
    """
    segment = get_segment(seg_id)
    plots = {}
    for name in plot_names:
        zoom = None
        plot = get_plot(segment, preprocessed, name, zoom)
        if not name and all_components and isinstance(plot, list):
            for seg in segment.siblings(include_self=False):
                plt = get_plot(seg, preprocessed, name, zoom)
                if isinstance(plt, str):
                    plot = plt
                    break
                else:
                    plot.extend(plt)
        plots[name] = plot
    return plots


def get_plot(segment, preprocessed, func_name, zoom):
    """Return a jsplot.Plot object corresponding to the given function
    applied on the given segment

    :param segment: a Segment instance
    :param preprocessed: boolean, whether the function has to be applied on
        the pre-processed trace of the segment
    :param func_name: the name of the function to be called. It is one
        implemented in the python module with the relative decorator, and must
        have signature: func(segment, config). "" denotes the default function
        (just print print the trace)
    """
    try:
        func = exec_func(segment, preprocessed, g_functions[func_name])
        return convert2plotly(func, zoom)
    except Exception as exc:
        return 'Error ' + str(exc)


def exec_func(segment, preprocessed, function):
    """Execute the given function, setting the internal stream
    to the preprocessed one of needed and restoring to its original
    before returning. `func` signature must be: func(segment, config)
    (config is the global g_config variable)
    """
    with prepare_for_function(segment, preprocessed):
        return function(segment, g_config)


@contextlib.contextmanager
def prepare_for_function(segment, preprocessed=False):
    """contextmanager to be used before applying a custom function on a
    segment
    """
    # side note: we might cache the stream, the preprocessed stream and so
    # on, so that each time we do not need to read or process the miniSEED,
    # but this has no big impact. Caching ALL subplots is a pain (we
    # already tried ending up with unmaintainable code). Note however that
    # within the same request timespan, in case the same segment stream is
    # needed, it is cached inside the Segment object (same holds for the
    # segment's inventory (obspy Response object)
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


def convert2plotly(funcres, zoom=None):
    """Convert the result of a function to a plot. Raises if `funcres` is not
    in any valid format

    :param funcres: the function result of :func:`exdc_func`
    :param zoom: x bounds to zoom (not used)
    """
    if isinstance(funcres, Trace):
        return stream2plotly(Stream([funcres]))
    elif isinstance(funcres, Stream):
        return stream2plotly(funcres)
    elif isinstance(funcres, dict):
        funcres = [{k: _jsonify(v) for k, v in funcres.items()}]
    elif isinstance(funcres, (list, tuple)):
        old_funcres, funcres = funcres, []
        for f in old_funcres:
            funcres.extend(convert2plotly(f, zoom))

    err = not isinstance(funcres, (list, tuple))
    if not err:
        err = any(not isinstance(_, dict) for _ in funcres)
    if not err:
        err = any(('y' not in _ for _ in funcres))
    if err:
        raise ValueError('Plot function output must be an obspy Trace, Stream, dict '
                         '(or any list of those objects).\nDicts must have at least the '
                         'key "y"\n(full list of keys: '
                         'https://plotly.com/javascript/reference/)')
    return funcres


def trace2plotly(trace):
    return stream2plotly(Stream([trace]))


def stream2plotly(stream):
    """Return a list[dict] where each dict holds the trace data to be displayed
    with plotly"""
    labels = [t.get_id() for t in stream]
    # add trace.get_id() + "[#1]", "[#2]" etcetera if some traces have
    # same id:
    for i, lbl in enumerate(labels):
        chunk = 1
        for j, lbl2 in enumerate(labels[i + 1:], i + 1):
            if lbl == lbl2:
                chunk += 1
                labels[j] = lbl2 + ('[#%d]' % chunk)
        if chunk > 1:
            labels[i] = lbl + '[#1]'
    return [
        {
            'x0': _jsonify(trace.stats.starttime),
            'dx': _jsonify(trace.stats.delta) * 1000,  # *1000? plotly requires msec
            'y': _jsonify(trace.data),
            'name': name
        } for name, trace in zip(labels, stream)
    ]


def _jsonify(obj):
    """jsonify `obj`"""
    if isinstance(obj, (UTCDateTime, date, datetime)):
        ret = UTCDateTime(obj).isoformat(sep='T')
        return ret + 'Z' if ret[-1] != 'Z' else ret
    try:
        is_ndarray = isinstance(obj, (np.ndarray, np.generic))
        if is_ndarray or isinstance(obj, (list, tuple)):
            obj2 = np.asarray(obj)
            nonfinite = ~np.isfinite(obj2)
            if nonfinite.any():
                obj2 = obj2.astype(object)
                obj2[nonfinite] = None
            return obj2.tolist() if is_ndarray else obj
        return None if obj != obj or obj in (-math.inf, math.inf) else obj
    except TypeError:  # raised by np.isfinite
        if isinstance(obj, (list, tuple)):
            # (we might have e.g. a list of UTCDateTimes):
            return [_jsonify(_) for _ in obj]
        return obj
    except ValueError:
        return obj


def get_sn_windows(segment, config):
    """Return returns the two tuples (s_start, s_end), (n_start, n_end)
    where all arguments are `UTCDateTime`s and the first tuple refers to the
    signal window, the latter to the noise window. Both windows are
    calculated on the given segment, according to the given config
    (dict)
    """
    if len(segment.stream()) != 1:
        raise ValueError(("Unable to get sn-windows: %d traces in stream "
                          "(possible gaps/overlaps)") % len(segment.stream()))
    wndw = config['sn_windows']
    arrival_time = \
        UTCDateTime(segment.arrival_time) + wndw['arrival_time_shift']
    return sn_split(segment.stream()[0], arrival_time, wndw['signal_window'],
                    return_windows=True)
