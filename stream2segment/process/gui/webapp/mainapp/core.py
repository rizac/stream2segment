"""
Core functionalities for the main GUI web application (show command)
"""
# :date: Jul 31, 2016
import math
import os
from dataclasses import asdict, fields
from datetime import datetime, date

from io import StringIO
from typing import Any

import yaml
import numpy as np
from obspy import Stream, Trace
from obspy.core.inventory import Inventory
from obspy.core.utcdatetime import UTCDateTime
from sqlalchemy import Engine, select, func
from sqlalchemy import insert, delete
from sqlalchemy.exc import IntegrityError

from stream2segment.io.db import create_engine, secure_dburl
from stream2segment.process import gui  # FIXME gui import what is it?!!!!
from stream2segment.process.gui.introspection import scan_module
from stream2segment.process.main import (
    get_obspy_stream, get_obspy_inventory, SegmentMetadata
)
from stream2segment.process.segments_selection import build_select, is_legacy_db, \
    get_orderby_columns

g_engine: Engine = None

# Note that the use of global variables like this should be investigated
# in production (which is not the intended goal of the web GUI for the moment):

g_config = {}  # global config

g_selection = {}  # segments selection conditions

# `g_segment_ids` below is a numpy Array that caches the segments ids if `g_selection` is
# not empty in order to avoid modifying the matching segments from the GUI. E.g.,
# labelling segments while `g_selection` is configured to show unlabelled segments only
g_segment_ids = None


def _default_preprocessfunc(trace, inventory, config):
    """Default pre-process function: remove the instrumental response with no pre_filt and
    water_level=60. If the channel instrument code is in ('N', 'G', 'L') output will be
    m/s**2 ('ACC'), otherwise m/s ('VEL')
    """  # FIXME normalize i_code also in the templates?
    i_code = trace.stats.segment_metadata.instrument_code
    output = 'VEL'
    if i_code in ('N', 'G', 'L'):
        output = 'ACC'
    trace.remove_response(inventory, water_level=60, output=output, pre_filt=None)
    return trace


# global variables (will be initialized in _reset_global_functions, see below):

_preprocessfunc = _default_preprocessfunc
g_functions = {}

userdefined_plots = {}

main_function_label = ""

g_station_id: int | None = None

g_station: Inventory | None = None


def _reset_global_functions():
    """mainly used for testing purposes and within the init method"""
    global _preprocessfunc
    _preprocessfunc = _default_preprocessfunc
    global g_functions
    g_functions = {main_function_label: lambda trace, cfg: trace}
    global userdefined_plots
    userdefined_plots = {}


_reset_global_functions()  # just initialize global vars


def init(
    app,
    db_url: str,
    py_module=None,
    config: dict | None = None,
    segments_selection: dict | None =None
):
    """Initialize global variables. This method must be called once
    after the Flask app has been created and before using it.

    :param py_module: Python module
    :param config: dict of the current configuration
    :param segments_selection: dict[str, str] of segment attributes mapped to a
        selection expression (str)
    """
    # db.init(app, db_url)
    global g_engine
    g_engine = create_engine(db_url)  # , check_same_thread=True)

    if py_module:
        _reset_global_functions()
        for function in scan_module(py_module, functions=True, classes=False):
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

    return reset_global_vars(config or {}, segments_selection or {})


def reset_global_vars(config=None, segments_selection = None) -> int:
    """Reset global variables (both dicts). None means: skip"""
    if config is not None:
        global g_config
        g_config = dict(config)
    global g_selection
    if segments_selection is not None:
        g_selection = dict(segments_selection)
    # reset ids:
    stmt = build_select(g_engine, g_selection, ['id'])
    with g_engine.connect() as conn:
        rows = conn.execute(stmt).scalars()
    global g_segment_ids
    g_segment_ids = np.fromiter(rows, dtype = int)
    return len(g_segment_ids)


def get_db_url():
    return secure_dburl(str(g_engine.url))


def get_preprocess_function():
    return _preprocessfunc


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
    return yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)


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


def get_segment_id(segment_index):
    """Return the segment id corresponding to the given segment index in
    the GUI

    :param segment_index: the segment index
    """
    global g_segment_ids
    return int(g_segment_ids[segment_index])


def set_class_id(seg_id, class_id, value):
    """Set the given class to the given segment (value=True), or removes it
    from the given segment (value=False)
    """
    try:
        import getpass
        annotator = str(getpass.getuser())
        if len(annotator) > 2:
            annotator = annotator[0] + '*' * (len(annotator) - 2) + annotator[-1]
        else:
            annotator = 'user id ' + str(os.getuid())
    except Exception:  # noqa
        annotator = 'anonymous labeller'

    if is_legacy_db(g_engine):
        from stream2segment.io.db.legacy.models import ClassLabelling as ClassLabeling
    else:
        from stream2segment.io.db.models import ClassLabeling

    with g_engine.begin() as conn:  # noqa
        if value:
            try:
                conn.execute(
                    insert(ClassLabeling).values(
                        segment_id=seg_id, class_label_id=class_id, annotator=annotator
                    )
                )
            except IntegrityError:
                pass  # pairing already exists
        else:
            conn.execute(
                delete(ClassLabeling).where(
                    ClassLabeling.segment_id == seg_id,
                    ClassLabeling.class_label_id == class_id,
                )
            )

    # return the class label count:
    stmt = select(func.count()).select_from(ClassLabeling).where(
        ClassLabeling.class_label_id == class_id
    )
    with g_engine.connect() as conn:
        return conn.execute(stmt).scalar_one()


def get_segment_data(
    seg_id,
    plot_names,
    all_components,
    preprocessed,
    zooms,
    classes=False
):
    """Return the segment data, depending on the arguments

    :param seg_id: the segment id (int)
    :param plot_names: a list of plot names to be calculated. "" indicates the default
        plot
    :param all_components: boolean, whether the returned plots should
        include all segments components (channel orientations). Ignored if 0 is
        not in `plot_indices`
    :param preprocessed: boolean, whether the plot should be returned on
        the pre-processing function defined in the config (if any), or on the
        raw ObsPy Stream
    :param zooms: the plot bounds, list or None. NOT used.
        If List, each element is either None,  or a tuple of [xmin, xmax] values
        (xmin and xmax can be both None, to conform python slicing behaviour).
        If None, defaults
        to a list of [None, None] elements (one for each plot)
    :param classes: boolean, whether to return the integers classes ids (if
        any) of the given segment
    """
    stmt = build_select(g_engine, {'id': str(seg_id)})
    orderby_cols = get_orderby_columns(g_engine).values()
    stmt = stmt.order_by(*orderby_cols)

    with g_engine.connect() as conn:
        db_row = conn.execute(stmt).one()
    stream = get_obspy_stream(db_row)
    seg_meta = stream[0].stats.segment_metadata

    if zooms is None and plot_names:
        zooms = [(None, None) for _ in plot_names]

    plots = {}
    layouts = {}
    if plot_names:
        if preprocessed:
            sta_id = stream[0].stats.segment_metadata.station_id
            global g_station, g_station_id
            if sta_id == g_station_id and g_station is not None:
                inv = g_station
            else:
                inv = get_obspy_inventory(g_engine, sta_id, is_legacy_db(g_engine))
                g_station = inv
                g_station_id = sta_id
            stream_p = Stream([_preprocessfunc(t, inv, g_config) for t in stream])
        else:
            stream_p = stream
        plots, layouts = get_plotly_data_and_layout(
            stream_p, plot_names, all_components, zooms
        )

    desc = (
        '&#9432; '
        f'Event magnitude: <b>{seg_meta.event_magnitude} '
        f'{seg_meta.event_magnitude_type}</b>. Recording station '
        f'distance: &#8776; <b>{round(seg_meta.event_distance_km, 2):,} '
        f'km</b>. ',
        'Recorded segment metadata:'
    )

    return {
        'plotData': plots,
        'plotLayout': layouts,
        'attributes': [
            {
                'label': f.name,
                'value': _jsonify(getattr(seg_meta, f.name))
            } for f in fields(seg_meta)],
        'classes': [] if not classes else get_segment_class_labels(seg_id),
        'description': desc
    }

def get_metadata(trace: Trace | None = None) -> list[dict[str, str]] | dict[str, Any]:
    if trace is None:
        sorted_fields = sorted(fields(SegmentMetadata), key=lambda f: f.name)
        return [
            {
                'label': f.name,
                'dtype': str(f.type)
            }
            for f in sorted_fields
        ]
    return asdict(trace.stats.segment_metadata)


def get_plotly_data_and_layout(
    stream: Stream, plot_names, all_components, zooms
):
    """Return the plots to display for the given segment, as the tuple:

     plots:dict[str, list[dict]], layout: dict[str, dict]

    both plots and layout keys are the elements of `plot_names`:
     - in plots, a plot name is mapped to a list of dicts,
       where each dict represents a plot trace
     - in layout, a plot name is associated to a dict defining
        the plot layout (e.g., shapes, annotations, and so on). Some of the properties
        defined here might be overwritten in the frontend (e.g. axis scale or type
        properties)

    :param plot_names: the name (key) of each plot to draw
    :param all_components: if 0 is not in plot_indices, it is ignored.
        Otherwise returns in plot[I] all components
        (where I = argwhere(plot_indices == 0)
    :param zooms: list of x bounds to zoom or None, one for each plot (not used)
    :return the tuple: plots:dict[str, list[dict]], layout: dict[str, dict]
    """
    seg_meta = stream[0].stats.segment_metadata
    plots = {}
    layouts = {}
    for name in plot_names:
        zoom = None
        plot = get_plot(stream, name, zoom)
        # FIXME: fix here all components!
        # if not name and all_components and isinstance(plot, list):
        #     for seg in segment.siblings(include_self=False):
        #         plt = get_plot(seg, preprocessed, name, zoom)
        #         if isinstance(plt, str):
        #             plot = plt
        #             break
        #         else:
        #             plot.extend(plt)
        plots[name] = plot
        if name == main_function_label:
            # main plot: add arrival time vertical line:
            layouts[name] = {
                'shapes': [{
                    'type': 'line',
                    'x0': _jsonify(seg_meta.arrival_time),
                    'y0': 0,
                    'x1': _jsonify(seg_meta.arrival_time),
                    'yref': 'paper',
                    'y1': 1,
                    'line': {
                        'color': '#008B8B',
                        'width': 2,
                        'dash': 'dot'
                    }
                }],
                'annotations': [{
                    'text': 'arrival<br>time',
                    'align': 'right',
                    'xanchor': "right",
                    'x': _jsonify(seg_meta.arrival_time),
                    'y': 1,
                    'yref': 'paper',
                    'yanchor': 'top',
                    'font': {'size': '10'},
                    'showarrow': False
                }]
            }
    return plots, layouts


def get_plot(stream: Stream, func_name, zoom):
    """Return a list of dicts where each dict represents a Plotly Trace.
    The dict is the result of applying the given function
    to the given segment

    :param stream: an ObsPy Stream instance
    :param func_name: the name of the function to be called. It is one
        implemented in the python module with the relative decorator, and must
        have signature: func(segment, config). "" denotes the default function
        (just print print the trace)
    """
    try:
        stream = stream.copy()
        function = g_functions[func_name]
        result = [function(t, g_config) for t in stream]
        return convert2plotly(result, zoom)
    except Exception as exc:
        return str(exc)


def convert2plotly(func_result, zoom=None):
    """Convert the result of a function to a plot. Raises if `funcres` is not
    in any valid format

    :param func_result: the function result of :func:`exdc_func`
    :param zoom: x bounds to zoom (not used)
    """
    if isinstance(func_result, Trace):
        return stream2plotly(Stream([func_result]))
    elif isinstance(func_result, Stream):
        return stream2plotly(func_result)
    elif isinstance(func_result, dict):
        func_result = [{k: _jsonify(v) for k, v in func_result.items()}]
    elif isinstance(func_result, (list, tuple)):
        old_funcres, func_result = func_result, []
        for f in old_funcres:
            func_result.extend(convert2plotly(f, zoom))

    err = not isinstance(func_result, (list, tuple))
    if not err:
        err = any(not isinstance(_, dict) for _ in func_result)
    if not err:
        err = any(('y' not in _ for _ in func_result))
    if err:
        raise ValueError('Plot function output must be an obspy Trace, Stream, dict '
                         '(or any list of those objects).\nDicts must have at least the '
                         'key "y"\n(full list of keys: '
                         'https://plotly.com/javascript/reference/)')
    return func_result


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
            return obj2.tolist()
        return None if obj != obj or obj in (-math.inf, math.inf) else obj
    except TypeError:  # raised by np.isfinite
        if isinstance(obj, (list, tuple)):
            # (we might have e.g. a list of UTCDateTimes):
            return [_jsonify(_) for _ in obj]
        return obj
    except ValueError:
        return obj


def get_class_labels() -> list[tuple[str, int]]:
    """Return [(label, count), ...] for every ClassLabel, including those with 0 segments."""
    engine = g_engine
    if is_legacy_db(engine):
        from stream2segment.io.db.legacy.models import ClassLabelling as ClassLabeling
        from stream2segment.io.db.legacy.models import Class as ClassLabel
    else:
        from stream2segment.io.db.models import ClassLabel, ClassLabeling

    stmt = (
        select(ClassLabel.label, func.count(ClassLabeling.id))
        .select_from(ClassLabel)
        .outerjoin(ClassLabeling, ClassLabeling.class_label_id == ClassLabel.id)
        .group_by(ClassLabel.id)
    )
    with engine.connect() as conn:
        return conn.execute(stmt).fetchall()


def get_segment_class_labels(seg_id: int) -> list[str]:
    """Return all ClassLabel rows (label, description) assigned to the given segment."""
    engine = g_engine
    if is_legacy_db(engine):
        from stream2segment.io.db.legacy.models import ClassLabelling as ClassLabeling
        from stream2segment.io.db.legacy.models import Class as ClassLabel
    else:
        from stream2segment.io.db.models import ClassLabel, ClassLabeling
    stmt = (
        select(ClassLabel.label, ClassLabel.description)
        .join(ClassLabeling, ClassLabeling.class_label_id == ClassLabel.id)
        .where(ClassLabeling.segment_id == seg_id)
    )
    with engine.connect() as conn:
        return conn.execute(stmt).fetchall()
