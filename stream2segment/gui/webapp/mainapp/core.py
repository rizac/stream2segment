"""
Core functionalities for the main GUI web application (show command)

:date: Jul 31, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import os

# make the following(s) behave like python3 counterparts if running from py2.7+
# (http://python-future.org/imports.html#explicit-imports):

from builtins import zip

from itertools import cycle
from io import StringIO
import contextlib

import yaml
from obspy import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime

from stream2segment.process import gui
from stream2segment.utils import load_source, iterfuncs, yaml_safe_dump
from stream2segment.gui.webapp.mainapp.jsplot import Plot, isoformat
from stream2segment.gui.webapp.mainapp import db
from stream2segment.process.lib.traces import sn_split

# number of points per plot. Used to resample points:
NPTS_WIDE = 900  # FIXME: automatic retrieve from the GUI?
NPTS_SHORT = 900  # FIXME: see above

# Note that the use of global variables like this should be investigted
# in production (which is not the intended goal of the web GUI for the moment):


g_config = {}  # noqa


g_selection = {}


def _default_preprocessfunc(segment, config):
    """Default pre-process function: remove the instrumental response
    assuming output unit in m/s and water level for deconvolution = 60
    """
    s = Stream()
    inventory = segment.inventory()
    for t in segment.stream():
        t.remove_response(inventory)
        s.append(t)
    return t if len(s) == 1 else s
    # raise Exception("No function decorated with '@gui.preprocess'")


_preprocessfunc = _default_preprocessfunc  # pylint: disable=invalid-name

g_functions = [lambda seg, cfg: seg.stream()]  # pylint: disable=invalid-name

userdefined_plots = []  # pylint: disable=invalid-name


def _reset_global_functions():
    """mainly used for testing purposes and within the init method"""
    global _preprocessfunc  # pylint: disable=global-statement, invalid-name
    _preprocessfunc = _default_preprocessfunc
    del g_functions[1:]
    del userdefined_plots[:]


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

    _reset_global_vars()

    if config:
        g_config.update(config)

    if segments_selection:
        g_selection.update(segments_selection)


def _reset_global_vars():
    """mainly used for testing purposes and within the init method"""
    g_config.clear()
    g_selection.clear()


def get_segments_count(segselect=None):
    """Compute the segment count to be shown according if the given
    `segments_selection` is given (dict), and returns the number if block=True
    otherwise returns None
    """
    if segselect is not None:
        num_segments = db.get_segments_count(segselect)
        set_select_conditions(segselect)
    else:
        num_segments = db.get_segments_count(get_select_conditions())
    return num_segments


def get_db_url(safe=True):
    return db.get_db_url(safe=safe)


def get_func_doc(index=-1):
    """Return the documentation for the given custom function.

    :param index: if negative, returns the doc for the preprocess function,
        otherwise is the index of the i-th function (index 0 refers to the main
        function plotting the segment stream)
    """
    if index < 0:
        return _escapedoc(getattr(_preprocessfunc, "__doc__", ''))
    return userdefined_plots[index]['doc']


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
    # qry = query4gui(session, conditions=conditions, orderby=None)
    return {'classes': classes, 'metadata': _metadata}


def get_config(asstr=False):
    """Returns the current config as YAML formatted string (if `asstr` is True)
    or as dict. The returned value does not include the segments selection,
    if given from the command line
    """
    config_dict = dict(g_config)
    if not asstr:
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


def set_select_conditions(newdict):
    """Set a new a dict representing the current select conditions (parameter
    'segments_selection' of the YAML file)

    :param newdict: a dict of new select expressions all in str format
    """
    # FIXME: handle concurrency with locks?
    g_selection.clear()
    g_selection.update(newdict)


def get_segment(segment_id):
    """Return  the Segment object of the given segment id"""
    return db.get_segment(segment_id)


def get_segment_id(segment_index, segment_count):
    """Return the segment id corresponding to the given segment index in
    the GUI"""
    return db.get_segment_id(segment_index, segment_count,
                             get_select_conditions())


def set_class_id(seg_id, class_id, value):
    """Set the given class to the given segment (value=True), or removes it
    from the given segment (value=False)
    """
    segment = get_segment(seg_id)
    annotator = 'web app labeller'  # FIXME: use a session or computer username?
    if value:
        segment.add_classes(class_id, annotator=annotator)
    else:
        segment.del_classes(class_id)
    return {}


def get_segment_data(seg_id, plot_indices, all_components, preprocessed,
                     zooms, metadata=False, classes=False, config=None):
    """Return the segment data, depending on the arguments

    :param seg_id: the segment id (int)
    :param plot_indices: a list of plots to be calculated
    :param all_components: boolean, whether or not the returned plots should
        include all segments components (channel orientations). Ignored if 0 is
        not in `plot_indices`
    :param preprocessed: boolean, whether or not the plot should be returned on
        the pre-processing function defined in the config (if any), or on the
        raw ObsPy Stream
    :param zooms: the plot bounds, list or None. If list, each element is
        either None,  or a tuple of [xmin, xmax] values (xmin and xmax can
        be both None, to conform python slicing behaviour). If None, defaults
        to a list of [None, None] elemeents (one for each plot)
    :param metadata: boolean, whether or not to return a list of the segment
        metadata. The list is a list of tuples ('column', value). A list is
        used to preserve order for client-side javascript parsing
    :param classes: boolean, whether to return the integers classes ids (if
        any) of the given segment
    :param config: a dict of new config values. Can be falsy to skip updating
        the config
    """
    plots = []
    zooms_ = parse_zooms(zooms, plot_indices)
    sn_windows = []
    if config:
        g_config.update(**config)

    if plot_indices:
        plots = get_plots(seg_id, plot_indices, preprocessed, all_components)
        try:
            # return always sn_windows, as we already calculated them. It is
            # better to call this method AFTER get_plots_func defined above
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
    """Parse the zoom received from the frontend.

    :param zooms: a list of 2-element tuples, or None's. The elements of the
        tuple can be number, Nones or strings (in datetime format)
    :return: an iterator over zooms
    """
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
    """Return the plots

    :param all_components: if 0 is not in plot_indices, it is ignored.
        Otherwise returns in plot[I] all components
        (where I = argwhere(plot_indices == 0)
    """
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
    """Return a jsplot.Plot object corresponding to the given function
    applied on the given segment

    :param segment: a Segment instance
    :param preprocessed: boolean, whether the function has to be applied on
        the pre-processed trace of the segment
    :param func_index: the index of the function to be called. It is one
        implemented in the python module with the relative decorator, and must
        have signature: func(segment, config)
    """
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


def convert2plot(funcres):
    """Convert the result of a function to a plot. Raises if `funcres` is not
    in any valid format

    :param funcres: the function result of :func:`exdc_func`
    """
    if isinstance(funcres, Plot):
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
                        raise ValueError(("Cannot create plot from tuple "
                                          "(length=%d): Expected (x0, dx, y) "
                                          "or (x0, dx, y, label)") % len(obj))
                plt.add(x0, dx, y, label)
            elif isinstance(obj, Trace):
                plt.addtrace(obj, label)
            elif isinstance(obj, Stream):
                for trace in obj:
                    plt.addtrace(trace, label)
            else:
                raise ValueError("Cannot create plot from %s (length=%d): " %
                                 str(type(obj)))
    return plt


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
