'''
Created on Feb 28, 2017

@author: riccardo
'''
import numpy as np
from datetime import datetime
from collections import OrderedDict
from sqlalchemy.orm.session import object_session
# from sqlalchemy.orm.session import Session
from obspy import Stream, Trace, UTCDateTime
from stream2segment.process.utils import get_stream, itercomponents
from stream2segment.analysis.mseeds import bandpass, utcdatetime, cumsum, cumtimes, fft
from stream2segment.analysis import amp_spec
from stream2segment.process.wrapper import get_inventory
from math import floor, log10
from itertools import cycle, izip


_functions = []
_ud_plots = OrderedDict()

if not _functions:
    # attach custom function. This might seem clumsy but it's to treat default functions
    # (fft and cumulative) as user defined functions
    # the arguments are: the function, the title, and if they accept a View as argument or the
    # view main trace. For user defined function, is the latter (see register_function below)
    # the last argument tells if the function has to be executed anyway. If False, and the
    # source trace has warnings (gaps etcetera) is not executed
    # If the function returns a trace, is title will be source_trace_title - second_argument
    # Otherwise it will be the title provided in the returned Plot
    _functions.append((lambda view: view._main_plot, "", True, True))
    _functions.append((lambda view: spectra(view),
                      "Spectra", True, False))
    _functions.append((lambda view: view.get_components()[0][0], "", True, True))
    _functions.append((lambda view: view.get_components()[1][0], "", True, True))
    _functions.append((lambda view: view._cum, "Cumulative", True, True))
    # set a default length to know if we added user defined plots
    _def_func_length = len(_functions)


def user_defined_plots():
    """
        Returns a dict with integer keys mapped to the relative user defined function name,
        if any was registered via `register_function`
        The keys (integer) of the returned dict are sorted according to the insertion order. The
        first keys is > 0, as there are surely some custom plots defined before user defined ones
        (currently, 4, but this might change)
    """
    ret = OrderedDict()
    for i in xrange(_def_func_length, len(_functions)):
        ret[i] = _functions[i][1]
    return ret


def register_function(func, name=None, execute_anyway=False):
    """
        Registers a new function for this module. The function will be called on any `Trace`
        which did not issue warnings. Warnings include inventory error (for filtered miniSEED),
        gaps and so on. In case of warnings, the program
        just shows an empty trace with the original warning issued for the parent trace (the
        argument to `func`). Note that the latter is a copy of the currently selected `Trace`
        so it's safe to manipulate it
        :param func: a function accepting the currently selected trace and returning another
        trace
        :param name: optional, the name to be shown in the GUI title. Defaults to `func.__name__`
        :param execute_anyway: boolean optional (default: False). Whether to execute the function
        if the source trace had warning/errors (e.g., gaps, or errors retrieving the station
        inventory, if the latter is needed). Note that executing a function on a trace with
        errors/warnings might lead to unexpected results or cause the code to hang
    """
    funcname = name or func.__name__
    _functions.append((func, funcname, False, execute_anyway))


def plot_title(src, title):
    """
    Creates a title for any object (trace, Plot) derived from `src`. Basically returns
    `"src's title" + "<title>"`
    :param src: trace or Plot, the source object
    :param title: the title of the object derived from `src`"""
    return "%s - %s" % (src.title if hasattr(src, 'title') else src.get_id(), title)


def exec_function(index, view):
    func, name, uses_view, execute_anyway = _functions[index]
    arg = view if uses_view else view._trace.copy()
    main_plot = view._main_plot
    title = plot_title(main_plot, name)  # "%s - %s" % (main_plot.title, name)
    try:
        if main_plot.warning and not execute_anyway:
            raise Exception("Not shown: %s" % main_plot.warning)
        funcres = func(arg)
        if isinstance(funcres, Plot):  # this should be called internally
            plt = funcres
        else:
            labels = cycle([None])
            if isinstance(funcres, Trace):
                itr = [funcres]
            elif isinstance(funcres, dict):
                labels = funcres.iterkeys()
                itr = funcres.itervalues()
            else:
                itr = funcres

            plt = Plot(title)
            for label, obj in izip(labels, itr):
                if isinstance(obj, Trace):
                    plt.addtrace(obj, label)
                else:
                    trace = view._trace
                    array = np.asarray(obj)
                    if array.size != trace.data.size:
                        raise ValueError("Expected array with %d elements, found %d" %
                                         (trace.data.size, array.size))
                    plt.addtrace(Trace(data=array, header=trace.stats.copy()), label)

#         isstream = isinstance(plt, Stream)
#         istrace = isinstance(plt, Trace)
#         isplot = isinstance(plt, Plot)
#         isdict = isinstance(plt, dict)
#         if not istrace and not isplot and not isstream and not isdict:
#             trace = view._trace
#             array = np.asarray(plt)
#             if array.size != trace.data.size:
#                 raise ValueError("Expected array with %d elements, found %d" %
#                                  (trace.data.size, array.size))
#             plt = Plot.fromtrace(Trace(data=array, header=trace.stats.copy()), title)
#         if istrace:  # fft below does not return trace(s) but plots
#             plt = Plot.fromtrace(plt, title)
#         elif isstream:
#             plt_ = Plot(title)
#             for t in plt:
#                 plt_.addtrace(t)
#             plt = plt_
            
    except Exception as exc:
        plt = Plot(title, warning=str(exc)).add(0, 1, [])

    return plt


def spectra(view):
    warning = ""
    trace = view._trace
    noisy_wdw, signal_wdw = view.get_spectra_windows()

    try:
        fft_noise = fft(trace, *noisy_wdw)
        fft_signal = fft(trace, *signal_wdw)

        df = fft_signal.stats.df
        f0 = 0
        amp_spec_noise, amp_spec_signal = amp_spec(fft_noise, True), amp_spec(fft_signal, True)
    except Exception as exc:
        warning = str(exc)
        f0 = 0
        df = 1
        amp_spec_noise = [0, 0]
        amp_spec_signal = [0, 0]

    # _functions[1][1] == 'Spectra'

    return Plot(plot_title(trace, _functions[1][1]), warning=warning).\
        add(f0, df, amp_spec_noise, "Noise").\
        add(f0, df, amp_spec_signal, "Signal")

# class CacheDict(odict):
#     """A dict which holds a limited size of data, after which for any entry added it will remove
#     one to keep size limit"""
#     def __init__(self, size_limit, *args, **kwds):
#         self.size_limit = size_limit
#         super(CacheDict, self).__init__(*args, **kwds)
#         # odict.__init__(self, *args, **kwds)
#         self._check_size_limit()
# 
#     def __setitem__(self, key, value):
#         super(CacheDict, self).__setitem__(key, value)
#         self._check_size_limit()
# 
#     def _check_size_limit(self):
#         rem_count = len(self) - self.size_limit
#         if self.size_limit is None or rem_count <= 0:
#             return
#         toremove = []
#         for i, k in enumerate(self.iterkeys()):
#             if i >= rem_count:
#                 break
#             toremove.append(k)
#         for key in toremove:
#             self.pop(key)


def set_spectra_config(arrival_time_shift, signal_window):
    lcl = dict(locals())
    if 'spectra' not in View.settings:
        View.settings['spectra'] = {}
    View.settings['spectra'].update(lcl)
    for v in View.views.itervalues():
        # force refresh of Spectra:
        v[1] = None
        v._noisy_wdw = None


def set_filter_config(remove_response_water_level, remove_response_output,
                      bandpass_freq_max, bandpass_max_nyquist_ratio, bandpass_corners):
    lcl = dict(locals())
    if 'filter' not in View.settings:
        View.settings['filter'] = {}
    View.settings['filter'].update(lcl)
    # force re-updating of filtered views:
    for key in View.views.keys():
        if ".filtered" in key:
            View.views.pop(key)


class Filter(object):
    """Class for filtering a trace. In the future, this might be customizable
    to account for different filters"""

    @classmethod
    def filter(cls, trace, segment, inventory, config):
        evt = segment.event
        trace = bandpass(trace, evt.magnitude, freq_max=config['bandpass_freq_max'],
                         max_nyquist_ratio=config['bandpass_max_nyquist_ratio'],
                         corners=config['bandpass_corners'], copy=False)
        trace.remove_response(inventory=inventory, output=config['remove_response_output'],
                              water_level=config['remove_response_water_level'])
        return trace


class View(list):
    """A View is a list of plots representing the view of a segment in the GUI.
    It is a list whose elements are Plot instances. view[0] returns the main plot
    (representing the segment), views[1] and so on the other plots (specrtra, cumulative
    and so on, including user defined functions, if any)"""
    inventories = {}  # CacheDict(50)
    views = {}  # CacheDict(50)
    settings = {}

    def __init__(self, segment, trace, warning=None):
        # append default custom functions. Use self cause they avoid copying the trace,
        # and in case of fft they need to access object attributes
        self._main_plot = Plot.fromtrace(trace, trace.get_id(), warning)
        super(View, self).__init__([None] * len(_functions))
        self._trace = trace
        self._segment = segment
        self._atime = UTCDateTime(segment.arrival_time)
        self._other_component_views = None
        # calculate default spectra windows.
        # cumulative needs to be stored as attribute so that we do not calculate it twice
        # when requesting for it
        self._cum = cumsum(trace)
        self._noisy_wdw = None
        self._signal_wdw = None

        # calculate the default window for the spectra
        # self.get_spectra_windows()

    @property
    def warning(self):
        """Returns if this view's main plot has warnings (e.g. gaps, error in repsonse if if
        filtered...)"""
        return self._main_plot.warning

    def get_spectra_windows(self):
        if self._noisy_wdw is None or self._signal_wdw is None:
            a_time = self._atime + self.settings['spectra']['arrival_time_shift']
            try:
                cum0, cum1 = self.settings['spectra']['signal_window']
                t0, t1 = cumtimes(self._cum, cum0, cum1)
                nsy, sig = [a_time, t0 - t1], [t0, t1 - t0]
            except TypeError:
                shift = self.settings['spectra']['signal_window']
                nsy, sig = [a_time, -shift], [a_time, shift]
            self._noisy_wdw = nsy
            self._signal_wdw = sig

        return self._noisy_wdw, self._signal_wdw

    def link(self, other_component_views):
        self._other_component_views = other_component_views

    @property
    def linked(self):
        return True if self._other_component_views else False

    def get_components(self):
        if not self.linked:
            raise ValueError("Component not loaded")
        return self._other_component_views

    @classmethod
    def _get_components(cls, segment, filtered):
        try:
            segments = list(itercomponents(segment))[:2]
            return [View.get(segment, filtered, False) for segment in segments]
        except Exception:
            return []

    def __getitem__(self, index):
        """Overrides self[i] to lazily calculate plots when needed"""
        if super(View, self).__getitem__(index) is None:
            self[index] = exec_function(index, self)
        return super(View, self).__getitem__(index)

    @classmethod
    def get(cls, segment, filtered=False, load_all_components=True):
        """Returns the trace for a givens segment
        The segment trace and potential errors to be displayed are saved as `View` class `dict`s
        """
        key = str(segment.id)  # it's integer, so str is safe. We use strings cause we mark
        # as key + ".filtered" the filtered traces (see below)
        view = cls.views.get(key, None)
        if view is None:
            warning = ""
            try:
                stream = get_stream(segment)
                prevlen = len(stream)
                stream = stream.merge(fill_value='latest')
                if len(stream) != 1:
                    raise ValueError("Unmergeable gaps/overlaps")
                if len(stream) != prevlen:
                    warning = "Merged gaps/overlaps"
            except Exception as exc:
                warning = str(exc)
                stream = Stream(Trace(np.array([0, 0]), {'starttime':
                                                          UTCDateTime(segment.start_time),
                                                          'delta': UTCDateTime(segment.end_time) -
                                                          UTCDateTime(segment.start_time),
                                                          'network': segment.station.network,
                                                          'station': segment.station.station,
                                                          'location': segment.channel.location,
                                                          'channel': segment.channel.channel
                                                          }))

            view = View(segment, stream[0], warning=warning)
            cls.views[key] = view

        if not filtered and not view.linked and load_all_components:  # load other components, too
            view.link(cls._get_components(segment, filtered))

        if not filtered:
            return view

        warning = view.warning
        key += ".filtered"
        source_view = view
        view = cls.views.get(key, None)
        if view is None:
            trace = source_view._trace
            if not warning:
                inv = cls.inventories.get(segment.station.id, None)
                if inv is None:
                    try:
                        inv = get_inventory(segment.station, True)
                        cls.inventories[segment.station.id] = inv
                    except Exception:
                        warning = ("Error getting inventory")

                if not warning:
                    try:
                        stream = Stream([Filter.filter(trace.copy(), segment, inv,
                                                       View.settings['filter'])])
                        trace = stream[0]
                    except Exception as exc:
                        warning = str(exc)
            else:
                warning = "%s (filter - if any - not applied)" % warning

            view = View(segment, trace, warning=warning or None)
            # write to cache dict:
            cls.views[key] = view

        if not view.linked and load_all_components:  # load other components, too
            view.link(cls._get_components(segment, filtered))

        return view


class Plot(object):
    """A plot is a class representing a Plot on the GUI"""

    def __init__(self, title=None, warning=None):
        self.title = title or ''
        self.data = []
        self.shapes = []
        # self.xrange = x_range
        self.warning = warning or ""

    def add(self, x0=None, dx=None, y=None, label=None):
        """Adds a new series (scatter line) to this plot. This method optimizes
        the data transfer and the line will be handled by the frontend plot library"""
        self.data.append([x0, dx, np.asarray(y), label])
        return self

    def addtrace(self, trace, label=None):
        return self.add(jsontimestamp(trace.stats.starttime),
                        1000 * trace.stats.delta, trace.data,
                        label or trace.get_id())

    @staticmethod
    def fromtrace(trace, label=None, warning=None):
        return Plot(label, warning).addtrace(trace)

    @staticmethod
    def get_slice(x0, dx, y, xbounds, npts):
#         print "before"
#         print "x0: %d" % x0
#         print "dx %f" % dx
#         print "x1 %f" % (x0 + dx * (len(y)-1))
#         print "min(y): %f " % np.min(y)
#         print "max(y): %f" % np.max(y)


        start, end = Plot.unpack_bounds(xbounds)
        idx0 = None if start is None else max(0,
                                              int(np.ceil(np.true_divide(start-x0, dx))))
        idx1 = None if end is None else min(len(y),
                                            int(np.floor(np.true_divide(end-x0, dx) + 1)))

        if idx0 is not None or idx1 is not None:
            y = y[idx0:idx1]
            if idx0 is not None:
                x0 += idx0 * dx

        size = len(y)
        if size > npts and npts > 0:
            # set dx to be an int, too many
            y, newdxratio = downsample(y, npts)
            if newdxratio > 1:
                dx *= newdxratio  # (dx * (size - 1)) / (len(y) - 1)

#         print "after"
#         print "x0: %d" % x0
#         print "dx %f" % dx
#         print "x1 %f" % (x0 + dx * (len(y)-1))
#         print "min(y): %f " % np.min(y)
#         print "max(y): %f" % np.max(y)

        return x0, dx, y

    @staticmethod
    def unpack_bounds(xbounds):
        try:
            start, end = xbounds
        except TypeError:
            start, end = None, None
        return start, end

    def tojson(self, xbounds, npts):  # this makes the current class json serializable
        data = []
        for x0, dx, y, label in self.data:
            x0, dx, y = self.get_slice(x0, dx, y, xbounds, npts)
            y = np.nan_to_num(y)  # replaces nan's with zeros and infinity with numbers
            data.append([x0.item() if hasattr(x0, 'item') else x0,
                         dx.item() if hasattr(dx, 'item') else dx,
                         y.tolist(), label or ''])

        # uncomment to have arough estimation of the file sizes (around 200 kb for 5 plots)
        # print len(json.dumps([self.title or '', data, "".join(self.warnings), self.xrange]))
        # set the title if there is only one item and a single label??
        return [self.title or '', data, self.warning, self.shapes]


def jsontimestamp(utctime, adjust_tzone=True):
    """Converts utctime (which MUST be in UTC!) by returning a timestamp for json transfer.
    Moreover, if `adjust_tzone=True` (the default), shifts utcdatetime timestamp so that
    a local browser can correctly display the time. This assumes that:
     - utctime is in UTC
     - the browser displays times in current timezone
    :param utctime: a timestamp (numeric) a datetime or an UTCDateTime. If numeric, the timestamp
    is assumed to be in *seconds**
    :return the unic timestamp (milliseconds)
    """
    try:
        time_in_sec = float(utctime)  # either a number or an UTCDateTime
    except TypeError:
        time_in_sec = float(UTCDateTime(utctime))  # maybe a datetime? convert to UTC
        # (this assumes the datetime is in UTC, which is the case)
    tdelta = 0 if not adjust_tzone else (datetime.fromtimestamp(time_in_sec) -
                                         datetime.utcfromtimestamp(time_in_sec)).total_seconds()

    return int(0.5 + 1000 * (time_in_sec - tdelta))


def downsample(array, npts):
    """Downsamples array for visualize it. Returns the tuple new_array, dx_ratio
    where new_array has at most 2*(npts+1) points and dx_ratio is a number defining the new dx
    in old dx units. This can be used to multiply the dx of array after calling this method
    Note that dx_ratio >=1, and if 1, array is returned as it is
    """
    # get minima and maxima
    # http://numpy-discussion.10968.n7.nabble.com/reduce-array-by-computing-min-max-every-n-samples-td6919.html
    offset = array.size % npts
    chunk_size = array.size / npts

    newdxratio = np.true_divide(chunk_size, 2)
    if newdxratio <= 1:
        return array, 1

    arr_slice = array[:array.size-offset] if offset > 0 else array
    arr_reshape = arr_slice.reshape((npts, chunk_size))
    array_min = arr_reshape.min(axis=1)
    array_max = arr_reshape.max(axis=1)

    # now 'interleave' min and max:
    # http://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    downsamples = np.empty((array_min.size + array_max.size + (2 if offset > 0 else 0),),
                           dtype=array.dtype)
    end = None if offset == 0 else -2
    downsamples[0:end:2] = array_min
    downsamples[1:end:2] = array_max

    # add also last element calculated in the remaining part (if offset=modulo is not zero)
    if offset > 0:
        arr_slice = array[array.size-offset:]
        downsamples[-2] = arr_slice.min()
        downsamples[-1] = arr_slice.max()

    return downsamples, newdxratio
