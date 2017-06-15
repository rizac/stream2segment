'''
Module to handle plots on the GUI. Most efforts are done here
in order to cache `obspy.traces` data and avoid re-querying it to the db. Due to the fact that
we might also want to show a segment trace and all other components, a synchronization is also
in place that does not download (nor re-processes) other component traces, if they are requested
together with the current trace.

First of all, this module defines a
```Plot```
class which represents a Plot on the GUI. A Plot
can be a time-series, can be derived from a trace or a numpy array, and has the method 'tojson'
for sending data to a given web request. A Plot has furthermore optimized algorithms for
returning only the relevant data, as the screen number of pixels, usually in the order of
thousands, does not require all the points to be sent. Also, a Plot supports slicing (useful if
zooming is applicable on the web page, to return only the given portion of data to display)

A single segment displayed on the GUI is a so-called View: i.e., what we view on the page at 
any given selected segment. A View is basically a collection of Plots. Due to the fact that
we need to synchronize when to load a segment and/or its components, when requesting a semgent
all components are loaded togehter and their Views (collection of plots) are stored into a
```
ViewManager
```
Finally, this module holds a base manager class:
```
PlotManager
```
which, roughly speaking, stores inside three dictionaries:
 - a dict for the inventories, if needed (with a cache system in order to avoid re-downloading them)
 - a dict for all `ViewManager`s of the db segments (for a recorded stream on three components
 s1, s2, s3, all components are mapped to the same ViewManager object)
 - a dict for all `ViewManager`s of the db segments, with filter applied (same as above)

The user has then simply to instantiate a PlotManager object (currently stored inside the
Flask app config, but maybe we should investigate if it's a good practice): after that the methods
```
PlotManager.getplots
PlotManager.getfplots
```
do all the work of returning the requested plots, without re-calculating already existing plots,
without re-downloading already existing traces or inventories.
Then the user can call `.tojson` on any given plot and a response can be sent across the web

Created on Jun 8, 2017

@author: riccardo
'''
from io import BytesIO
from itertools import cycle, izip
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.sql.expression import and_, or_
from sqlalchemy.orm.session import object_session
from sqlalchemy.orm.util import aliased
from obspy.core import Stream, Trace, UTCDateTime, read
from obspy.geodetics.base import locations2degrees

from stream2segment.analysis.mseeds import cumsum, cumtimes, fft, get_bounds
from stream2segment.io.db.models import Channel, Segment
from stream2segment.analysis import amp_spec, pow_spec
from stream2segment.io.db.queries import getallcomponents
from stream2segment.download.utils import get_inventory
from stream2segment.utils import iterfuncs
from sqlalchemy.orm import load_only


def get_stream(segment):
    """Returns a Stream object relative to the given segment.
        :param segment: a model ORM instance representing a Segment (waveform data db row)
    """
    data = segment.data
    if not data:
        raise ValueError('no data')
    return read(BytesIO(segment.data))


def exec_function(func, trace, segment, inventory, config, warning, check_return_value=True):
    '''Executes the given function on the given trace
    This function should be called by *all* functions returning a Plot object
    '''
    title = plot_title(trace, segment, func)  # "%s - %s" % (main_plot.title, name)
    try:
        assertnoexc(trace)  # will raise if trace instanceof exception
        try:  # for inventories, append to existing warning on exception:
            assertnoexc(inventory)
        except Exception as exc:
            warning = "%s%s%s" % (warning, "\n" if warning else "",
                                  "Inventory error: %s" % str(exc))

        funcres = func(trace, segment, inventory, config, warning)
        if not check_return_value:
            return funcres

        # this should be called internally (not anymore, leave functionality):
        if isinstance(funcres, Plot):
            plt = funcres
        else:
            labels = cycle([None])
            if isinstance(funcres, Trace):
                itr = [funcres]
            elif isinstance(funcres, dict):
                labels = funcres.iterkeys()
                itr = funcres.itervalues()
            elif type(funcres) == np.ndarray:
                itr = [funcres]
            else:
                itr = funcres

            plt = Plot(title, warning)
            istimeseries = None
            mixintypeserr = ValueError("Return *either* custom-domain functions (x0, dx, y) "
                                       "*or* timeseries (traces or numpy arrays)")
            for label, obj in izip(labels, itr):
                if isinstance(obj, tuple):
                    if istimeseries is True:
                        raise mixintypeserr
                    istimeseries = False
                    if len(obj) != 3:
                        raise ValueError(("Expected tuple (x0, dx, y), "
                                          "found %d-elements tuple") % len(obj))
                    x0, dx, y = obj
                    plt.add(x0, dx, y, label)
                else:
                    if istimeseries is False:
                        raise mixintypeserr
                    istimeseries = True
                    if isinstance(obj, Trace):
                        plt.addtrace(obj, label)
                    else:
                        array = np.asarray(obj)
                        if array.size != trace.data.size:
                            raise ValueError("Expected array with %d elements, found %d" %
                                             (trace.data.size, array.size))
                        plt.addtrace(Trace(data=array, header=trace.stats.copy()), label)
    except Exception as exc:
        plt = Plot(title, warning=str(exc)).add(0, 1, [])

    return plt


def assertnoexc(obj):
    '''Raises obj if the latter is an Exception, otherwise does nothing.
    Convenience function used in `exec_function` for those objects (trace, inventory)
    which can also be an Exception'''
    if isinstance(obj, Exception):
        raise obj


def _get_me(trace, segment, inventory, config, warning):
    '''function returning the trace in the form of a Plot object'''
    # note that returning a Plot does not set the title cause we provide it here
    # note also that all other exception handling still works (e.g., if trace is an exception)
    return Plot.fromtrace(trace, trace.get_id(), warning)


# def _spectra(spectrum_func, trace, segment, inventory, config, warning, postmanager):
#     '''function returning the spectra (noise, signal) of a trace in the form of a Plot object'''
#     # warning = ""
#     noisy_wdw_args, signal_wdw_args = get_spectra_windows(config, segment.arrival_time, trace)
# 
#     noise_trace = trace.copy().trim(starttime=noisy_wdw_args[0], endtime=noisy_wdw_args[1],
#                                     pad=True, fill_value=0)
#     signal_trace = trace.copy().trim(starttime=signal_wdw_args[0], endtime=signal_wdw_args[1],
#                                      pad=True, fill_value=0)
# 
#     noise_trace
#     noise_trace.trim()
#     plot1 = exec_function(spectrum_func, signal_trace, segment, inventory, config, warning,
#                           check_return_value=False)
#     plot2 = exec_function(spectrum_func, noise_trace, segment, inventory, config, warning,
#                           check_return_value=False)
# 
#     plot1.title = 'Signal'
#     plot2.title = 'Noise'
#     merged = plot1.merge(plot2)
#     merged.title = plot_title(trace, segment, "spectra")
#     return merged

#     try:
#     fft_noise = fft(trace, *noisy_wdw)
#     fft_signal = fft(trace, *signal_wdw)
# 
#     df = fft_signal.stats.df
#     f0 = 0
# 
#     if config.get('spectra', {}).get('type', 'amp') == 'pow':
#         func = pow_spec
#     else:
#         func = amp_spec
#     # note below: amp_spec(array, True) (or pow_spec, it's the same) simply returns the
#     # abs(array) which apparently works also if array is an obspy Trace. To avoid problems pass
#     # the data value
#     spec_noise, spec_signal = func(fft_noise.data, True), func(fft_signal.data, True)
# 
#     if postprocess_func is not None:
#         f0noise, dfnoise, spec_noise = postprocess_func(f0, df, spec_noise)
#         f0signal, dfsignal, spec_signal = postprocess_func(f0, df, spec_signal)
#     else:
#         f0noise, dfnoise, f0signal, dfsignal = f0, df, f0, df
# 
#     return Plot(plot_title(trace, segment, "spectra"), warning=warning).\
#         add(f0noise, dfnoise, spec_noise, "Noise").\
#         add(f0signal, dfsignal, spec_signal, "Signal")


# def get_spectra_windows(config, a_time, trace):
#     '''Returns the spectra windows from a given arguments. Used by `_spectra`
#     :return the tuple (start, end), (start, end) where all arguments are `UTCDateTime`s
#     and the first tuple refers to the noisy window, the latter to the signal window
#     '''
#     try:
#         a_time = UTCDateTime(a_time) + config['spectra']['arrival_time_shift']
#         # Note above: UTCDateTime +float considers the latter in seconds
#         # we need UTcDateTime cause the spectra function we implemented accepts that object type
#         try:
#             cum0, cum1 = config['spectra']['signal_window']
#             t0, t1 = cumtimes(cumsum(trace), cum0, cum1)
#             nsy, sig = [a_time, t0 - t1], [t0, t1 - t0]
#         except TypeError:
#             shift = config['spectra']['signal_window']
#             nsy, sig = [a_time, -shift], [a_time, shift]
#         return nsy, sig
#     except Exception as err:
#         raise ValueError("%s (check config)" % str(err))


def plot_title(trace, segment, func_or_funcname):
    """
    Creates a title for the given function (or function name)
    from the given trace. Basically returns
    ```trace.get_id() + " - " + func_or_funcname```
    if trace is an Exception (resulting from an error when reading the trace), the segment is
    used to retrieve trace.get_id() in the classical form:
    ```network.station.location.channel```
    """
    try:
        id_ = trace.get_id()
    except AttributeError:
        id_ = segment.seed_identifier
        if not id:
            id_ = ".".join([segment.station.network, segment.station.station,
                           segment.channel.location, segment.channel.channel])
    try:
        funcname = func_or_funcname.__name__
    except AttributeError:
        funcname = str(func_or_funcname)
    return "%s - %s" % (id_, funcname)


class ViewManager(object):
    """A ViewManager is a class handling all the components traces from the same event
    on the same station and channel. The components are usually the last channel letter
    (e.g. HHZ, HHE, HHN)
    """

    def __init__(self, traces, seg_ids, warnings, functions):
        """Builds a new ViewManager
        :param functions: functions which must return a `Plot` (or an Plot-convertible object).
        **the first item MUST be `_getme` by default**
        """
        # super(View, self).__init__((None for _ in functions))
        # append default custom functions. Use self cause they avoid copying the trace,
        # and in case of fft they need to access object attributes
        self.functions = functions
        self.data = {segid: [t,  [None] * len(self.functions), w] for
                     t, segid, w in izip(traces, seg_ids, warnings)}
#         self.data = {seg.id: [traces[i], [None] * len(self.functions), warnings[i]]
#                      for i, seg in enumerate(segments)}

    def get_plots(self, session, seg_id, inv, config, all_components=False, *indices):
        '''
        Returns the `Plot`s representing the the custom functions of
        the segment identified by `seg_id
        :param seg_id: (integer) a valid segment id (i.e., must be the id of one of the segments
        passed in the constructor)
        :param inv: (intventory object) an object either inventory or exception
        (will be handled by `exec_function`, which is called internally)
        :param config: (dict) the plot config parsed from a user defined yaml file
        :param all_components: (bool) if True, a list of N `Plot`s will be returned, where the first
        item is the plot of the segment whose id is `seg_id`, and all other plots are the
        other components. N is the length of the `segments` list passed in the constructor. If
        False, a list of a single `Plot` (relative to `seg_id`) will be returned
        :return: a list of `Plot`s
        '''
        segments = getsegs(session, seg_id, all_components=all_components, as_dict=True)
        index_of_traceplot = 0  # the index of the function returning the
        # trace plot (main plot returning the trace as it is)
        trace, plots, warning = self.data[seg_id]
        ret = []
        for i in indices:
            if plots[i] is None:
                # trace either trace or exception (will be handled by exec_function:
                # skip calculation and return empty trace with err message)
                # inv: either trace or exception (will be handled by exec_function:
                # append err mesage to warnings and proceed to calculation normally)
                plots[i] = exec_function(self.functions[i], trace, segments[seg_id],
                                         inv, config, warning)
            plot = plots[i]
            if i == index_of_traceplot and all_components:
                # get all other components:
                other_comp_plots = []
                for segid, _data in self.data.iteritems():
                    if segid == seg_id:
                        continue
                    _trace, _plots, _warning = _data
                    if _plots[index_of_traceplot] is None:
                        # see comments above
                        _plots[index_of_traceplot] = \
                            exec_function(self.functions[index_of_traceplot], _trace,
                                          segments[segid], inv, config, _warning)
                    other_comp_plots.append(_plots[index_of_traceplot])
                if other_comp_plots:
                    plot = plot.merge(*other_comp_plots)  # returns a copy

            ret.append(plot)
        return ret


class PlotManager(object):
    """
    PlotManager is a class which handles (with cache mechanism) all Plots of the program
    """
    def __init__(self, pymodule, config):
        self._views = {}  # seg_id (str) to view
        self._fviews = {}  # seg_id (str) to view
        self.config = config
        self.functions = [_get_me, _spectra]
        self._def_func_count = len(self.functions)
        # by default, filter and spectrum function raise an exception: 'no func set'
        # if they are defined in the config, they will be overridden below
        # meanwhile they raise, and as lambda function cannot raise, we make use of our
        # 'assertnoexc' function which is used in execfunction and comes handy here:
        self.filterfunc = lambda *a, **v: assertnoexc(Exception("No `filter` function set"))
        self.soectrumfunc = lambda *a, **v: assertnoexc(Exception("No `spectrum` function set"))
        for f in iterfuncs(pymodule):
            if f.__name__.startswith('_'):
                continue
            if f.__name__ == 'filter':
                self.filterfunc = f
            elif f.__name__ == 's2n_spectra':
                self.spectrumfunc = f
            else:
                self.functions.append(f)
        self._inv_cache = {}  # station_id -> obj (inventory, exception or None)
        # this is used to skip download if already present
        self.segid2inv = {}  # segment_id -> obj (inventory, exception or None)
        # this is used to get an inventory given a segment.id avoiding looking up its station
        self.use_inventories = config.get('inventory', False)
        self.save_inventories = config.get('save_downloaded_inventory', False)

    @property
    def userdefined_plotnames(self):
        return [x.__name__ for x in self.functions[self._def_func_count:]]

    def getplots(self, session, seg_id, all_components=False, *indices):
        """Returns the plots representing the trace of the segment `seg_id` (more precisely,
        the segment whose id is `seg_id`). The returned plots will be the results of
        returning in a list `self.functions[i]` applied on the trace, for all `i` in `indices`

        :return: a list of `Plot`s according to `indices`. The index of the function returning
        the trace as-it-is is currently 0. If you want to display it, 0 must be in `indices`:
        note that in this case, if `all_components=True` the plot will have also the
        trace(s) of all the other components of the segment `seg_id`, if any.
        """
        return self._getplots(session, seg_id, self._getviewmanager(session, seg_id), all_components,
                              *indices)

    def _loadsegment(self, session, seg_id):
        # load segments:
        # traces = []
        warnings = []
        segments = getsegs(session, seg_id)
        traces = []
        seg_ids = []
        for segment in segments:
            seg_ids.append(segment.id)
            warning = ''
            try:
                stream = get_stream(segment)
                prevlen = len(stream)
                if prevlen > 1:
                    stream = stream.merge(fill_value='latest')
                    if len(stream) != 1:
                        raise ValueError("Gaps/overlaps (unmergeable)")
                    if len(stream) != prevlen:
                        warning = "Gaps/overlaps (merged)"
                traces.append(stream[0])
            except Exception as exc:
                traces.append(exc)
            warnings.append(warning)

        # set the inventory if needed from the config. Otherwise, store it as None:
        inventory = None
        if self.config.get('inventory', False):
            segment = segments[0]
            sta_id = segment.channel.station_id  # should be faster than segment.station.id
            inventory = self._inv_cache.get(sta_id, None)
            if inventory is None:
                try:
                    inventory = get_inventory(segment.station,
                                              self.config['save_downloaded_inventory'])
                except Exception as exc:
                    inventory = exc
                self._inv_cache[sta_id] = inventory
        for segid in seg_ids:
            self.segid2inv[segid] = inventory

        view = ViewManager(traces, seg_ids, warnings, self.functions)
        for segid in seg_ids:
            self._views[segid] = view
        return view

    def getfplots(self, session, seg_id, all_components=False, *indices):
        """Returns the plots representing the filtered trace of the segment `seg_id`
        (more precisely, the segment whose id is `seg_id`).
        The filtered trace is a trace where the custom filter function is applied on.
        The returned plots will be the results of
        returning in a list `self.functions[i]` applied on the filtered trace,
        for all `i` in `indices`

        :return: a list of `Plot`s according to `indices`. The index of the function returning
        the (filtered) trace as-it-is is currently 0. If you want to display it, 0 must be in
        `indices`: note that in this case, if `all_components=True` the plot will have also the
        (filtered) trace(s) of all the other components of the segment `seg_id`, if any.
        """
        viewmanager = self._fviews.get(seg_id, None)
        if viewmanager is None:
            orig_viewmanager = self._getviewmanager(session, seg_id)
            viewmanager = self._filter(session, orig_viewmanager)  # also adds to self._fviews

        return self._getplots(session, seg_id, viewmanager, all_components, *indices)

    def _getviewmanager(self, session, seg_id):
        viewmanager = self._views.get(seg_id, None)
        if viewmanager is None:
            viewmanager = self._loadsegment(session, seg_id)  # adds to the internal dict
        return viewmanager

    def _filter(self, session, viewmanager):
        filt_traces = []
        # need to create new arrays also for segments and warnings to preserve order:
        segments = None
        seg_ids = []
        warnings = []
        for segid, _data in viewmanager.data.iteritems():
            tra, _, warn = _data
            if segments is None:
                segments = getsegs(session, segid, as_dict=True)

            seg_ids.append(segid)  # we might avoid this, it's just to be passed to ViewManager...
            warnings.append(warn)
            try:
                if not isinstance(tra, Exception) and self.filterfunc is None:
                    tra = Exception("No filter function set")
                assertnoexc(tra)
                filttrace = self.filterfunc(tra, segments[segid], self.segid2inv[segid], self.config, warn)  # FIXME: warn here should do nothing actually. Skip it?
                if not isinstance(filttrace, Trace):
                    raise Exception('filter function must return a Trace object')
            except Exception as exc:
                filttrace = exc
            filt_traces.append(filttrace)

        view = ViewManager(filt_traces, seg_ids, warnings, self.functions)
        for segid in seg_ids:
            self._fviews[segid] = view
        return view

    def _getplots(self, session, seg_id, viewmanager, all_components=False, *indices):
        """Returns the View for a givens segment
        The segment trace and potential errors to be displayed are saved as `View` class `dict`s
        """
        return viewmanager.get_plots(session, seg_id, self.segid2inv[seg_id], self.config, all_components,
                                     *indices)
        # return [allcomps, viewmanager.get_custom_plots(seg_id, inv, config, *indices)]

    def get_spectra_windows(self, session, seg_id):
        viewmanager = self._getviewmanager(seg_id)
        for t, s in izip(viewmanager.traces, viewmanager.segments):
            if s.id == seg_id:
                return get_spectra_windows(self.config, s.arrival_time, t)
        raise ValueError('%d not found' % seg_id)


def getsegs(session, seg_id, all_components=True, load_only_ids=True, as_dict=False):
    """Returns a list (`as_dict=False`) of segments or a dict of {segment.id: segment} key-values
    (`as_dict=True`).
    The list/dict will have just one item if
    all_components=False, load_only_ids does what the name says"""
    segquery = (getallcomponents(session, seg_id) if all_components else
                session.query(Segment).filter(Segment.id == seg_id))
    if load_only_ids:
        segquery = segquery.options(load_only(Segment.id))
    return {s.id: s for s in segquery} if as_dict else segquery.all()


class Plot(object):
    """A plot is a class representing a Plot on the GUI"""

    def __init__(self, title=None, warning=None):
        self.title = title or ''
        self.data = []  # a list of linesereies. Each linesereies is a list [x0, dx, np.asarray(y), label]
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

    def merge(self, *plots):
        ret = Plot(title=self.title, warning=self.warning)
        ret.data = list(self.data)
        ret.shapes = list(self.shapes)
        # _warnings = []
        for p in plots:
            ret.data.extend(p.data)
            ret.shapes.extend(p.shapes)
        #   ret.warnings.append(p.warning)
        # ret.data = data
        # ret.shapes = shapes
        # ret.warning = '\n'.join(set(warnings))  # set warnings gets only unique warnings
        return ret

    def tojson(self, xbounds, npts):  # this makes the current class json serializable
        data = []
        for x0, dx, y, label in self.data:
            x0, dx, y = self.get_slice(x0, dx, y, xbounds, npts)
            y = np.nan_to_num(y)  # replaces nan's with zeros and infinity with numbers
            data.append([x0.item() if hasattr(x0, 'item') else x0,
                         dx.item() if hasattr(dx, 'item') else dx,
                         y.tolist(), label or ''])

        # uncomment to have a rough estimation of the file sizes (around 200 kb for 5 plots)
        # print len(json.dumps([self.title or '', data, "".join(self.warnings), self.xrange]))
        # set the title if there is only one item and a single label??
        return [self.title or '', data, self.warning, self.shapes]

    @staticmethod
    def get_slice(x0, dx, y, xbounds, npts):
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

        return x0, dx, y

    @staticmethod
    def unpack_bounds(xbounds):
        try:
            start, end = xbounds
        except TypeError:
            start, end = None, None
        return start, end

    @staticmethod
    def fromtrace(trace, label=None, warning=None):
        return Plot(label, warning).addtrace(trace)


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
