'''
Module to handle plots on the GUI. Most efforts are done here
in order to cache `obspy.traces` data and avoid re-querying it to the db, and caching `Plot`'s
objects (the objects representing a plot on the GUI) to avoid re-calculations, when possible.

First of all, this module defines a
```Plot```
class which represents a Plot on the GUI. A Plot is basically a tuple (x0, dx, y, error_message)
and can be constructed from an obspy trace object very easily (`Plot.fromtrace`) or Stream 
(`Plot.fromstream`). `Plot.tojson` handles the conversion to json for sending the plot data as
web response: the method has optimized algorithms for
returning slices (in case of zooms) and only the relevant data for visualization by means of a
special down-sampling (if given, the screen number of pixels is usually much lower than the
plot data points)

Due to the fact that we need to synchronize when to load a segment and/or its components,
when requesting a segment all components are loaded together and their Plots are stored into a
```
ChannelComponentsPlotsHandler
```
object, which also stores and cache data (like the sn_windows data: signal-to-noise windows
according to the current config). All data in a ChannelComponentsPlotsHandler is cached whenever
possible. Note that if pre-process is
enabled, a clone of ChannelComponentsPlotsHandler is done, which works on the pre-processed stream (plots
are not shared across these ChannelComponentsPlotsHandler instances, whereas the sn_window data is)

Finally, this module holds a base manager class:
```
PlotManager
```
which, roughly speaking, stores all the ChannelComponentsPlotsHandler (s) loaded (with a size limit
for memory performances)

The user has then simply to instantiate a PlotManager object (currently stored inside the
Flask app config, but maybe we should investigate if it's a good practice): after that the method
```
PlotManager.getplots
```
do all the work of returning the requested plots, without re-calculating already existing plots,
without re-downloading already existing traces or inventories.
Then the user can call `.tojson` on any given plot and a response can be sent across the web

:date: Jun 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import zip, range, object, dict

from io import BytesIO
from itertools import cycle
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import load_only
# from sqlalchemy.sql.expression import and_, or_
# from sqlalchemy.orm.session import object_session
# from sqlalchemy.orm.util import aliased
from obspy.core import Stream, Trace, UTCDateTime, read

from stream2segment.analysis.mseeds import cumsum, cumtimes, fft
from stream2segment.io.db.models import Channel, Segment, Station
# from stream2segment.analysis import ampspec, powspec
from stream2segment.io.db.queries import getallcomponents
from stream2segment.utils.postdownload import SegmentWrapper, InventoryCache, LimitedSizeDict
from stream2segment.utils import iterfuncs


def exec_func(func, session, seg_id, plotscache, invcache, config):
    '''Executes the given function on the given segment identified by seg_id
        Returns the function result (or exception) after updating plotscache, if needed
    '''
    segwrapper = SegmentWrapper(config).reinit(session, seg_id,
                                               stream=plotscache.get_cache(seg_id, 'stream', None),
                                               inventory=invcache.get(seg_id, None),
                                               sn_windows=plotscache.get_cache(seg_id, 'sn_windows', None))

    try:
        return func(segwrapper, config)
    except Exception as exc:
        return exc
    finally:
        # set back values if needed, even if we had exceptions.
        # Any of these values might be also an exception. Call the
        # 'private' attribute cause the relative method, if exists, most likely raises
        # the exception, it does not return it
        if segwrapper._SegmentWrapper__stream is not None:
            plotscache.set_cache(seg_id, 'stream', segwrapper._SegmentWrapper__stream)
        if segwrapper._SegmentWrapper__sn_windows is not None:
            plotscache.set_cache(seg_id, 'sn_windows', segwrapper._SegmentWrapper__sn_windows)
        # allocate the segment if we need to set the title (might be None):
        segment = segwrapper._SegmentWrapper__segment
        if segwrapper._SegmentWrapper__inv is not None and segment is not None:
            # if inventory is not None and segment is None, we did pass inventory before func(...)
            # call, so no need to set it again
            invcache[segment] = segwrapper._SegmentWrapper__inv  # might be exception
        if not plotscache.get_cache(seg_id, 'plot_title_prefix', None):
            title = None
            if isinstance(segwrapper._SegmentWrapper__stream, Stream):
                title = segwrapper._SegmentWrapper__stream[0].get_id()
                for trace in segwrapper._SegmentWrapper__stream:
                    if trace.get_id() != title:
                        title = None
                        break
            if title is None and isinstance(segment, Segment):
                title = segment.strid
            if title is None:
                title = session.query(Segment).filter(Segment.id == seg_id).strid
            if title is not None:
                # try to get it from the stream. Otherwise, get it from the segment
                plotscache.set_cache(seg_id, 'plot_title_prefix', title)


def get_plot(func, session, seg_id, plotscache, invcache, config, func_name=None):
    '''Executes the given function on the given trace
    This function should be called by *all* functions returning a Plot object
    '''
    try:
        funcres = exec_func(func, session, seg_id, plotscache, invcache, config)
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
            elif type(funcres) == tuple:
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
                    raise ValueError(("Cannot create plot from %s (length=%d): ") % str(type(obj)))

    except Exception as exc:
        plt = Plot('', message=str(exc)).add(0, 1, [])

    # set title:
    title_prefix = plotscache.get_cache(seg_id, 'plot_title_prefix', '')
    if not func_name:
        func_name = getattr(func, "__name__", "")
    if title_prefix or func_name:
        sep = " - " if title_prefix and func_name else ''
        plt.title = "%s%s%s" % (title_prefix, sep, func_name)
    return plt


class ChannelComponentsPlotsHandler(object):
    """A ChannelComponentsPlotsHandler is a class handling all the components
    traces and plots from the same event
    on the same station and channel. The components are usually the last channel letter
    (e.g. HHZ, HHE, HHN). The need of such a class is because shared data across different 
    components on the same channel is sometimes required
    and needs not to be recalculated twice when possible
    """

    # set here what to set to None when calling invalidate (see __init__ for details):
    _cahce_values_to_invalidate = ['sn_windows']

    def __init__(self, segids, functions):
        """Builds a new ChannelComponentsPlotsHandler
        :param functions: functions which must return a `Plot` (or an Plot-convertible object).
        **the first item MUST be `_getme` by default**
        """
        self.functions = functions
        self.data = dict()  # make py2 compatible (see future imports at module's top)
        # data is a dict of type:
        # { ...,
        #   seg_id: {
        #            'plots': [plot1,...,plotN],
        #            'cache': {'stream': None, 'plot_title_prefix': None, 'sn_windows': None, ...}  # whatever will be added
        #           },
        #  ...
        # }

        # set data:
        for (segid,) in segids:
            self.data[segid] = dict(plots=[None] * len(self.functions),
                                    cache=dict(stream=None, sn_windows=None, plot_title_prefix=''))

    def copy(self):
        '''copies this ChannelComponentsPlotsHandler with empty plots.
        All other data is shared with this object'''
        return ChannelComponentsPlotsHandler([(s,) for s in self.data.keys()], self.functions)

    def invalidate(self):
        '''invalidates all the plots and other stuff which must be calculated to get them
        (setting what has to be invalidated to None) except the main trace plot'''
        index_of_traceplot = 0  # the index of the function returning the
        # trace plot (main plot returning the trace as it is)
        for segid in self.data:
            self.data[segid]['cache'] = self.data[segid]['cache'].copy()
            for key in self._cahce_values_to_invalidate:
                self.data[segid]['cache'][key] = None
            for i in range(len(self.data[segid]['plots'])):
                if i != index_of_traceplot:
                    self.data[segid]['plots'][i] = None

    def get_cache(self, segment_id, cachekey, default_if_missing=None):
        return self.data[segment_id]['cache'].get(cachekey, default_if_missing)

    def set_cache(self, segment_id, cachekey, value):
        self.data[segment_id]['cache'][cachekey] = value

#     def get_sn_windows(self, session, segment_id, config):
#         '''Returns the sn_windows for the stream of the given segment identified by its id
#         `seg_id` as the tuple
#         ```
#         [noise_start, noise_end], [signal_start, signal_end]
#         ```
#         all elements are `obspy` `UTCDateTime`s
#         raises `Exception` if the stream has more than one trace, or the
#         config values are not properly set
#         '''
#         try:
#             data = self.data
#             segw = SegmentWrapper().reinit(session, segment_id, config,
#                                            stream=data[segment_id]['stream'])
#             segw.stream('signal')  # calculates the sn windows
#             return segw._SegmentWrapper_sn_windows
#         except Exception as exc:
#             return exc

    def get_plots(self, session, seg_id, plot_indices, inv_cache, config,
                  all_components_on_main_plot=False):
        '''
        Returns the `Plot`s representing the the custom functions of
        the segment identified by `seg_id
        :param seg_id: (integer) a valid segment id (i.e., must be the id of one of the segments
        passed in the constructor)
        :param inv: (intventory object) an object either inventory or exception
        (will be handled by `exec_function`, which is called internally)
        :param config: (dict) the plot config parsed from a user defined yaml file
        :param all_components_on_main_plot: (bool) if True, and the index of the main plot
        (usually 0) is in plot_indices, then the relative plot will show the stream and
        and all other components together
        '''
        index_of_main_plot = 0  # the index of the function returning the
        # trace plot (main plot returning the trace as it is)
        plots = self.data[seg_id]['plots']

        # here we should build the warnings: check gaps /overlaps, check inventory exception
        # (pass None in case)
        # check that the SNWindow has a stream with one trace ()
        ret = []
        for i in plot_indices:
            if plots[i] is None:
                # trace either trace or exception (will be handled by exec_function:
                # skip calculation and return empty trace with err message)
                # inv: either trace or exception (will be handled by exec_function:
                # append err mesage to warnings and proceed to calculation normally)
                plots[i] = get_plot(self.functions[i], session, seg_id, self,
                                    inv_cache, config,
                                    func_name='' if i == index_of_main_plot else None)
            plot = plots[i]
            if i == index_of_main_plot and all_components_on_main_plot:
                # get all other components and merge them with the main plot
                other_comp_plots = []
                for other_comp_segid, _data in self.data.items():
                    if other_comp_segid == seg_id:
                        continue
                    _plots = _data['plots']
                    if _plots[index_of_main_plot] is None:
                        # see comments above
                        _plots[index_of_main_plot] = \
                            get_plot(self.functions[index_of_main_plot], session,
                                     other_comp_segid, self, inv_cache, config)
                    other_comp_plots.append(_plots[index_of_main_plot])
                if other_comp_plots:
                    plot = plot.merge(*other_comp_plots)  # returns a copy

            ret.append(plot)
        return ret


class PlotManager(object):
    """
    PlotManager is a class which handles (with cache mechanism) all Plots of the program. It
    wraps a dict of segment ids mapped to ChannelComponentsPlotHandlers and returns the required
    plots to be displayed
    """
    def __init__(self, pymodule, config):
        self._plotscache = LimitedSizeDict(size_limit=30)
        self.config = config
        self.functions = []
        self.inv_cache = InventoryCache(10)

        self._def_func_count = 2  # CHANGE THIS IF YOU CHANGE SOME FUNC BELOW

        # define default functions if not found:

        def preprocess_func(segment, config):
            raise Exception("No 'pre_process' function implemented")

        def sn_specrtra_func(segment, config):
            raise Exception("No 'sn_spectra' function implemented")

        def main_function(segment, config):
            return Plot.fromstream(segment.stream(), check_same_seedid=True)

        self.preprocessfunc = preprocess_func

        for f in iterfuncs(pymodule):
            if f.__name__ == 'pre_process':
                self.preprocessfunc = f
            elif f.__name__ == 'sn_spectra':
                sn_spectrafunc = f
            elif f.__name__.startswith('_'):
                continue
            else:
                self.functions.append(f)
        self.functions = [main_function, sn_spectrafunc] + self.functions

    @property
    def userdefined_plotnames(self):
        return [x.__name__ for x in self.functions[self._def_func_count:]]

    @property
    def get_preprocessfunc_doc(self):
        try:
            ret = self.preprocessfunc.__doc__
            if not ret.strip():
                ret = "No function doc found: check GUI python file"
        except Exception as exc:
            ret = "Error getting function doc:\n%s" % str(exc)
        return ret

    def getplots(self, session, seg_id, plot_indices, preprocessed=False, all_components=False):
        """Returns the plots representing the trace of the segment `seg_id` (more precisely,
        the segment whose id is `seg_id`). The returned plots will be the results of
        returning in a list `self.functions[i]` applied on the trace, for all `i` in `indices`

        :return: a list of `Plot`s according to `indices`. The index of the function returning
        the trace as-it-is is currently 0. If you want to display it, 0 must be in `indices`:
        note that in this case, if `all_components=True` the plot will have also the
        trace(s) of all the other components of the segment `seg_id`, if any.
        """
        return self._getplotscache(session, seg_id, preprocessed).\
            get_plots(session, seg_id, plot_indices, self.inv_cache, self.config, all_components)

#     def getpplots(self, session, seg_id, plot_indices, all_components=False):
#         """Returns the plots representing the pre-processed trace of the segment `seg_id`
#         (more precisely, the segment whose id is `seg_id`).
#         The pre-processed trace is a trace where the custom pre-process function is applied on.
#         The returned plots will be the results of
#         returning in a list `self.functions[i]` applied on the pre-processed trace,
#         for all `i` in `indices`
# 
#         :return: a list of `Plot`s according to `indices`. The index of the function returning
#         the (pre-processed) trace as-it-is is currently 0. If you want to display it, 0 must be in
#         `indices`: note that in this case, if `all_components=True` the plot will have also the
#         (pre-processed) trace(s) of all the other components of the segment `seg_id`, if any.
#         """
#         return self._getplotscache(session, seg_id, True).\
#             get_plots(session, seg_id, plot_indices, self.invcache, self.config, all_components)

    def _getplotscache(self, session, seg_id, preprocessed=False):
        plotscache, p_plotscache = self._plotscache.get(seg_id, [None, None])

        if plotscache is None:
            plotscache = ChannelComponentsPlotsHandler(getallcomponents(session, seg_id),
                                                       self.functions)
            # adds to the internal dict:
            for segid in plotscache.data:
                self._plotscache[segid] = [plotscache, None]

        if not preprocessed:
            return plotscache

        if p_plotscache is None:
            # Appliy a pre-process to the given plotscache,
            # creating a copy of it and adding to the second element of self._plotscache
            p_plotscache = plotscache.copy()
            for seg_id in plotscache.data.keys():
                stream = None
                try:
                    stream = exec_func(self.preprocessfunc, session, seg_id, plotscache,
                                       self.inv_cache, self.config)
                    if isinstance(stream, Trace):
                        stream = Stream([stream])
                    elif isinstance(stream, Exception):
                        raise stream
                    elif not isinstance(stream, Stream):
                        raise Exception('_pre_process function must return a Trace or Stream object')
                except Exception as exc:
                    stream = exc
                p_plotscache.set_cache(seg_id, 'stream', stream)
                self._plotscache[seg_id][1] = p_plotscache

        return p_plotscache

#     def get_warnings(self, seg_id, preprocessed):
#         '''Returns a list of strings denoting the warnings for the given 'seg_id'. Warnings
#         can be of three types:
#         1. Gaps/overlaps in trace
#         2. Inventory error (if inventories are required)
#         3. S/N window calculation errors (e.g., because of point 1., or malformed input params)
# 
#         The returned list will have at most three elements, where each element is a string
#         associated to one of the three problem above
# 
#         Note: if sn_windows are not calculated yet, this method calculates them
#         '''
#         ww = []
#         pltcache = self._pplots.get(seg_id, None) if preprocessed else self._plots.get(seg_id, None)
#         if pltcache is None:
#             return ww
# 
#         stream, _, _ = pltcache.data[seg_id]
#         if not isinstance(stream, Exception) and len(stream) != 1:
#             ww.append("%d traces (probably gaps/overlaps)" % len(stream))
# 
#         inv = self.inv_cache.get(seg_id, None)
#         if isinstance(inv, Exception):
#             ww.append("Inventory N/A: %s" % str(inv))
#             inv = None
# 
#         try:
#             pltcache.get_sn_windows(seg_id, self.config)
#         except Exception as exc:
#             ww.append(str(exc))
# 
#         return ww

    def get_cache(self, segment_id, cachekey, preprocessed, default_if_missing=None):
        try:
            plots_cache = self._plotscache[0 if not preprocessed else 1]
        except (KeyError, IndexError):
            return default_if_missing
        return plots_cache[segment_id].get_cache(segment_id, cachekey, default_if_missing)

#     def get_sn_windows(self, session, seg_id, preprocessed):
#         '''Returns the sn_windows for the stream of the given segment identified by its id
#         `seg_id` as the tuple
#         ```
#         ['noise_start], [noise_end], [signal_start, signal_end]
#         ```
#         all elements are `obspy` `UTCDateTime`s
#         raises `Exception` if the stream has more than one trace, or the
#         config values are not properly set
# 
#         :param preprocessed: whether or not to return the sn-window on the pre-processed
#         plotscache, if a `_pre_process` function is defined
#         '''
#         plots_cache = self._plots if not preprocessed else self._pplots
#         return plots_cache[seg_id].get(seg_id, 'sn_windows', [])

    def update_config(self, **values):
        '''updates the current config and invalidates all plotcahce's, so that a query to
        their plots forces a recalculation (except for the main stream plot,
        currently at index 0)'''
        self.config.update(**values)
        for v in self._plotscache.values():
            v[0].invalidate()
            v[1].invalidate()


# def getsegs(session, seg_id, all_components=True, load_only_ids=True, as_dict=False):
#     """Returns a list (`as_dict=False`) of segments or a dict of {segment.id: segment} key-values
#     (`as_dict=True`).
#     The list/dict will have just one item if
#     all_components=False, load_only_ids does what the name says"""
#     segquery = (getallcomponents(session, seg_id) if all_components else
#                 session.query(Segment).filter(Segment.id == seg_id))
#     if load_only_ids:
#         segquery = segquery.options(load_only(Segment.id))
#     return {s.id: s for s in segquery} if as_dict else segquery.all()


class Plot(object):
    """A plot is a class representing a Plot on the GUI"""

#     colors = cycle(["#1f77b4",
#                     "#aec7e8",
#                     "#ff710e",
#                     "#ffbb78",
#                     "#2ca02c",
#                     "#98df8a",
#                     "#d62728",
#                     "#ff9896",
#                     "#9467bd",
#                     "#c5b0d5",
#                     "#8c564b",
#                     "#c49c94",
#                     "#e377c2",
#                     "#f7b6d2",
#                     "#7f7f7f",
#                     "#c7c7c7",
#                     "#bcbd22",
#                     "#dbdb8d",
#                     "#17becf",
#                     "#9edae5"])

    def __init__(self, title=None, message=None):
        self.title = title or ''
        self.data = []  # a list of series. Each series is a list [x0, dx, np.asarray(y), label]
        self.is_timeseries = False
        self.message = message or ""

    def add(self, x0=None, dx=None, y=None, label=None):
        """Adds a new series (scatter line) to this plot. This method optimizes
        the data transfer and the line will be handled by the frontend plot library"""
        verr = ValueError("mixed x-domain types (e.g., times and numeric)")
        if isinstance(x0, datetime) or isinstance(x0, UTCDateTime) or isinstance(dx, timedelta):
            x0 = x0 if isinstance(x0, UTCDateTime) else UTCDateTime(x0)
            if isinstance(dx, timedelta):
                dx = dx.total_seconds()
            x0 = jsontimestamp(x0)
            dx = 1000 * dx
            if not self.is_timeseries and self.data:
                raise verr
            self.is_timeseries = True
        else:
            if self.is_timeseries:
                raise verr
        self.data.append([x0, dx, np.asarray(y), label])
        return self

    def addtrace(self, trace, label=None):
        return self.add(trace.stats.starttime,
                        trace.stats.delta, trace.data,
                        trace.get_id() if label is None else label)

    def merge(self, *plots):
        ret = Plot(title=self.title, message=self.message)
        ret.data = list(self.data)
        # _warnings = []
        for p in plots:
            ret.data.extend(p.data)
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
        # print len(json.dumps([self.title or '', data, "".join(self.message), self.xrange]))
        # set the title if there is only one item and a single label??
        return [self.title or '', data, self.message, self.is_timeseries]

    @staticmethod
    def get_slice(x0, dx, y, xbounds, npts):
        start, end = Plot.unpack_bounds(xbounds)
        if (start is not None and start >= x0 + dx * (len(y) - 1)) or \
                (end is not None and end <= x0):  # out of bounds. Treat it now cause maintaining
            # it below is a mess FIXME: we should write some tests here ...
            return x0, dx, []

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
    def fromstream(stream, title=None, message=None, check_same_seedid=False):
        p = Plot(title, message)
        seedid = None
        for i, t in enumerate(stream):
            p.addtrace(t)
            if i == 0:
                seedid = t.get_id()
            elif seedid is not None and seedid != t.get_id():
                msg = 'Different traces (seed id) in stream'
                if p.message:
                    p.messaage += "\n%s" % msg
                else:
                    p.messaage = msg
                seedid = None
        if title is None and seedid is not None:
            p.title = seedid
            for i, d in enumerate(p.data, 1):  # clear label for all line series (traces):
                d[-1] = 'chunk %d' % i
        return p

    @staticmethod
    def fromtrace(trace, title=None, label=None, message=None):
        if title is None and label is None:
            title = trace.get_id()
            label = ''
        return Plot(title, message).addtrace(trace, label)


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
    chunk_size = int(array.size / npts)

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
