'''
Module to handle plots on the GUI. Most efforts are done here
in order to cache `obspy.traces` data and avoid re-querying it to the db, and caching `Plot`'s
objects (the objects representing a plot on the GUI) to avoid re-calculations, when possible.

The class implement a `PlotManager` which, as the name says, is a cache-like dict
(i.e., with limited size for memory performances, discarding old items first). The class
handles all the plots currently loaded in the GUI

The key and values of a `PlotManager` are each segment id, mapped to a list of two elements:
a) the `SegmentPlotList` of the unprocessed (raw) segment, and b) the `SegmentPlotList` of the
pre-processed one

A `SegmentPlotList` is a list-like class which stores all the plots (`jsplot.Plot` object)
for a given segment, according
to the config. Index 0 is always the plot of the segment obspy Stream (raw in case a)
or preprocessed in case b)), index 1 is always the stream signal and noise spectra, whose
windows are also defined in the config


:date: Jun 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import zip, range, object, dict

from collections import OrderedDict
from itertools import cycle

from obspy.core import Stream, Trace

from stream2segment.io.db.models import Segment
from stream2segment.utils import iterfuncs
from stream2segment.gui.webapp.mainapp.plots.jsplot import Plot
from stream2segment.process.utils import enhancesegmentclass, getseg, gui


class SegmentPlotList(list):
    """A SegmentPlotList is a class handling all the components
    traces and plots from the same event
    on the same station and channel. The components are usually the last channel letter
    (e.g. HHZ, HHE, HHN). The need of such a class is because shared data across different
    components on the same channel is sometimes required
    and needs not to be recalculated twice when possible
    """

    # set here what to set to None when calling invalidate (see __init__ for details):
    _data_to_invalidate = ['sn_windows']

    def __init__(self, segid, functions, other_components_id_list=None):
        """Builds a new SegmentPlotList
        :param functions: functions which must return a `Plot` (or an Plot-convertible object).
        **the first item MUST be `_getme` by default**
        """
        super(SegmentPlotList, self).__init__([None] * len(functions))
        self.functions = functions
        # use dict instead of {} to make it py2 compatible (see future imports at module's top)
        self.data = dict(stream=None, plot_title_prefix='',
                         **({k: None for k in self._data_to_invalidate}))
        self.segment_id = segid
        self.oc_segment_ids = other_components_id_list or []

    def copy(self):
        '''copies this SegmentPlotList with empty plots.
        All other data is shared with this object'''
        return SegmentPlotList(self.segment_id, self.functions, self.oc_segment_ids)

    def invalidate(self, hard=False):
        '''invalidates (sets to None) this object. Invalidate a value means setting it to None:
         -  if `hard=False` (the default) leaves the plot resulting from the segment
             without applying any function (usually at self[0]) and only `self.data` values whose
             keys are not in `self._data_to_invalidate` (this means usually setting to None all
             but data['stream'])
         -  if `hard=True` (currently not used but left for potential new features),
             invalidates all plots (elements of this list) and all `self.data`
             values
        '''
        index_of_main_plot = 0  # the index of the function returning the
        # trace plot (main plot returning the trace as it is)
        for key in self.data if hard else self._data_to_invalidate:
            self.data[key] = None
        for i in range(len(self)):
            if not hard and i == index_of_main_plot:
                continue
            self[i] = None

    def get_plots(self, session, plot_indices, inv_cache, config):
        '''
        Returns the list of `Plot`s representing the the custom functions of
        the segment identified by `seg_id. The length of the returned list equals
        `len(plot_indices)`. The list can be manipulated without affecting the stored internal list,
        the elements are passed by reference and thus each element modification affects the
        stored element
        :param seg_id: (integer) a valid segment id (i.e., must be the id of one of the segments
        passed in the constructor)
        :param inv: (inventory object) an object either inventory or exception
        (will be handled by `exec_function`, which is called internally)
        :param config: (dict) the plot config parsed from a user defined yaml file
        :param all_components_on_main_plot: (bool) if True, and the index of the main plot
        (usually 0) is in plot_indices, then the relative plot will show the stream and
        and all other components together
        '''
        with enhancesegmentclass(config):
            index_of_main_plot = 0  # the index of the function returning the
            # trace plot (main plot returning the trace as it is)

            ret = []
            for i in plot_indices:
                if self[i] is None:
                    # trace either trace or exception (will be handled by exec_function:
                    # skip calculation and return empty trace with err message)
                    # inv: either trace or exception (will be handled by exec_function:
                    # append err mesage to warnings and proceed to calculation normally)
                    self[i] = self.get_plot(self.functions[i], session, inv_cache, config,
                                            func_name='' if i == index_of_main_plot else None)
                ret.append(self[i])
            return ret

    def get_plot(self, func, session, invcache, config, func_name=None):
        '''Executes the given function on the given trace
        This function should be called by *all* functions returning a Plot object
        '''
        with enhancesegmentclass(config):
            try:
                funcres = self.exec_func(func, session, invcache, config)
                plt = self.convert2plot(funcres)
            except Exception as exc:
                # add dummy series (empty):
                plt = Plot('', warnings=str(exc)).add(0, 1, [])
            # set title:
            title_prefix = self.data.get('plot_title_prefix', '')
            if func_name is None:
                func_name = getattr(func, "__name__", "")
            if title_prefix or func_name:
                sep = " - " if title_prefix and func_name else ''
                plt.title = "%s%s%s" % (title_prefix, sep, func_name)
            return plt

    def exec_func(self, func, session, invcache, config):
        '''Executes the given function on the given segment identified by seg_id
           Returns the function result (or exception) after updating self, if needed.
           Raises if func raises
        '''
        with enhancesegmentclass(config):
            seg_id = self.segment_id
            segment = getseg(session, seg_id)
            # if stream has not been loaded, do it now in order to pass a copy of it to
            # `func`: this prevents in-place modifications:
            stream = self.data.get('stream', None)
            if stream is None:
                try:
                    stream = segment.stream()
                except Exception as exc:
                    stream = exc
                self.data['stream'] = stream
            segment._stream = stream.copy() if isinstance(stream, Stream) else stream
            inventory = invcache.get(seg_id, None)
            segment._inventory = inventory

            try:
                return func(segment, config)
            finally:
                # set back values if needed, even if we had exceptions.
                # Any of these values might be also an exception. Call the
                # 'private' attribute cause the relative method, if exists, most likely raises
                # the exception, it does not return it
                sn_windows = self.data.get('sn_windows', None)
                if sn_windows is None:
                    try:
                        self.data['sn_windows'] = segment.sn_windows()
                    except Exception as exc:
                        self.data['sn_windows'] = exc

                if inventory is None:
                    invcache[segment] = segment._inventory  # might be exc, or None
                # reset segment stream to None, for safety: we do not know if it refers
                # to a pre-processed stream or not, and thus segment._stream needs to be set from
                # self.data each time we are here. Note that this should not be a problem as
                # the web app re-initializes the session each time (thus each segment SHOULD have
                # no _stream attribute), but for safety we remove it:
                segment._stream = None
                if not self.data.get('plot_title_prefix', None):
                    title = None
                    if isinstance(segment._stream, Stream):
                        title = segment._stream[0].get_id()
                        for trace in segment._stream:
                            if trace.get_id() != title:
                                title = None
                                break
                    if title is None:
                        title = segment.seed_id
                    if title is not None:
                        # try to get it from the stream. Otherwise, get it from the segment
                        self.data['plot_title_prefix'] = title

    @staticmethod
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


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        super(LimitedSizeDict, self).__init__(*args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        super(LimitedSizeDict, self).__setitem__(key, value)
        self._check_size_limit()

    def update(self, *args, **kwargs):  # python2 compatibility (python3 calls __setitem__)
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 argument, got %d" % len(args))
            other = dict(args[0])
            for key in other:
                super(LimitedSizeDict, self).__setitem__(key, other[key])
        for key in kwargs:
            super(LimitedSizeDict, self).__setitem__(key, kwargs[key])
        self._check_size_limit()

    def setdefault(self, key, value=None):  # python2 compatibility (python3 calls __setitem__)
        if key not in self:
            self[key] = value
        return self[key]

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self._popitem_size_limit()

    def _popitem_size_limit(self):
        return self.popitem(last=False)


class InventoryCache(LimitedSizeDict):

    def __init__(self, size_limit=30):
        super(InventoryCache, self).__init__(size_limit=size_limit)
        self._segid2staid = dict()

    def __setitem__(self, segment, inventory_or_exception):
        if inventory_or_exception is None:
            return
        super(InventoryCache, self).__setitem__(segment.station.id, inventory_or_exception)
        self._segid2staid[segment.id] = segment.station.id

    def __getitem__(self, segment_id):
        inventory = None
        staid = self._segid2staid.get(segment_id, None)
        if staid is not None:
            inventory = super(InventoryCache, self).get(staid, None)
            if inventory is None:  # expired, remove the key:
                self._segid2staid.pop(segment_id)
        return inventory


class PlotManager(LimitedSizeDict):
    """
    PlotManager is a class which handles (with cache mechanism) all Plots of the program. It
    wraps a dict of segment ids mapped to ChannelComponentsPlotHandlers and returns the required
    plots to be displayed
    """
    def __init__(self, pymodule, config, size_limit=30):
        super(PlotManager, self).__init__(size_limit=size_limit)
        self.config = config
        self.functions = []
        self._functions_atts = []
        self.inv_cache = InventoryCache(10)

        # define default functions if not found:

        def preprocess_func(segment, config):
            raise Exception("No function decorated with '@gui.preprocess'")

        def main_function(segment, config):
            '''Returns the segment stream'''
            return Plot.fromstream(segment.stream())

        self.preprocessfunc = preprocess_func

        index = 1
        for func in iterfuncs(pymodule):
            att, pos, xaxis, yaxis = gui.get_func_attrs(func)
            if att == 'gui.preprocess':
                self.preprocessfunc = func
            elif att == 'gui.plot':
                self.functions.append(func)
                self._functions_atts.append({'name': func.__name__, 'index': index,
                                             'position': pos, 'xaxis': xaxis, 'yaxis': yaxis,
                                             'doc': func.__doc__})
                index += 1

        self.functions = [main_function] + self.functions

    @property
    def userdefined_plots(self):
        '''Returns a list of dicts, each dict denotes the properties of a user defined plot
        and has the key 'name', 'position', 'xaxis' and
        'yaxis'
        '''
        return self._functions_atts

    def get_doc(self, index=-1):
        '''Returns the documentation for the given custom function.
        :param index: if negative, returns the doc for the preprocess function, otherwise
        is the index of the i-th function (index 0 refers to the main function plotting the
        segment stream)
        '''
        try:
            ret = self.preprocessfunc.__doc__ if index < 0 else self._functions_atts[index].doc
            if not ret.strip():
                ret = "No function doc found: check GUI python file"
        except Exception as exc:  # pylint: disable=broad-except
            ret = "Error getting function doc:\n%s" % str(exc)
        return ret

    def get_plots(self, session, seg_id, plot_indices, preprocessed=False,
                  all_components_in_segment_plot=False):
        """Returns the plots representing the trace of the segment `seg_id` (more precisely,
        the segment whose id is `seg_id`). The returned plots will be the results of
        returning in a list `self.functions[i]` applied on the trace, for all `i` in `indices`

        :return: a list of `Plot`s according to `indices`. The index of the function returning
        the trace as-it-is is currently 0. If you want to display it, 0 must be in `indices`:
        note that in this case, if `all_components=True` the plot will have also the
        trace(s) of all the other components of the segment `seg_id`, if any.
        """
        with enhancesegmentclass(self.config):
            plotlist = self._getplotlist(session, seg_id, preprocessed)
            plots = plotlist.get_plots(session, plot_indices, self.inv_cache, self.config)
            index_of_main_plot = 0
            if index_of_main_plot in plot_indices and all_components_in_segment_plot:
                stream0 = plotlist.data['stream']
                if isinstance(stream0, Stream):
                    stream0 = stream0.copy()
                    warnings = []
                    title = plots[index_of_main_plot].title
                    for segid in plotlist.oc_segment_ids:
                        oc_plotlist = self._getplotlist(session, segid, preprocessed)
                        # force calculation of the main plot, which stores also the stream:
                        # (this will not computed twice if already computed)
                        _ = oc_plotlist.get_plots(session, [index_of_main_plot],
                                                  self.inv_cache, self.config)[0]
                        _stream = oc_plotlist.data['stream']
                        if isinstance(_stream, Stream):
                            stream0 += _stream
                        else:
                            streamid = oc_plotlist.data.get('plot_title_prefix', '')
                            if streamid:
                                streamid = streamid + ": "
                            warnings += ["%s%s" % (streamid, str(_stream))]
                    # get all other orientations (components) and merge them with the main plot:
                    plots[index_of_main_plot] = Plot.fromstream(stream0, title=title,
                                                                warnings=warnings or None)
            return plots

    def _getplotlist(self, session, seg_id, preprocessed=False):
        with enhancesegmentclass(self.config):
            plotlist, p_plotlist = self.get(seg_id, [None, None])

            if plotlist is None:
                seg = getseg(session, seg_id)
                segids = set([_[0]
                              for _ in seg._query_to_other_orientations(Segment.id)] + [seg_id])
                for segid in segids:
                    tmp = SegmentPlotList(segid, self.functions, segids - set([segid]))
                    self[segid] = [tmp, None]
                    if seg_id == segid:
                        plotlist = tmp

            if not preprocessed:
                return plotlist

            if p_plotlist is None:
                # Apply a pre-process to the given plotlist,
                # creating a copy of it and adding to the second element of self.plotlists
                stream = None
                try:
                    stream = plotlist.exec_func(self.preprocessfunc, session,
                                                self.inv_cache, self.config)
                    if isinstance(stream, Trace):
                        stream = Stream([stream])
                    elif not isinstance(stream, Stream):
                        raise Exception('pre_process function must return '
                                        'a Trace or Stream object')
                except Exception as exc:
                    stream = exc
                p_plotlist = plotlist.copy()
                p_plotlist.data['stream'] = stream
                self[seg_id][1] = p_plotlist

            return p_plotlist

    def get_data(self, segment_id, key, preprocessed, default_if_missing=None):
        try:
            plotlist = self[segment_id][0 if not preprocessed else 1]
        except (KeyError, IndexError):
            return default_if_missing
        return plotlist.data.get(key, default_if_missing)

    def update_config(self, **values):
        '''updates the current config and invalidates all plotcahce's, so that a query to
        their plots forces a recalculation (except for the main stream plot,
        currently at index 0)'''
        self.config.update(**values)
        for val in self.values():
            if val[0] is not None:
                val[0].invalidate()
            # pre-processed plotlist is set to None to force re-calculate all
            # as we cannot be sure if the main stream (resulting from preprocess func)
            # needed a config value
            val[1] = None

    def _popitem_size_limit(self):
        '''Called when super._check_size_limit is called. Remove also other components
        segmentPlotList(s)'''
        _, item = super(PlotManager, self)._popitem_size_limit()
        plotlist = item[0]
        for sid in plotlist.oc_segment_ids:
            self.pop(sid)
