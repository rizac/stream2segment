'''
Module to handle plots on the GUI. Most efforts are done here
in order to cache `obspy.traces` data and avoid re-querying it to the db, and caching `Plot`'s
objects (the objects representing a plot on the GUI) to avoid re-calculations, when possible.

The class implement a `PlotMAnager` which, as the name says, is a cache-like dict
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

from itertools import cycle, chain

from obspy.core import Stream, Trace

from stream2segment.io.db.models import Channel, Segment, Station
from stream2segment.io.db.queries import getallcomponents
from stream2segment.utils.postdownload import SegmentWrapper, InventoryCache, LimitedSizeDict
from stream2segment.utils import iterfuncs
from stream2segment.gui.webapp.plots.jsplot import Plot


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

    def __init__(self, segid, functions, other_components_id_list=[]):
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
        self.oc_segment_ids = other_components_id_list

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
        for key in (self.data if hard else self._data_to_invalidate):
            self.data[key] = None
        for i in range(len(self)):
            if not hard and i == index_of_main_plot:
                continue
            self[i] = None

    def get_plots(self, session, plot_indices, inv_cache, config):
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

        ret = []
        for i in plot_indices:
            if self[i] is None:
                # trace either trace or exception (will be handled by exec_function:
                # skip calculation and return empty trace with err message)
                # inv: either trace or exception (will be handled by exec_function:
                # append err mesage to warnings and proceed to calculation normally)
                self[i] = self.get_plot(self.functions[i], session, inv_cache,
                                        config, func_name='' if i == index_of_main_plot else None)
            ret.append(self[i])
        return ret

    def get_plot(self, func, session, invcache, config, func_name=None):
        '''Executes the given function on the given trace
        This function should be called by *all* functions returning a Plot object
        '''
        try:
            funcres = self.exec_func(func, session, invcache, config)
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
                        raise ValueError(("Cannot create plot from %s (length=%d): ") %
                                         str(type(obj)))

        except Exception as exc:
            plt = Plot('', warnings=str(exc)).add(0, 1, [])  # add dummy series (empty)

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
        seg_id = self.segment_id
        stream = self.data.get('stream', None)
        inventory = invcache.get(seg_id, None)
        segwrapper = SegmentWrapper(config).reinit(session, seg_id,
                                                   stream=stream,
                                                   inventory=inventory)

        try:
            return func(segwrapper, config)
        finally:
            # set back values if needed, even if we had exceptions.
            # Any of these values might be also an exception. Call the
            # 'private' attribute cause the relative method, if exists, most likely raises
            # the exception, it does not return it
            if stream is None:
                self.data['stream'] = segwrapper._SegmentWrapper__stream  # might be exc, or None
            sn_windows = self.data.get('sn_windows', None)
            if sn_windows is None:
                try:
                    self.data['sn_windows'] = segwrapper.sn_windows()
                except Exception as exc:
                    self.data['sn_windows'] = exc
                    
            if self.data.get('sn_windows', None) is None:
                sdf = 9
            # allocate the segment if we need to set the title (might be None):
            segment = segwrapper._SegmentWrapper__segment
            if inventory is None and segment is not None:
                invcache[segment] = segwrapper._SegmentWrapper__inv  # might be exc, or None
            if not self.data.get('plot_title_prefix', None):
                title = None
                if isinstance(segwrapper._SegmentWrapper__stream, Stream):
                    title = segwrapper._SegmentWrapper__stream[0].get_id()
                    for trace in segwrapper._SegmentWrapper__stream:
                        if trace.get_id() != title:
                            title = None
                            break
                if title is None and isinstance(segment, Segment):
                    title = segment.seed_identifier
                if title is None:
                    _ = session.query(Segment).filter(Segment.id == seg_id).first()
                    if _:
                        title = _.seed_identifier
                if title is not None:
                    # try to get it from the stream. Otherwise, get it from the segment
                    self.data['plot_title_prefix'] = title


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
        self.inv_cache = InventoryCache(10)

        self._def_func_count = 2  # CHANGE THIS IF YOU CHANGE SOME FUNC BELOW

        # define default functions if not found:

        def preprocess_func(segment, config):
            raise Exception("No function decorated with '@gui.preprocess'")

        def side_func(segment, config):
            raise Exception("No function decorated with '@gui.sideplot'")

        def main_function(segment, config):
            return Plot.fromstream(segment.stream(), check_same_seedid=True)

        self.preprocessfunc = preprocess_func

        for f in iterfuncs(pymodule):
            att = getattr(f, "_s2s_att", "")
            if att == 'gui.preprocess':
                self.preprocessfunc = f
            elif att == 'gui.sideplot':
                side_func = f
            elif att == 'gui.customplot':
                self.functions.append(f)
        self.functions = [main_function, side_func] + self.functions

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
        segplotlist = self._getsegplotlist(session, seg_id, preprocessed)
        plots = segplotlist.get_plots(session, plot_indices, self.inv_cache, self.config)
        index_of_main_plot = 0
        if index_of_main_plot in plot_indices and all_components_in_segment_plot:
            other_comp_plots = []
            for segid in segplotlist.oc_segment_ids:
                oc_segplotlist = self._getsegplotlist(session, segid, preprocessed)
                oc_plot = oc_segplotlist.get_plots(session, [index_of_main_plot], self.inv_cache,
                                                   self.config)[0]
                other_comp_plots.append(oc_plot)
                # get all other components and merge them with the main plot
            plots[index_of_main_plot] = plots[index_of_main_plot].merge(*other_comp_plots)
        return plots

    def _getsegplotlist(self, session, seg_id, preprocessed=False):
        segplotlist, p_segplotlist = self.get(seg_id, [None, None])

        if segplotlist is None:
            segids = set(_[0] for _ in getallcomponents(session, seg_id))
            for segid in segids:
                tmp = SegmentPlotList(segid, self.functions, segids - set([segid]))
                self[segid] = [tmp, None]
                if seg_id == segid:
                    segplotlist = tmp

        if not preprocessed:
            return segplotlist

        if p_segplotlist is None:
            # Appliy a pre-process to the given segplotlist,
            # creating a copy of it and adding to the second element of self.segplotlists
            for segplotlist in chain([segplotlist],
                                     (self[_][0] for _ in segplotlist.oc_segment_ids)):
                stream = None
                try:
                    stream = segplotlist.exec_func(self.preprocessfunc, session,
                                                   self.inv_cache, self.config)
                    if isinstance(stream, Trace):
                        stream = Stream([stream])
                    elif not isinstance(stream, Stream):
                        raise Exception('pre_process function must return a Trace or Stream object')
                except Exception as exc:
                    stream = exc
                tmp = segplotlist.copy()
                tmp.data['stream'] = stream
                self[tmp.segment_id][1] = tmp
                if tmp.segment_id == seg_id:
                    p_segplotlist = tmp

        return p_segplotlist

    def get_data(self, segment_id, key, preprocessed, default_if_missing=None):
        try:
            segplotlist = self.segplotlists[segment_id][0 if not preprocessed else 1]
        except (KeyError, IndexError):
            return default_if_missing
        return segplotlist.data.get(key, default_if_missing)

    def update_config(self, **values):
        '''updates the current config and invalidates all plotcahce's, so that a query to
        their plots forces a recalculation (except for the main stream plot,
        currently at index 0)'''
        self.config.update(**values)
        for v in self.values():
            if v[0] is not None:
                v[0].invalidate()
            # pre-processed segplotlist is set to None to force re-calculate all
            # as we cannot be sure if the main stream (resulting from preprocess func)
            # needed a config value
            v[1] = None

    def _popitem_size_limit(self):
        '''Called when super._check_size_limit is called. Remove also other components
        segmentPlotList(s)'''
        _, segplotlist = super(PlotManager, self)._popitem_size_limit()
        for sid in segplotlist.oc_segment_ids:
            self.pop(sid)
