'''
Module implementing the jsplot a class which acts as a bridge between python objects
(traces, stream, numpy arrays) and their representation on a web page in javascript format

:date: Sep 22, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object

from datetime import datetime, timedelta

import numpy as np
from obspy.core import UTCDateTime  # , Stream, Trace, read


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

    def __init__(self, title=None, warnings=None):
        '''initialize a new Plot object. Use `add`, `addtrace` to populate it later (the methods
        can be chained as in jQuery syntax: plot = `Plot(...),addtrace(...)`).
        Have a look also at the static methods `fromtrace` and `fromstream` for creating Plot
        objects directly.
        Here you can customize the plot title and warning messages.
        Note that it is up to the frontend library managing `tojson` to display titles and warnings
        :param title: string, the plot title
        :param warnings: a string or a list of strings for one or more
            warnings. If falsy, the list wil be empty. `plot.warnings` is stored internally as a
            list of strings
        '''
        self.title = title or ''
        self.data = []  # a list of series. Each series is a list [x0, dx, np.asarray(y), label]
        self.is_timeseries = False
        if warnings:
            if isinstance(warnings, bytes):
                warnings = warnings.decode('utf8')
            if isinstance(warnings, str):
                warnings = [warnings]
        self.warnings = warnings or []

    def add(self, x0=None, dx=None, y=None, label=None):
        """Adds a new series (scatter line) to this plot.
        :param x0: (numeric, datetime, `:ref:obspy.UTCDateTime`) the x value of the first point.
            If `x0` **or** `dx` are time-domain values, then the following conversion will take
            place: `x0=UtcDateTime(x0)` (if `x0` is not an `UTCDateTime` object) and
            `dx=1000*dx.total_seconds()` (if `dx` is not a `timedelta` object).
            Note that a Plot can have series with all the same domain type, otherwise this method
            raises a ValueError
        :param dx: (numeric, timedelta): the value of the distance of two points on the x axis. See
            note above for `x0`.
        :param y: (numeric iterable, numpy array) the y values
        :param label: (string or None): the label of this series. Typically, this is the string
            displayed on the plot legend, if any
        :raise: ValueError if series with different domain types are added
        """
        verr = ValueError("conflicting x-domain types "
                          "(e.g., adding time-series and numeric-series to the same plot)")
        if isinstance(x0, (datetime, UTCDateTime)) or isinstance(dx, timedelta):
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
        '''Adds a trace to this plot.
        :param: label: the trace label (typically the name displayed on the plot legend). If None
            it defaults to `trace.get_id()` (the trace seed id)
        :raise: ValueError if this plot time domain (depending on what has already been added)
        is not compatible with time-series
        '''
        return self.add(trace.stats.starttime,
                        trace.stats.delta, trace.data,
                        trace.get_id() if label is None else label)

#     def merge(self, *plots):
#         '''Merge this Plot with all other plots. Preserves the `warnings` of the current plot
#         discarding potential `warnings` in the other plots
#         :return: A new Plot with all plots series
#         '''
#         ret = Plot(title=self.title, warnings=self.warnings)
#         ret.data = list(self.data)
#         # _warnings = []
#         for p in plots:
#             ret.data.extend(p.data)
#         return ret

    def tojson(self, xbounds=None, npts=-1):  # this makes the current class json serializable
        '''Returns a json-serializable representation of this Plot. Basically, it returns
         ```[self.title or '', data, "\n".join(self.warnings), self.is_timeseries]```
        where `data` (python list) is a serialized version of `self.data`
        (basically, `self.data` after converting numpy arrays to list): each `data` element
        represents a series of the plot in the form:
        ```[x0, dx, y, label]```
        :param xbounds: 2-element numeric tuple/list, or None (default:None): restrict all series
        of this plot to be trimmed between the given xbounds. if None, no trim is performed.
        Otherwise a tuple/list of [xstart, xend] values
        :param npts: integer (default:-1): resample all series to have for the given number of
        points. This number is useful for downsampling data to be displayed in devices whose
        pixel width is smaller than the series number of points: downsampling it here is
        faster as it avoids sending and processing redundant data: the new series of the plot
        will display for each "bin" (of a given length depending on `npts`) only the minimum and
        the maximum of that bin.
        '''
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
        return [self.title or '', data, "\n".join(self.warnings), self.is_timeseries]

    @staticmethod
    def get_slice(x0, dx, y, xbounds, npts):
        start, end = Plot.unpack_bounds(xbounds)

        if start is None and end is None and npts < 0:
            return x0, dx, y

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
    def fromstream(stream, title=None, warnings=None):
        p = Plot(title, warnings)
        same_trace = len(set([t.get_id() for t in stream])) == 1
        for i, t in enumerate(stream, 1):
            p.addtrace(t, 'chunk %d' % i if same_trace else t.get_id())
        if title is None and same_trace:
            p.title = stream[0].get_id()
        if same_trace and len(stream) > 1:
            p.warnings += ['gaps/overlaps']
        return p

    @staticmethod
    def fromtrace(trace, title=None, label=None, warnings=None):
        if title is None and label is None:
            title = trace.get_id()
            label = ''
        return Plot(title, warnings).addtrace(trace, label)

    def __str__(self, *args, **kwargs):
        return "js.Plot('%s')" % (self.title) + "\n%d line-series" % len(self.data) +\
             "\nis_timeseries: %s" % (self.is_timeseries) + "\nwarnings: " +"\tn".join(self.warnings)


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
