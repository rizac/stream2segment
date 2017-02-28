'''
Created on Jul 31, 2016

@author: riccardo
'''
# from stream2segment.utils import get_session
# # from flask import 
# from stream2segment.io.db.models import Segment, Processing, Event, Station, Channel,\
#     DataCenter, Run, SegmentClassAssociation, Class
# # from stream2segment.classification import class_labels_df
# 
# import numpy as np
# # from numpy import interp
# from stream2segment.analysis.mseeds import cumsum, env, bandpass, amp_ratio,\
#     cumtimes, interpolate, dfreq
# from stream2segment.io.utils import loads    
# # from stream2segment.classification.handlabelling import ClassAnnotator
# from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing as kos
# from itertools import izip
# from obspy.core.stream import Stream
# from obspy.core.trace import Trace
# from obspy.core.utcdatetime import UTCDateTime
# import yaml
# from stream2segment.io.db.pd_sql_utils import commit, colnames
# from sqlalchemy.sql.sqltypes import Binary, DateTime
# from stream2segment.analysis import amp_spec, freqs, interp as analysis_interp
# from stream2segment.main import yaml_load
import time
from datetime import datetime

from stream2segment.io.db.models import Segment, Class, ClassLabelling
from stream2segment.io.db.pd_sql_utils import colnames
from stream2segment.process.utils import get_stream, itercomponents
import numpy as np
from obspy.core.stream import Stream
from stream2segment.process.wrapper import get_inventory
from stream2segment.analysis.mseeds import bandpass, remove_response, stream_compliant, utcdatetime,\
    snr, cumsum, cumtimes, fft, dfreq
from obspy.core.utcdatetime import UTCDateTime
from itertools import cycle, chain
from sqlalchemy.orm.session import Session
from stream2segment.analysis import amp_spec
import json
from stream2segment.gui.webapp import get_session
from collections import OrderedDict as odict


NPTS_WIDE = 900
NPTS_SHORT = 900


def get_ids():
    session = get_session()
    return {'segment_ids': [seg[0] for seg in session.query(Segment.id)]}


def get_num_custom_plots():
    return 1


def get_classes():
    session = get_session()
    clazzes = session.query(Class).all()
    ret = []
    colz = colnames(Class)
    for c in clazzes:
        row = {}
        for col in colz:
            row[col] = getattr(c, col)
        row['count'] = session.query(ClassLabelling).\
            filter(ClassLabelling.class_id == c.id).count()
        ret.append(row)
    return ret


def get_data(seg_id, filtered, zooms):
    session = get_session()
    seg = session.query(Segment).filter(Segment.id == seg_id).first()
    ret = [Plot.fromsegment(seg, zooms[0], NPTS_WIDE, filtered, copy=True)]

    for i, seg in enumerate(itercomponents(seg, session), 1):
        if i > 2:
            break
        ret.append(Plot.fromsegment(seg, zooms[i], NPTS_WIDE, False, False))

    try:
        trace = Traces.get(seg, filtered)
        cumtrace = cumsum(trace)
        ret.append(Plot.fromtrace_spectra(trace, zooms[3], NPTS_SHORT, seg.arrival_time,
                                          cumulative=cumtrace))
        ret.append(Plot.fromtrace(cumtrace, zooms[4], NPTS_SHORT,
                                  title=trace.get_id() + " - Cumulative"))
    except Exception as exc:  # @UnusedVariable
        ret.append(Plot())
        ret.append(Plot())

    return [r.tojson() for r in ret]


def downsample(array, npts):
    if array.size <= npts:
        return array

    # get minima and maxima
    # http://numpy-discussion.10968.n7.nabble.com/reduce-array-by-computing-min-max-every-n-samples-td6919.html
    offset = array.size % npts
    chunk_size = array.size / npts
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

    return downsamples


def jsontimestamp(utctime):
    """Converts utctime by returning a shifted timestamp such as
    browsers which assume times are local, will display the correct utc time
    :param utctime: a timestamp (numeric) a datetime or an UTCDateTime. If numeric, the timestamp
    is assumed to be in *seconds**
    :return the unic timestamp (milliseconds)
    """
    try:
        time_in_sec = float(utctime)  # either a number or an UTCDateTime
    except TypeError:
        time_in_sec = float(UTCDateTime(utctime))  # maybe a datetime? convert to UTC
        # (this assumes the datetime is in UTC, which is the case)

    tdelta = (datetime.fromtimestamp(time_in_sec) -
              datetime.utcfromtimestamp(time_in_sec)).total_seconds()

    return int(0.5 + 1000 * (time_in_sec - tdelta))


class CacheDict(odict):
    def __init__(self, size_limit, *args, **kwds):
        self.size_limit = size_limit
        super(CacheDict, self).__init__(*args, **kwds)
        # odict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        super(CacheDict, self).__setitem__(key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        rem_count = len(self) - self.size_limit
        if self.size_limit is None or rem_count <= 0:
            return
        toremove = []
        for i, k in enumerate(self.iterkeys()):
            if i >= rem_count:
                break
            toremove.append(k)
        for key in toremove:
            self.pop(key)


class Traces(object):

    inventories = CacheDict(50)
    streams = CacheDict(50)
    filter_func = None
    config = dict(remove_response_water_level=60, remove_response_output='ACC',
                  bandpass_freq_max=20,  # the max frequency, in Hz
                  bandpass_max_nyquist_ratio=0.9,  # the amount of freq_max to be taken. low-pass corner = max_nyquist_ratio * freq_max (defined above)
                  bandpass_corners=2)

    @classmethod
    def init(cls, filter_func=None, config=None):
        cls.filter_func = filter_func
        cls.config = config
        # re-set all filtered signals, as we changed the function
        for val in cls.streams.itervalues():
            val[1] = None

    @classmethod
    def get(cls, segment, filtered=False):
        streams = cls.streams.get(segment.id, None)

        stream0 = None if streams is None else streams[0]
        stream1 = None if streams is None else streams[1]  # unfiltered

        # streams is none
        if stream0 is None:  # if None, calculate stream0 in any case
            try:
                stream0 = get_stream(segment)
                prevlen = len(stream0)
                stream0 = stream0.merge(fill_value=0)
                if len(stream0) != 1:
                    raise ValueError("Has gaps/overlaps")
                if len(stream0) != prevlen:
                    trace0 = stream0[0]
                    trace0.stats._warning = ("Had gaps/overlaps (merged)")
                    trace1 = trace0.copy()
                    trace1.stats._warning = ("Had gaps/overlaps (merged): "
                                             "showing unfiltered trace")
                    stream1 = Stream([trace1])
                    cls.streams[segment.id] = [stream0, stream1]
                else:
                    stream1 = None  # this is True in any case: if filter=False, we will
                    # not calculate the branch below. If filter is True, we will force
                    # recalculation cause stream0 has just been calculated
                    if not filtered:  # we will not calculate the branch below, store:
                        cls.streams[segment.id] = [stream0, stream1]
            except Exception as exc:
                stream0, stream1 = exc, exc
                cls.streams[segment.id] = [stream0, stream1]

        if stream1 is None and filtered:
            # now do filtering
            # get response
            inv = cls.inventories.get(segment.station.id, None)
            if inv is None:
                try:
                    stream1 = Stream([stream0[0].copy()])
                    try:
                        inv = get_inventory(segment.station, Session.object_session(segment))
                    except Exception:
                        stream1[0].stats._warning = ("Error getting inventory. "
                                                     "showing unfiltered trace")
                    else:
                        stream1 = Stream([cls.filter(stream1[0], segment, inv, cls.config or {})])
                except Exception as exc:
                    stream1 = exc
            # write to cache dict:
            cls.streams[segment.id] = [stream0, stream1]

        stream = stream1 if filtered else stream0
        if isinstance(stream, Exception):
            raise stream

        return stream[0]

    @classmethod
    def filter(cls, trace, segment, inventory, config):
        if cls.filter_func is not None:
            return cls.filter_func(trace, segment, inventory, config)
        evt = segment.event
        trace = bandpass(trace, evt.magnitude, freq_max=config['bandpass_freq_max'],
                         max_nyquist_ratio=config['bandpass_max_nyquist_ratio'],
                         corners=config['bandpass_corners'], copy=False)
        trace.remove_response(inventory=inventory, output=config['remove_response_output'],
                              water_level=config['remove_response_water_level'])
        return trace


class Plot(object):

    inventories = CacheDict(50)
    segments = CacheDict(50)

    def __init__(self, title=None, x_range=None, warning=None):
        self.title = title or ''
        self.data = []
        self.xrange = x_range
        self.warnings = [warning] if warning else []

    def add(self, x0=None, dx=None, y=None, label=None, warning=None):
        if warning:
            self.warnings.append(warning)
        self.data.append([x0, dx, y, label])
        return self

    @classmethod
    def fromsegment(cls, segment, xbounds, npts, filtered=False, copy=False):
        """Returns trace, Plot. The first can be none, the second is the object
        which has a 'tojson' method to convert it to a json array

        If copy is False and either dofilter or remresp, then the trace is permanently
        modified when further accessed. So basically use copy=True for the main trace,
        and false otherwise assuming that we do not filter other components
        """
        try:
            trace = Traces.get(segment, filtered)
        except Exception as exc:
            return Plot(title=segment.channel_id,
                        x_range=[jsontimestamp(segment.start_time),
                                 jsontimestamp(segment.end_time)],
                        warning=str(exc))

        return Plot(title=segment.channel_id, warning=None if not hasattr(trace.stats, "_warning")
                    else trace.stats._warning).addtrace(trace, xbounds, npts)

    @staticmethod
    def fromtrace_spectra(trace, xbounds, npts, arrival_time,
                          atime_window='auto5-95%', cumulative=None, **kwargs):
        if atime_window == 'auto5-95%' or atime_window == 'auto10-90%':
            if cumulative is None:
                cumulative = cumsum(trace)
            if atime_window == 'auto5-95%':
                t0, t1 = cumtimes(cumulative, 0.05, 0.95)
            else:
                t0, t1 = cumtimes(cumulative, 0.1, 0.9)
            window_in_sec = t1 - t0
            range_signal = [t0, window_in_sec]
            range_noise = [arrival_time, -window_in_sec]
        else:
            range_signal = [arrival_time, atime_window]
            range_noise = [arrival_time, atime_window]

        fft_noise = fft(trace, *range_noise)
        fft_signal = fft(trace, *range_signal)

        df = dfreq(trace)
        f0 = 0

        amp_spec_noise, amp_spec_signal = amp_spec(fft_noise, True), amp_spec(fft_signal, True)
        start, end = Plot.unpack_bounds(xbounds)

        i0 = None if start is None else np.searchsorted(amp_spec_signal, start, side='left')
        i1 = None if end is None else np.searchsorted(amp_spec_signal, end, side='right')
        if i0 is not None or i1 is not None:
            amp_spec_noise, amp_spec_signal = amp_spec_noise[i0:i1], amp_spec_signal[i0:i1]
            f0 = start

        df_noise, amp_spec_noise = Plot.downsample(amp_spec_noise, df, npts)
        df_signal, amp_spec_signal = Plot.downsample(amp_spec_signal, df, npts)

        if "title" not in kwargs:
            kwargs["title"] = trace.get_id() + " - Spectra"

        return Plot(**kwargs).\
            add(f0, df_noise, amp_spec_noise, "Noise").\
            add(f0, df_signal, amp_spec_signal, "Signal")

    @staticmethod
    def _fromtrace(trace, xbounds, npts, **kwargs):
        starttime, endtime = Plot.unpack_bounds(xbounds)
        start = utcdatetime(starttime)
        end = utcdatetime(endtime)

        if start is not None or end is not None:
            trace = trace.copy()
            trace.trim(start, end, pad=True, nearest_sample=False, fill_value=0)

        x0 = jsontimestamp(trace.stats.starttime)
#         y = downsample(trace.data, npts)
#         # adjust sampling rate:
#         dx = (trace.stats.endtime - trace.stats.starttime) / (y.size-1)
#         # utcdatetime to float returns the seconds (as float). Js libraries want unix timestamps
#         # (milliseconds). x0 has already been converted, the same for delta time:
#         dx *= 1000
#         # print datarow[-1]  # this is just for printing and inspecting the size via:
#         # http://bytesizematters.com/

        dx, y = Plot.downsample(trace.data, trace.stats.delta, npts)

        # utcdatetime to float returns the seconds (as float). Js libraries want unix timestamps
#         # (milliseconds). x0 has already been converted, the same for delta time:
        dx *= 1000

        trace_id = trace.get_id()
        if "title" not in kwargs:
            kwargs["title"] = trace_id
        return x0, dx, y, trace_id

    @staticmethod
    def fromtrace(trace, xbounds, npts, **kwargs):
        x0, dx, y, trace_id = Plot._fromtrace(trace, xbounds, npts, **kwargs)
        if "title" not in kwargs:
            kwargs["title"] = trace_id
        return Plot(**kwargs).add(x0, dx, y, trace_id)

    def addtrace(self, trace, xbounds, npts):
        x0, dx, y, trace_id = Plot._fromtrace(trace, xbounds, npts)
        return self.add(x0, dx, y, trace_id)

    @staticmethod
    def unpack_bounds(xbounds):
        try:
            start, end = xbounds
        except TypeError:
            start, end = None, None
        return start, end

    @staticmethod
    def downsample(data, dx, npts):
        size = len(data)
        if size <= npts or npts < 1:
            return dx, data
        y = downsample(data, npts)
        new_dx = (dx * (size - 1)) / (len(y) - 1)
        return new_dx, y

    @staticmethod
    def float(val, default):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def tojson(self):  # this makes the current class json serializable
        data = []
        for x0, dx, y, label in self.data:
            data.append([Plot.float(x0, 0), Plot.float(dx, 1),
                         y.tolist() if hasattr(y, 'tolist') else y,
                         label or ''])
        # set the title if there is only one item and a single label??
        return [self.title or '',
                data,
                "".join(self.warnings),
                self.xrange
                ]
