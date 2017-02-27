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
from stream2segment.utils import get_session
from stream2segment.io.db.models import Segment, Class, ClassLabelling
from stream2segment.io.db.pd_sql_utils import colnames
from stream2segment.process.utils import get_stream, itercomponents
import numpy as np
from obspy.core.stream import Stream
from stream2segment.process.wrapper import get_inventory
from stream2segment.analysis.mseeds import bandpass, remove_response, stream_compliant, utcdatetime
from obspy.core.utcdatetime import UTCDateTime
from itertools import cycle, chain

MAX_NUM_PTS_TIMESCALE = 800
MAX_NUM_PTS_FREQSCALE = 300

def _get_session(app):
    # maybe not nicest way to store the session, but we want to avoid flask-sqlalchemy
    # for such a simple app
    key = '__DBSESSION__'
    if not app.config.get(key, None):
        sess = get_session(app.config['DATABASE_URI'])
        app.config[key] = sess

    return app.config[key]


def get_ids(session):
    segs = session.query(Segment.id).all()
    return {'segment_ids': [seg[0] for seg in segs]}


def get_num_custom_plots():
    return 1


def get_classes(session):
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

inventories = {}

config = dict(remove_response_water_level=60, remove_response_output='ACC',
              bandpass_freq_max=20,  # the max frequency, in Hz
              bandpass_max_nyquist_ratio=0.9,  # the amount of freq_max to be taken. low-pass corner = max_nyquist_ratio * freq_max (defined above)
              bandpass_corners=2 )



def get_data(session, seg_id, filtered, zooms):
    seg = session.query(Segment).filter(Segment.id == seg_id).first()
    segments = itercomponents(seg, session)
    try:
        stream0 = stream(seg)
    except Exception as exc:
        emptyjson = PlotData().tojson()
        return [emptyjson for _ in get_num_custom_plots()+3], exc

    prevlen = len(stream0)
    stream = get_stream(seg, session, True).merge(fill_value=0)
    if len(stream) != prevlen:
        emptyjson = PlotData().tojson()
        return [emptyjson for _ in get_num_custom_plots()+3], exc


    for stream in iterstream(seg, session, include_segment=True):
        if filtered:
            evt = seg.event
            stream[0] = bandpass(stream[0], evt.magnitude, freq_max=config['bandpass_freq_max'],
                                 max_nyquist_ratio=config['bandpass_max_nyquist_ratio'],
                                 corners=config['bandpass_corners'])
    
            # remove response
            inv = inventories.get(seg.station.id, None)
            if inv is None:
                try:
                    inv = get_inventory(seg.station, session)
                    inventories[seg.station.id] = inv
                except:
                    raise ValueError("Error while getting inventory")
            stream[0] = remove_response(stream[0], inv, output=config['remove_response_output'],
                                        water_level=config['remove_response_water_level'])

    return jsonify_(stream, zooms[0])  # use the zoom of the first plot


# @stream_compliant
# def jsonify_mseed(trace, zoom, compress=True):

#     starttime, endtime = zoom
#     start = utcdatetime(starttime)
#     end = utcdatetime(endtime)
# 
#     if start or end:
#         trace.trim(start, end, pad=True, nearest_sample=True, fill_value=0)
# 
#     # print trace.get_id()
#     datarow = [trace.get_id(),
#                toutc(float(trace.stats.starttime)),
#                float(trace.stats.delta),
#                None]
# 
#     datarow[-1] = downsample(trace.data, MAX_NUM_PTS_TIMESCALE)
#     # adjust sampling rate:
#     datarow[2] = (trace.stats.endtime - trace.stats.starttime) / (datarow[-1].size-1)
# 
#     datarow[-1] = datarow[-1].tolist()
#     # utcdatetime to float returns the seconds (as float). Js libraries want unix timestamps
#     # (milliseconds):
#     datarow[1] = int(1000*float(datarow[1])+0.5)
#     # the same for delta time:
#     datarow[2] *= 1000
#     # print datarow[-1]  # this is just for printing and inspecting the size via:
#     # http://bytesizematters.com/
#     return datarow




def downsample(array, npts):
    if array.size <= npts:
        return array

    # get minima and maxima
    # http://numpy-discussion.10968.n7.nabble.com/reduce-array-by-computing-min-max-every-n-samples-td6919.html
    offset = array.size % npts
    chunk_size = array.size / npts
    arr_slice = array[offset:] if offset > 0 else array
    arr_reshape = arr_slice.reshape((npts, chunk_size))
    array_min = arr_reshape.min(axis=1)
    array_max = arr_reshape.max(axis=1)

    shift = 0 if arr_slice is array else 2
    # now 'interleave' min and max:
    # http://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    downsamples = np.empty((array_min.size + array_max.size + shift,), dtype=array.dtype)
    downsamples[shift::2] = array_min
    downsamples[shift+1::2] = array_max

    # add also first part (if modulo is not zero)
    if shift != 0:
        arr_slice = array[:offset]
        downsamples[0] = arr_slice.min()
        downsamples[1] = arr_slice.max()

    return downsamples


def toutc(time_in_sec):
    tdelta = (datetime.fromtimestamp(time_in_sec) -
              datetime.utcfromtimestamp(time_in_sec)).total_seconds()

    return time_in_sec - tdelta


class PlotData(object):
    def __init__(self, title=None, x0=None, dx=None, y=None, label=None,
                 warnings=None):
        self.title = title or ''
        self.x0 = x0
        self.dx = dx
        self.y = y
        self.label = label
        self.warnings = warnings

    @staticmethod
    def fromsegment(segment, dofilter=False, remresp=False, zoom):
        try:
            stream = get_stream(segment)
        except Exception as exc:
            return PlotData(title=segment.channel_id, warnings=str(exc))

        prevlen = len(stream)
        stream = stream.merge(fill_value=0)
        if len(stream) != prevlen or len(stream) != 1:
            return PlotData(title=segment.channel_id, warnings="has gaps/overlaps")

        if dofilter:
            evt = segment.event
            stream[0] = bandpass(stream[0], evt.magnitude, freq_max=config['bandpass_freq_max'],
                                 max_nyquist_ratio=config['bandpass_max_nyquist_ratio'],
                                 corners=config['bandpass_corners'])

        if remresp:
            # remove response
            inv = inventories.get(segment.station.id, None)
            if inv is None:
                try:
                    inv = get_inventory(segment.station, segment.object_session())
                    inventories[segment.station.id] = inv
                except:
                    raise ValueError("Error while getting inventory")
            stream[0] = remove_response(stream[0], inv, output=config['remove_response_output'],
                                        water_level=config['remove_response_water_level'])

    @staticmethod
    def fromtrace(trace, zoom):
        starttime, endtime = zoom
        start = utcdatetime(starttime)
        end = utcdatetime(endtime)

        if start or end:
            trace.trim(start, end, pad=True, nearest_sample=True, fill_value=0)

        # print trace.get_id()
        datarow = [trace.get_id(),
                   toutc(float(trace.stats.starttime)),
                   float(trace.stats.delta),
                   None,
                   '',
                   '']

        datarow[-1] = downsample(trace.data, MAX_NUM_PTS_TIMESCALE)
        # adjust sampling rate:
        datarow[2] = (trace.stats.endtime - trace.stats.starttime) / (datarow[-1].size-1)

        datarow[-1] = datarow[-1].tolist()
        # utcdatetime to float returns the seconds (as float). Js libraries want unix timestamps
        # (milliseconds):
        datarow[1] = int(1000*float(datarow[1])+0.5)
        # the same for delta time:
        datarow[2] *= 1000
        # print datarow[-1]  # this is just for printing and inspecting the size via:
        # http://bytesizematters.com/
        return PlotData(*datarow)

    @staticmethod
    def float(val, default):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def tojson(self):
        return [self.title or '',
                PlotData.float(self.x0, 0),
                PlotData.float(self.dx, 1),
                self.y.tolist() if hasattr(self.y, 'tolist') else self.y,
                self.label or '',
                self.ann or ''
                ]

