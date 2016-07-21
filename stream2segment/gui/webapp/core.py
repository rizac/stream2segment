'''
Created on Jun 20, 2016

@author: riccardo
'''
from stream2segment.s2sio.db import ListReader
import numpy as np
from stream2segment.analysis import moving_average
from numpy import interp
from stream2segment.analysis.mseeds import cumsum, snr, env, bandpass, freq_stream, amp_ratio,\
    cumtimes, interpolate
from stream2segment.classification.handlabelling import ClassAnnotator
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing as kos
from itertools import izip
from obspy.signal.util import smooth
# from numpy import nanmax
# from obspy.core.utcdatetime import UTCDateTime
# from future.backports.datetime import timezone
listreader = None
class_ids = None
classannotator = None
SNR_WINDOW_SIZE_IN_SECS = 40  # FIXME: add to the config!!


def get_listreader(db_uri, class_ids_=None):
    global listreader
    global class_ids
    if listreader is None or class_ids_ != class_ids:
        if class_ids_:
            def filter_func(dframe):
                return dframe[dframe['ClassId'].isin(class_ids)]
        else:
            filter_func = None
        class_ids = class_ids_
        listreader = ListReader(db_uri, filter_func=filter_func,
                                sort_columns=["#EventID", "EventDistance/deg"],
                                sort_ascending=[True, True])

#         listreader2 = ListReader(db_uri, filter_func=None,
#                                  sort_columns=["#EventID", "EventDistance/deg"],
#                                  sort_ascending=[True, True])
#         listreader2.filter(listreader2.T_EVT.Magnitude.between(3, 3.1))
        global classannotator
        classannotator = ClassAnnotator(listreader)
    return listreader


def get_ids(db_uri, class_ids=[]):
    listreader = get_listreader(db_uri, class_ids)

    # NOTE: we need conversion to string cause apparently jsonify does some rounding on big ints
    # FIXME: CHECK!!!
    return {'segment_ids': tojson(listreader.mseed_ids['Id'].values.astype(str)),
            'classes':  classannotator.get_classes_df().to_dict('records')}


def get_data(db_uri, seg_id):
    listreader = get_listreader(db_uri)
    db_row = listreader.get(seg_id, listreader.T_SEG)
    stream = listreader.get_stream(seg_id, include_same_channel=True)
    filtered_stream = bandpass(stream)
    cumulative_trace = cumsum(filtered_stream[0])
    snr_stream = freq_stream(filtered_stream[0], db_row.iloc[0]['ArrivalTime'],
                             SNR_WINDOW_SIZE_IN_SECS)
    evlp_trace = env(filtered_stream[0])

    # calculate "Numbers" (scalar info):
    _cum_t5, _cum_t95 = cumtimes(cumulative_trace, 0.05, 0.95)
    _amp_ratio = amp_ratio(stream[0])
    _snr = snr(snr_stream[1], snr_stream[0])

    # define interpolation values
    MAX_NUM_PTS_TIMESCALE = 1100
    MAX_NUM_PTS_FREQSCALE = 200

    # interpolate and return:
    times, stream = interpolate(stream, MAX_NUM_PTS_TIMESCALE, align_if_stream=True,
                                return_x_array=True)

    filtered_stream = interpolate(filtered_stream, times, align_if_stream=True)
    cumulative_trace = interpolate(cumulative_trace, times)
    bwd = 1200000000
#     evlp_trace_data = kos(interpolate(evlp_trace, times).data,
#                           times, bandwidth=bwd)
    evlp_trace_data = moving_average(interpolate(evlp_trace, times).data, 50)

    time_data = {'labels': tojson(np.round(times * 1000.0)), 'datasets': []}
    datasets = time_data['datasets']

    title = stream[0].id
    datasets.append(to_chart_dataset(stream[0].data, title))
    datasets.append(to_chart_dataset(filtered_stream[0].data, title + " (Filtered)"))
    # append other two traces:
    datasets.append(to_chart_dataset(stream[1].data, stream[1].id))
    datasets.append(to_chart_dataset(filtered_stream[1].data, stream[1].id + " (Filtered)"))
    datasets.append(to_chart_dataset(stream[2].data, stream[2].id))
    datasets.append(to_chart_dataset(filtered_stream[2].data, stream[2].id + " (Filtered)"))

    # cumulative:
    datasets.append(to_chart_dataset(cumulative_trace.data, title + " (Cumulative)"))
    # envelope
    datasets.append(to_chart_dataset(evlp_trace_data, title + " (Envelope)"))

    # calculate frequencies and return
    freqs, snr_stream = interpolate(snr_stream, MAX_NUM_PTS_FREQSCALE, align_if_stream=True,
                                    return_x_array=True)

    freqs_log = np.log10(freqs[1:])  # FIXME: REMOVE!
    freq_data = {'datasets': []}
    datasets = freq_data['datasets']
    # smooth signal:
    bwd = 100
    noisy_amps = kos(snr_stream[0].data, freqs, bandwidth=bwd)
    sig_amps = kos(snr_stream[1].data, freqs, bandwidth=bwd)
    datasets.append(to_chart_dataset(noisy_amps[1:], title + " (Noise)", freqs_log))
    datasets.append(to_chart_dataset(sig_amps[1:], title + " (Signal)", freqs_log))

    # add metadata:
    mag = listreader.get(seg_id, listreader.T_EVT, ['Magnitude'])
    metadata = []
    metadata.append(("Mag", str(mag.iloc[0]['Magnitude'])))
    for key in ("#EventID", "EventDistance/deg", "DataStartTime", "ArrivalTime",
                "DataEndTime", "#Network", "Station", "Location", "Channel", "SampleRate"):
        value = db_row.iloc[0][key]
        if "time" in key.lower():
            # here we don't have obpsy UTCDatetimes, but PANDAS timestamops
            # FIXME: check, use unix timestamps?
            value = round(value.value / 1000000)
            if key == "DataStartTime":
                # insert arrival time DATE. Words with Date will be parsed as Date
                # (avoiding time info):
                metadata.insert(0, ("DataStartDate", value))
        else:
            value = str(value)
        # store as timestamp
        metadata.append((key, value))
        if key == "ArrivalTime":
            # set arrival time. This is a pandas Timestamp object and
            # uses microseconds. We want seconds
            metadata.append(('SnrWindow/sec', SNR_WINDOW_SIZE_IN_SECS))

    metadata.append(('---', ""))
    metadata.append(('SNR', _snr))
    # reminder: # setting the word 'time' will convert to timestamp in web page
    metadata.append(('Cum_time( 5%)', _cum_t5.timestamp * 1000))
    metadata.append(('Cum_time(95%)', _cum_t95.timestamp * 1000))
    metadata.append(('Amplitude_Ratio', _amp_ratio))

    return {'time_data': time_data, 'freq_data': freq_data, 'metadata': metadata,
            'class_id': classannotator.get_class(seg_id)}


def to_chart_dataset(np_array_y, title=None, np_array_x=None):
    """Converts the array to a dataset dict usable for Chart.js
        IF title is NOT NONE. Otherwise, converts array to a json serializable list
        :param array: a numpy array
    """

    if np_array_y is None:
        array = []
    else:
        if np_array_x is not None:
            array = []
            for x, y in izip(np_array_x, np_array_y):
                array.append({'x': x.item(), 'y': y.item()})
        else:
            array = np_array_y.tolist()
    return {'label': title, 'data': array}


def tojson(array):
    return array.tolist()


def to_chart_data(np_xvalues, chart_datasets_list):
    """Converts the array to a dataset dict usable for Chart.js
        IF title is NOT NONE. Otherwise, converts array to a json serializable list
        :param array: a numpy array
    """
    return {'labels': tojson(np_xvalues), 'datasets': chart_datasets_list}

# def interp(newxarray, oldxarray, yarray, numpoints=1000, return_json_serializable=True):
#     """Calls numpy.interp(newxarray, oldxarray, yarray), with the difference that oldxarray can be
#     None (in this case nothing is interpolated
#     :param return_json_serializable: converts the returned array to a python list, so that is
#     json serializable
#     """
#     if oldxarray is None:
#         newy = yarray
#     else:
#         newy = np.interp(newxarray, oldxarray, yarray)
#     return newy if not return_json_serializable else newy.tolist()
# 
# 
# def get_other_components(segment_series, listreader):
#     # get other components
#     def filter_func(df):
#         return df[(df['#Network'] == segment_series['#Network']) &
#                   (df['Station'] == segment_series['Station']) &
#                   (df['Location'] == segment_series['Location']) &
#                   (df['DataStartTime'] == segment_series['DataStartTime']) &
#                   (df['DataEndTime'] == segment_series['DataEndTime']) &
#                   (df['Channel'].str[:2] == segment_series['Channel'][:2]) &
#                   (df['Channel'] != segment_series['Channel'])]
# 
#     other_components = listreader.read(ListReader.T_SEG, filter_func=filter_func)
#     return other_components


def set_class(seg_id, class_id):
    old_class_id = classannotator.get_class(seg_id)
    if old_class_id != class_id:
        classannotator.set_class(seg_id, class_id)
    return old_class_id