'''
Created on Jun 20, 2016

@author: riccardo
'''
from stream2segment.s2sio.db import ListReader
import numpy as np
from stream2segment.analysis.mseeds import cumsum, snr, env, bandpass
from stream2segment.classification.handlabelling import ClassAnnotator
# from obspy.core.utcdatetime import UTCDateTime
# from future.backports.datetime import timezone
listreader = None
classannotator = None
SNR_WINDOW_SIZE_IN_SECS = 40  # FIXME: add to the config!!


def get_listreader(db_uri):
    if listreader is None:
        global listreader
        listreader = ListReader(db_uri, filter_func=None,
                                sort_columns=["#EventID", "EventDistance/deg"],
                                sort_ascending=[True, True])
        global classannotator
        classannotator = ClassAnnotator(listreader)
    return listreader


def get_ids(db_uri):
    listreader = get_listreader(db_uri)
    # NOTE: we need conversion to string cause apparently jsonify does some rounding on big ints
    # FIXME: CHECK!!!
    return {'segment_ids': listreader.mseed_ids['Id'].values.astype(str).tolist(),
            'classes':  classannotator.get_classes_df().to_dict('records')}


def get_data(db_uri, seg_id):
    listreader = get_listreader(db_uri)
    db_row = listreader.get(seg_id, listreader.T_SEG)
    stream = listreader.get_stream(seg_id, include_same_channel=True)
    filtered_stream = bandpass(stream)
    cumulative_trace = cumsum(filtered_stream[0])
    snr_stream = snr(filtered_stream[0], db_row.iloc[0]['ArrivalTime'], SNR_WINDOW_SIZE_IN_SECS)
    evlp_trace = env(filtered_stream[0])

    ret_data = {'data': [], 'metadata': [], 'class_id': None}

    MAX_NUM_PTS = 1200

    metadata = []  # store each type of metadata here
    s_n_r = "N/A"
    cum_t5 = 0
    cum_t95 = 0
    for i in xrange(3):
        try:
            data_list = ret_data['data']
            starttime = stream[i].stats.starttime
            delta = stream[i].stats.delta
            endtime = stream[i].stats.endtime
            datalen = len(stream[i].data)
            datalen2 = min(datalen, MAX_NUM_PTS)

            timez = np.linspace(starttime.timestamp, endtime.timestamp, num=datalen2, endpoint=True)

            orig_timez = None if datalen2 == datalen else \
                np.linspace(starttime.timestamp, endtime.timestamp, num=datalen, endpoint=True)

            # put data in chartjs "format":
            data_list.append({
                              # round to milliseconds for chart.js time scale:
                              'labels': np.round(timez * 1000.0).tolist(),
                              'datasets': [{
                                           'label': stream[i].id,
                                           'data': interp(timez, orig_timez, stream[i].data)
                                           },
                                           {
                                           'label': stream[i].id + " (filtered)",
                                           'data': interp(timez, orig_timez,
                                                          filtered_stream[i].data)
                                           }
                                           ]
                            })

            if i != 0:
                continue

            cum_t5 = (starttime +
                      (np.where(cumulative_trace.data <= 0.05)[0][-1]) * delta).timestamp * 1000
            cum_t95 = (starttime +
                       (np.where(cumulative_trace.data >= 0.95)[0][0]) * delta).timestamp * 1000

            # add other plots and stuff:
            data_list[-1]['datasets'].\
                extend([{
                         'label': stream[i].id + " (Cumulative)",
                         'data': interp(timez, orig_timez, cumulative_trace.data)
                         },
                        {
                         'label': stream[i].id + " (Envelope)",
                         'data': interp(timez, orig_timez, evlp_trace.data)
                         }])

            # add frequencies (actually, power spectra)
            df = snr_stream[0].df
            datalen = max(len(snr_stream[0].data), len(snr_stream[1].data))
            datalen2 = min(datalen, MAX_NUM_PTS)
            freqz = np.linspace(0, df*datalen2, num=datalen2, endpoint=False)

            orig_freqz = None if datalen == datalen2 else \
                np.linspace(0, df*datalen, num=datalen, endpoint=False)

            s_n_r = 10 * np.log10(np.sum(snr_stream[1].data) / np.sum(snr_stream[0].data))

            data_list.append({
                              # round to 2 decimals cause chart.js x axis scale is nicer so:
                              'labels': np.round(freqz, 2).tolist(),
                              'datasets': [{
                                           'label': stream[i].id + " (Noise)",
                                           'data': interp(freqz, orig_freqz, snr_stream[0].data)
                                           },
                                           {
                                           'label': stream[i].id + " (Sig.)",
                                           'data': interp(freqz, orig_freqz, snr_stream[1].data)
                                           }
                                           ]
                               })
        except (ValueError, IndexError) as _:
            raise

    # add metadata:
    mag = listreader.get(seg_id, listreader.T_EVT, ['Magnitude'])
    metadata.append(("Mag", str(mag.iloc[0]['Magnitude'])))
    for key in ("#EventID", "EventDistance/deg", "DataStartTime", "ArrivalTime",
                "DataEndTime", "#Network", "Station", "Location", "Channel"):  # , "", "RunId"):
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

    metadata.append(('SNR', s_n_r))
    # reminder: # setting the word 'time' will convert to timestamp in web page
    metadata.append(('Cum_time( 5%)', cum_t5))
    metadata.append(('Cum_time(95%)', cum_t95))

    ret_data['class_id'] = classannotator.get_class(seg_id)
    ret_data['metadata'] = metadata

    return ret_data


def interp(newxarray, oldxarray, yarray, numpoints=1000, return_json_serializable=True):
    """Calls numpy.interp(newxarray, oldxarray, yarray), with the difference that oldxarray can be
    None (in this case nothing is interpolated
    :param return_json_serializable: converts the returned array to a python list, so that is
    json serializable
    """
    if oldxarray is None:
        newy = yarray
    else:
        newy = np.interp(newxarray, oldxarray, yarray)
    return newy if not return_json_serializable else newy.tolist()


def get_other_components(segment_series, listreader):
    # get other components
    def filter_func(df):
        return df[(df['#Network'] == segment_series['#Network']) &
                  (df['Station'] == segment_series['Station']) &
                  (df['Location'] == segment_series['Location']) &
                  (df['DataStartTime'] == segment_series['DataStartTime']) &
                  (df['DataEndTime'] == segment_series['DataEndTime']) &
                  (df['Channel'].str[:2] == segment_series['Channel'][:2]) &
                  (df['Channel'] != segment_series['Channel'])]

    other_components = listreader.read(ListReader.T_SEG, filter_func=filter_func)
    return other_components


def set_class(seg_id, class_id):
    old_class_id = classannotator.get_class(seg_id)
    if old_class_id != class_id:
        classannotator.set_class(seg_id, class_id)
    return old_class_id