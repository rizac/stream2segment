'''
Created on Jun 20, 2016

@author: riccardo
'''
from stream2segment.s2sio.db import ListReader
import numpy as np
from stream2segment.analysis.mseeds import cumsum, snr, env, bandpass
from obspy.core.utcdatetime import UTCDateTime
listreader = None
SNR_WINDOW_SIZE_IN_SECS = 40  # FIXME: add to the config!!


def get_listreader(db_uri):
    if listreader is None:
        global listreader
        listreader = ListReader(db_uri, filter_func=None,
                                sort_columns=["#EventID", "EventDistance/deg"],
                                sort_ascending=[True, True])
    return listreader


def get_ids(db_uri):
    listreader = get_listreader(db_uri)
    # NOTE: we need conversion to string cause apparently jsonify does some rounding on big ints
    # FIXME: CHECK!!!
    # NOTE 2: WE ALSO REPLACE THE MINUS SIGN WITH "NEG" CAUSE ANGULAR.JS HAS PROBLEM SENDING THE 
    # URL IF STARTS WITH "-". ALSO CHECK! GOOGLING DIDN't GIVE ME ANY HINTS
    return np.core.defchararray.replace(listreader.mseed_ids['Id'].values.astype(str), "-", "NEG",
                                        count=1).tolist()
    # return listreader.mseed_ids['Id'].values.astype(str).tolist()


def get_data(db_uri, id):
    listreader = get_listreader(db_uri)
    db_row = listreader.get(id, listreader.T_SEG)
    stream = listreader.get_stream(id, include_same_channel=True)
    filtered_stream = bandpass(stream)
    cumulative_trace = cumsum(filtered_stream[0])
    snr_stream = snr(filtered_stream[0], db_row.iloc[0]['ArrivalTime'], SNR_WINDOW_SIZE_IN_SECS)
    evlp_trace = env(filtered_stream[0])

    ret_data = {'freqs': [], 'data': [], 'spectrum_bounds': [0, 1, 0, 1]}

    MAX_NUM_PTS = 1200

    metadata = []  # store each type of metadata here
    for i in xrange(3):
        signal_found = i < len(stream)
        if signal_found:

            # set the labels for the times
            starttime = stream[i].stats.starttime
            delta = stream[i].stats.delta
            timez = np.linspace(starttime.timestamp, stream[i].stats.endtime.timestamp,
                                num=len(stream[i].data),
                                endpoint=True)

            newtimez = None if len(timez) <= MAX_NUM_PTS else \
                np.linspace(timez[0], timez[-1], num=MAX_NUM_PTS, endpoint=True)

            times_rounded_to_millisecs = np.round((timez if newtimez is None else newtimez) *
                                                  1000.0)

            ret_data['data'].append(
                            {
                             'times': times_rounded_to_millisecs.tolist(),  # [UTCDateTime(x).isoformat()[11:] for x in newtimez],
                             'id': stream[i].id,
                             'trace': interp(newtimez, timez, stream[i].data),
                             'trace_bandpass': interp(newtimez, timez, filtered_stream[i].data),
                             }
                            )
        else:
            ret_data['data'].append(
                            {
                             'id': 'Signal not found',
                             'trace': [],
                             'trace_bandpass': [],
                            }
                            )

        if i == 0:
            if signal_found:
                if not ret_data['freqs']:
                    # set the labels for the times
                    starttime = stream[i].stats.starttime
                    delta = snr_stream[0].df
                    max_len = max(len(snr_stream[0].data), len(snr_stream[1].data))
                    freqz = np.arange(0, max_len, delta)[0:max_len]

                    newfreqz = None if len(freqz) <= MAX_NUM_PTS else \
                        np.linspace(freqz[0], freqz[-1], num=MAX_NUM_PTS, endpoint=True)

                    ret_data['freqs'] = freqz.tolist() if len(freqz) <= MAX_NUM_PTS else \
                        newfreqz.tolist()

                    snr_noise = interp(newfreqz, freqz, snr_stream[0].data,
                                       return_json_serializable=False)
                    snr_sig = interp(newfreqz, freqz, snr_stream[1].data,
                                     return_json_serializable=False)

                    metadata.append(['SNR',  \
                        np.sum(snr_stream[1].data) / np.sum(snr_stream[0].data)])

                    # set frequency bounds. ChartJs requires them to properly scale the log axes:
                    # FIXME: REMOVE!!
                    ret_data['spectrum_bounds'][0] = freqz[0]
                    ret_data['spectrum_bounds'][1] = freqz[-1]
                    ret_data['spectrum_bounds'][2] = np.nanmin([snr_noise, snr_sig])
                    ret_data['spectrum_bounds'][3] = np.nanmax([snr_noise, snr_sig])

                ret_data['data'][-1].update(
                                    {
                                      'env': interp(newtimez, timez, evlp_trace.data),
                                      'snr_noise': snr_noise.tolist(),
                                      'snr_sig': snr_sig.tolist(),
                                      'cum': interp(newtimez, timez, cumulative_trace.data),
                                      }
                                    )
            else:
                ret_data['data'][-1].update(
                                    {
                                      'env': [],
                                      'snr_noise': [],
                                      'snr_sig': [],
                                      'cum': [],
                                      }
                                    )

    # add metadata:
    mag = listreader.get(id, listreader.T_EVT, ['Magnitude'])
    metadata.append(("Mag", str(mag.iloc[0]['Magnitude'])))
    for key in ("#EventID", "EventDistance/deg", "DataStartTime", "ArrivalTime",
                "DataEndTime", "#Network", "Station", "Location", "Channel"):  # , "", "RunId"):
        metadata.append((key, str(db_row.iloc[0][key])))
        if key == "ArrivalTime":
            # set arrival time. This is a pandas Timestamp object and
            # uses microseconds. We want seconds
            ret_data['arrival_time'] = round(db_row.iloc[0][key].value / 1000000)
            ret_data['snr_dt_in_sec'] = SNR_WINDOW_SIZE_IN_SECS

    ret_data['metadata'] = metadata

    return ret_data


def interp(newxarray, oldxarray, yarray, numpoints=1000, return_json_serializable=True):
    if newxarray is None:
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
