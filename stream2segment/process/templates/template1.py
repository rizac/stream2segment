'''
========================
Processing file template
========================

Template for processing downloaded waveform segments
Edit this file (plus additional config file, if any) and run processing via the command:

```
>>> stream2segment p $FILE $CONFIG $OUTPUT
```

where:
  - $FILE is the path of this file,
  - $CONFIG is a path to an (otpional) configuration yaml file (if this file was auto generated,
    it should be a file named $FILE.yaml), and
  - $OUTPUT is the csv file where data (one row per segment) will to be saved

This module must implement a `main` function (processing function) that will be called by the
program for each database segment. The processing function:

  - Takes as argument a single segment and a configuration dict (resulting from the $CONFIG file):
    ```
    def main(segment, config):
        ... code here ...
    ```
  - Must implement the actual processing for the segment and must return an
    iterable (list, tuple, numpy array, dict...) of values. The returned iterable
    will be written as a row of the resulting csv file $OUTPUT. If dict, the keys of the dict 
    will populate the first row header of the resulting csv file, otherwise the csv file will have
    no header. Please be consistent: always return the same type of iterable for all segments,
    if dict, always return the same keys for all dicts, if list, always return the same length,
    etcetera
  - Should return numeric or string data only. For instance, in case of obspy `UTCDateTime`s you
    should return either `float(utcdatetime)` (numeric) or `utcdatetime.isoformat()` (string).
    Returning other types of object *should* be safe (not tested) but will most lilely convert
    the values to string according to python `__str__` function and might be out of control for
    the user
  - Can raise any Exception, or return None. In both cases, the segment will not be written to the
    csv file. In the former case, the exception message (with the segment id) will be written to a
    log file `$OUTPUT.log`. Otherwise, return None to silently skip the segment (with no messages)
  - Does not actually need to be called `main`: if you wish to implement more than one function, say
    `main1` and `main2`, you can call them via the command line by specifying it in $FILE with a
    semicolon:
    ```
    >>> stream2segment p $FILE:main1 $CONFIG $OUTPUT
    ```
    or
    ```
    >>> stream2segment p $FILE:main2 $CONFIG $OUTPUT
    ```

**IMPORTANT: please note** that the first three columns of the resulting csv will be *always*
populated with the segment channel id, the segment start time and the segment end time
(these three values identify univocally the segment). Thus the first value returned by the iterable
of `main` will be in the csv file fourth column, the second in the fifth column, and so on ...

Please refer to the docstring of the `main` function below for further details on how to implement
the processing function

Created on Feb 2, 2017

@author: riccardo
'''
# import numpy for fatser numeric array processing:
import numpy as np
# strem2segment functions for processing mseeds
# If you need to use them, import them like this:
from stream2segment.analysis.mseeds import remove_response, get_gaps, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, get_multievent, snr, dfreq
# when working with times, use obspy UTCDateTime:
from obspy.core.utcdatetime import UTCDateTime
# stream2segment function for processing numpy arrays (such as stream.traces[0])
# If you need to to use them, import them:
from stream2segment.analysis import amp_spec, freqs


def main(seg, config):
    """
    Main processing function, where the user must implement the processing for a single segment
    which will populate a csv file row.
    This function can return None or raise any Exception. In both cases, the segment will not be
    written to the csv file. None will silently ignore the segment, while in case of Exceptions,
    the exception message (with the segment id) will be written to a `logger`.
    If this file is run for csv output, the logger output will be a .log file in the same
    folder than the output csv file.

    :param: segment (ptyhon object): An object representing a waveform data to be processed,
    reflecting the relative database table row.

    Technically it's an 'SqlAlchemy` (modified) ORM instance but for the user it is enough to
    consider and treat as a normal python object whose attributes are "simple" python types
    (boolean, numeric, string, datetime, bytes) returning the relative db table column value.
    E.g.: `segment.arrival_time` returns the segment arrival time as a datetime object.

    `segment` is intented to be used to retrieve properties (you don't need and you should not
    set any attribute on it) and has two special methods:

    * `segment.stream()` which returns the waveform data in the form of
      an `obspy.Stream` object, and
    * `segment.inventory()` which returns an `obspy.Inventory`
    object (e.g., for removing the instrumental response from the stream data)

    Moreover, it has some special attributes not returning simple python types but other objects.
    These objects reflect the rows of other database tables related to the segment table row. E.g.,
    `segment.event`, `segment.channel` access the segment event and the segment table,
    respectively. Thus, for accessing the magnitude of the event originating the waveform data,
    use `segment.event.magnitude` (float).
    To access the station db object, don't use `segment.station` but `segment.channel.station`


    :param: config (python dict): a dictionary reflecting what has been implemented in $CONFIG.
    You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
    will be accessible as usual via `config['propertyname']`

    Has a single "special" argument, 'inventory', which if True, will save all inventories in
    the database, if not already present. This might slow down the processing the first time,
    but might be handy if you did not save inventories during download and you want to try to
    speed up further processing requiring the inventory


    :return: an iterable (list, tuple, numpy array, dict...) of values. The returned iterable
    will be written as a row of the resulting csv file. If dict, the keys of the dict
    will populate the first row header of the resulting csv file, otherwise the csv file
    will have no header. Please be consistent: always return the same type of iterable for
    all segments, if dict, always return the same keys for all dicts, if list, always
    return the same length, etcetera

    The iterable should return numeric or string data only. For instance, in case of obspy
    `UTCDateTime`s you should return either `float(utcdatetime)` (numeric) or
    `utcdatetime.isoformat()` (string). Returning other types of object *should* be safe
    (not tested) but will most lilely convert
    the values to string according to python `__str__` function and might be out of control
    for the user

    """

    # if the segment has no data downloaded, no need to proceed:
    if not seg.data:
        raise ValueError('empty data')

    # get the obpsy Stream. Calling stream() several times might be time consuming, so better
    # do it once:
    stream = seg.stream()

    # discard streams with gaps
    if get_gaps(stream):
        raise ValueError('has gaps')

    # discard streams with more than one trace:
    if len(stream) != 1:
        raise ValueError('more than one obspy.Trace')

    # work on the trace now. All functions will return Traces or scalars, which is better
    # so we can write them to database more easily
    trace = stream[0]

    # discard saturated signals (according to the threshold set in the config file):
    ampratio = amp_ratio(trace)
    if ampratio >= config['amp_ratio_threshold']:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # convert to UTCDateTime for operations later:
    a_time = UTCDateTime(seg.arrival_time) + config['arrival_time_delay']

    # bandpass the trace, according to the event magnitude
    evt = seg.event
    trace = bandpass(trace, evt.magnitude, freq_max=config['bandpass_freq_max'],
                     max_nyquist_ratio=config['bandpass_max_nyquist_ratio'],
                     corners=config['bandpass_corners'])

    # remove response
    inventory = seg.inventory()
    trace_rem_resp = remove_response(trace, inventory, output=config['remove_response_output'],
                                     water_level=config['remove_response_water_level'])

    # calculate cumulative:
    cum_trace = cumsum(trace_rem_resp)
    # and then calculate t005, t010, t025, t050, t075, t90, t95 (converting as float):
    cum_times = [float(t) for t in cumtimes(cum_trace, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)]
    # then, for instance:
    # mseed_rem_resp_t05_t95 = trace_rem_resp.slice(t05, t95)

    # calculate PGA and times of occurrence (t_PGA):
    t95 = cum_times[-1]
    t_PGA, PGA = maxabs(trace_rem_resp, a_time, t95)

    # calculates amplitudes at the frequency bins given in the config file:
    fft_rem_resp = fft(trace_rem_resp, a_time, config['snr_window_length'],
                       taper_max_percentage=config['taper_max_percentage'])
    ampspec = amp_spec(fft_rem_resp.data, True)
    ampspec_freqs = freqs(ampspec, dfreq(fft_rem_resp))
    required_freqs = config['freqs_interp']
    required_amplitudes = np.interp(required_freqs, ampspec_freqs, ampspec)

    # save as csv row fft amplitudes, times of cumulative, t_PGA and PGA:
    return np.concatenate(required_amplitudes, cum_times, [float(t_PGA), PGA])
