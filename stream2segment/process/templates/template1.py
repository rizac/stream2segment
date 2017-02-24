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
    it should be a file named $FILE.yaml)
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
    Returning other types of object *should* be safe (not tested) but will most likely convert
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
(these three values identify uniquely the segment). Thus the first value returned by the iterable
of `main` will be in the csv file fourth column, the second in the fifth column, and so on ...

Please refer to the docstring of the `main` function below for further details on how to implement
the processing function

Created on Feb 2, 2017

@author: riccardo
'''
# import numpy for fatser numeric array processing:
import numpy as np
# import ordered dict if you want to create a csv with the header columns ordered as you want
from collections import OrderedDict as odict
# Example from above: python dicts do not preserve the keys order (as they where inserted), so
# returning {'a':7, 'b':56} might write 'b' as first column and 'a' as second in the csv.
# To set the order you want:
# dic = odict()
# dic['a'] = 7
# dic['b'] = 56

# strem2segment functions for processing mseeds. This is just a list of possible functions
# to show how to import them:
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
    For info about possible function to use, please have a look at `stream2segment.analysis.mseeds`
    and obviously at `obpsy <https://docs.obspy.org/packages/index.html>`_, in particular:

    *  `obspy.core.Stream <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html#obspy.core.stream.Stream>_`
    *  `obspy.core.Trace <https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html#obspy.core.trace.Trace>_`

    :param: segment (ptyhon object): An object representing a waveform data to be processed,
    reflecting the relative database table row.

    Technically it's an 'SqlAlchemy` (modified) ORM instance but for the user it is enough to
    consider and treat it as a normal python object. It has three special methods:

    * `segment.has_data()` which returns a boolean indicating if the segment has waveform data
      on the database. Some queries resulted in errors or returned zero bytes, so this is usually
      the first check to do
    * `segment.stream()` which returns the waveform data in the form of an `obspy.Stream` object
    * `segment.inventory()` which returns an `obspy.core.inventory.inventory.Inventory` object
      (e.g., for removing the instrumental response from the stream data)

    `segment` has the following attributes (with relative python type) which return the values
    of the relative database table columns. The attributes are mainly self-explanatory
    (Note: 'bytes' attributes, if accessed, are time consuming and you should not usually need them)

    segment.data                            bytes (the raw data for building `segment.stream()`)
    segment.id                              int
    segment.event_distance_deg              float
    segment.start_time                      datetime.datetime
    segment.arrival_time                    datetime.datetime
    segment.end_time                        datetime.datetime

    segment.event                           object (attributes below)
    segment.event.id                        str
    segment.event.time                      datetime.datetime
    segment.event.latitude                  float
    segment.event.longitude                 float
    segment.event.depth_km                  float
    segment.event.author                    str
    segment.event.catalog                   str
    segment.event.contributor               str
    segment.event.contributor_id            str
    segment.event.mag_type                  str
    segment.event.magnitude                 float
    segment.event.mag_author                str
    segment.event.event_location_name       str

    segment.channel                         object (attributes below)
    segment.channel.id                      str
    segment.channel.location                str
    segment.channel.channel                 str
    segment.channel.depth                   float
    segment.channel.azimuth                 float
    segment.channel.dip                     float
    segment.channel.sensor_description      str
    segment.channel.scale                   float
    segment.channel.scale_freq              float
    segment.channel.scale_units             str
    segment.channel.sample_rate             float
    segment.channel.station                 object (same as segment.station, see below)

    segment.station                         object (attributes below)
    segment.station.id                      str
    segment.station.network                 str
    segment.station.station                 str
    segment.station.latitude                float
    segment.station.longitude               float
    segment.station.elevation               float
    segment.station.site_name               str
    segment.station.start_time              datetime.datetime
    segment.station.end_time                datetime.datetime
    segment.station.inventory_xml           bytes (the raw data for building `segment.inventory()`)
    segment.station.datacenter              object (same as segment.datacenter, see below)

    segment.datacenter                      object (attributes below)
    segment.datacenter.id                   int
    segment.datacenter.station_query_url    str
    segment.datacenter.dataselect_query_url str

    segment.run                             object (attributes below)
    segment.run.id                          int
    segment.run.run_time                    datetime.datetime
    segment.run.log                         str
    segment.run.warnings                    int
    segment.run.errors                      int
    segment.run.config                      str
    segment.run.program_version             str


    :param: config (python dict): a dictionary reflecting what has been implemented in $CONFIG.
    You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
    will be accessible as usual via `config['propertyname']`

    The config has a single "special" property name, 'inventory', which if True, will save all
    inventories in the database, if not already present (i.e., if you did not save inventories
    during download). This will slow down the processing the first time, but will most likely
    speed up further processing runs


    :return: an iterable (list, tuple, numpy array, dict...) of values. The returned iterable
    will be written as a row of the resulting csv file. If dict, the keys of the dict
    will populate the first row header of the resulting csv file, otherwise the csv file
    will have no header. Please be consistent: always return the same type of iterable for
    all segments; if dict, always return the same keys for all dicts; if list, always
    return the same length, etcetera. If you want to preserve the order of the dict keys as
    inserted in the code, use `OrderedDict` instead of `dict` or `{}`

    Pay attention when setting complex objects (e.g., everything neither string nor numeric) as
    elements of the returned iterable: the values will be most likely converted to string according
    to python `__str__` function and might be out of control for the user.
    Thus, it is suggested to convert everything to string or number. E.g., for `UTCDateTime`s
    you could return either `float(utcdatetime)` (numeric) or `utcdatetime.isoformat()` (string)
    """

    # if the segment has no data downloaded, no need to proceed:
    if not seg.has_data():
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
    # and then calculate t005, t010, t025, t050, t075, t90, t95 (UTCDateTime objects):
    cum_times = cumtimes(cum_trace, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
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

    # convert cum_times to float for saving
    cum_times_float = [float(t) for t in cum_times]
    # save as csv row fft amplitudes, times of cumulative, t_PGA and PGA:
    return np.hstack((required_amplitudes, cum_times_float, [float(t_PGA), PGA]))
