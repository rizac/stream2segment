'''
=============================================================
Stream2segment: Processing+Visualization python file template
=============================================================

This is a template python module for processing downloaded waveform segments and defining the
segment plots to be visualized in the web GUI (Graphical user interface).

This file can be edited and passed to the program commands `s2s v` (visualize) and `s2s p` (process)
as -p or --pyfile option, together with an associated configuration .yaml file (-c option):
```
    s2s show -p [thisfilepath] -c [configfilepath] ...
    s2s process -p [thisfilepath] -c [configfilepath] ...
```
This module needs to implement one or more functions which will be described here (for full details,
look at their doc-string). All these functions must have the same signature:
```
    def myfunction(segment, config):
```
where `segment` is the python object representing a waveform data segment to be processed
and `config` is the python dictionary representing the given configuration .yaml file.

Processing
==========

When invoked via `s2s process ...`, the program will search for a function called "main", e.g.:
```
def main(segment, config)
```
the program will iterate over each selected segment (according to 'segment_select' parameter
in the config) and execute the function, writing its output to the given .csv file

Visualization (web GUI)
=======================

When invoked via `s2s show ...`, the program will search for all functions decorated with
"@gui.preprocess", "@gui.sideplot" or "@gui.customplot".
The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plot functions (i.e., all other functions decorated with either '@sideplot' or '@customplot')
are re-executed with the only difference that `segment.stream()`
will return the pre-processed stream, instead of the "raw" unprocessed stream. Thus, this
function must return a Stream or Trace object.
The function decorated with "@gui.sideplot", e.g.:
```
@gui.sideplot
def sn_spectra(segment, config)
```
will be associated to (i.e., its output will be displayed in) the right side plot,
next to the raw / pre-processed segment stream plot.
Finally, the functions decorated with "@gui.customplot", e.g.:
```
@gui.customplot
def cumulative(segment, config)
@gui.customplot
def first_derivative(segment, config)
...
```
will be associated to the bottom plot, below the raw / pre-processed segment stream plot, and can
be selected (one at a time) from the GUI with a radio-button.
All plot functions should return objects of certain types (more details in their doc-strings
in this module).

Important notes
===============

1) This module is designed to force the DRY (don't repeat yourself) principle, thus if a portion
of code implemented in "main" should be visualized for inspection, it should be moved, not copied,
to a separated function (decorated with '@gui.customplot') and called from within "main"

2) All functions here can safely raise Exceptions, as all exceptions will be caught by the caller:
- displaying the error message on the plot if the function is called for visualization,
- printing it to a log file, if teh function is called for processing into .csv
  (More details on this in the "main" function doc-string).
Thus do not issue print statement in any function because, to put it short,
it's useless (and if you are used to do it extensively for debugging, consider changing this habit
and use a logger): if any information should be given, simply raise a base exception, e.g.:
`raise Exception("segment sample rate too low")`.

Functions arguments
===================

As said, all functions needed or implemented for processing and visualization must have the same
signature:
```
    def myfunction(segment, config):
```
In details, the two arguments passed to those functions are:

segment (object)
~~~~~~~~~~~~~~~~

Technically it's like an 'SqlAlchemy` ORM instance but for the user it is enough to
consider and treat it as a normal python object. It has special methods and several
attributes returning python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

segment methods:
----------------

* segment.stream(): the `obspy.Stream` object representing the waveform data
  associated to the segment. Please remember that many obspy functions modify the
  stream in-place:
  ```
      s = segment.stream()
      s_rem_resp = s.remove_response(segment.inventory())
      segment.stream() is s  # False!!!
      segment.stream() is s_rem_resp  # True!!!
  ```
  When visualizing plots, where efficiency is less important, each function is executed on a
  copy of segment.stream(). However, from within the `main` function, the user has to handle when
  to copy the segment's stream or not. For info see:
  https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.copy.html

* segment.inventory(): the `obspy.core.inventory.inventory.Inventory`. This object is useful e.g.,
  for removing the instrumental response from `segment.stream()`

* segment.sn_windows(): returns the signal and noise time windows:
  (s_start, s_end), (n_start, n_end)
  where all elements are `UTCDateTime`s. The windows are computed according to
  the settings of the associated yaml configuration file: `config['sn_windows']`). Example usage:
  `
  sig_wdw, noise_wdw = segment.sn_windows()
  stream_noise = segment.stream().copy().trim(*noise_wdw, ...)
  stream_signal = segment.stream().copy().trim(*sig_wdw, ...)
  `
  If segment's stream has more than one trace, the method raises.

* segment.other_orientations(): returns a list of segment objects representing the same recorded
  event on other channel's orientations. E.g., if `segment` refers to an event E recorded by a
  station channel with code 'HHZ', this method returns the segments recorded on 'HHE' and
  'HHN' (relative to the same event on the same station and location codes).

* segment.del_classes(*ids_or_labels): Deletes the given classes of the segment. The argument is
  a comma-separated list of class labels (string) or class ids (int). As classes are given in the
  config as a dict of label:description values, usually string labels are passed here.
  E.g.: `segment.del_classes('class1')`, `segment.del_classes('class1', 'class2', 'class3')`

* segment.set_classes(*ids_or_labels, annotator=None): Sets the given classes on the segment,
  deleting all already assigned classes, if any. `ids_or_labels` is a comma-separated list of class
  labels (string) or class ids (int). As classes are given in the config as a dict of
  label:description values, usually string labels are passed here. `annotator` is a string name
  which, if given (not None) denotes that the class labelling is a human hand-labelled class
  assignment (vs. a statistical classifier class assignment).
  E.g.: `segment.set_classes('class1')`, `segment.set_classes('class1', 'class2', annotator='Jim')`

* segment.add_classes(*ids_or_labels, annotator=None): Same as `segment.set_classes` but already
  assigned classes will neither be deleted first, nor added again if already assigned

* segment.dbsession(): WARNING: this is for advanced users experienced with Sql-Alchemy library:
  returns the database session for IO operations with the database


segment attributes:
-------------------

========================================= ================================================
attribute                                 python type and description (if any)
========================================= ================================================
segment.id                                int: segment (unique) db id
segment.event_distance_deg                float: distance between the segment's station and
\                                         the event, in degrees
segment.event_distance_km                 float: distance between the segment's station and
\                                         the event, in km, assuming a perfectly spherical earth
\                                         with a radius of 6371 km
segment.start_time                        datetime.datetime: the waveform data start time
segment.arrival_time                      datetime.datetime
segment.end_time                          datetime.datetime: the waveform data end time
segment.request_start                     datetime.datetime: the requested start time of the data
segment.request_end                       datetime.datetime: the requested end time of the data
segment.duration_sec                      float: the waveform data duration, in seconds
segment.missing_data_sec                  float: the number of seconds of missing data, with respect
\                                         to the request time window. E.g. if we requested 5
\                                         minutes of data and we got 4 minutes, then
\                                         missing_data_sec=60; if we got 6 minutes, then
\                                         missing_data_sec=-60. This attribute is particularly
\                                         useful in the config to select only well formed data and
\                                         speed up the processing, e.g.: missing_data_sec: '< 120'
segment.missing_data_ratio                float: the portion of missing data, with respect
\                                         to the request time window. E.g. if we requested 5
\                                         minutes of data and we got 4 minutes, then
\                                         missing_data_ratio=0.2 (20%); if we got 6 minutes, then
\                                         missing_data_ratio=-0.2. This attribute is particularly
\                                         useful in the config to select only well formed data and
\                                         speed up the processing, e.g.: missing_data_ratio: '< 0.5'
segment.has_data                          boolean: tells if the segment has data saved (at least
\                                         one byte of data). This attribute useful in the config to
\                                         select only well formed data and speed up the processing,
\                                         e.g. has_data: 'true'.
segment.sample_rate                       float: the waveform data sample rate.
\                                         It might differ from the segment channel's sample_rate
segment.download_code                     int: the download code (extends HTTP status codes).
\                                         Typically, values between 200 and 399 denote
\                                         successful download. Values >=400 and lower than 500
\                                         denote client errors, values >=500 server errors, -1
\                                         indicates a general download error - e.g. no Internet
\                                         connection, -2 that the waveform data is corrupted,
\                                         -200 a successful download where some waveform data has
\                                         been discarded because outside the requested time span,
\                                         -204 a successful download where no data has been saved
\                                         because all response data was outside the requested time
\                                         span, and finally None denotes a general unknown error
\                                         not in the previous categories
segment.maxgap_numsamples                 float: the maximum gap found in the waveform data, in
\                                         in number of points.
\                                         If the value is positive, the max is a gap. If negative,
\                                         it's an overlap. If zero, no gaps/overlaps were found.
\                                         This attribute is particularly useful in the config to
\                                         select only well formed data and speed up the processing,
\                                         e.g.: maxgap_numsamples: '[-0.5, 0.5]'.
\                                         This number is a float because it is the ratio between
\                                         the waveform data's max gap/overlap and its sampling
\                                         period (both in seconds). Thus, non-zero float values
\                                         in (-1, 1) are difficult to interpret: a rule of thumb
\                                         is to consider a segment with gaps/overlaps when this
\                                         attribute's absolute value exceeds 0.5. The user can
\                                         always perform a check in the processing for
\                                         safety, e.g., via `len(segment.stream())` or
\                                         `segment.stream().get_gaps()`)
segment.data_seed_id                      str: the seed identifier in the typical format
\                                         'Network.Station.Location.Channel' as read from the data.
\                                         It might be null if the data is empty or null because of
\                                         a download error. See also 'segment.meed_identifier'
segment.seed_id                           str: the seed identifier in the typical format
\                                         'Network.Station.Location.Channel': it is the same as
\                                         'segment.data_seed_id', but it is assured not to be,
\                                         null, as the segment meta-data is used if needed: in this
\                                         case the query might perform more poorly at the SQL level
segment.has_class                         boolean: tells if the segment has (at least one) class
\                                         assigned
segment.data                              bytes: the waveform (raw) data. You don't generally need
\                                         to access this attribute which is also time-consuming
\                                         to fetch. Used by `segment.stream()`
----------------------------------------- ------------------------------------------------
segment.event                             object (attributes below)
segment.event.id                          int
segment.event.event_id                    str: the id returned by the web service
segment.event.time                        datetime.datetime
segment.event.latitude                    float
segment.event.longitude                   float
segment.event.depth_km                    float
segment.event.author                      str
segment.event.catalog                     str
segment.event.contributor                 str
segment.event.contributor_id              str
segment.event.mag_type                    str
segment.event.magnitude                   float
segment.event.mag_author                  str
segment.event.event_location_name         str
----------------------------------------- ------------------------------------------------
segment.channel                           object (attributes below)
segment.channel.id                        int
segment.channel.location                  str
segment.channel.channel                   str
segment.channel.depth                     float
segment.channel.azimuth                   float
segment.channel.dip                       float
segment.channel.sensor_description        str
segment.channel.scale                     float
segment.channel.scale_freq                float
segment.channel.scale_units               str
segment.channel.sample_rate               float
segment.channel.band_code                 str: the first letter of channel.channel
segment.channel.instrument_code           str: the second letter of channel.channel
segment.channel.orientation_code          str: the third letter of channel.channel
segment.channel.station                   object: same as segment.station (see below)
----------------------------------------- ------------------------------------------------
segment.station                           object (attributes below)
segment.station.id                        int
segment.station.network                   str
segment.station.station                   str
segment.station.latitude                  float
segment.station.longitude                 float
segment.station.elevation                 float
segment.station.site_name                 str
segment.station.start_time                datetime.datetime
segment.station.end_time                  datetime.datetime
segment.station.inventory_xml             bytes. The station inventory (raw) data. You don't
\                                         generally need to access this attribute which is also
\                                         time-consuming to fetch. Used by `segment.inventory()`
segment.station.has_inventory             boolean: tells if the segment's station inventory has
\                                         data saved (at least one byte of data).
\                                         This attribute useful in the config to select only
\                                         segments with inventory downloaded and speed up the
\                                         processing,
\                                         e.g. has_inventory: 'true'.
segment.station.datacenter                object (same as segment.datacenter, see below)
----------------------------------------- ------------------------------------------------
segment.datacenter                        object (attributes below)
segment.datacenter.id                     int
segment.datacenter.station_url            str
segment.datacenter.dataselect_url         str
segment.datacenter.organization_name      str
----------------------------------------- ------------------------------------------------
segment.download                          object (attributes below): the download execution
segment.download.id                       int
segment.download.run_time                 datetime.datetime
segment.download.log                      str: The log text of the segment's download execution.
\                                         You don't generally need to access this
\                                         attribute which is also time-consuming to fetch.
\                                         Useful for advanced debugging / inspection
segment.download.warnings                 int
segment.download.errors                   int
segment.download.config                   str
segment.download.program_version          str
========================================= ================================================

config (dict)
~~~~~~~~~~~~~

This is the dictionary representing the chosen .yaml config file (usually, via command line).
By design, we strongly encourage to decouple code and configuration, so that you can easily
and safely experiment different configurations on the same code, if needed.
The config default file is documented with all necessary information, and you can put therein
whatever you want to be accessible as a python dict key, e.g. `config['mypropertyname']`
'''

from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

# OrderedDict is a python dict that returns its keys in the order they are inserted
# (a normal python dict returns its keys in arbitrary order)
# Useful e.g. in  "main" if we want to control the *order* of the columns in the output csv
from collections import OrderedDict
from datetime import datetime, timedelta  # always useful
from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
# import obspy core classes (when working with times, use obspy UTCDateTime when possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.sideplot, @gui.preprocess @gui.customplot:
from stream2segment.process.utils import gui
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
from stream2segment.process.math.traces import ampratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, utcdatetime, ampspec, powspec, timeof, respspec
# stream2segment function for processing numpy arrays:
from stream2segment.process.math.ndarrays import triangsmooth, snr, linspace


def assert1trace(stream):
    '''asserts the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps'''
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))


def main(segment, config):
    """
    Main processing function for generating output in a .csv file
    See `return` below for a detailed explanation of what this function should return after the
    processing is completed

    This function is called by executing the command:
    ```
        >>> stream2segment -p $PYFILE -c $CONFIG $OUTPUT
    ```
    where:
      - $PYFILE is the path of this file,
      - $CONFIG is a path to the .yaml configuration file (if this file was auto generated,
        it should be a file named $FILE.yaml)
      - $OUTPUT is the csv file where data (one row per segment) will to be saved

    For info about possible functions to use, please have a look at
    `stream2segment.process.math.traces`
    and obviously at `obpsy <https://docs.obspy.org/packages/index.html>`_, in particular:

    *  `obspy.core.Stream <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html#obspy.core.stream.Stream>_`
    *  `obspy.core.Trace <https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html#obspy.core.trace.Trace>_`

    :param: segment (ptyhon object): An object representing a waveform data to be processed,
    reflecting the relative database table row. See module docstring above for a detailed list
    of attributes and methods

    :param: config (python dict): a dictionary reflecting what has been implemented in $CONFIG.
    You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
    will be accessible as usual via `config['propertyname']`

    :return: an iterable (list, tuple, numpy array, dict...) of values. The returned iterable
    will be written as a row of the resulting csv file. If dict, the keys of the dict
    will populate the first row header of the resulting csv file, otherwise the csv file
    will have no header. Please be consistent: always return the same type of iterable for
    all segments; if dict, always return the same keys for all dicts; if list, always
    return the same length, etc.
    If you want to preserve the order of the dict keys as inserted in the code, use `OrderedDict`
    instead of `dict` or `{}`.
    Please note that the first column of the resulting csv will be *always* the segment id
    (an integer stored in the database uniquely identifying the segment). Thus the first value
    returned by the iterable of `main` will be in the csv file second column, the second in the
    third, and so on.
    If this function (or module, when imported) or any of the functions called raise any of the
    following:
    `TypeError`, `SyntaxError`, `NameError`, `ImportError`, `AttributeError`
    then the whole process will **stop**, because those exceptions are most likely caused
    by code errors which might affect all segments and the user can fix them without waiting
    for all segments to be processed.
    Otherwise, the function can **raise** any *other* Exception, or **return** None.
    In both cases, the iteration will not stop but will go on processing the following segment.
    None will silently ignore the segment, otherwise
    the exception message (with the segment id) will be written to a .log file in the same folder
    than the output csv file.
    Pay attention when setting complex objects (e.g., everything neither string nor numeric) as
    elements of the returned iterable: the values will be most likely converted to string according
    to python `__str__` function and might be out of control for the user.
    Thus, it is suggested to convert everything to string or number. E.g., for obspy's
    `UTCDateTime`s you could return either `float(utcdatetime)` (numeric) or
    `utcdatetime.isoformat()` (string)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now

    # discard saturated signals (according to the threshold set in the config file):
    amp_ratio = ampratio(trace)
    if amp_ratio >= config['amp_ratio_threshold']:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # bandpass the trace, according to the event magnitude.
    # WARNING: For efficiency reasons, this modifies `segment.stream()` permanently!
    # But this is not a requirement and the user can change this behavior. With the current
    # implementation, if you want to preserve the original stream, store trace.copy()
    # and later set segment.stream()[0] = trace
    trace = bandpass_remresp(segment, config)
    # From now on, segment.stream()[0] will return `trace`

    spectra = sn_spectra(segment, config)
    normal_f0, normal_df, normal_spe = spectra['Signal']
    noise_f0, noise_df, noise_spe = spectra['Noise']
    evt = segment.event
    fcmin = mag2freq(evt.magnitude)
    fcmax = config['preprocess']['bandpass_freq_max']  # used in bandpass_remresp
    snr_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
               fmin=fcmin, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    if snr_ < config['snr_threshold']:
        raise ValueError('low snr %f' % snr_)

    # calculate cumulative
    cum_labels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cum_trace = cumulative(segment, config)
    cum_times = cumtimes(cum_trace, *cum_labels)

    # double event
    pstart = 1
    pend = -2
    pref = 3
    (score, t_double, tt1, tt2) = \
        get_multievent_sg(cum_trace, cum_times[pstart], cum_times[pend], cum_times[pref],
                          config['threshold_inside_tmin_tmax_percent'],
                          config['threshold_inside_tmin_tmax_sec'],
                          config['threshold_after_tmax_percent'])
    if score in {1, 3}:
        raise ValueError('Double event detected %d %s %s %s' % (score, t_double, tt1, tt2))

    # calculate PGA and times of occurrence (t_PGA):
    t_PGA, PGA = maxabs(trace)  # note: you can also provide tstart tend for slicing
    trace_int = trace.copy()
    trace_int.integrate()
    t_PGV, PGV = maxabs(trace_int)

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = linspace(start=normal_f0, delta=normal_df, num=len(normal_spe))
    required_amplitudes = np.interp(required_freqs, ampspec_freqs, normal_spe) / segment.sample_rate

    # compute synthetic WA. Note: modifies the segment trace in-place!
    trace_wa = synth_wa(segment, config)
    t_WA, maxWA = maxabs(trace_wa)

    # write stuff to csv:
    ret = OrderedDict()

    ret['snr'] = snr_
    for cum_lbl, cum_t in zip(cum_labels, cum_times):
        ret['cum_t%d' % cum_lbl] = float(cum_t)  # convert cum_times to float for saving

    ret['dist_deg'] = segment.event_distance_deg        # dist
    ret['dist_km'] = d2km(segment.event_distance_deg)  # dist_km
    ret['t_PGA'] = t_PGA                  # peak info
    ret['PGA'] = PGA
    ret['t_PGV'] = t_PGV                  # peak info
    ret['PGV'] = PGV
    ret['t_WA'] = t_WA
    ret['maxWA'] = maxWA
    ret['channel'] = segment.channel.channel
    ret['channel_component'] = segment.channel.channel[-1]
    ret['ev_id'] = segment.event.id           # event metadata
    ret['ev_lat'] = segment.event.latitude
    ret['ev_lon'] = segment.event.longitude
    ret['ev_dep'] = segment.event.depth_km
    ret['ev_mag'] = segment.event.magnitude
    ret['ev_mty'] = segment.event.mag_type
    ret['st_id'] = segment.station.id         # station metadata
    ret['st_name'] = segment.station.station
    ret['st_net'] = segment.station.network
    ret['st_lat'] = segment.station.latitude
    ret['st_lon'] = segment.station.longitude
    ret['st_ele'] = segment.station.elevation

    for f, a in zip(required_freqs, required_amplitudes):
        ret['f_%.5f' % f] = float(a)

    return ret


@gui.preprocess
def bandpass_remresp(segment, config):
    """Applies a pre-process on the given segment waveform by
    filtering the signal and removing the instrumental response.

    The filter algorithm has the following steps:
    1. Sets the max frequency to 0.9 of the Nyquist frequency (sampling rate /2)
    (slightly less than Nyquist seems to avoid artifacts)
    2. Offset removal (subtract the mean from the signal)
    3. Tapering
    4. Pad data with zeros at the END in order to accommodate the filter transient
    5. Apply bandpass filter, where the lower frequency is set according to the magnitude
    6. Remove padded elements
    7. Remove the instrumental response

    IMPORTANT NOTES:
    - Being decorated with '@gui.preprocess', this function:
      * returns the *base* stream used by all plots whenever the relative check-box is on
      * must return either a Trace or Stream object

    - In this implementation THIS FUNCTION DOES MODIFY `segment.stream()` IN-PLACE: from within
      `main`, further calls to `segment.stream()` will return the stream returned by this function.
      However, MODIFYING THE STREAM IN-PLACE IS NOT A REQUIREMENT AND THE USER CAN CHANGE THIS
      BEHAVIOUR. In any case, you can use `segment.stream().copy()` before this call to keep the
      old "raw" stream

    :return: a Trace object.
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    inventory = segment.inventory()
    trace = stream[0]
    # define some parameters:
    evt = segment.event
    conf = config['preprocess']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, mag2freq(evt.magnitude), freq_max=conf['bandpass_freq_max'],
                     max_nyquist_ratio=conf['bandpass_max_nyquist_ratio'],
                     corners=conf['bandpass_corners'], copy=False)
    trace.remove_response(inventory=inventory, output=conf['remove_response_output'],
                          water_level=conf['remove_response_water_level'])
    return trace


def mag2freq(magnitude):
    if magnitude <= 4.5:
        freq_min = 0.4
    elif magnitude <= 5.5:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size-1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def get_multievent_sg(cum_trace, tmin, tmax, tstart,
                      threshold_inside_tmin_tmax_percent,
                      threshold_inside_tmin_tmax_sec, threshold_after_tmax_percent):
    """
        Returns the tuple (or a list of tuples, if the first argument is a stream) of the
        values (score, UTCDateTime of arrival)
        where scores is: 0: no double event, 1: double event inside tmin_tmax,
            2: double event after tmax, 3: both double event previously defined are detected
        If score is 2 or 3, the second argument is the UTCDateTime denoting the occurrence of the
        first sample triggering the double event after tmax
        :param trace: the input obspy.core.Trace
    """
    tmin = utcdatetime(tmin)
    tmax = utcdatetime(tmax)
    tstart = utcdatetime(tstart)

    # split traces between tmin and tmax and after tmax
    traces = [cum_trace.slice(tmin, tmax), cum_trace.slice(tmax, None)]

    # calculate second derivative and normalize:
    second_derivs = []
    max_ = np.nan
    for ttt in traces:
        ttt.taper(type='cosine', max_percentage=0.05)
        sec_der = savitzky_golay(ttt.data, 31, 2, deriv=2)
        sec_der_abs = np.abs(sec_der)
        idx = np.nanargmax(sec_der_abs)
        # get max (global) for normalization:
        max_ = np.nanmax([max_, sec_der_abs[idx]])
        second_derivs.append(sec_der_abs)

    # normalize second derivatives:
    for der in second_derivs:
        der /= max_

    result = 0

    # case A: see if after tmax we exceed a threshold
    indices = np.where(second_derivs[1] >= threshold_after_tmax_percent)[0]
    if len(indices):
        result = 2

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the duration
    deltatime = 0
    indices = np.where(second_derivs[0] >= threshold_inside_tmin_tmax_percent)[0]
    starttime = endtime = None
    if len(indices) >= 2:
        idx0 = indices[0]
        idx1 = indices[-1]
        starttime = timeof(traces[0], idx0)
        endtime = timeof(traces[0], idx1)
        deltatime = endtime - starttime
        if deltatime >= threshold_inside_tmin_tmax_sec:
            result += 1

    return result, deltatime, starttime, endtime


@gui.customplot
def synth_wa(segment, config):
    '''compute synthetic WA. This method does NOT remove the segment's stream instrumental response.
    Does not modify the segment's stream or traces in-place.

    IMPORTANT NOTES:

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return:  an obspy Trace
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    # compute synthetic WA,NOTE: keep as last action, it modifies trace!!
    config_wa = dict(config['paz_wa'])
    # parse complex string to complex numbers:
    zeros_parsed = map(complex, (c.replace(' ', '') for c in config_wa['zeros']))
    config_wa['zeros'] = list(zeros_parsed)
    poles_parsed = map(complex, (c.replace(' ', '') for c in config_wa['poles']))
    config_wa['poles'] = list(poles_parsed)
    # compute synthetic WA response. This modifies the trace in-place!
    return trace.copy().simulate(paz_remove=None, paz_simulate=config_wa)


@gui.customplot
def derivcum2(segment, config):
    """
    compute the second derivative of the cumulative function using savitzy-golay.
    Does not modify the segment's stream or traces in-place

    IMPORTANT NOTES:

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return: the tuple (starttime, timedelta, values)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    cum = cumulative(segment, config)
    sec_der = savitzky_golay(cum.data, 31, 2, deriv=2)
    sec_der_abs = np.abs(sec_der)
    sec_der_abs /= np.nanmax(sec_der_abs)  # FIXME: this should be sec_der_abs /= mmm
    # the stream object has surely only one trace (see 'cumulative')
    return segment.stream()[0].stats.starttime, segment.stream()[0].stats.delta, sec_der_abs


@gui.customplot
def cumulative(segment, config):
    '''Computes the cumulative of a trace in the form of a Plot object.
    Does not modify the segment's stream or traces in-place

    IMPORTANT NOTES:

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return: an obspy.Trace

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    return cumsum(trace)


@gui.sideplot
def sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place

    IMPORTANT NOTES:

    -Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
     a numeric sequence y taken at successive equally spaced points in any of these forms:
        - a Trace object
        - a Stream object
        - the tuple (x0, dx, y) or (x0, dx, y, label), where
            - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
            - dx (numeric or `timedelta`) is the sampling period
            - y (numpy array or numeric list) are the sequence values
            - label (string, optional) is the sequence name to be displayed on the plot legend.
              (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
              to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is
              numeric, then `dx` will be converted to `timedelta(seconds=dx)`)
        - a dict of any of the above types, where the keys (string) will denote each sequence
          name to be displayed on the plot legend.

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    signal_wdw, noise_wdw = segment.sn_windows()
    x0_sig, df_sig, sig = _spectrum(stream[0], config, *signal_wdw)
    x0_noi, df_noi, noi = _spectrum(stream[0], config, *noise_wdw)
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def _spectrum(trace, config, starttime=None, endtime=None):
    '''Calculate the spectrum of a trace. Returns the tuple (0, df, values), where
    values depends on the config dict parameters.
    Does not modify the trace in-place
    '''
    taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
    taper_type = config['sn_spectra']['taper']['type']
    if config['sn_spectra']['type'] == 'pow':
        func = powspec  # copies the trace if needed
    elif config['sn_spectra']['type'] == 'amp':
        func = ampspec  # copies the trace if needed
    else:
        # raise TypeError so that if called from within main, the iteration stops
        raise TypeError("config['sn_spectra']['type'] expects either 'pow' or 'amp'")

    df_, spec_ = func(trace, starttime, endtime,
                      taper_max_percentage=taper_max_percentage, taper_type=taper_type)

    # if you want to implement your own smoothing, change the lines below before 'return'
    # and implement your own config variables, if any
    smoothing_wlen_ratio = config['sn_spectra']['smoothing_wlen_ratio']
    if smoothing_wlen_ratio > 0:
        spec_ = triangsmooth(spec_, winlen_ratio=smoothing_wlen_ratio)

    return (0, df_, spec_)
