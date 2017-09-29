'''
=============================================================
Stream2segment: Processing+Visualization python file template
=============================================================

This is a template python module for processing downloaded waveform segments and defining the
segment plots to be visualized in the web GUI (Graphical user interface).

This file can be edited and passed to the program commands `s2s v` (visualize) and `s2s p` (process)
as -p or --pyfile option, together with an associated configuration .yaml file (-c option):
```
    s2s v -p [thisfilepath] -c [configfilepath] ...
    s2s p -p [thisfilepath] -c [configfilepath] ...
```
This module needs to implement few functions which will be described here (for full details,
look at their doc-string). All these functions must have the same signature:
```
    def myfunction(segment, config):
```
where `segment` is the python object representing a waveform data segment to be processed
and `config` is the python dictionary representing the given configuration .yaml file.

Processing
==========

When invoked via `s2s p ...`, the program will search for a function called "main", e.g.:
```
def main(segment, config)
```
the program will iterate over each selected segment (according to 'segment_select' parameter
in the config) and execute the function, writing its output to the given .csv file

Visualization (web GUI)
=======================

When invoked via `s2s v ...`, the program will search for all functions decorated with
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

This module is designed to force the DRY (don't repeat yourself) principle, thus if a portion
of code implemented in "main" should be visualized for inspection, it should be moved, not copied,
to a separated function (decorated with '@gui.customplot') and called from within "main"

All functions here can safely raise Exceptions, as all exceptions will be caught by the caller
displaying the error message on the plot if the function is called for visualization,
printing it to a log file if called for processing into .csv (More details on this in the "main"
function doc-string). Thus do not issue print statement in any function because, to put it short,
it's useless (and if you are used to do it extensively for debugging, consider changing this habit
and use a logger): if any information should be given, simply raise a base exception, e.g.:
`raise Exception("segment sample rate too low")`.

Functions arguments
===================

As said, all functions needed or implemented for processing and visualization must have the same
signature. In details, the two arguments passed to those functions are:

segment (object)
~~~~~~~~~~~~~~~~

Technically it's like an 'SqlAlchemy` ORM instance but for the user it is enough to
consider and treat it as a normal python object. It has special methods and several
attributes returning python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

segment methods:
----------------

segment.stream(): the `obspy.Stream` object representing the waveform data (raw or pre-processed)
associated to the segment

segment.timewindow('s') or segment.timewindow('signal'): a list of two `UTCDateTime`s denoting the
start and end time of the computed window on the waveform 'signal' part (opposed to waveform
'noise' part). The window is computed according to the settings of the associated yaml
configuration file, available under `config['sn_windows']`). You can use it to, e.g., trim
`segment.stream()` or any other Stream/Trace object via its `trim` method:
`trace_signal = trace.copy().trim(*segment.timewindow('s'), ...)`

segment.timewindow('n') or segment.timewindow('noise'): same as segment.timewindow('s'),
but returns the computed window on the waveform 'noise' part

segment.inventory(): the `obspy.core.inventory.inventory.Inventory`. This object is useful e.g.,
for removing the instrumental response from `segment.stream()`


segment attributes:
-------------------

========================================= ================================================
attribute                                 python type
========================================= ================================================
segment.id                                int
segment.event_distance_deg                float (distance between segment station and event,
\                                         in degrees)
segment.start_time                        datetime.datetime
segment.arrival_time                      datetime.datetime
segment.end_time                          datetime.datetime
segment.sample_rate                       float (as written in the segment bytes data,
\                                         might differ from segment.channel.sample_rate)
segment.download_status_code              int (typically, values between 200 and 399 denote
\                                         ok download. Values >=400 or lower than zero denote
\                                         errors)
segment.max_gap_overlap_ratio             float (denotes the maximum number of missing points,
\                                         if positive, or overlapping points, if negative,
\                                         in the waveform data.
\                                         As this number is easy to obtain while downloading
\                                         waveform bytes data, it is saved to the database so
\                                         that the user can speed up processing or visualization
\                                         by discarding malformed segments, if gaps/overlaps are a
\                                         concern. A value of 0 denotes no gaps/overlaps,
\                                         a value >= 1 denotes gaps, and a value <= 1 denotes
\                                         overlaps in the waveform data. However, as this number
\                                         is the ratio between the maximum gap/overlap interval
\                                         found, in seconds, and the waveform data sampling period,
\                                         in seconds, there is no way to safely assess
\                                         max_gap_overlap_ratio values in (-1, 1): a rule of
\                                         thumb is to select segments whose max_gap_overlap_ratio
\                                         is in the interval [-0.5, 0.5] and perform a check for
\                                         safety, e.g., via `len(segment.stream())` or
\                                         `segment.stream().get_gaps()`)
segment.seed_identifier                   str (string in the typical
\                                         Network.Station.Location.Channel format. Do not rely
\                                         on this value because it might be None - e.g. when
\                                         download errors occurred)
segment.data                              bytes (you don't generally need to access this
\                                         attribute which is also time-comsuming to fetch. It is
\                                         the raw data for building `stream()`)
----------------------------------------- ------------------------------------------------
segment.event                             object (attributes below)
segment.event.id                          str
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
segment.channel.id                        str
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
segment.channel.station                   object (same as segment.station, see below)
----------------------------------------- ------------------------------------------------
segment.station                           object (attributes below)
segment.station.id                        str
segment.station.network                   str
segment.station.station                   str
segment.station.latitude                  float
segment.station.longitude                 float
segment.station.elevation                 float
segment.station.site_name                 str
segment.station.start_time                datetime.datetime
segment.station.end_time                  datetime.datetime
segment.station.inventory_xml             bytes* (you don't generally need to access this
\                                         attribute which is also time-comsuming to fetch. It is
\                                         the raw data for building `inventory()`)
segment.station.datacenter                object (same as segment.datacenter, see below)
----------------------------------------- ------------------------------------------------
segment.datacenter                        object (attributes below)
segment.datacenter.id                     int
segment.datacenter.station_url            str
segment.datacenter.dataselect_url         str
segment.datacenter.node_organization_name str
----------------------------------------- ------------------------------------------------
segment.download                          object (attributes below): the download execution
segment.download.id                       int
segment.download.run_time                 datetime.datetime
segment.download.log                      str  (you don't generally need to access this
\                                         attribute which is also time-comsuming to fetch.
\                                         It is the log text written during download,
\                                         useful for debugging / inspection)
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
from stream2segment.utils.postdownload import gui
# strem2segment functions for processing mseeds. This is just a list of possible functions
# to show how to import them:
from stream2segment.analysis.mseeds import ampratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, utcdatetime, ampspec, powspec, timeof
# stream2segment function for processing numpy arrays:
from stream2segment.analysis import triangsmooth, snr


def main_typeerr(segment, config, wrong_argument):
    return [6]


def main_retlist(segment, config):
    """
    Main processing function for generating output in a .csv file
    See `return` below for a detailed explanation of what this function should return after the
    processing is completed

    This function is called by executing the command:
    ```
        >>> stream2segment p $PYFILE $CONFIG $OUTPUT
    ```
    where:
      - $PYFILE is the path of this file,
      - $CONFIG is a path to the .yaml configuration file (if this file was auto generated,
        it should be a file named $FILE.yaml)
      - $OUTPUT is the csv file where data (one row per segment) will to be saved

    For info about possible functions to use, please have a look at `stream2segment.analysis.mseeds`
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
    return the same length, etcetera.
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

    # discard streams with more than one trace:
    if len(stream) != 1:
        raise ValueError('more than one obspy.Trace. Possible cause: gaps')

    # work on the trace now. All functions will return Traces or scalars, which is better
    # so we can write them to database more easily
    trace = stream[0]

    # discard saturated signals (according to the threshold set in the config file):
    amp_ratio = ampratio(trace)
    if amp_ratio >= config['amp_ratio_threshold']:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # bandpass the trace, according to the event magnitude
    trace = bandpass_remresp(segment, config)

    normal_f0, normal_df, normal_spe = spectrum(trace, config, *segment.timewindow('signal'))
    noise_f0, noise_df, noise_spe = spectrum(trace, config, *segment.timewindow('noise'))
    evt = segment.event
    fcmin = mag2freq(evt.magnitude)
    fcmax = fcmax = config['preprocess']['bandpass_freq_max']  # was also used in bandpass_remresp
    snr_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
               fmin=fcmin, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    if snr_ < config['snr_threshold']:
        raise ValueError('low snr %f' % snr_)

    # remove response, note: modify trace!!
#     trace.remove_response(inventory=inventory, output=config['preprocess']['remove_response_output'],
#                                      water_level=config['preprocess']['remove_response_water_level'])

    # calculate cumulative
    cum_labels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cum_trace = cumulative(segment, config)
    cum_times = cumtimes(cum_trace, *cum_labels)

    # double event
    pstart = 1
    pend = -2
    pref = 3  # non mi ricordo ma lo uso!!!! (Grande Dino :))
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
    ampspec_freqs = np.linspace(start=normal_f0, stop=normal_df * len(normal_spe),
                                num=len(normal_spe), endpoint=False)
    required_amplitudes = np.interp(required_freqs, ampspec_freqs, normal_spe) / segment.sample_rate

    # compute synthetic WA. NOTE: keep as last action, it modifies trace!!
    trace_wa = synth_wa(segment, config)
    t_WA, maxWA = maxabs(trace_wa)

    # write stuff to csv:
    ret = [float(cum_t) for cum_lbl, cum_t in zip(cum_labels, cum_times)]

    ret += [segment.event_distance_deg,        # dist
            d2km(segment.event_distance_deg),  # dist_km
            t_PGA,                  # peak info
            PGA,
            t_PGV,                  # peak info
            PGV,
            t_WA,
            maxWA,
            segment.channel.channel,
            segment.event.id,           # event metadata
            segment.event.latitude,
            segment.event.longitude,
            segment.event.depth_km,
            segment.event.magnitude,
            segment.event.mag_type,
            segment.station.id,    # station metadata
            segment.station.station,
            segment.station.network,
            segment.station.latitude,
            segment.station.longitude,
            segment.station.elevation]

    ret += [float(a) for f, a in zip(required_freqs, required_amplitudes)]
    return ret


@gui.preprocess
def bandpass_remresp(segment, config):
    """Applies a pre-process on the given segment waveform by
    filtering the signal and removing the instrumental response
    The filter algorithm has the following steps:
    1. Sets the max frequency to 0.9 of the nyquist freauency (sampling rate /2)
    (slightly less than nyquist seems to avoid artifacts)
    2. Offset removal (substract the mean from the signal)
    3. Tapering
    4. Pad data with zeros at the END in order to accomodate the filter transient
    5. Apply bandpass filter, where the lower frequency is set according to the magnitude
    6. Remove padded elements
    7. Remove the instrumental response

    Being decorated with '@gui.preprocess', this function must return either a Trace or Stream
    object

    :return: a Trace object.
    """
    stream = segment.stream()
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))

    inventory = segment.inventory()

    trace = stream[0]
    # define some parameters:
    evt = segment.event
    conf = config['preprocess']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, mag2freq(evt.magnitude), freq_max=conf['bandpass_freq_max'],
                     max_nyquist_ratio=conf['bandpass_max_nyquist_ratio'],
                     corners=conf['bandpass_corners'], copy=True)
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

    # what's happen if threshold_inside_tmin_tmax_percent > tmax-tmin?
    # twin = tmax-tmin
    # if (threshold_inside_tmin_tmax_sec > twin):
    #    threshold_inside_tmin_tmax_sec = 0.8*twin
    ##

    double_event_after_tmax_time = None
    deltatime= None
    d_order = 2

    # split traces between tmin and tmax and after tmax
    traces = [cum_trace.slice(tmin, tmax), cum_trace.slice(tmax, None)]

    # calculate second derivative and normalize:
    derivs = []
    max_ = np.nan
    for ttt in traces:
        ttt.taper(type='cosine', max_percentage=0.05)
        sec_der = savitzky_golay(ttt.data,31,2,deriv=2)
        sec_der_abs = np.abs(sec_der)
        idx = np.nanargmax(sec_der_abs)
        max_ = np.nanmax([max_, sec_der_abs[idx]])  # get max (global) for normalization (see below):
        derivs.append(sec_der_abs)

    # normalize second derivatives:
    for der in derivs:
        der /= max_

    result = 0

    # case A: see if after tmax we exceed a threshold
    indices = np.where(derivs[1] >= threshold_after_tmax_percent)[0]
    if len(indices):
        result = 2
        double_event_after_tmax_time = timeof(traces[1], indices[0])  # FIXME

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the duration
    indices = np.where(derivs[0] >= threshold_inside_tmin_tmax_percent)[0]
    if len(indices) >= 2:
        idx0 = indices[0]
        idx1 = indices[-1]
        deltatime = (idx1 - idx0) * cum_trace.stats.delta
        # deltatime = timeof(traces[0], indices[-1]) - timeof(traces[0], indices[0])
        # deltatime = timeof(traces[1], indices[-1]) - tstart

        if deltatime >= threshold_inside_tmin_tmax_sec:
            result += 1

    return result, None, None, None
    # FIXME:
    # return result, deltatime, timeof(traces[0], indices[-1]), timeof(traces[0], indices[0])


@gui.customplot
def synth_wa(segment, config):
    '''compute synthetic WA. NOTE: keep as last action, it modifies trace!!

    Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
    a numeric sequence y taken at successive equally spaced points in any of these forms:
    - a Trace object
    - a Stream object
    - the tuple (x0, dx, y) or (x0, dx, y, label), where
        - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
        - dx (numeric or `timedelta`) is the sampling period
        - y (numpy array or numeric list) are the sequence values
        - label (string, optional) is the sequence name to be displayed on the plot legend.
          (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
          to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is numeric,
          then `dx` will be converted to `timedelta(seconds=dx)`)
    - a dict of any of the above types, where the keys (string) will denote each sequence
      name to be displayed on the plot legend.

    :return:  an obspy Trace
    '''
    stream = segment.stream()
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))

    trace = stream[0]
    # compute synthetic WA,NOTE: keep as last action, it modifies trace!!
    config_wa = dict(config['paz_wa'])
    # parse complex string to complex numbers:
    zeros_parsed = map(complex, (c.replace(' ', '') for c in config_wa['zeros']))
    config_wa['zeros'] = list(zeros_parsed)
    poles_parsed = map(complex, (c.replace(' ', '') for c in config_wa['poles']))
    config_wa['poles'] = list(poles_parsed)
    # compute synthetic WA response
    trace_wa = trace.simulate(paz_remove=None, paz_simulate=config_wa)

    return trace_wa


@gui.customplot
def derivcum2(segment, config):
    """
    compute the second derivative of the cumulative function using savitzy-golay

    Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
    a numeric sequence y taken at successive equally spaced points in any of these forms:
    - a Trace object
    - a Stream object
    - the tuple (x0, dx, y) or (x0, dx, y, label), where
        - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
        - dx (numeric or `timedelta`) is the sampling period
        - y (numpy array or numeric list) are the sequence values
        - label (string, optional) is the sequence name to be displayed on the plot legend.
          (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
          to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is numeric,
          then `dx` will be converted to `timedelta(seconds=dx)`)
    - a dict of any of the above types, where the keys (string) will denote each sequence
      name to be displayed on the plot legend.

    :return: the tuple (starttime, timedelta, values)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    cum = cumulative(segment, config)
    sec_der = savitzky_golay(cum.data, 31, 2, deriv=2)
    sec_der_abs = np.abs(sec_der)
    mmm = np.nanmax(sec_der_abs)
    sec_der /= mmm  # FIXME: this should be sec_der_abs /= mmm
    
    return segment.stream().stats.starttime, segment.stream().stats.delta, sec_der_abs


@gui.customplot
def cumulative(segment, config):
    '''Computes the cumulative of a trace in the form of a Plot object.

    Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
    a numeric sequence y taken at successive equally spaced points in any of these forms:
    - a Trace object
    - a Stream object
    - the tuple (x0, dx, y) or (x0, dx, y, label), where
        - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
        - dx (numeric or `timedelta`) is the sampling period
        - y (numpy array or numeric list) are the sequence values
        - label (string, optional) is the sequence name to be displayed on the plot legend.
          (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
          to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is numeric,
          then `dx` will be converted to `timedelta(seconds=dx)`)
    - a dict of any of the above types, where the keys (string) will denote each sequence
      name to be displayed on the plot legend.

    :return: an obspy.Trace

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    '''
    stream = segment.stream()
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))
    trace = stream[0]

    return cumsum(trace)


@gui.sideplot
def sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).

    Being decorated with '@gui.sideplot' or '@gui.customplot', this function must return
    a numeric sequence y taken at successive equally spaced points in any of these forms:
    - a Trace object
    - a Stream object
    - the tuple (x0, dx, y) or (x0, dx, y, label), where
        - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point
        - dx (numeric or `timedelta`) is the sampling period
        - y (numpy array or numeric list) are the sequence values
        - label (string, optional) is the sequence name to be displayed on the plot legend.
          (if x0 is numeric and `dx` is a `timedelta` object, then x0 will be converted
          to `UTCDateTime(x0)`; if x0 is a `datetime` or `UTCDateTime` object and `dx` is numeric,
          then `dx` will be converted to `timedelta(seconds=dx)`)
    - a dict of any of the above types, where the keys (string) will denote each sequence
      name to be displayed on the plot legend.

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))

    x0_sig, df_sig, sig = spectrum(stream[0], config, *segment.timewindow('signal'))
    x0_noi, df_noi, noi = spectrum(stream[0], config, *segment.timewindow('noise'))
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def spectrum(trace, config, starttime=None, endtime=None):
    '''Calculate the spectrum of a trace. Returns the tuple 0, df, values, where
    values depends on the config dict parameters'''
    taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
    taper_type = config['sn_spectra']['taper']['type']
    if config['sn_spectra']['type'] == 'pow':
        func = powspec
    elif config['sn_spectra']['type'] == 'amp':
        func = ampspec
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
