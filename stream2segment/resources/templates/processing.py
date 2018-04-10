'''
====================================================================
Stream2segment: Processing and/or Visualization python file template
====================================================================

This is a template python module for processing downloaded waveform segments and optionally defining
functions that can be visualized as plots in the web GUI (Graphical user interface).

This module needs to implement one or more functions which will be described in the sections below.
**All these functions must have the same signature**:
```
    def myfunction(segment, config):
```
where `segment` is the python object representing a waveform data segment to be processed
and `config` is the python dictionary representing the given configuration .yaml file.

After editing, this file can be invoked from the command line commands `s2s process` and `s2s show`
with the `-p` / `--pyfile` option (type `s2s show --help` or `s2s show --help` for details).
In the first case, see section 'Processing' below, otherwise see section 'Visualization (web GUI)'.
In both cases, please read the remaining of this documentation.


Processing
==========

When processing, the program will search for a function called "main", e.g.:
```
def main(segment, config)
```
the program will iterate over each selected segment (according to 'segment_select' parameter
in the config) and execute the function, writing its output to the given .csv file, if given.
If you do not need to use this module for visualizing stuff, skip the section below and go to the
next one.


Visualization (web GUI)
=======================

When visualizing, the program will fetch all segments (according
to 'segment_select' parameter in the config), and open a web page where the user can browse and
visualize each segment one at a time.
The page shows by default on the upper left corner a plot representing the segment trace(s).
The GUI can be customized by providing here functions decorated with
"@gui.preprocess" or "@gui.plot".
Plot functions can return only special 'plottable' values (basically arrays,
more details in their doc-strings).
The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plots of the page will be re-calculated with the output of this function,
which **must thus return an obspy Stream or Trace object**.
The function decorated with "@gui.plot", e.g.:
```
@gui.plot
def cumulative(segment, config)
...
```
will be associated to (i.e., its output will be displayed in) the plot below the main plot.
You can also call @gui.plot with arguments, e.g.:
```
@gui.plot(position='r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def spectra(segment, config)
...
```
The first one controls where the plot
will be placed in the GUI ('b' means bottom, the default, 'r' means right to the main plot)
and the other two, `xaxis` and `yaxis`, are dict (defaulting to the empty dict {}) controlling
the x and y axis of the plot. For info, see:
https://plot.ly/python/axes/
When not given, axis types will be inferred from the function return type (see below) and in most
cases defaults to 'date' (i.e., date-times on the x values).
Functions decorated with '@gui.plot' must return
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

Functions implementation
========================

The implementation of the functions is user-dependent. As said, all functions needed for
processing and visualization must have the same signature:
```
    def myfunction(segment, config):
```

all functions can safely raise Exceptions, as all exceptions will be caught by the caller:

* displaying the error message on the plot if the function is called for visualization,

* printing it to a log file, if the function is called for processing into .csv
  (More details on this in the "main" function doc-string).
  Issuing `print` statements for debugging it's thus useless (and a bad practice overall):
  if any information should be given, simply raise a base exception, e.g.:
  `raise Exception("segment sample rate too low")`.

Conventions and suggestions
---------------------------

1) This module is designed to encourage the decoupling of code and configuration, so that you can
easily and safely experiment different configurations on the same code, if needed. We strongly
discuourage to implement a python file and copy/paste it by changing some parameters only,
as it is very unmantainable and bug-prone. That said, it's up to you (e.g. a script-like
processing file for saving once some selected segments to a file might have the output directory
hard-coded)

2) This module is designed to force the DRY (don't repeat yourself) principle. This is particularly
important when using the GUI to visually debug / inspect some code for processing
implemented in `main`: we strongly encourage to *move* the portion of code into a separate
function F and call F from 'main' AND decorate it with '@gui.plot'. That said, it's up to you
also in this case (e.g. a file becoming too big might be separated into processing and
visualization, paying attention that modification of the code in one file might need
synchronization with the other file)


Functions arguments
-------------------

config (dict)
~~~~~~~~~~~~~

This is the dictionary representing the chosen .yaml config file (usually, via command line).
As said, we strongly encourage to decouple code and configuration, so that you can easily
and safely experiment different configurations on the same code, if needed.
The config default file is documented with all necessary information, put therein
whatever property you want, e.g.:
```
outfile: '/home/mydir/root'
mythreshold: 5.67
```
and it will be accessible via `config['outfile']`, `config['mythreshold']`

segment (object)
~~~~~~~~~~~~~~~~

Technically it's like an 'SqlAlchemy` ORM instance but for the user it is enough to
consider and treat it as a normal python object. It features special methods and
several attributes returning python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

segment methods:
----------------

* segment.stream(): the `obspy.Stream` object representing the waveform data
  associated to the segment. Please remember that many obspy functions modify the
  stream in-place:
  ```
      stream = segment.stream()
      stream_remresp = s.remove_response(segment.inventory())
      stream is segment.stream()  # False
      stream_remresp is segment.stream()  # True
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

* segment.seiscomp_path(root='.'): Returns a file system path where to store
  the given segment or any data associated to it in the seiscomp-compatible format:
     <root>/<net>/<sta>/<loc>/<cha>.D/<net>.<sta>.<loc>.<cha>.<year>.<day>.<event_id>
  The optional root argument, when missing, defaults to '.' (current working directory)
  The directory name of the path is accessible via `os.path.dirname(segment.seiscomp_path())`
  The file name of the path is accessible via `os.path.basename(segment.seiscomp_path())`. Note
  that the file name has intentionally no extension because the user might be interested to save
  different types of segment's data (metadata, spectra, etcetera). In the typical case where the
  segment's stream has to be saved you can type:
      segment.stream().write(segment.seiscomp_path() + '.mseed', format='MSEED')

* segment.dbsession(): WARNING: this is for advanced users experienced with Sql-Alchemy library:
  returns the database session for custom IO operations with the database


segment attributes:
-------------------

========================================= ================================================
attribute                                 python type and description (if any)
========================================= ================================================
segment.id                                int: segment (unique) db id
segment.event_distance_deg                float: distance between the segment's station and
                                          the event, in degrees
segment.event_distance_km                 float: distance between the segment's station and
                                          the event, in km, assuming a perfectly spherical earth
                                          with a radius of 6371 km
segment.start_time                        datetime.datetime: the waveform data start time
segment.arrival_time                      datetime.datetime
segment.end_time                          datetime.datetime: the waveform data end time
segment.request_start                     datetime.datetime: the requested start time of the data
segment.request_end                       datetime.datetime: the requested end time of the data
segment.duration_sec                      float: the waveform data duration, in seconds
segment.missing_data_sec                  float: the number of seconds of missing data, with respect
                                          to the request time window. E.g. if we requested 5
                                          minutes of data and we got 4 minutes, then
                                          missing_data_sec=60; if we got 6 minutes, then
                                          missing_data_sec=-60. This attribute is particularly
                                          useful in the config to select only well formed data and
                                          speed up the processing, e.g.: missing_data_sec: '< 120'
segment.missing_data_ratio                float: the portion of missing data, with respect
                                          to the request time window. E.g. if we requested 5
                                          minutes of data and we got 4 minutes, then
                                          missing_data_ratio=0.2 (20%); if we got 6 minutes, then
                                          missing_data_ratio=-0.2. This attribute is particularly
                                          useful in the config to select only well formed data and
                                          speed up the processing, e.g.: missing_data_ratio: '< 0.5'
segment.sample_rate                       float: the waveform data sample rate.
                                          It might differ from the segment channel's sample_rate
segment.has_data                          boolean: tells if the segment has data saved (at least
                                          one byte of data). This attribute useful in the config to
                                          select only well formed data and speed up the processing,
                                          e.g. has_data: 'true'.
segment.download_code                     int: the download code (for experienced users). As for
                                          any HTTP status code,
                                          values between 200 and 399 denote a successful download
                                          (this does not tell anything about the segment's data,
                                          which might be empty anyway. See 'segment.has_data'.
                                          Conversely, a download error assures no data has been
                                          saved), whereas
                                          values >=400 and < 500 denote client errors and
                                          values >=500 server errors.
                                          Moreover,
                                          -1 indicates a general download error - e.g. no Internet
                                          connection,
                                          -2 a successful download with corrupted waveform data,
                                          -200 a successful download where some waveform data chunks
                                          (miniSeed records) have been discarded because completely
                                          outside the requested time span,
                                          -204 a successful download where no data has been saved
                                          because all chunks were completely outside the requested
                                          time span, and finally:
                                          None denotes a successful download where no data has been
                                          saved because the given segment wasn't found in the
                                          server response (note: this latter case is NOT the case
                                          when the server returns no data with an appropriate
                                          'No Content' message with download_code=204)
segment.maxgap_numsamples                 float: the maximum gap found in the waveform data, in
                                          number of points. This attribute is particularly useful
                                          in the config to select only well formed data and speed
                                          up the processing.
                                          If this attribute is zero, the segment has no
                                          gaps/overlaps, if >=1 the segment has gaps, if <=-1,
                                          the segment has overlaps.
                                          Values in (-1, 1) are difficult to interpret: as this
                                          number is the ratio between
                                          the waveform data's max gap/overlap and its sampling
                                          period (both in seconds), a rule of thumb is to
                                          consider a segment with gaps/overlaps when this
                                          attribute's absolute value exceeds 0.5, e.g. you can
                                          discard segments with gaps overlaps by inputting in the
                                          config "maxgap_numsamples:  '[-0.5, 0.5]'" and, if you
                                          absolutely want no segment with gaps/overlaps,
                                          perform a further check in the processing via
                                          `len(segment.stream())` (zero if no gaps/overlaps) or
                                          `segment.stream().get_gaps()` (see obspy doc)
segment.data_seed_id                      str: the seed identifier in the typical format
                                          [Network.Station.Location.Channel] stored in the
                                          segment's data. It might be null if the data is empty
                                          or null (e.g., because of a download error).
                                          See also 'segment.seed_id'
segment.seed_id                           str: the seed identifier in the typical format
                                          [Network.Station.Location.Channel]: it is the same as
                                          'segment.data_seed_id' if the latter is not null,
                                          otherwise it is fetched from the segment's metadata
                                          (in this case, the operation might more time consuming)
segment.has_class                         boolean: tells if the segment has (at least one) class
                                          assigned
segment.data                              bytes: the waveform (raw) data. You don't generally need
                                          to access this attribute which is also time-consuming
                                          to fetch. Used by `segment.stream()`
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
                                          generally need to access this attribute which is also
                                          time-consuming to fetch. Used by `segment.inventory()`
segment.station.has_inventory             boolean: tells if the segment's station inventory has
                                          data saved (at least one byte of data).
                                          This attribute useful in the config to select only
                                          segments with inventory downloaded and speed up the
                                          processing,
                                          e.g. has_inventory: 'true'.
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
                                          You don't generally need to access this
                                          attribute which is also time-consuming to fetch.
                                          Useful for advanced debugging / inspection
segment.download.warnings                 int
segment.download.errors                   int
segment.download.config                   str
segment.download.program_version          str
----------------------------------------- ------------------------------------------------
segment.classes.id                        int: the id(s) of the classes assigned to the segment
segment.classes.label                     int: the label(s) of the classes assigned to the segment
segment.classes.description               int: the description(s) of the classes assigned to the
                                          segment
========================================= ================================================
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
# decorators needed to setup this module @gui.preprocess @gui.plot:
from stream2segment.process.utils import gui
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
from stream2segment.process.math.traces import ampratio, bandpass, cumsumsq,\
    cumtimes, fft, maxabs, utcdatetime, ampspec, powspec, timeof
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
    Main processing function. The user should implement here the processing steps for any given
    selected segment. Useful links for functions, libraries and utilities:

    - `stream2segment.analysis.mseeds` (small processing library implemented in this program,
      most of its functions are imported here by default)
    - `obpsy <https://docs.obspy.org/packages/index.html>`_
    - `obspy Stream object <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html>_`
    - `obspy Trace object <https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html>_`

    IMPORTANT: The output messages of this program will be redirected either to standard error or
    to a log file (see documentation of `s2s process` for details). This includes exceptions
    raised by this function: note that any of the following exceptions:
        `TypeError`, `SyntaxError`, `NameError`, `ImportError`, `AttributeError`
    is most likely a bug: if raised by this function, then the whole process will **stop**.
    On the other hand, any other exception will just skip the current segment and can be
    raised programmatically. E.g. one could write:
    ```
        if snr <0.4:
            raise Exception('SNR ratio to low')
    ```
    to print the current segment id with the message 'SNR ratio too low' in the
    log file or standard error (the segment id is added automatically by the program).

    :param: segment (ptyhon object): An object representing a waveform data to be processed,
    reflecting the relative database table row. See module docstring above for a detailed list
    of attributes and methods

    :param: config (python dict): a dictionary reflecting what has been implemented in $CONFIG.
    You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
    will be accessible as usual via `config['propertyname']`

    :return: If the processing routine calling this function needs not generate output in a .csv
    file, the return value of this function will not be processed and can be whatever.
    Otherwise, this function must return an iterable (list, tuple, numpy array, dict...
    obviously, the same type should be returned for all segments, with the same number of elements).
    The iterable will be written as a row of the resulting csv file. The .csv file will have a
    row header only if `dict`s are returned: in this case, the dict keys are used as row header
    columns.
    If you want to preserve in the .csv the order of the dict keys as the were inserted
    in the dict, use `OrderedDict` instead of `dict` or `{}`.
    Returning None is also valid: in this case the segment will be silently skipped

    NOTES: The first column of the resulting csv will be *always* the segment id
    (an integer stored in the database uniquely identifying the segment)

    Pay attention when setting complex objects (e.g., everything neither string nor numeric) as
    elements of the iterable: the values will be most likely converted to string according
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
    # WARNING: this modifies the segment.stream() permanently!
    # If you want to preserve the original stream, store trace.copy()
    trace = bandpass_remresp(segment, config)

    spectra = sn_spectra(segment, config)
    normal_f0, normal_df, normal_spe = spectra['Signal']
    noise_f0, noise_df, noise_spe = spectra['Noise']
    evt = segment.event
    fcmin = mag2freq(evt.magnitude)
    fcmax = config['preprocess']['bandpass_freq_max']  # used in bandpass_remresp
    snr_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
               fmin=fcmin, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    snr1_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
                fmin=fcmin, fmax=1, delta_signal=normal_df, delta_noise=noise_df)
    snr2_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
                fmin=1, fmax=10, delta_signal=normal_df, delta_noise=noise_df)
    snr3_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
                fmin=10, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    if snr_ < config['snr_threshold']:
        raise ValueError('low snr %f' % snr_)

    # calculate cumulative

    cum_labels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cum_trace = cumsumsq(trace, copy=True)  # prevent original trace from being modified
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
    # note: you can also provide tstart tend for slicing
    t_PGA, PGA = maxabs(trace, cum_times[1], cum_times[-2])
    trace_int = trace.copy()
    trace_int.integrate()
    t_PGV, PGV = maxabs(trace_int, cum_times[1], cum_times[-2])
    meanoff = meanslice(trace_int, 100, cum_times[-1], trace_int.stats.endtime)

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = linspace(start=normal_f0, delta=normal_df, num=len(normal_spe))
    required_amplitudes = np.interp(np.log10(required_freqs),
                                    np.log10(ampspec_freqs), normal_spe) / segment.sample_rate

    # compute synthetic WA.
    # IMPORTANT: modifies the segment trace in-place!
    trace_wa = synth_wa(segment, config)
    t_WA, maxWA = maxabs(trace_wa)

    # write stuff to csv:
    ret = OrderedDict()

    ret['snr'] = snr_
    ret['snr1'] = snr1_
    ret['snr2'] = snr2_
    ret['snr3'] = snr3_
    for cum_lbl, cum_t in zip(cum_labels[slice(1, 8, 3)], cum_times[slice(1, 8, 3)]):
        ret['cum_t%f' % cum_lbl] = float(cum_t)  # convert cum_times to float for saving

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
    ret['score'] = score
    ret['d2max'] = float(tt1)
    ret['offset'] = np.abs(meanoff/PGV)
    for freq, amp in zip(required_freqs, required_amplitudes):
        ret['f_%.5f' % freq] = float(amp)

    return ret


@gui.preprocess
def bandpass_remresp(segment, config):
    """Applies a pre-process on the given segment waveform by
    filtering the signal and removing the instrumental response.
    DOES modify the segment stream in-place (see below).

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
      However, In any case, you can use `segment.stream().copy()` before this call to keep the
      old "raw" stream

    :return: a Trace object.
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]

    inventory = segment.inventory()

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
    '''returns a magnitude dependent frequency (in Hz)'''
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
    except ValueError:
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
    #   starttime = timeof(traces[0], indices[0])

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the duration
    deltatime = 0
    starttime = tmin
    endtime = None
    indices = np.where(second_derivs[0] >= threshold_inside_tmin_tmax_percent)[0]
    if len(indices) >= 2:
        idx0 = indices[0]
        starttime = timeof(traces[0], idx0)
        idx1 = indices[-1]
        endtime = timeof(traces[0], idx1)
        deltatime = endtime - starttime
        if deltatime >= threshold_inside_tmin_tmax_sec:
            result += 1

    return result, deltatime, starttime, endtime


@gui.plot
def synth_wa(segment, config):
    '''compute synthetic WA. See ``_synth_wa``.
    DOES modify the segment's stream or traces in-place.

    :return:  an obspy Trace
    '''
    return _synth_wa(segment, config, config['preprocess']['remove_response_output'])


def _synth_wa(segment, config, trace_input_type=None):
    '''
    Low-level function to calculate the synthetic wood-anderson of `trace`.
    The dict ``config['simulate_wa']`` must be implemented
    and houses the wood-anderson configuration 'sensitivity', 'zeros', 'poles' and 'gain'

    :param trace_input_type:
        None: trace is unprocessed and trace.remove_response(.. output="DISP"...)
            will be applied on it before applying `trace.simulate`
        'ACC': trace is already processed, e.g..
            it's the output of trace.remove_response(..output='ACC')
        'VEL': trace is already processed,
            it's the output of trace.remove_response(..output='VEL')
        'DISP': trace is already processed,
            it's the output of trace.remove_response(..output='DISP')

    Warning: modifies the trace in place
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]

    conf = config['preprocess']
    config_wa = dict(config['paz_wa'])
    # parse complex string to complex numbers:
    zeros_parsed = map(complex, (c.replace(' ', '') for c in config_wa['zeros']))
    config_wa['zeros'] = list(zeros_parsed)
    poles_parsed = map(complex, (c.replace(' ', '') for c in config_wa['poles']))
    config_wa['poles'] = list(poles_parsed)
    # compute synthetic WA response. This modifies the trace in-place!

    if trace_input_type in ('VEL', 'ACC'):
        trace.integrate()
    if trace_input_type == 'ACC':
        trace.integrate()

    if trace_input_type is None:
        pre_filt = (0.005, 0.006, 40.0, 45.0)
        trace.remove_response(inventory=segment.inventory(), output="DISP",
                              pre_filt=pre_filt, water_level=conf['remove_response_water_level'])

    return trace.simulate(paz_remove=None, paz_simulate=config_wa)


@gui.plot
def velocity(segment, config):
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    trace_int = trace.copy()
    return trace_int.integrate()


@gui.plot
def derivcum2(segment, config):
    """
    compute the second derivative of the cumulative function using savitzy-golay.
    DOES modify the segment's stream or traces in-place

    :return: the tuple (starttime, timedelta, values)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    cum = cumulative(segment, config)
    sec_der = savitzky_golay(cum.data, 31, 2, deriv=2)
    sec_der_abs = np.abs(sec_der)
    sec_der_abs /= np.nanmax(sec_der_abs)
    # the stream object has surely only one trace (see 'cumulative')
    return segment.stream()[0].stats.starttime, segment.stream()[0].stats.delta, sec_der_abs


@gui.plot
def cumulative(segment, config):
    '''Computes the cumulative of the squares of the segment's trace in the form of a Plot object.
    DOES modify the segment's stream or traces in-place. Normalizes the returned trace values
    in [0,1]

    :return: an obspy.Trace

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return cumsumsq(stream[0], copy=False)


# def _cumulative(trace):
#     '''Computes the cumulative of the squares of the segment's trace in the form of a Plot object.
#     DOES modify the segment's stream or traces in-place. Normalizes the returned trace values
#     in [0,1]'''
#     return cumsumsq(trace, normalize=True, copy=False)


@gui.plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place

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


def meanslice(trace, nptmin=100, starttime=None, endtime=None):
    """
    at least nptmin points
    """
    if starttime is not None or endtime is not None:
        trace = trace.slice(starttime, endtime)
    if trace.stats.npts < nptmin:
        return np.nan
    val = np.nanmean(trace.data)
    return val
