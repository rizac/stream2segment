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

    url = stream.datancenter.dataselect_url + \
        "?network=%s&station=%s&location=%s&channel=%s&start=%s&end=%s" % \
        (segment.network, segment.station, segment.location, segment.channel,
         segment.request_start.isoformat(), segment.request_end.isoformat())

    try:
        obspy_stream = read(url)
    except Exception as exc:
      raise ValueError("Cannot fetch mseed: " + str(exc)) 

    
    assert1trace(obspy_stream)

    res_stime = obspy_stream[0].stats.starttime
    res_etime = obspy_stream[0].stats.endtime
    # let's work with UTCDateTime's
    req_stime = UTCDateTime(segment.request_start)
    req_etime = UTCDateTime(segment.request_end)

    # -204: everything out of request
    # -200: something out of request

    # % of response outside request (over total response length): percout
    # % of response inside request (over total response length): percin

    res_len = res_etime - res_stime
    percout = max(0, req_stime - res_stime) + max(0, res_etime - req_etime)
    
    

    # write stuff to csv:
    ret = OrderedDict()

    ret['\% resp outside req'] = percout
    ret['url'] = url

    return ret


