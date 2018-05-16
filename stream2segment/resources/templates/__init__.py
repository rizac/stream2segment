'''doc strings to be injected via jinja2 into the templates when running `s2s init`.
Any NON PRIVATE variable name (i.e., without leading underscore '_') of this module
can be injected in a template file in the usual way, e.g.:
{{ PROCESS_PY_BANDPASSFUNC }}
For any new variable name to be implemented here, note also that: the variables values are
stripped before being assigned to the gobal DOCVARS (the dict passed to to jinja). *_PY_* variable
names are for python docs, *_YAML_* variable names for yaml docs. In the latter case,
yaml variables values do not need a leading '# ' on the first line, as it is usually input in
the template file, e.g.:
# {{ PROCESS_YAML_MAIN }}
'''


def _linecomment(string, prefix, first=True):
    '''prefixes any line of string with `prefix` and returns the resulting string'''
    if first is False:
        return ('\n' + prefix).join(line for line in string.splitlines())
    return '\n'.join(prefix + line for line in string.splitlines())


_SEGMENT_ATTRS = '''
===================================== ================================================
segment attribute                     python type and description (if any)
===================================== ================================================
segment.id                            int: segment (unique) db id
segment.event_distance_deg            float: distance between the segment's station and
                                      the event, in degrees
segment.event_distance_km             float: distance between the segment's station and
                                      the event, in km, assuming a perfectly spherical earth
                                      with a radius of 6371 km
segment.start_time                    datetime.datetime: the waveform data start time
segment.arrival_time                  datetime.datetime
segment.end_time                      datetime.datetime: the waveform data end time
segment.request_start                 datetime.datetime: the requested start time of the data
segment.request_end                   datetime.datetime: the requested end time of the data
segment.duration_sec                  float: the waveform data duration, in seconds
segment.missing_data_sec              float: the number of seconds of missing data, with respect
                                      to the request time window. E.g. if we requested 5
                                      minutes of data and we got 4 minutes, then
                                      missing_data_sec=60; if we got 6 minutes, then
                                      missing_data_sec=-60. This attribute is particularly
                                      useful in the config to select only well formed data and
                                      speed up the processing, e.g.: missing_data_sec: '< 120'
segment.missing_data_ratio            float: the portion of missing data, with respect
                                      to the request time window. E.g. if we requested 5
                                      minutes of data and we got 4 minutes, then
                                      missing_data_ratio=0.2 (20%); if we got 6 minutes, then
                                      missing_data_ratio=-0.2. This attribute is particularly
                                      useful in the config to select only well formed data and
                                      speed up the processing, e.g.: missing_data_ratio: '< 0.5'
segment.sample_rate                   float: the waveform data sample rate.
                                      It might differ from the segment channel's sample_rate
segment.has_data                      boolean: tells if the segment has data saved (at least
                                      one byte of data). This attribute useful in the config to
                                      select only well formed data and speed up the processing,
                                      e.g. has_data: 'true'.
segment.download_code                 int: the download code (for experienced users). As for
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
segment.maxgap_numsamples             float: the maximum gap found in the waveform data, in
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
segment.data_seed_id                  str: the seed identifier in the typical format
                                      [Network.Station.Location.Channel] stored in the
                                      segment's data. It might be null if the data is empty
                                      or null (e.g., because of a download error).
                                      See also 'segment.seed_id'
segment.seed_id                       str: the seed identifier in the typical format
                                      [Network.Station.Location.Channel]: it is the same as
                                      'segment.data_seed_id' if the latter is not null,
                                      otherwise it is fetched from the segment's metadata
                                      (in this case, the operation might more time consuming)
segment.has_class                     boolean: tells if the segment has (at least one) class
                                      assigned
segment.data                          bytes: the waveform (raw) data. You don't generally need
                                      to access this attribute which is also time-consuming
                                      to fetch. Used by `segment.stream()`
------------------------------------- ------------------------------------------------
segment.event                         object (attributes below)
segment.event.id                      int
segment.event.event_id                str: the id returned by the web service
segment.event.time                    datetime.datetime
segment.event.latitude                float
segment.event.longitude               float
segment.event.depth_km                float
segment.event.author                  str
segment.event.catalog                 str
segment.event.contributor             str
segment.event.contributor_id          str
segment.event.mag_type                str
segment.event.magnitude               float
segment.event.mag_author              str
segment.event.event_location_name     str
------------------------------------- ------------------------------------------------
segment.channel                       object (attributes below)
segment.channel.id                    int
segment.channel.location              str
segment.channel.channel               str
segment.channel.depth                 float
segment.channel.azimuth               float
segment.channel.dip                   float
segment.channel.sensor_description    str
segment.channel.scale                 float
segment.channel.scale_freq            float
segment.channel.scale_units           str
segment.channel.sample_rate           float
segment.channel.band_code             str: the first letter of channel.channel
segment.channel.instrument_code       str: the second letter of channel.channel
segment.channel.orientation_code      str: the third letter of channel.channel
segment.channel.station               object: same as segment.station (see below)
------------------------------------- ------------------------------------------------
segment.station                       object (attributes below)
segment.station.id                    int
segment.station.network               str
segment.station.station               str
segment.station.latitude              float
segment.station.longitude             float
segment.station.elevation             float
segment.station.site_name             str
segment.station.start_time            datetime.datetime
segment.station.end_time              datetime.datetime
segment.station.inventory_xml         bytes. The station inventory (raw) data. You don't
                                      generally need to access this attribute which is also
                                      time-consuming to fetch. Used by `segment.inventory()`
segment.station.has_inventory         boolean: tells if the segment's station inventory has
                                      data saved (at least one byte of data).
                                      This attribute useful in the config to select only
                                      segments with inventory downloaded and speed up the
                                      processing,
                                      e.g. has_inventory: 'true'.
segment.station.datacenter            object (same as segment.datacenter, see below)
------------------------------------- ------------------------------------------------
segment.datacenter                    object (attributes below)
segment.datacenter.id                 int
segment.datacenter.station_url        str
segment.datacenter.dataselect_url     str
segment.datacenter.organization_name  str
------------------------------------- ------------------------------------------------
segment.download                      object (attributes below): the download execution
segment.download.id                   int
segment.download.run_time             datetime.datetime
segment.download.log                  str: The log text of the segment's download execution.
                                      You don't generally need to access this
                                      attribute which is also time-consuming to fetch.
                                      Useful for advanced debugging / inspection
segment.download.warnings             int
segment.download.errors               int
segment.download.config               str
segment.download.program_version      str
------------------------------------- ------------------------------------------------
segment.classes.id                    int: the id(s) of the classes assigned to the segment
segment.classes.label                 int: the label(s) of the classes assigned to the segment
segment.classes.description           int: the description(s) of the classes assigned to the
                                      segment
===================================== ================================================
'''.strip()


PROCESS_PY_BANDPASSFUNC = """
Applies a pre-process on the given segment waveform by
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

PROCESS_PY_MAINFUNC = '''
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
    if snr < 0.4:
        raise Exception('SNR ratio to low')
```
to print the current segment id with the message 'SNR ratio too low' in the
log file or standard error (the segment id is added automatically by the program).

:param: segment (ptyhon object): An object representing a waveform data to be processed,
reflecting the relative database table row. See module docstring above for a detailed list
of attributes and methods

:param: config (python dict): a dictionary reflecting what has been implemented in the configuration
file. You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
will be accessible as usual via `config['propertyname']`

:return: If the processing routine calling this function needs not generate a .csv file output,
the return value of this function will not be processed and can be whatever.
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
'''

PROCESS_PY_MAIN = '''
============================================================================================
stream2segment python file to implement the processing/visualization subroutines: User guide
============================================================================================

This module needs to implement one or more functions which will be described in the sections below.
**All these functions must have the same signature**:
```
    def myfunction(segment, config):
```
where `segment` is the python object representing a waveform data segment to be processed
and `config` is the python dictionary representing the given configuration file.

After editing, this file can be invoked from the command line commands `s2s process` and `s2s show`
with the `-p` / `--pyfile` option (type `s2s process --help` or `s2s show --help` for details).
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
Functions decorated this way (Plot functions) can return only special 'plottable' values
(basically arrays, more details in their doc-strings, if provided in this template).
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
will be placed in the GUI ('b' means bottom, the default, 'r' means next to the main plot, on its
right) and the other two, `xaxis` and `yaxis`, are dict (defaulting to the empty dict {})
controlling the x and y axis of the plot. For info, see:
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

This is the dictionary representing the chosen configuration file (usually, via command line)
in YAML format (see documentation therein). Any property defined in the configuration file, e.g.:
```
outfile: '/home/mydir/root'
mythreshold: 5.67
```
will be accessible via `config['outfile']`, `config['mythreshold']`

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
      segment.stream() is stream          # False !!!
      segment.stream() is stream_remresp  # True !!!
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
  the settings of the associated configuration file: `config['sn_windows']`). Example usage:
  `
  sig_wdw, noise_wdw = segment.sn_windows()
  stream_noise = segment.stream().copy().trim(*noise_wdw, ...)
  stream_signal = segment.stream().copy().trim(*sig_wdw, ...)

* segment.siblings(parent=None): returns an iterable of siblings of this segment. `parent` can be
  any of the following: missing or None: returns all segments of the same recorded event, on the
  other channel components / orientations. 'stationname': returns all segments of the
  same station, identified by the tuple of the codes (newtwork, station). 'networkname':
  returns all segments of the same network (network code). 'datacenter',
  'event', 'station', 'channel': returns all segments of the same datacenter, event,
  station or channel, all identified by the associated id.
  NOTES: 1. The returned segment list is always a subset of the segments selected for processing
  (see configuration file). Thus, if there are N sibling segments in the
  database, this method returns 0 <= M <= N siblings.
  2. Use with care (or with multi-processing enabled) when providing a `parent` argument,
  as the amount of segments might be huge (up to hundreds of thousands of segments).
  The amount of returned segments is (almost exponentially) increasing according to the following
  order of the `parent` argument:
  'channel', 'station', 'stationname', 'networkname', 'event' or 'datacenter'

* segment.del_classes(*labels): Deletes the given classes of the segment. The argument is
  a comma-separated list of class labels (string). See configuration file for setting up the
  desired classes.
  E.g.: `segment.del_classes('class1')`, `segment.del_classes('class1', 'class2', 'class3')`

* segment.set_classes(*labels, annotator=None): Sets the given classes on the segment,
  deleting first all segment classes, if any. The argument is
  a comma-separated list of class labels (string). See configuration file for setting up the
  desired classes. `annotator` is a keyword argument (optional): if given (not None) denotes the
  user name that annotates the class.
  E.g.: `segment.set_classes('class1')`, `segment.set_classes('class1', 'class2', annotator='Jim')`

* segment.add_classes(*labels, annotator=None): Same as `segment.set_classes` but does not
  delete segment classes first. If a label is already assigned to the segment, it is not added again

* segment.seiscomp_path(root='.'): Returns a file system path where to store
  the given segment or any data associated to it in the seiscomp-compatible format:
     <root>/<net>/<sta>/<loc>/<cha>.D/<net>.<sta>.<loc>.<cha>.<year>.<day>.<event_id>
  The optional root argument, when missing, defaults to '.' (current working directory). Note
  that the paths ends intentionally with no extension because the user might be interested to save
  different types of segment's data (metadata, spectra, etcetera). In the typical case where the
  segment's stream has to be saved you can type:
      segment.stream().write(segment.seiscomp_path() + '.mseed', format='MSEED')

* segment.dbsession(): WARNING: this is for advanced users experienced with Sql-Alchemy library:
  returns the database session for custom IO operations with the database


segment attributes:
-------------------

''' + _SEGMENT_ATTRS

YAML_WARN = '''
NOTE: **this file is written in YAML syntax**, which uses Python-style indentation to
# indicate nesting, keep it in mind when editing. You can also use a more compact format that
# uses [] for lists and {} for maps/objects.
# For info see http://docs.ansible.com/ansible/latest/YAMLSyntax.html
'''

PROCESS_YAML_MAIN = '''
==========================================================================
# stream2segment config file to tune the processing/visualization subroutine
# ==========================================================================
#
# This editable template defines the configuration parameters which will
# be accessible in the associated processing / visualization python file.
#
# You are free to implement here anything you need: there are no mandatory parameters, although few
# ('segment_select', 'save_inventory' and 'sn_windows') are in most cases useful. Moreover,
# 'segment_select' and 'sn_windows', if present, add also special features to the GUI..
'''

# yamelise _SEGMENT_ATTRS (first line not commented, see below)
_SEGMENT_ATTRS_YAML = "\n# ".join(s[8:] for s in _SEGMENT_ATTRS.splitlines())


PROCESS_YAML_SEGMENTSELECT = '''
The parameter 'segment_select' defines what segments to be processed or
# visualized. If this argument is missing, all segments will be processed or
# (from within the GUI) visualized. The selection is made via the list-like argument:
#
# segment_select:
#   <att>: "<expression>"
#   <att>: "<expression>"
#   ...
#
# where each <att> is a segment attribute and <expression> is a simplified SQL-select string
# expression. Example:
#
# 1. To select and work on segments of stations activated in 2017 only:
# segment_select:
#   station.start_time: "[2017-01-01, 2018-01-01T00:00:00)"
# (brackets denote intervals. Square brackets include end-points, round brackets exclude endpoints)
#
# 2. To select segments from specified ids, e.g. 1, 4, 342, 67 (e.g., ids which raised errors during
# a previous run and whose id where logged might need inspection in the GUI):
# segment_select:
#   id: "1 4 342 67"
#
# 3. To select segments whose event magnitude is greater than 4.2:
# segment_select:
#   event.magnitude: ">4.2"
# (the same way work the operators: =, >=, <=, <, !=)
#
# 4. To select segments with a particular channel sensor description:
# segment_select:
#   channel.sensor_description: "'GURALP CMG-40T-30S'"
# (note: for attributes with str values and spaces, we need to quote twice, as otherwise
# "GURALP CMG-40T-30S" would match 'GURALP' and 'CMG-40T-30S', but not the whole string.
# See attribute types below)
#
# The list of segment attribute names and types is:
#
# ''' + _SEGMENT_ATTRS_YAML + '''
# '''


PROCESS_YAML_SAVEINVENTORY = '''
Boolean flag indicating whether station inventories (in xml format) should be saved to the database
# when they need to be downloaded during processing. Defaults to False when not given.
# From within the GUI, this parameter is not used
'''

PROCESS_YAML_SNWINDOWS = '''
Settings for computing the 'signal' and 'noise' time windows on a segment waveform.
# This parameter defines the signal and noise windows of each segment obtained from
# `segment.sn_windows()` (see associated python module help).
# If you do not need the functionality, you can safely remove this parameter.
# From within the GUI, signal and noise windows will be visualized as shaded areas
# on the plot of the currently selected segment. If this parameter is
# missing, the areas will not be shown.
#
# Arrival time shift: shifts the calculated arrival time of
# each segment by the specified amount of time (in seconds). Negative values are allowed.
# The arrival time is used to split a segment into segment's noise (before the arrival time)
# and segment's signal (after)
#
# Signal window: specifies the time window of the segment's signal, in seconds from the
# arrival time. If not numeric it must be a 2-element numeric array, denoting the
# start and end points, relative to the squares cumulative of the segment's signal.
# E.g.: [0.05, 0.95]
# sets the signal window from the time the cumulative reaches 5% of its maximum, until
# the time it reaches 95% of its maximum.
# The segment's noise window will be set equal to the signal window (i.e., same duration) and
# shifted in order to always end on the segment's arrival time
'''

PROCESS_YAML_CLASSLABELS = '''
If you want to use the GUI as hand labelling tool (for e.g. supervised classification problems)
# you can provide the parameter 'class_labels' which is a dictionary of label names mapped
# to their description. If provided, the labels will first be added to the database
# (updating the description, if the label name is already present) and then will show up in the GUI
# where one or more classes can be assigned to a given segment via check boxes.
# If missing, no class labels will show up in the GUI, unless already set by a previous config.
# Example:
#class_labels:
#  Discarded: "Segment which does not fall in any other cathegory (e.g., unknown artifacts)"
#  Unknown: "Segment which is either: unlabeled (not annotated) or unclassified"
#  Ok: "Segment with no artifact"
#  LowS2N: "Segment has a low signal-to-noise ratio"
#  Aftershock: "Segment with non overlapping multi-events recorded (aftershock)"
#  MultiEvent: "Segment with overlapping multi-events recorded (no aftershock)"
#  BadCoda: "Segment with a bad coda (bad decay)"
'''

PROCESS_YAML_ADVANCEDSETTINGS = '''
If you want to setup advanced settings, uncomment
# (i.e., remove the leading '#' from each line) and edit the text block below
#advanced_settings:
#  # Although each segment is processed separately, loading from the database one segment at a time
#  # is time consuming. The number below defines how many segments will be loaded in a chunk.
#  # When missing, the value defaults to 1200. If multi_process is true (see below), this
#  # number defines how many segments will be loaded in each python sub-process. Increasing
#  # this number might speed up the execution but consumes more memory
#  segments_chunksize: 1200
#  # Use parallel sub-processes to speed up the execution. When missing, it defaults to false
#  multi_process: true
#  # The number of sub-processes. If missing, it is set as the the number of CPUs in the system.
#  # This option is ignored if multi_process is not given
#  num_processes: 4
'''

# setting up DOCVARS:
DOCVARS = {k: v.strip() for k, v in globals().items()
           if hasattr(v, 'strip') and not k.startswith('_')}
