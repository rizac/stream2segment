'''This module holds doc strings to be injected via jinja2 into the templates when running
`s2s init`.
Any NON PRIVATE variable name (i.e., without leading underscore '_') of this module
can be injected in a template file in the usual way, e.g.:
{{ PROCESS_PY_BANDPASSFUNC }}
For any new variable name to be implemented here in the future, note also that:
1. the variables
values are stripped before being assigned to the gobal DOCVARS (the dict passed to to jinja).
2. By convention, *_PY_* variable names are for Python docs, *_YAML_* variable names for yaml docs.
2b. In the latter case, yaml variables values do not need a leading '# ' on the first line,
as it is usually input in the template file, e.g.:
# {{ PROCESS_YAML_MAIN }}

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from stream2segment.download.utils import EVENTWS_MAPPING
from stream2segment.process.writers import SEGMENT_ID_COLNAME, HDF_DEFAULT_CHUNKSIZE


# REMEMBER:this list does not comprise ALL attributes  (look at _SEGMENT_ATTRS_REMOVED below)
_SEGMENT_ATTRS = '''
===================================== ==============================================================
Segment attribute                     Python type and (optional) description
===================================== ==============================================================
segment.id                            int: segment (unique) db id
segment.event_distance_deg            float: distance between the segment's station and
                                      the event, in degrees
segment.event_distance_km             float: distance between the segment's station and
                                      the event, in km, assuming a perfectly spherical earth
                                      with a radius of 6371 km
segment.start_time                    datetime.datetime: the waveform data start time
segment.arrival_time                  datetime.datetime: the station's arrival time of the waveform.
                                      Value between 'start_time' and 'end_time'
segment.end_time                      datetime.datetime: the waveform data end time
segment.request_start                 datetime.datetime: the requested start time of the data
segment.request_end                   datetime.datetime: the requested end time of the data
segment.duration_sec                  float: the waveform data duration, in seconds
segment.missing_data_sec              float: the number of seconds of missing data, with respect
                                      to the requested time window. It might also be negative
                                      (more data received than requested). This parameter is useful
                                      when selecting segments: e.g., if we requested 5
                                      minutes of data and we want to process segments with at
                                      least 4 minutes of downloaded data, then:
                                      missing_data_sec: '< 60'
segment.missing_data_ratio            float: the portion of missing data, with respect
                                      to the request time window. It might also be negative
                                      (more data received than requested). This parameter is useful
                                      when selecting segments: e.g., if you want to process
                                      segments whose real time window is at least 90% of the
                                      requested one, then: missing_data_ratio: '< 0.1'
segment.sample_rate                   float: the waveform data sample rate.
                                      It might differ from the segment channel's sample_rate
segment.has_data                      boolean: tells if the segment has data saved (at least
                                      one byte of data). This parameter is useful when selecting
                                      segments (in most cases, almost necessary), e.g.:
                                      has_data: 'true'
segment.download_code                 int: the code reporting the segment download status. This
                                      parameter is useful to further refine the segment selection
                                      skipping beforehand segments with malformed data (code -2):
                                      has_data: 'true'
                                      download_code: '!=-2'
                                      (All other codes are generally of no interest for the user.
                                      However, for details see Table 2 in
                                      https://doi.org/10.1785/0220180314#tb2)
segment.maxgap_numsamples             float: the maximum gap or overlap found in the waveform data,
                                      in number of points. If 0, the segment has no gaps/overlaps.
                                      Otherwise, if >=1: the segment has gaps, if <=-1: the segment
                                      has overlaps. Values in (-1, 1) are difficult to interpret: a
                                      rule of thumb is to consider half a point a gap / overlap
                                      (maxgap_numsamples > 0.5 or maxgap_numsamples < -0.5).
                                      This parameter is useful when selecting segments: e.g.,
                                      to select segments with no gaps/overlaps, then:
                                      maxgap_numsamples: '(-0.5, 0.5)'
segment.seed_id                       str: the seed identifier in the typical format
                                      [Network].[Station].[Location].[Channel]. For segments
                                      with waveform data, `data_seed_id` (see below) might be
                                      faster to fetch.
segment.data_seed_id                  str: same as 'segment.seed_id', but faster to get because it
                                      reads the value stored in the waveform data. The drawback
                                      is that this value is null for segments with no waveform data
segment.has_class                     boolean: tells if the segment has (at least one) class
                                      assigned
segment.data                          bytes: the waveform (raw) data. Used by `segment.stream()`
------------------------------------- ------------------------------------------------
segment.event                         object (attributes below)
segment.event.id                      int
segment.event.event_id                str: the id returned by the web service or catalog
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
segment.station.network               str: the station's network code, e.g. 'AZ'
segment.station.station               str: the station code, e.g. 'NHZR'
segment.station.netsta_code           str: the network + station code, concatenated with
                                      the dot, e.g.: 'AZ.NHZR'
segment.station.latitude              float
segment.station.longitude             float
segment.station.elevation             float
segment.station.site_name             str
segment.station.start_time            datetime.datetime
segment.station.end_time              datetime.datetime
segment.station.has_inventory         boolean: tells if the segment's station inventory has
                                      data saved (at least one byte of data).
                                      This parameter is useful when selecting segments: e.g.,
                                      to select only segments with inventory downloaded:
                                      station.has_inventory: 'true'
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
------------------------------------- ------------------------------------------------
segment.classes.id                    int: the id(s) of the classes assigned to the segment
segment.classes.label                 int: the label(s) of the classes assigned to the segment
segment.classes.description           int: the description(s) of the classes assigned to the
                                      segment
===================================== ================================================
'''.strip()

# the variable below IS NOT USED ANYWHERE, it just collects the attributes removed from
# _SEGMENT_ATTRS. Add them back in _SEGMENT_ATTRS (in the right place) at your choice:
_SEGMENT_ATTRS_REMOVED = '''
segment.station.inventory_xml         bytes. The station inventory (raw) data. You don't
                                      generally need to access this attribute which is also
                                      time-consuming to fetch. Used by `segment.inventory()`
segment.download.log                  str: The log text of the segment's download execution.
                                      You don't generally need to access this
                                      attribute which is also time-consuming to fetch.
                                      Useful for advanced debugging / inspection
segment.download.warnings             int
segment.download.errors               int
segment.download.config               str
segment.download.program_version      str
'''


PROCESS_PY_BANDPASSFUNC = """
Applies a pre-process on the given segment waveform by
filtering the signal and removing the instrumental response.
Modifies the segment stream in-place (see below).

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
  old unprocessed stream

:return: a Trace object.
"""


PROCESS_PY_MAINFUNC = '''
Main processing function. The user should implement here the processing for any given
selected segment. Useful links for functions, libraries and utilities:

- `stream2segment.process.math.traces` (small processing library implemented in this program,
  most of its functions are imported here by default)
- `obpsy <https://docs.obspy.org/packages/index.html>`_
- `obspy Stream object <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html>_`
- `obspy Trace object <https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html>_`

IMPORTANT: Any exception raised by this routine will be logged to file for inspection.
    All exceptions will interrupt the whole exectution, only exceptions of type `ValueError`
    will interrupt the execution of the currently processed segment and continue to the
    next segment, as ValueErrors might not always denote critical code errors. This feature can
    also be triggered programmatically to skip the currently processed segment
    and log the message for later inspection, e.g.:
    ```
    if snr < 0.4:
        raise ValueError('SNR ratio too low')
    ```

:param: segment (ptyhon object): An object representing a waveform data to be processed,
    reflecting the relative database table row. See above for a detailed list
    of attributes and methods

:param: config (Python dict): a dictionary reflecting what has been implemented in the configuration
    file. You can write there whatever you want (in yaml format, e.g. "propertyname: 6.7" ) and it
    will be accessible as usual via `config['propertyname']`

:return: If the processing routine calling this function needs not to generate a file output,
    the returned value of this function, if given, will be ignored.
    Otherwise:

    * For CSV output, this function must return an iterable that will be written as a row of the
      resulting file (e.g. list, tuple, numpy array, dict. You must always return the same type
      of object, e.g. not lists or dicts conditionally).

      Returning None or nothing is also valid: in this case the segment will be silently skipped

      The CSV file will have a row header only if `dict`s are returned (the dict keys will be the
      CSV header columns). For Python version < 3.6, if you want to preserve in the CSV the order
      of the dict keys as the were inserted, use `OrderedDict`.

      A column with the segment database id (an integer uniquely identifying the segment)
      will be automatically inserted as first element of the iterable, before writing it to file.

      SUPPORTED TYPES as elements of the returned iterable: any Python object, but we
      suggest to use only strings or numbers: any other object will be converted to string
      via `str(object)`: if this is not what you want, convert it to the numeric or string
      representation of your choice. E.g., for Python `datetime`s you might want to set
      `datetime.isoformat()` (string), for obspy's `UTCDateTime`s `float(utcdatetime)` (numeric)

   * For HDF output, this function must return a dict, pandas Series or pandas DataFrame
     that will be written as a row of the resulting file (or rows, in case of DataFrame).

     Returning None or nothing is also valid: in this case the segment will be silently skipped.

     A column named '{0}' with the segment database id (an integer uniquely identifying the segment)
     will be automatically added to the dict / Series, or to each row of the DataFrame,
     before writing it to file.

     SUPPORTED TYPES as elements of the returned dict/Series/DataFrame: all types supported
     by pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes

     For info on hdf and the pandas library (included in the package), see:
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
     https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-hdf5

'''.format(SEGMENT_ID_COLNAME)

PROCESS_PY_MAIN = '''
============================================================================================
stream2segment Python file to implement the processing/visualization subroutines: User guide
============================================================================================

This module needs to implement one or more functions which will be described in the sections below.
**All these functions must have the same signature**:
```
    def myfunction(segment, config):
```
where `segment` is the Python object representing a waveform data segment to be processed
and `config` is the Python dictionary representing the given configuration file.

After editing, this file can be invoked from the command line commands `s2s process` and `s2s show`
with the `-p` / `--pyfile` option (type `s2s process --help` or `s2s show --help` for details).
In the first case, see section 'Processing' below, otherwise see section 'Visualization (web GUI)'.


Processing
==========

When processing, the program will search for a function called "main", e.g.:
```
def main(segment, config)
```
the program will iterate over each selected segment (according to 'segment_select' parameter
in the config) and execute the function, writing its output to the given file, if given.
If you do not need to use this module for visualizing stuff, skip the section 'Visualization'
below and go to the next one.


Visualization (web GUI)
=======================

When visualizing, the program will fetch all segments (according
to 'segment_select' parameter in the config), and open a web page where the user can browse and
visualize each segment one at a time.
The page shows by default on the upper left corner a plot representing the segment trace(s).
The GUI can be customized by providing here functions decorated with
"@gui.preprocess" or "@gui.plot".
Functions decorated this way (Plot functions) can return only special 'plottable' values
(see 'Plot functions' below for details).

Pre-process function
--------------------

The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plots of the page will be re-calculated with the output of this function,
which **must thus return an obspy Stream or Trace object**.

Plot functions
--------------

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
The 'position' argument controls where the plot will be placed in the GUI ('b' means bottom,
the default, 'r' means next to the main plot, on its right) and the other two, `xaxis` and
`yaxis`, are dict (defaulting to the empty dict {}) controlling the x and y axis of the plot
(for info, see: https://plot.ly/python/axes/). When not given, axis types will be inferred
from the function's return type (see below) and in most cases defaults to 'date' (i.e.,
date-times on the x values).

Functions decorated with '@gui.plot' must return a numeric sequence y taken at successive
equally spaced points in any of these forms:

- a obspy Trace object

- a obspy Stream object

- the tuple (x0, dx, y) or (x0, dx, y, label), where

    - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point.
      For time-series abscissas, UTCDateTime is quite flexible with several input formats.
      For info see: https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html

    - dx (numeric or `timedelta`) is the sampling period. If x0 has been given as date-time
      or UTCDateTime object and 'dx' is numeric, its unit is in seconds
      (e.g. 45.67 = 45 seconds and 670000 microseconds). If `dx` is a timedelta object and
      x0 has been given as numeric, then x0 will be converted to UtcDateTime(x0).

    - y (numpy array or numeric list) are the sequence values, numeric

    - label (string, optional) is the sequence name to be displayed on the plot legend.

- a dict of any of the above types, where the keys (string) will denote each sequence
  name to be displayed on the plot legend (and will override the 'label' argument, if provided)

Functions implementation
========================

The implementation of the functions is user-dependent. As said, all functions needed for
processing and visualization must have the same signature:
```
    def myfunction(segment, config):
```

any Exception raised will be handled this way:

* if the function is called for visualization, the exception will be caught and its message
  displayed on the plot

* if the function is called for processing, the exception will raise as usual, interrupting
  the routine, with one special case: `ValueError`s will interrupt the currently processed segment
  only (the exception message will be logged) and continue the execution to the next segment.
  This feature can also be triggered programmatically to skip the currently processed segment and
  log the error for later insopection, e.g.:
    `raise ValueError("segment sample rate too low")`
  (thus, do not issue `print` statements for debugging as it's useless, and a bad practice overall)

Conventions and suggestions:

This module is designed to encourage the decoupling of code and configuration, so that you can
easily and safely experiment different configurations on the same code of the same Python module,
instead of having duplicated modules with different hard coded parameters.

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
consider and treat it as a normal Python object. It features special methods and
several attributes returning Python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

segment methods:
----------------

* segment.stream(): the `obspy.Stream` object representing the waveform data
  associated to the segment. Please remember that many obspy functions modify the
  stream in-place:
  ```
      stream_remresp = segment.stream().remove_response(segment.inventory())
      segment.stream() is stream_remresp  # == True: segment.stream() is modified permanently!
  ```
  When visualizing plots, where efficiency is less important, each function is executed on a
  copy of segment.stream(). However, from within the `main` function, the user has to handle when
  to copy the segment's stream or not, e.g.:
  ```
      stream_remresp = segment.stream().copy().remove_response(segment.inventory())
      segment.stream() is stream_remresp  # False: segment.stream() is still the original object
  ```
  For info see https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.copy.html

* segment.inventory(): the `obspy.core.inventory.inventory.Inventory`. This object is useful e.g.,
  for removing the instrumental response from `segment.stream()`: note that it will be available
  only if the inventories in xml format were downloaded in the downloaded subroutine

* segment.sn_windows(length, shift=0): returns the signal and noise time windows:
  (s_start, s_end), (n_start, n_end) where all elements are obspy `UTCDateTime`s.
  The windows are calculated as follows: the segment arrival time A is retrieved and shifted
  by the given seconds (`shift` argument, float, defaults to 0).
  Then, the length of the signal window is computed:
  - if `length` is numeric, it is the window length in seconds: s_start = A, s_end = A + length
  - If `length` is a list of two numbers (VAL0, VAL1) both in [0, 1], then the waveform's cumulative
    sum of squares (CUMSS) is calculated from A and normalized in [0, 1]: s_start is the time
    where CUMSS reaches VAL0, s_end is the time where CUMSS reaches VAL1.
  Once the signal window has been calculated, the noise window will have the same length L
  and moved 'backwards' in order to end on A: n_start = A - L, n_end = A
  In the YAML templates, you will see the parameter 'sn_windows': you can
  tune it to configure your signal and noise window calculation. Example:
  ```
  snw = config['sn_windows']
  sig_wdw, noise_wdw = segment.sn_windows(snw['signal_window'], snw['arrival_time_shift'])
  stream_noise = segment.stream().copy().trim(*noise_wdw, ...)
  stream_signal = segment.stream().copy().trim(*sig_wdw, ...)
  ```

* segment.siblings(parent=None, condition): returns an iterable of siblings of this segment.
  `parent` can be any of the following:
  - missing or None: returns all segments of the same recorded event, on the
    other channel components / orientations
  - 'stationname': returns all segments of the same station, identified by the tuple of the
    codes (newtwork, station)
  - 'networkname': returns all segments of the same network (network code)
  - 'datacenter', 'event', 'station', 'channel': returns all segments of the same datacenter, event,
    station or channel, all identified by the associated database id.
  `condition` is a dict of expression to filter the returned element. the argument
  `config['segment_select]` can be passed here to return only siblings selected for processing.
  NOTE: Use with care when providing a `parent` argument, as the amount of segments might be huge
  (up to hundreds of thousands of segments). The amount of returned segments is increasing
  (non linearly) according to the following order of the `parent` argument:
  'channel', 'station', 'stationname', 'networkname', 'event' and 'datacenter'

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
  delete segment classes first. If a label is already assigned to the segment, it is not added
  again (regardless of whether the 'annotator' changed or not)

* segment.sds_path(root='.'): Returns the segment's file path in a seiscomp data
  structure (SDS) format:
     <root>/<event_id>/<net>/<sta>/<loc>/<cha>.D/<net>.<sta>.<loc>.<cha>.<year>.<day>
  See https://www.seiscomp3.org/doc/applications/slarchive/SDS.html for details.
  Example: to save the segment's waveform as miniSEED you can type (explicitly
  adding the file extension '.mseed' to the output path):
  ```
      segment.stream().write(segment.sds_path() + '.mseed', format='MSEED')
  ```

* segment.dbsession(): returns the database session for custom IO operations with the database.
  WARNING: this is for advanced users experienced with SQLAlchemy library. If you want to
  use it you probably want to import stream2segment in custom code. See the github documentation
  in case

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
# be accessible in the associated processing / visualization Python file.
#
# You are free to implement here anything you need: there are no mandatory parameters but we
# strongly suggest to keep 'segment_select' and 'sn_windows', which add also special features
# to the GUI.
'''

# yamelise _SEGMENT_ATTRS (first line not commented, see below)
_SEGMENT_ATTRS_YAML = "\n# ".join(s[8:] for s in _SEGMENT_ATTRS.splitlines())


PROCESS_YAML_SEGMENTSELECT = '''
The parameter 'segment_select' defines which segments to be processed or visualized. PLEASE USE
# THIS PARAMETER. If missing, all segments will be loaded, including segment with no
# (or malformed) waveform data: this is in practically always useless and slows down considerably
# the processing or visualization routine. The selection is made via the list-like argument:
#
# segment_select:
#   <att>: "<expression>"
#   <att>: "<expression>"
#   ...
#
# where each <att> is a segment attribute and <expression> is a simplified SQL-select string
# expression. Example:
#
# 1. To select and work on segments with downloaded data (at least one byte of data):
# segment_select:
#   has_data: "true"
#
# 2. To select and work on segments of stations activated in 2017 only:
# segment_select:
#   station.start_time: "[2017-01-01, 2018-01-01T00:00:00)"
# (brackets denote intervals. Square brackets include end-points, round brackets exclude endpoints)
#
# 3. To select segments from specified ids, e.g. 1, 4, 342, 67 (e.g., ids which raised errors during
# a previous run and whose id where logged might need inspection in the GUI):
# segment_select:
#   id: "1 4 342 67"
#
# 4. To select segments whose event magnitude is greater than 4.2:
# segment_select:
#   event.magnitude: ">4.2"
# (the same way work the operators: =, >=, <=, <, !=)
#
# 5. To select segments with a particular channel sensor description:
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

PROCESS_YAML_SNWINDOWS = '''
Settings for computing the 'signal' and 'noise' time windows on a segment waveform.
# From within the GUI, signal and noise windows will be visualized as shaded areas on the plot
# of the currently selected segment. If this parameter is missing, the areas will not be shown.
# This parameter can also be used to define the arguments of `segment.sn_windows()` (see associated
# Python module help).
#
# Arrival time shift: shifts the calculated arrival time of
# each segment by the specified amount of time (in seconds). Negative values are allowed.
# The arrival time is used to split a segment into segment's noise (before the arrival time)
# and segment's signal (after)
#
# Signal window: specifies the time window of the segment's signal, in seconds from the
# arrival time. If not numeric it must be a 2-element numeric array, denoting the
# start and end points, relative to the squares cumulative of the segment's signal portion.
# E.g.: [0.05, 0.95] sets the signal window from the time the cumulative reaches 5% of its
# maximum, until the time it reaches 95% of its maximum.
# The segment's noise window will be set equal to the signal window (i.e., same duration) and
# shifted in order to always end on the segment's arrival time
'''

PROCESS_YAML_CLASSLABELS = '''
If you want to use the GUI as hand labelling tool (for e.g. supervised classification problems)
# or setup classes before processing, you can provide the parameter 'class_labels' which is a
# dictionary of label names mapped to their description. If provided, the labels will first be
# added to the database (updating the description, if the label name is already present) and
# then will show up in the GUI where one or more classes can be assigned to a given segment via
# check boxes. If missing, no class labels will show up in the GUI, unless already set by a
# previous config. Example:
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
Advanced settings tuning the process routine:
advanced_settings:
  # Use parallel sub-processes to speed up the execution.
  multi_process: false
  # The number of sub-processes. If null, it is set as the the number of CPUs in the
  # system. This option is ignored if multi_process is false
  num_processes: null
  # Although each segment is processed one at a time, loading segments in chunks from the
  # database is faster: the number below defines the chunk size. If multi_process is true,
  # the chunk size also defines how many segments will be loaded in each Python sub-process.
  # Increasing this number might speed up execution but increases the memory usage.
  # When null, the chunk size defaults to 1200 if the number N of
  # segments to be processed is > 1200, otherwise N/10.
  segments_chunksize: null
  # Optional arguments for the output writer. Ignored for CSV output, for HDF output see:
  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.append.html
  # (the parameters 'append' and 'value' will be ignored, if given here)
  writer_options:
    chunksize: {0:d}
    # hdf needs a fixed length for all columns: for variable-length string columns,
    # you need to tell in advance how many bytes to allocate with 'min_itemsize'.
    # E.g., if you have two string columns 'col1' and 'col2' and you assume to store
    # at most 10 ASCII characters in 'col1' and 20 in 'col2', then:
    # min_itemsize:
    #   col1: 10
    #   col2: 20
'''.format(HDF_DEFAULT_CHUNKSIZE)

DOWNLOAD_EVENTWS_LIST = '\n'.join('%s"%s": %s' % ('# ' if i > 0 else '', str(k), str(v))
                                  for i, (k, v) in enumerate(EVENTWS_MAPPING.items()))

# setting up DOCVARS:
DOCVARS = {k: v.strip() for k, v in globals().items()
           if hasattr(v, 'strip') and not k.startswith('_')}
