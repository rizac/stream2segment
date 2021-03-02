"""This module holds doc strings to be injected via jinja2 into the templates when running
`s2s init`.
Any NON PRIVATE variable name (i.e., without leading underscore '_') of this module
can be injected in a template file in the usual way, e.g.:
{{ PROCESS_PY_BANDPASSFUNC }}
For any new variable name to be implemented here in the future, note also that:
1. the variables values are stripped before being assigned to the global DOCVARS (the
   dict passed to to Jinja).
2. By convention, *_PY_* variable names are for Python docs, *_YAML_* variable names for
   YAML docs. In the latter case, YAML variables values do not need a leading '# ' on the
   first line, as it is usually input in the template file, e.g.:
   # {{ PROCESS_YAML_MAIN }}

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from stream2segment.download.utils import EVENTWS_MAPPING
from stream2segment.process.main import _get_chunksize_defaults
from stream2segment.process.writers import SEGMENT_ID_COLNAME, HDF_DEFAULT_CHUNKSIZE


# REMEMBER:this list does not comprise ALL attributes  (look at _SEGMENT_ATTRS_REMOVED
# below)


_SEGMENT_ATTRS = """
===================================== ===================================================
Segment attribute                     Python type and (optional) description
===================================== ===================================================
segment.id                            int: segment (unique) db id
segment.has_data                      boolean: if the segment has waveform data saved (at
                                      least one byte of data). Useful (often *mandatory*)
                                      in segment selection: e.g., to skip processing
                                      segments with no data, then:
                                      has_data: 'true'
segment.event_distance_deg            float: distance between the segment station and
                                      the event, in degrees
segment.event_distance_km             float: distance between the segment station and the
                                      event, in km, assuming a perfectly spherical earth
                                      with a radius of 6371 km
segment.start_time                    datetime.datetime: waveform start time
segment.arrival_time                  datetime.datetime: waveform arrival time (value
                                      between 'start_time' and 'end_time')
segment.end_time                      datetime.datetime: waveform end time
segment.request_start                 datetime.datetime: waveform requested start time
segment.request_end                   datetime.datetime: waveform requested end time
segment.duration_sec                  float: waveform data duration, in seconds
segment.missing_data_sec              float: number of seconds of missing data, as ratio
                                      of the requested time window. It might also be 
                                      negative (more data received than requested). 
                                      Useful in segment selection: e.g., if we requested 
                                      5 minutes of data and we want to process segments 
                                      with at least 4 minutes of  downloaded data, then:
                                      missing_data_sec: '< 60'
segment.missing_data_ratio            float: portion of missing data, as ratio of the 
                                      requested time window. It might also be negative
                                      (more data received than requested). Useful in 
                                      segment selection: e.g., to process segments whose
                                      time window is at least 90% of the requested one: 
                                      missing_data_ratio: '< 0.1'
segment.sample_rate                   float: waveform sample rate. It might differ from 
                                      the segment channel sample_rate
segment.download_code                 int: the segment download status. For advanced
                                      users. Useful in segment selection. E.g., to
                                      process segments with non malformed waveform data:
                                      has_data: 'true'
                                      download_code: '!=-2'
                                      (for details on all download codes, see Table 2 in 
                                      https://doi.org/10.1785/0220180314)
segment.maxgap_numsamples             float: maximum gap/overlap (G/O) found in the 
                                      waveform, in number of points. If
                                         0: segment has no G/O
                                       >=1: segment has Gaps
                                      <=-1: segment has Overlaps. 
                                      Values in (-1, 1) are difficult to interpret: a 
                                      rule of thumb is to consider no G/O if values are 
                                      within -0.5 and 0.5. Useful in segment selection: 
                                      e.g., to process segments with no gaps/overlaps:
                                      maxgap_numsamples: '(-0.5, 0.5)'
segment.seed_id                       str: the seed identifier in the typical format
                                      [Network].[Station].[Location].[Channel]. For 
                                      segments with waveform data, `data_seed_id` (see 
                                      below) might be faster to fetch.
segment.data_seed_id                  str: same as 'segment.seed_id', but faster to get 
                                      because it reads the value stored in the waveform 
                                      data. The drawback is that this value is null for 
                                      segments with no waveform data
segment.has_class                     boolean: tells if the segment has (at least one)
                                      class label assigned
segment.data                          bytes: the waveform (raw) data. Used by
                                      `segment.stream()`
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
segment.station.has_inventory         boolean: tells if the segment's station inventory 
                                      has data saved (at least one byte of data).
                                      Useful in segment selection. E.g., to process only 
                                      segments with inventory downloaded:
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
segment.classes.id                    int: the id(s) of the class labels assigned to the 
                                      segment
segment.classes.label                 int: the unique name(s) of the class labels 
                                      assigned to the segment
segment.classes.description           int: the description(s) of the class labels 
                                      assigned to the segment
===================================== ================================================
""".strip()

# the variable below IS NOT USED ANYWHERE, it just collects the attributes removed from
# _SEGMENT_ATTRS. Add them back in _SEGMENT_ATTRS (in the right place) at your choice:
_SEGMENT_ATTRS_REMOVED = """
segment.station.inventory_xml         bytes. The station inventory (raw) data. You don't
                                      generally need to access this attribute which is 
                                      also time-consuming to fetch. Used by 
                                      `segment.inventory()`
segment.download.log                  str: The log text of the segment's download 
                                      execution. You don't generally need to access this
                                      attribute which is also time-consuming to fetch.
                                      Useful for advanced debugging / inspection
segment.download.warnings             int
segment.download.errors               int
segment.download.config               str
segment.download.program_version      str
"""


PROCESS_PY_BANDPASSFUNC = """
Apply a pre-process on the given segment waveform by filtering the signal and
removing the instrumental response. 

This function is used for processing (see `main` function) and visualization
(see the `@gui.preprocess` decorator and its documentation above)

The steps performed are:
1. Sets the max frequency to 0.9 of the Nyquist frequency (sampling rate /2)
   (slightly less than Nyquist seems to avoid artifacts)
2. Offset removal (subtract the mean from the signal)
3. Tapering
4. Pad data with zeros at the END in order to accommodate the filter transient
5. Apply bandpass filter, where the lower frequency is magnitude dependent
6. Remove padded elements
7. Remove the instrumental response

IMPORTANT: This function modifies the segment stream in-place: further calls to 
`segment.stream()` will return the pre-processed stream. During visualization, this
is not an issue because Stream2segment always caches a copy of the raw trace.
During processing (see `main` function) you need to be more careful: if needed, you
can store the raw stream beforehand (`raw_trace=segment.stream().copy()`) or reload
the segment stream afterwards with `segment.stream(reload=True)`.

:return: a Trace object (a Stream is also valid value for functions decorated with
    `@gui.preprocess`)
"""


PROCESS_PY_MAINFUNC = """
Main processing function. The user should implement here the processing for any
given selected segment. Useful links for functions, libraries and utilities:

- `stream2segment.process.math.traces` (small processing library implemented in
   this program, most of its functions are imported here by default)
- `ObpPy <https://docs.obspy.org/packages/index.html>`_
- `ObsPy Stream <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html>_`
- `ObsPy Trace <https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.html>_`

IMPORTANT: any exception raised here or from any sub-function will interrupt the
whole processing routine with one special case: `stream2segment.process.SkipSegment`
exceptions will be logged to file and the execution will resume from the next 
segment. Raise them to programmatically skip a segment, e.g.:
```
if segment.sample_rate < 60: 
    raise SkipSegment("segment sample rate too low")`
```

Handling exceptions at any point of the processing is non trivial: some have to
be skipped to save precious time, some must not be ignored and should interrupt 
the routine to fix bugs preventing output mistakes.
Therefore, we recommend to try to run your code on a smaller and possibly 
heterogeneous dataset first: change temporarily the segment selection in the
configuration file, and then analyze any exception raised, if you want to ignore 
the exception (e.g., it's not due to a bug in your code), then you can wrap only 
the part of code affected in a “try ... catch" statement, and raise a `SkipSegment`.
Also, please spend some time on the configuration file segment selection: you might
find that your code runs smoothly and faster by simply skipping certain segments in 
the first place.

:param: segment (Python object): An object representing a waveform data to be
    processed, reflecting the relative database table row. See above for a detailed
    list of attributes and methods

:param: config (Python dict): a dictionary reflecting what has been implemented in
    the configuration file. You can write there whatever you want (in yaml format,
    e.g. "propertyname: 6.7" ) and it will be accessible as usual via
    `config['propertyname']`

:return: If the processing routine calling this function needs not to generate a
    file output, the returned value of this function, if given, will be ignored.
    Otherwise:

    * For CSV output, this function must return an iterable that will be written
      as a row of the resulting file (e.g. list, tuple, numpy array, dict. You must
      always return the same type of object, e.g. not lists or dicts conditionally).

      Returning None or nothing is also valid: in this case the segment will be
      silently skipped

      The CSV file will have a row header only if `dict`s are returned (the dict
      keys will be the CSV header columns). For Python version < 3.6, if you want
      to preserve in the CSV the order of the dict keys as the were inserted, use
      `OrderedDict`.

      A column with the segment database id (an integer uniquely identifying the
      segment) will be automatically inserted as first element of the iterable, 
      before writing it to file.

      SUPPORTED TYPES as elements of the returned iterable: any Python object, but
      we suggest to use only strings or numbers: any other object will be converted
      to string via `str(object)`: if this is not what you want, convert it to the
      numeric or string representation of your choice. E.g., for Python `datetime`s
      you might want to set `datetime.isoformat()` (string), for ObsPy `UTCDateTime`s
      `float(utcdatetime)` (numeric)

   * For HDF output, this function must return a dict, pandas Series or pandas
     DataFrame that will be written as a row of the resulting file (or rows, in case
     of DataFrame).

     Returning None or nothing is also valid: in this case the segment will be
     silently skipped.

     A column named '{0}' with the segment database id (an integer uniquely
     identifying the segment) will be automatically added to the dict / Series, or
     to each row of the DataFrame, before writing it to file.

     SUPPORTED TYPES as elements of the returned dict/Series/DataFrame: all types 
     supported by pandas: 
     https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes

     For info on hdf and the pandas library (included in the package), see:
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
     https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-hdf5

""".format(SEGMENT_ID_COLNAME)

PROCESS_PY_MAIN = """
==========================================================
Stream2segment processing+visualization module: User guide
==========================================================

The module implements the necessary code to process and visualize downloaded data
in a web Graphical User Interface (GUI). Edit this file and pass its path 
`<module_path>` to the following commands from the terminal:

`s2s process -p <module_path> -c <config_path>`  (data processing)

`s2s show -p <module_path> -c <config_path>`     (data visualization / web GUI)

You can always type `s2s process --help` or `s2s show --help` for details
(`<config_path>` is the path of the associated a configuration file in YAML 
format). You can also separate visualization and process routines in two different
Python modules, as long as in each single file the requirements described below 
are provided.


Processing
==========

When processing, Stream2segment will search for a function called "main", e.g.:
```
def main(segment, config)
```
and execute the function on each selected segment (according to 'segment_selection'
parameter in the config). See the function docstring of this module for implementation
details.


Visualization (web GUI)
=======================

When visualizing, Stream2segment will open a web page where the user can browse 
and visualize the data. Contrarily to the processing, the `show` command can be
invoked with no argument: this will show by default all database segments, 
their metadata and a plot of their raw waveform (main plot).
When `show` is invoked with module and config files, Stream2segment will fetch
all segments (according to 'segment_selection' parameter in the config) and search
for all module functions with signature:
```
def function_name(segment, config)
```
and decorated with either "@gui.preprocess" or "@gui.plot". In the former case,
the function will be recognized as pre-process function, in all other cases, the
functions will be recognized as plot functions. Note that any Exception raised 
anywhere by any function will be caught and its message displayed on the plot.

Pre-process function
--------------------

The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plots of the page will be re-calculated with the output of this function,
which **must thus return an ObsPy Stream or Trace object**.

Plot functions
--------------

The functions decorated with "@gui.plot", e.g.:
```
@gui.plot
def cumulative(segment, config)
...
```
will be associated to (i.e., its output will be displayed in) the plot below 
the main plot. You can also call @gui.plot with arguments, e.g.:
```
@gui.plot(position='r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def spectra(segment, config)
...
```
The 'position' argument controls where the plot will be placed in the GUI ('b' means 
bottom, the default, 'r' means next to the main plot, on its right) and the other two,
`xaxis` and `yaxis`, are dict (defaulting to the empty dict {}) controlling the x and y 
axis of the plot (for info, see: https://plot.ly/python/axes/).

When not given, axis types (e.g., date time vs numeric) will be inferred from the
function's returned value which *must* be a numeric sequence (y values) taken at 
successive equally spaced points (x values) in any of these forms:

- ObsPy Trace object

- ObsPy Stream object

- the tuple (x0, dx, y) or (x0, dx, y, label), where

    - x0 (numeric, `datetime` or `UTCDateTime`) is the abscissa of the first point.
      For time-series abscissas, UTCDateTime is quite flexible with several input
      formats. For info see:
      https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html

    - dx (numeric or `timedelta`) is the sampling period. If x0 has been given as
      datetime or UTCDateTime object and 'dx' is numeric, its unit is in seconds
      (e.g. 45.67 = 45 seconds and 670000 microseconds). If `dx` is a timedelta object
      and x0 has been given as numeric, then x0 will be converted to UtcDateTime(x0).

    - y (numpy array or numeric list) are the sequence values, numeric

    - label (string, optional) is the sequence name to be displayed on the plot legend.

- a dict of any of the above types, where the keys (string) will denote each sequence
  name to be displayed on the plot legend (and will override the 'label' argument, if
  provided)


Functions arguments
===================

As described above, all functions needed for processing and visualization must have the
same signature, i.e. they accepts the same arguments `(segment, config)` (in this order).

config (dict)
-------------

This is the dictionary representing the chosen configuration file, in YAML format.
Any property defined in the file, e.g.:
```
outfile: '/home/mydir/root'
mythreshold: 5.67
```
will be accessible via `config['outfile']`, `config['mythreshold']`.

The purpose of the `config` is to encourage decoupling of code and configuration for
better and more maintainable code: try to avoid many similar Python modules differing 
by few hard-coded parameters. Try instead to implement a single Python module
with the program functionality, and put those parameters in different config YAML
files to run the same module in different scenarios.

segment (object)
----------------

Technically it's like an 'SqlAlchemy` ORM instance but for the user it is enough to
consider and treat it as a normal Python object. It features special methods and
several attributes returning Python "scalars" (float, int, str, bool, datetime, bytes).
Each attribute can be considered as segment metadata: it reflects a segment column
(or an associated database table via a foreign key) and returns the relative value.

### segment methods: ###

* segment.stream(reload=False): the `obspy.Stream` object representing the waveform data
  associated to the segment. Please remember that many ObsPy functions modify the
  stream in-place:
  ```
      stream_remresp = segment.stream().remove_response(segment.inventory())
      # any call to segment.stream() returns from now on `stream_remresp`
  ```
  For any case where you do not want to modify `segment.stream()`, copy the stream
  (or its traces) first, e.g.:
  ```
      stream_raw = segment.stream()
      stream_remresp = stream_raw.copy().remove_response(segment.inventory())
      # any call to segment.stream() will still return `stream_raw`
  ```
  You can also pass a boolean value (False by default when missing) to `stream` to force
  reloading it from the database (this is less performant as it resets the cached value):
  ```
      stream_remresp = segment.stream().remove_response(segment.inventory())
      stream_reloaded = segment.stream(True)
      # any call to segment.stream() returns from now on `stream_reloaded`
  ```
  (In visualization functions, i.e. those decorated with '@gui', any modification
  to the segment stream will NOT affect the segment's stream in other functions)

  For info see https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.copy.html

* segment.inventory(reload=False): the `obspy.core.inventory.inventory.Inventory`.
  This object is useful e.g., for removing the instrumental response from
  `segment.stream()`: note that it will be available only if the inventories in xml
  format were downloaded (by default, they are). As for `stream`, you can also pass
  a boolean value (False by default when missing) to `inventory` to force reloading it
  from the database.

* segment.siblings(parent=None, condition): returns an iterable of siblings of this
  segment. `parent` can be any of the following:
  - missing or None: returns all segments of the same recorded event, on the
    other channel components / orientations
  - 'stationname': returns all segments of the same station, identified by the tuple of
    the codes (newtwork, station)
  - 'networkname': returns all segments of the same network (network code)
  - 'datacenter', 'event', 'station', 'channel': returns all segments of the same
    datacenter, event, station or channel, all identified by the associated database id.
  `condition` is a dict of expression to filter the returned element. the argument
  `config['segment_selection']` can be passed here to return only siblings selected for
  processing. NOTE: Use with care when providing a `parent` argument, as the amount of
  segments might be huge (up to hundreds of thousands of segments). The amount of
  returned segments is increasing (non linearly) according to the following order of the
  `parent` argument: 'channel', 'station', 'stationname', 'networkname', 'event' and
  'datacenter'

* segment.del_classes(*labels): Deletes the given classes of the segment. The argument is
  a comma-separated list of class labels (string). See configuration file for setting up
  the desired classes. E.g.:
  `segment.del_classes('class1')`
  `segment.del_classes('class1', 'class2', 'class3')`

* segment.set_classes(*labels, annotator=None): Sets the given classes on the segment,
  deleting first all segment classes, if any. The argument is a comma-separated list of
  class labels (string). See configuration file for setting up the desired classes.
  `annotator` is a keyword argument (optional): if given (not None) denotes the user name
  that annotates the class. E.g.:
  `segment.set_classes('class1')`
  `segment.set_classes('class1', 'class2', annotator='Jim')`

* segment.add_classes(*labels, annotator=None): Same as `segment.set_classes` but does
  not delete segment classes first. If a label is already assigned to the segment, it is
  not added again (regardless of whether the 'annotator' changed or not)

* segment.sds_path(root='.'): Returns the segment's file path in a seiscomp data
  structure (SDS) format:
     <root>/<event_id>/<net>/<sta>/<loc>/<cha>.D/<net>.<sta>.<loc>.<cha>.<year>.<day>
  See https://www.seiscomp3.org/doc/applications/slarchive/SDS.html for details.
  Example: to save the segment's waveform as miniSEED you can type (explicitly
  adding the file extension '.mseed' to the output path):
  `segment.stream().write(segment.sds_path() + '.mseed', format='MSEED')`

* segment.dbsession(): (for advanced users) the database session for custom IO operations
  with the database.
  WARNING: this is for users experienced with SQLAlchemy library. If you want to use it
  you probably want to import stream2segment in custom code. See the github documentation
  in case

### segment attributes ###

""" + _SEGMENT_ATTRS

YAML_WARN = """
NOTE: **this file is written in YAML syntax**, which uses Python-style indentation to
# indicate nesting, keep it in mind when editing. You can also use a more compact format
# that uses [] for lists and {} for maps/objects.
# For info see http://docs.ansible.com/ansible/latest/YAMLSyntax.html
"""

PROCESS_YAML_MAIN = """
==========================================================================
# Stream2segment config file to tune the processing/visualization subroutine
# ==========================================================================
#
# This editable template defines the configuration parameters which will
# be accessible in the associated processing / visualization Python file.
#
# You are free to implement here anything you need: there are no mandatory parameters but
# we strongly suggest to keep 'segment_selection' and 'sn_windows', which add also special 
# features to the GUI.
"""

# yamelise _SEGMENT_ATTRS (first line not commented, see below)
_SEGMENT_ATTRS_YAML = "\n# ".join(s[8:] for s in _SEGMENT_ATTRS.splitlines())


PROCESS_YAML_SEGMENTSELECT = """
The parameter 'segment_selection' defines which segments to be processed or visualized.
# PLEASE USE THIS PARAMETER. If missing, all segments will be loaded, including segment
# with no (or malformed) waveform data: this is in practically always useless and slows
# down considerably the processing or visualization routine. The selection is made via
# the list-like argument:
#
# segment_selection:
#   <att>: "<expression>"
#   <att>: "<expression>"
#   ...
#
# where each <att> is a segment attribute and <expression> is a simplified SQL-select
# string expression. Example:
#
# 1. To select and work on segments with downloaded data (at least one byte of data):
# segment_selection:
#   has_data: "true"
#
# 2. To select and work on segments of stations activated in 2017 only:
# segment_selection:
#   station.start_time: "[2017-01-01, 2018-01-01T00:00:00)"
# (brackets denote intervals. Square brackets include end-points, round brackets exclude
# endpoints)
#
# 3. To select segments from specified ids, e.g. 1, 4, 342, 67 (e.g., ids which raised
# errors during a previous run and whose id where logged might need inspection in the GUI):
# segment_selection:
#   id: "1 4 342 67"
#
# 4. To select segments whose event magnitude is greater than 4.2:
# segment_selection:
#   event.magnitude: ">4.2"
# (the same way work the operators: =, >=, <=, <, !=)
#
# 5. To select segments with a particular channel sensor description:
# segment_selection:
#   channel.sensor_description: "'GURALP CMG-40T-30S'"
# (note: for attributes with str values and spaces, we need to quote twice, as otherwise
# "GURALP CMG-40T-30S" would match 'GURALP' and 'CMG-40T-30S', but not the whole string.
# See attribute types below)
#
# The list of segment attribute names and types is:
#
# """ + _SEGMENT_ATTRS_YAML + """
# """

PROCESS_YAML_SNWINDOWS = """
Settings for computing the 'signal' and 'noise' time windows on a segment waveform.
# From within the GUI, signal and noise windows will be visualized as shaded areas on the
# plot of the currently selected segment. If this parameter is missing, the areas will
# not be shown.
#
# Arrival time shift: shifts the calculated arrival time of
# each segment by the specified amount of time (in seconds). Negative values are allowed.
# The arrival time is used to split a segment into segment's noise (before the arrival
# time) and segment's signal (after)
#
# Signal window: specifies the time window of the segment's signal, in seconds from the
# arrival time. If not numeric it must be a 2-element numeric array, denoting the start
# and end points, relative to the squares cumulative of the segment's signal portion.
# E.g.: [0.05, 0.95] sets the signal window from the time the cumulative reaches 5% of
# its maximum, until the time it reaches 95% of its maximum.
# The segment's noise window will be set equal to the signal window (i.e., same duration)
# and shifted in order to always end on the segment's arrival time
"""

PROCESS_YAML_ADVANCEDSETTINGS = """
Advanced settings tuning the process routine:
advanced_settings:
  # Use parallel sub-processes to speed up the execution (true or false). Advanced users
  # can also provide a numeric value > 0 to tune the number of processes in the Pool 
  # (https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool)
  multi_process: false
  # Set the size, in number of segments, of each chunk of data that will be loaded from 
  # the database. Increasing this number speeds up the load but also increases memory 
  # usage. Null means: set the chunk size automatically ({1:d} if the number N of 
  # segments to be processed is > {1:d}, otherwise N/{2:d}). If multi_process is on, the
  # chunk size also defines how many segments will be loaded in each Python sub-process.
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
""".format(HDF_DEFAULT_CHUNKSIZE, *_get_chunksize_defaults())

DOWNLOAD_EVENTWS_LIST = '\n'.join('%s"%s": %s' % ('# ' if i > 0 else '', str(k), str(v))
                                  for i, (k, v) in enumerate(EVENTWS_MAPPING.items()))

# setting up DOCVARS:
DOCVARS = {k: v.strip() for k, v in globals().items()
           if hasattr(v, 'strip') and not k.startswith('_')}
