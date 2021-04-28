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
from stream2segment.download.modules.utils import EVENTWS_MAPPING
from stream2segment.process.main import _get_chunksize_defaults
from stream2segment.process.writers import SEGMENT_ID_COLNAME, HDF_DEFAULT_CHUNKSIZE
from stream2segment.process.inputvalidation import SEGMENT_SELECT_PARAM_NAMES
SEGSEL_PARAMNAME = SEGMENT_SELECT_PARAM_NAMES[0]


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

- `stream2segment.process.funclib.traces` (small processing library implemented in
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


_THE_SEGMENT_OBJECT_URL = 'https://github.com/rizac/stream2segment/wiki/the-segment-object'


_THE_SEGMENT_OBJECT_ATTRS_AND_METHS = _THE_SEGMENT_OBJECT_URL + '#attributes-and-methods'


_THE_SEGMENT_OBJECT_SEGSEL = _THE_SEGMENT_OBJECT_URL + '#segments-selection'


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
and execute the function on each selected segment (according to the 
'%(seg_sel)s' parameter in the config). See the function docstring of this 
module for implementation details.


Visualization (web GUI)
=======================

When visualizing, Stream2segment will open a web page where the user can browse 
and visualize the data. Contrarily to the processing, the `show` command can be
invoked with no argument: this will show by default all database segments, 
their metadata and a plot of their raw waveform (main plot).
When `show` is invoked with module and config files, Stream2segment will fetch
all segments (according to the '%(seg_sel)s' parameter in the config) and 
search for all module functions with signature:
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
consider and treat it as a normal Python object with several attributes and methods.
All details can be found here:
%(url)s
""" % {'seg_sel': SEGSEL_PARAMNAME, 'url': _THE_SEGMENT_OBJECT_ATTRS_AND_METHS}

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
# we strongly suggest to keep '{0}' and 'sn_windows', which add also special 
# features to the GUI.
""".format(SEGSEL_PARAMNAME)


PROCESS_YAML_SEGMENTSELECT = """
The parameter '{0}' defines which segments to be processed or visualized.
# PLEASE USE THIS PARAMETER. If missing, all segments will be loaded, including segment
# with no (or malformed) waveform data: this is in practically always useless and slows
# down considerably the processing or visualization routine. For details, see:
# {0}
# (scroll to the top of the page for the full list of selectable attributes)
#
# """.format(_THE_SEGMENT_OBJECT_SEGSEL) + """
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
