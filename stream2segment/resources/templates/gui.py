"""
=========================================================================
Stream2segment Graphical User Interface (GUI) Python module
=========================================================================

This Python module defines the plots to be displayed in a web browser GUI via
the `show` command, e.g.: `s2s show -d download.yaml -p <this_file_path>`


GUI functions implementation
============================

GUI functions are Python functions with specific decorators attached. Regardless
of the decorator type (described in details below), remember that all GUI functions:
- must have two arguments:
   - segment: the Segment object (for details, see {{ THE_SEGMENT_OBJECT_WIKI_URL }})
   - config: a Python `dict` representing the parameters set in the associated YAML file.
- can raise any Exception, in which case the exception message will be displayed
  as text on the corresponding plot area
- get a new copy of `segment.stream()` (the ObsPy Stream object holding the segment
  waveform), which means you do not necessarily need to copy the Stream because any
  inplace modification will not affect other GUI plots


1. Pre-process function
-----------------------

The function decorated with "@gui.preprocess", e.g.:
```
@gui.preprocess
def applybandpass(segment, config)
```
will be associated to a check-box in the GUI. By clicking the check-box,
all plots of the page will be re-calculated with the output of this function
instead of the raw segment waveform. Consequently, this function
**must thus return an ObsPy Stream or Trace object**


2. Plot functions
-----------------

The functions decorated with "@gui.plot", e.g.:
```
@gui.plot
def cumulative(segment, config)
```
will be associated to (i.e., its output will be displayed in) the plot below
the main plot.

You can also call @gui.plot with arguments, e.g.:
```
@gui.plot(position='r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def spectra(segment, config)
```
The 'position' argument controls where the plot will be placed in the GUI ('b' means
bottom, the default, 'r' means next to the main plot, on its right) and the other two,
`xaxis` and `yaxis`, are dict (defaulting to the empty dict: {}) controlling the x and y
axis of the plot (for info, see: https://plotly.com/javascript/axes/).

When not given, axis types (e.g., date time vs numeric) will be inferred from the
function's returned value which *must* be either:
- an ObsPy Trace object
- an ObsPy Stream object
- a dict compatible with the plot browser library (usually, at least the 'y' key must
  be present - see examples in this module. For a full list of keys, see
  https://plotly.com/javascript/reference/)
- a list of any object type described above
"""

# import numpy for efficient computation:
# import obspy core classes (when working with times, use obspy UTCDateTime when
# possible):
from obspy import UTCDateTime
# decorators needed to setup this module @gui.preprocess @gui.plot:
from stream2segment.process import gui
# stream2segment functions for processing obspy Traces:
from stream2segment.process.traces import bandpass, cumsumsq, \
    ampspec, powspec, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.ndarrays import triangsmooth


def assert1trace(stream):
    """Assert the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps
    """
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise ValueError("%d traces (probably gaps/overlaps)" % len(stream))


@gui.preprocess
def bandpass_remresp(segment, config):
    """{{ PROCESS_PY_BANDPASSFUNC | indent }}
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]

    inv = segment.inventory()

    # define some parameters:
    evt = segment.event
    bp_conf = config['bandpass']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, mag2freq(evt.magnitude), freq_max=bp_conf['freq_max'],
                     max_nyquist_ratio=bp_conf['max_nyquist_ratio'],
                     corners=bp_conf['corners'], copy=False)
    trace.remove_response(inventory=inv, output='ACC', water_level=None, pre_filt=None)
    return trace


def mag2freq(magnitude):
    """return a magnitude dependent frequency (in Hz)"""
    if magnitude <= 4.5:
        freq_min = 0.4
    elif magnitude <= 5.5:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min


def _spectrum(trace, config):
    """Calculate the spectrum of a trace. Returns the tuple (0, df, values), where
    values depends on the config dict parameters
    """
    taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
    taper_type = config['sn_spectra']['taper']['type']
    if config['sn_spectra']['type'] == 'pow':
        func = powspec  # copies the trace if needed
    elif config['sn_spectra']['type'] == 'amp':
        func = ampspec  # copies the trace if needed
    else:
        # raise TypeError so that if called from within main, the iteration stops
        raise TypeError("config['sn_spectra']['type'] expects either 'pow' or 'amp'")

    df_, spec_ = func(trace, taper_max_percentage=taper_max_percentage,
                      taper_type=taper_type)

    # Smoothing (if you want to implement your own smoothing, change the lines below):
    smoothing_wlen_ratio = config['sn_spectra']['smoothing_wlen_ratio']
    if smoothing_wlen_ratio > 0:
        spec_ = triangsmooth(spec_, winlen_ratio=smoothing_wlen_ratio)

    return 0, df_, spec_


def signal_noise_traces(segment, config):
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    arrival_time = UTCDateTime(segment.arrival_time) + \
        config['sn_windows']['arrival_time_shift']
    win_len = config['sn_windows']['signal_window']
    # assumes stream has only one trace:
    return sn_split(stream[0].copy(), arrival_time, win_len)


######################################
# GUI functions for displaying plots #
######################################


@gui.plot
def cumulative(segment, config):
    """Compute the cumulative of the squares of the segment's trace in the form of a
    Plot object. Normalizes the returned trace values in [0,1]

    :return: an obspy.Trace
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return cumsumsq(stream[0], normalize=True, copy=False)


@gui.plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def sn_spectra(segment, config):
    """Compute the signal and noise spectra, as dict of strings mapped to tuples
    (x0, dx, y).

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, frequencies)
    """
    # assumes stream has only one trace:
    signal_trace, noise_trace = signal_noise_traces(segment, config)
    x0_sig, df_sig, sig = _spectrum(signal_trace, config)
    x0_noi, df_noi, noi = _spectrum(noise_trace, config)
    signal = {'x0': x0_sig, 'dx': df_sig, 'y': sig, 'name': 'Signal'}
    noise = {'x0': x0_noi, 'dx': df_noi, 'y': noi, 'name': 'Noise'}
    return [signal, noise]


@gui.plot('r')
def sn_windows(segment, config):
    """Compute the signal and noise windows and return two traces

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, frequencies)
    """
    signal_trace, noise_trace = signal_noise_traces(segment, config)
    signal = {
        'x0': signal_trace.stats.starttime,
        'dx': signal_trace.stats.delta * 1000,  # dt in plots must be msec
        'y': signal_trace.data,
        'name': 'Signal'
    }
    noise = {
        'x0': noise_trace.stats.starttime,
        'dx': noise_trace.stats.delta * 1000,  # dt in plots must be msec
        'y': noise_trace.data,
        'name': 'Noise'
    }
    all_traces = [signal, noise]
    # add the underlying trace, but to avoid plotting overlapping points
    # and show only the chunks not in signal and noise
    original_trace = segment.stream()[0]
    times = (
        (original_trace.stats.starttime, noise_trace.stats.starttime),
        (noise_trace.stats.endtime, signal_trace.stats.starttime),
        (signal_trace.stats.endtime, original_trace.stats.endtime)
    )
    showlegend = True
    for stime, etime in times:
        if etime <= stime:
            continue
        trace = original_trace.slice(stime, etime)
        all_traces.append({
            'x0': trace.stats.starttime,
            'dx': trace.stats.delta * 1000,  # delta is in msec
            'y': trace.data,
            'line': {
                'color': 'rgba(100,100,100,1)'
            },
            'name': 'segment remainder',
            'legendgroup': 'segment remainder',
            'showlegend': showlegend
        })
        showlegend = False  # show only one legend for all chunks
    return all_traces
    # note: order matters for colors and placement (last traces on top):
    # return [signal, noise, trace]
