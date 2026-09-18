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
   - segment: the ObsPy Stream object denoting a waveform segment. Note that for each
     ObsPy Tracey in the Stream (stream[i]) you can access the stream2segment metadata
     via stream[0].stats.segment_metadata (object with attributes)
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
def applybandpass(trace, station, config)
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
`xaxis` and `yaxis`, are dict (defaulting to the empty dict: {}) controlling the x and
y-axis of the plot (for info, see: https://plotly.com/javascript/axes/).

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
from obspy import UTCDateTime, Trace, Inventory
# decorators needed to setup this module @gui.preprocess @gui.plot:
from stream2segment.process import gui
# stream2segment functions for processing obspy Traces:
from stream2segment.process.traces import bandpass, cumsumsq, \
    ampspec, powspec, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.ndarrays import triangsmooth


@gui.preprocess
def bandpass_remresp(trace: Trace, station: Inventory, config: dict):
    """Preprocess the given segment waveform by filtering the signal and
    removing the instrumental response, returning a new Trace in acceleration units
    (meters/second**2)

    The steps performed are:
    1. Sets the max frequency to 0.9 of the Nyquist frequency (sampling rate /2)
       (slightly less than Nyquist seems to avoid artifacts)
    2. Offset removal (subtract the mean from the signal)
    3. Tapering
    4. Pad data with zeros at the END in order to accommodate the filter transient
    5. Apply bandpass filter, where the lower frequency is magnitude dependent
    6. Remove padded elements
    7. Remove the instrumental response. For info see:
       https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html

    IMPORTANT: This function modifies the Trace in-place

    :return: a Trace object
    """
    magnitude = trace.stats.segment_metadata.event_magnitude
    bp_config = config['bandpass']
    trace = bandpass(
        trace,
        mag2freq(magnitude),
        freq_max=bp_config['freq_max'],
        max_nyquist_ratio=bp_config['max_nyquist_ratio'],
        corners=bp_config['corners'],
        copy=False
    )
    trace.remove_response(
        inventory=station,
        output='ACC',
        water_level=None,
        pre_filt=None
    )
    return trace


@gui.plot
def cumulative(trace: Trace, config: dict):
    """Compute the cumulative of the squares of the segment's trace in the form of a
    Plot object. Normalizes the returned trace values in [0,1]

    :return: an obspy.Trace
    """
    return cumsumsq(trace, normalize=True, copy=False)


@gui.plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def sn_spectra(trace: Trace, config: dict):
    """Compute the signal and noise spectra, as dict of strings mapped to tuples
    (x0, dx, y).

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, frequencies)
    """
    # assumes stream has only one trace:
    signal_trace, noise_trace = signal_noise_traces(trace, config)
    x0_sig, df_sig, sig = _spectrum(signal_trace, config)
    x0_noi, df_noi, noi = _spectrum(noise_trace, config)
    signal = {'x0': x0_sig, 'dx': df_sig, 'y': sig, 'name': 'Signal'}
    noise = {'x0': x0_noi, 'dx': df_noi, 'y': noi, 'name': 'Noise'}
    return [signal, noise]


@gui.plot('r')
def sn_windows(trace: Trace, config: dict):
    """Compute the signal and noise windows and return two traces

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, frequencies)
    """
    signal_trace, noise_trace = signal_noise_traces(trace, config)
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
    times = (
        (trace.stats.starttime, noise_trace.stats.starttime),
        (noise_trace.stats.endtime, signal_trace.stats.starttime),
        (signal_trace.stats.endtime, trace.stats.endtime)
    )
    showlegend = True
    for stime, etime in times:
        if etime <= stime:
            continue
        new_trace = trace.slice(stime, etime)
        all_traces.append({
            'x0': new_trace.stats.starttime,
            'dx': new_trace.stats.delta * 1000,  # delta is in msec
            'y': new_trace.data,
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


####################
# Helper functions #
####################


def mag2freq(magnitude):
    """Return the magnitude-dependent frequency (Hz)"""
    if magnitude <= 4.5:
        freq_min = 0.4
    elif magnitude <= 5.5:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min


def signal_noise_traces(trace: Trace, config: dict) -> tuple[Trace, Trace]:
    arrival_time = (
        UTCDateTime(trace.stats.segment_metadata.arrival_time) +
        config['sn_windows']['arrival_time_shift']
    )
    win_len = config['sn_windows']['signal_window']
    # assumes stream has only one trace:
    return sn_split(trace.copy(), arrival_time, win_len)


def _spectrum(trace: Trace, config: dict):
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


