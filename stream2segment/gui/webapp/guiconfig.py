"""
Config file for the GUI.
Here you can edit options and add custom plots. The latter will be available under the plot
of the currently selected segments

Options
=======

Options are `dict`s whose keys are the GUI properties (mapped to their protperty values).
Currently there are two options:

- spectra: define the windows for the spectra. Dict of two keys:

  - Arrival time shift: shifts programmatically the calculated arrival time of
    each segment by the specified amount of time (in seconds). Negative values are allowed.
    The arrival time sets the <i>end</i> of the noise window, whose
    length (duration) is always set equal to the signal window (see below)
  - Signal window: specifies the signal window. It can be a number, specifying the
    window duration, in seconds starting from the arrival time, or a 2-element array
    specifying the window time start and end, relative to the cumulative. E.g., a value of
    [0.05, 0.95] sets the signal window from the time the cumulative reaches 5% of its maximum,
    until the time it reaches 95% of its maximum. Note that in this case the noise window and
    the signal window might overlap.

- filter: defines the filter arguments. Filters the signal and removes the instrumental
  response. The filter algorithm has the following steps:

  1. Sets the max frequency to 0.9 of the nyquist freauency (sampling rate /2)
     (slightly less than nyquist seems to avoid artifacts)
  2. Offset removal (substract the mean from the signal)
  3. Tapering
  4. Pad data with zeros at the END in order to accomodate the filter transient
  5. Apply bandpass filter, where the lower frequency is set according to the magnitude
  6. Remove padded elements
  7. Remove the instrumental response

Custom plots
============

A custom plot can be associated to a function that must be implemented here.
There are two steps.

Custom plot function
--------------------

First define a function, e.g.:
```
def myfunc(trace):
    ...
```
where `trace` is the `obspy.Trace` currently shown in the GUI, and the function must return either:

1. a new Trace
2. an obspy Stream
3. a dict of strings mapped to a trace or a numpy array. This option allows to show legend titles
   on the plot (the dict keys). Note that python dict does not preserve the insertion order, to do
   that return `collections.OrderedDict`
4. a list of arrays (numeric lists, numpy arrays)
5. a numpy array or anything convertible with `numpy.asarray`. E.g. `data=np.array(...)`.

For any returned trace t or numpy array a, `len(t.data)` and `len(a)` **must** be equal to
`len(trace.data)`

Register function
-----------------

Then, you need to call:

```register_function(func, name, execute_anyway)```

where:
 - `func` (python function) is the custom function just defined
 - `name` (string, optional) is the plot title as it will be displayed on the GUI.
   If missing, it defaults to the python function name
 - `execute_anyway` (boolean, optional) whether to execute the function anyway: if False,
   the function will not be executed if the source trace had gaps/errors (such as errors
   retrieving the station inventory) and the relative plot will be empty. False is the default
   when missing because we seldom experienced problems (GUI hangs and becomes unresponsive)
   when traces have gaps. You can always set this argument to True and then move to False if
   you experience the same problems
"""

import numpy as np
from obspy.core import Trace, Stream, UTCDateTime
from stream2segment.gui.webapp.plots import register_function
from scipy.optimize import curve_fit
from stream2segment.analysis import cumsum
from collections import OrderedDict

# settings for spectra and filter. You can change their values but you should not delete the keys:
spectra = {
    'arrival_time_shift': 0,  # in seconds
    'signal_window': [0.1, 0.9]  # either a number (in seconds) or interval relative to the % of the cumulative
}

filter = {  # @ReservedAssignment
    'remove_response_water_level': 60,
    'remove_response_output': 'ACC',
    'bandpass_freq_max': 20,  # the max frequency, in Hz:
    'bandpass_max_nyquist_ratio': 0.9,
    'bandpass_corners': 2
}


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y


def sigmoid_fit(trace):
    xdata = np.linspace(float(trace.stats.starttime),
                        float(trace.stats.endtime),
                        trace.stats.npts, endpoint=False, dtype=float) - float(trace.stats.starttime)
    ydata = trace.data
    ydata = cumsum(trace.data)  # - np.nanmean(trace.data))

    idx0 = np.searchsorted(ydata, 0.1)
#     idx1 = np.searchsorted(ydata, 0.99)
#     xdata = xdata[idx0:idx1]
#     ydata = ydata[idx0:idx1]

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    sigm = sigmoid(xdata, *popt)

    absdiff = np.abs(sigm-ydata)
#     firstd_absdiff = np.diff(trace.data)
#     firstd_absdiff = np.append(firstd_absdiff, firstd_absdiff[-1])
#     secdiff_absdiff = np.diff(firstd_absdiff)
#     secdiff_absdiff = np.append(secdiff_absdiff, secdiff_absdiff[-1])
# 
#     val = secdiff_absdiff

    val = np.append(np.zeros(idx0), absdiff[idx0:])
    title = "abs(cum-sigm): %.3e " % np.trapz(val)

    sigm_data = (popt[1], np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1])

    o = OrderedDict()
    o['cum'] = Trace(data=ydata, header=trace.stats.copy())
    o['sigm %.2e %.2e %.2e' % sigm_data] = Trace(data=sigm, header=trace.stats.copy())
    o[title] = Trace(data=val, header=trace.stats.copy())
    return o


register_function(sigmoid_fit, "Sigmoid fit", True)


# Now we can write a custom function. We will implement the 1st derivative
# A custom function accepts the currently selected obspy Trace T, and MUST return another Trace,
# or a numpy array with the same number of points as T:
def derivative1st(trace):
    deriv = np.diff(trace.data)
    # append last point (deriv.size = trace.data.size-1):
    deriv = np.append(deriv, deriv[-1])
    # and return our array:
    return deriv
    # or, alternatively:
    # return Trace(data=deriv, header=trace.stats.copy())

# to see the first derivative in the GUI, un-comment the line below:
register_function(derivative1st, "1st deriv", True)

