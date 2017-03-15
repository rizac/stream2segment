"""
Config file for the GUI. Please follow the instructions for customizing it
"""
import numpy as np
from obspy.core import Trace, Stream, UTCDateTime
from stream2segment.gui.webapp.plots import register_function

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
# register_function(derivative1st, "1st deriv", True)

# To implement your custom functions, follow the procedure above and call `register_function(func, name, execute_anyway)`, where:
# `func` (python function) is the custom function
# `name` (string, optional) is the function name as it will be displayed on the GUI.
#        If missing it defaults to the python function name
# `execute_anyway` (boolean, optional) whether to execute the function anyway: if False,
#        the function will not be executed if the source trace had gaps/errors (such as errors retrieving
#        the station inventory) and the relative plot will be empty. False is the default when missing
#        because we seldom experienced problems (GUI hangs and becomes unresponsive) when traces have gaps
#        You can always set this argument to True and then move to False if you experience the same problems
