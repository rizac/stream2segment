"""
============================================================================
Stream2segment processing module generating a segment-based parametric table
============================================================================

This file exemplifies how to process downloaded data and can be run as Python script
from the terminal:
`python <this_file_path>`
See section `if __name__ == "__main__"` at the end of the module for details

For a general overview on segment processing (applicable e.g., in custom code, Jupyter
Notebook), see {{ USING_S2S_IN_YOUR_PYTHON_CODE_WIKI_URL }}
"""
import os.path
# From Python >= 3.6, dicts keys are returned (and thus, written to file) in the order
# they are inserted. Prior to that version, to preserve insertion order you needed to
# use OrderedDict:
from collections import OrderedDict
from datetime import datetime, timedelta
from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
# import obspy core classes (when working with times, use obspy UTCDateTime when
# possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.core.util.obspy_types import ObsPyException
from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.preprocess @gui.plot:
from stream2segment.process import SkipSegment
# straem2segment functions for processing obspy Traces. This is just a list of possible
# functions to show how to import them:
from stream2segment.process.funclib.traces import bandpass, cumsumsq,\
    fft, ampspec, powspec, timeof, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.funclib.ndarrays import triangsmooth, snr


def main(segment, config):
    """Main processing function, called iteratively for any segment selected from `imap`
    or `process` functions of stream2segment. If you just created this file with
    `s2s init`, see section `if __name__ == "__main__"` at the end of the module for
    details.

    IMPORTANT: Any exception raised here or from any sub-function will interrupt the
    whole processing routine (`imap` or `process`) with one special case:
    `stream2segment.process.SkipSegment` will resume from the next segment.
    Raise it to programmatically skip a segment, e.g.:
    ```
    if segment.sample_rate < 60:
        raise SkipSegment("segment sample rate too low")`
    ```
    Hint: Because handling exceptions at any point of a time-consuming processing is
    complex, we recommend to try to run your code on a smaller and possibly
    heterogeneous dataset first: change temporarily the segment selection (See section
    `if __name__ == "__main__"` at the end of the module), and inspect the logfile:
    for any exception that is not a bug and should simply be ignored, wrap only
    the part of code affected in a "try ... except" statement, and raise a `SkipSegment`.
    Also, please spend some time on refining the selection of segments: you might
    find that your code runs smoothly and faster by simply skipping certain segments in
    the first place.

    :param: segment: the object describing a downloaded waveform segment and its metadata,
        with a full set of useful attributes and methods detailed here:
        {{ THE_SEGMENT_OBJECT_WIKI_URL }}

    :param: config: a dictionary representing the configuration parameters
        accessible globally by all processed segments. The purpose of the `config`
        is to encourage decoupling of code and configuration for better and more
        maintainable code, avoiding, e.g., many similar processing functions differing
        by few hard-coded parameters (this is one of the reasons why the config is
        given as separate YAML file to be passed to the `s2s process` command)

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

         A column named '{{ SEGMENT_ID_COLNAME }}' with the segment database id (an integer
         uniquely identifying the segment) will be automatically added to the dict /
         Series, or to each row of the DataFrame, before writing it to file.

         SUPPORTED TYPES as elements of the returned dict/Series/DataFrame: all types
         supported by pandas:
         https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes

         For info on hdf and the pandas library (included in the package), see:
         https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
         https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-hdf5

    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now

    # discard saturated signals (according to the threshold set in the config file):
    amp_ratio = np.true_divide(np.nanmax(np.abs(trace.data)), 2**23)
    if amp_ratio >= config['amp_ratio_threshold']:
        raise SkipSegment('possibly saturated (amp. ratio exceeds)')

    # bandpass the trace, according to the event magnitude.
    # WARNING: this modifies the segment.stream() permanently!
    # If you want to preserve the original stream, store trace.copy() beforehand.
    # Also, use a 'try catch': sometimes Inventories are corrupted and ObsPy raises
    # a TypeError, which would break the WHOLE processing execution.
    # Raising a SkipSegment will stop the execution of the currently processed
    # segment only (logging the error message):
    try:
        trace = bandpass_remresp(segment, config)
    except (TypeError, ObsPyException) as resp_error:
        raise SkipSegment("Error in 'bandpass_remresp': %s" % str(resp_error))

    spectra = signal_noise_spectra(segment, config)
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
        raise SkipSegment('low snr %f' % snr_)

    # calculate cumulative

    cum_trace = cumsumsq(trace, normalize=True, copy=True)
    # Note above: copy=True prevent original trace from being modified
    # get times where cumulative reaches specific values/labels
    _cumlabels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    _cumtimes = (timeof(cum_trace, i) for i in np.searchsorted(cum_trace.data, _cumlabels))
    cumtime = {c: t for c, t in zip(_cumlabels, _cumtimes)}

    # double event (heuristic algorithm to filter out malformed data)
    try:
        (score, t_double, tt1, tt2) = \
            get_multievent_sg(
                cum_trace, cumtime[0.05], cumtime[0.95],
                config['savitzky_golay'], config['multievent_thresholds']
            )
    except IndexError as _ierr:
        raise SkipSegment("Error in 'get_multievent_sg': %s" % str(_ierr))
    if score in {1, 3}:
        raise SkipSegment('Double event detected %d %s %s %s' %
                         (score, t_double, tt1, tt2))

    # calculate PGA and times of occurrence (t_PGA):
    # note: you can also provide tstart tend for slicing
    trace_cut = trace.slice(cumtime[0.05], cumtime[0.95])
    try:
        _argmax = np.nanargmax(np.abs(trace_cut.data))
    except ValueError as verr:
        raise SkipSegment('Unable to compute PGA: ' + str(verr))
    t_PGA = timeof(trace_cut, _argmax)
    PGA = trace_cut.data[_argmax]

    # PGV:
    trace_cut_vel = trace_cut.copy()
    trace_cut_vel.integrate()
    try:
        _argmax = np.nanargmax(np.abs(trace_cut_vel.data))
    except ValueError as verr:
        raise SkipSegment('Unable to compute PGV: ' + str(verr))
    t_PGV = timeof(trace_cut_vel, _argmax)
    PGV = trace_cut_vel.data[_argmax]
    meanoff = meanslice(trace_cut_vel, 100, cumtime[0.05], trace_cut_vel.stats.endtime)

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = normal_f0 + np.arange(len(normal_spe)) * normal_df
    required_amplitudes = np.interp(np.log10(required_freqs),
                                    np.log10(ampspec_freqs),
                                    normal_spe) / segment.sample_rate

    # compute synthetic WA.
    trace_wa = synth_wood_anderson(trace.copy(), segment.inventory(), config)
    try:
        _argmax = np.nanargmax(np.abs(trace_wa.data))
    except ValueError as verr:
        raise SkipSegment('Unable to compute max WoodAnderson: ' + str(verr))
    t_WA = timeof(trace_wa, _argmax)
    maxWA = trace_wa.data[_argmax]

    # write stuff to csv / hdf:
    ret = {}

    ret['snr'] = snr_
    ret['snr1'] = snr1_
    ret['snr2'] = snr2_
    ret['snr3'] = snr3_

    # cumulative times:
    for _cumlabel in [0.05, 0.5, 0.95]:
        ret['cumtime__%.2f' % _cumlabel] = cumtime[_cumlabel].datetime

    ret['dist_deg'] = segment.event_distance_deg        # dist
    ret['dist_km'] = d2km(segment.event_distance_deg)  # dist_km
    # t_PGA is a obspy UTCDateTime. This type is not supported in HDF output, thus
    # convert it to Python datetime. Note that in CSV output, the value will be written
    # as str(t_PGA.datetime): another option might be to store it as string with
    # str(t_PGA) (returns the iso-formatted string, supported in all output formats):
    ret['t_PGA'] = t_PGA.datetime  # peak info
    ret['PGA'] = PGA
    # (for t_PGV, see note above for t_PGA)
    ret['t_PGV'] = t_PGV.datetime  # peak info
    ret['PGV'] = PGV
    # (for t_WA, see note above for t_PGA)
    ret['t_WA'] = t_WA.datetime
    ret['maxWA'] = maxWA
    ret['channel'] = segment.channel.channel
    ret['channel_component'] = segment.channel.channel[-1]
    # event metadata:
    ret['ev_id'] = segment.event.id
    ret['ev_lat'] = segment.event.latitude
    ret['ev_lon'] = segment.event.longitude
    ret['ev_dep'] = segment.event.depth_km
    ret['ev_mag'] = segment.event.magnitude
    ret['ev_mty'] = segment.event.mag_type
    # station metadata:
    ret['st_id'] = segment.station.id
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


def assert1trace(stream):
    """Assert the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps
    """
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise SkipSegment("%d traces (probably gaps/overlaps)" % len(stream))


def bandpass_remresp(segment, config):
    """{{ PROCESS_PY_BANDPASSFUNC | indent }}
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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
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
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise TypeError("window_size and order have to be of type int")
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


def get_multievent_sg(cum_trace, tmin, tmax, sg_params, multievent_thresholds):
    """Return the tuple (or a list of tuples, if the first argument is a stream) of the
    values (score, UTCDateTime of arrival)
    where scores is: 0: no double event, 1: double event inside tmin_tmax,
        2: double event after tmax, 3: both double event previously defined are detected
    If score is 2 or 3, the second argument is the UTCDateTime denoting the occurrence of
    the first sample triggering the double event after tmax
    """
    if tmin is not None:
        tmin = UTCDateTime(tmin)
    if tmax is not None:
        tmax = UTCDateTime(tmax)

    # split traces between tmin and tmax and after tmax
    traces = [cum_trace.slice(tmin, tmax), cum_trace.slice(tmax, None)]

    # calculate second derivative and normalize:
    second_derivs = []
    max_ = np.nan
    for ttt in traces:
        sec_der = savitzky_golay(
            ttt.data,
            sg_params['wsize'],
            sg_params['order'],
            sg_params['deriv']
        )
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
    indices = np.where(second_derivs[1] >=
                       multievent_thresholds['after_tmax_inpercent'])[0]
    if len(indices):
        result = 2

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the
    # duration
    deltatime = 0
    starttime = tmin
    endtime = None
    indices = np.where(second_derivs[0] >=
                       multievent_thresholds['inside_tmin_tmax_inpercent'])[0]
    if len(indices) >= 2:
        idx0 = indices[0]
        starttime = timeof(traces[0], idx0)
        idx1 = indices[-1]
        endtime = timeof(traces[0], idx1)
        deltatime = endtime - starttime
        if deltatime >= multievent_thresholds['inside_tmin_tmax_insec']:
            result += 1

    return result, deltatime, starttime, endtime


def synth_wood_anderson(trace, inventory, config):
    """Low-level function to calculate the synthetic wood-anderson of `trace`. The dict
    `config['simulate_wa']` must be implemented and houses the Wood-Anderson parameters:
    'sensitivity', 'zeros', 'poles' and 'gain'. Modifies the trace in place
    """
    trace_input_type = config['preprocess']['remove_response_output']

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
        trace.remove_response(inventory=inventory, output="DISP",
                              pre_filt=pre_filt,
                              water_level=conf['remove_response_water_level'])

    return trace.simulate(paz_remove=None, paz_simulate=config_wa)


def signal_noise_spectra(segment, config):
    """Compute the signal and noise spectra, as dict of strings mapped to tuples
    (x0, dx, y). Does not modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, frequencies)
    """
    arrival_time = UTCDateTime(segment.arrival_time) + \
                   config['sn_windows']['arrival_time_shift']
    win_len = config['sn_windows']['signal_window']
    # assumes stream has only one trace:
    signal_trace, noise_trace = sn_split(segment.stream()[0], arrival_time, win_len)
    x0_sig, df_sig, sig = _spectrum(signal_trace, config)
    x0_noi, df_noi, noi = _spectrum(noise_trace, config)
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def _spectrum(trace, config):
    """Calculate the spectrum of a trace. Returns the tuple (0, df, values), where
    values depends on the config dict parameters.
    Does not modify the trace in-place
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


def meanslice(trace, nptmin=100, starttime=None, endtime=None):
    """Return the numpy nanmean of the trace data, optionally slicing the trace first.
    If the trace number of points is lower than `nptmin`, returns NaN (numpy.nan)
    """
    if starttime is not None or endtime is not None:
        trace = trace.slice(starttime, endtime)
    if trace.stats.npts < nptmin:
        return np.nan
    val = np.nanmean(trace.data)
    return val


if __name__ == "__main__":
    # execute the code below only if this module is run as a script
    # (python <this_file_path>)

    # Remove the following line and edit the remaining code
    raise ValueError('The module is not yet implemented to be run as script. '
                     'Please open the file and edit the code in the script '
                     'section at the end of the module ')

    # Example code: Check and customize before run
    # ------------------------------------
    # Setup config: you can build your own dict of parameters or load it from a YAML
    # file as in the example below (change path according to your needs):
    import yaml
    config_path = os.path.splitext(os.path.abspath(__file__))[0] + '.yaml'
    with open(config_path, 'r') as fpt:
        config = yaml.safe_load(fpt)
    # get the database URL. Do NOT TYPE anywhere URLs with passwords (e.g. postgres), or
    # if you do, do not COMMIT the file and keep it local. A good solution is to read
    # the db URL used for downloading the data from its config. Example:
    download_path = os.path.join(os.path.dirname(__file__), 'download.yaml')
    with open(download_path, 'r') as fpt:
        dburl = yaml.safe_load(fpt)['dburl']
    # segments to process
    # For details, see {{ THE_SEGMENT_OBJECT_WIKI_URL_SEGMENT_SELECTION }}
    # The variable below can also be a list/numpy array of integers denoting the
    # database IDs of the segments to process (e.g., IDs read from a file)
    segments_selection = {
        'has_valid_data': 'true',
        'maxgap_numsamples': '[-0.5, 0.5]',
    }
    # output file
    outfile = 'enter_your_csv_or_hdf_path_here'
    # provide a log file path to track all skipped segment (SkipSegment exceptions).
    # Here we input the boolean True, which automatically creates a log file in the
    # same directory 'outfile' above. To skip logging, type "" or False
    logfile = True
    # show progressbar on the terminal and additional info
    verbose = True
    # overwrite existing outfile, if present. If True and outfile exists, already
    # processed segments will be skipped
    append = False
    # csv or hdf options. Type help(process) on terminal or notebook for details
    writer_options = {}
    # use sub-processes to speed up the routine
    multiprocess = True
    # segment chunk size to load. Type help(process) on terminal or notebook for details.
    chunksize = None

    from stream2segment.process import imap, process

    # run imap or process here. Example with process:
    process(main, dburl, segments_selection=segments_selection, config=config,
            outfile=outfile, append=append, writer_options=writer_options,
            logfile=logfile, verbose=verbose, multi_process=multiprocess,
            chunksize=chunksize)
