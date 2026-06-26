"""
============================================================================
Stream2segment processing module generating a segment-based parametric table
============================================================================

Customize the section `if __name__ == "__main__"` at the end of the module and then
run it as Python script from the terminal:
`python <this_file_path>`

Remember that you are not bound to table generations or output files, you can use this
module as backbone for any kind of processing.

For a general overview on segment processing (applicable e.g., in custom code, Jupyter
Notebook), see {{ USING_S2S_IN_YOUR_PYTHON_CODE_WIKI_URL }}

General hint with big datasets: if you run this module with a lot of segments, handling
and tracking exceptions might become complex. We recommend in this case to try to run
your code on a smaller and possibly heterogeneous dataset first: change temporarily the
segment selection (See section `if __name__ == "__main__"` at the end of the module),
and inspect the logfile: for any exception that is not a bug and should simply be
ignored, wrap only the part of code affected in a "try ... except" statement, and
raise a `SkipSegment` (see details in the `main` function). Also, please spend some
time on refining the selection of segments: you might find that your code runs smoothly
and faster by simply skipping unwanted segments in the first place
"""
from datetime import datetime
from math import factorial  # for savitzky_golay function

import numpy as np
try:
    from numpy.matrixlib.defmatrix import asmatrix  # FIXME CHECK!!!
except ImportError:
    from numpy import mat as asmatrix  # numpy < 2

from obspy import Trace, Stream, UTCDateTime, Inventory
from obspy.core.event import Event
from obspy.core.util.obspy_types import ObsPyException
from stream2segment.process import SkipSegment, SegmentMetadata
# functions to show how to import them:
from stream2segment.process.traces import (
    bandpass, cumsumsq, ampspec, powspec, timeof, sn_split
)
from stream2segment.process.ndarrays import triangsmooth, snr


def main(
    segment: Stream,
    station: Inventory | None,
    event: Event | None,
    config: dict
):
    """
    Main processing function, called iteratively for any segment selected from `imap`
    or `process` functions of stream2segment.

    IMPORTANT: Any exception raised here or from any sub-function will interrupt the
    whole processing routine (`imap` or `process`) with one special case:
    `stream2segment.process.SkipSegment` will resume from the next segment.
    Raise it to programmatically skip a segment, e.g.:
    ```
    if segment.sample_rate < 60:
        raise SkipSegment("segment sample rate too low")`
    ```

    :param: segment: an ObsPy `Stream` object, a container of ObsPy `Trace`s each
        representing a Segment on the DB. Depending on the users input configuration ,
        this object should contain only a single Trace (accessible via `segment[0]`), or
        all (usually three) components of a recorded waveform segment.
        For each Trace, the metadata stored in the DB is accessible via
        the `Trace.stats.segment_metadata` attribute (for details, see
        {{ THE_SEGMENT_OBJECT_WIKI_URL }})
        Please note that any Trace with gaps or overlaps will be
        included separately in this object, so the total Trace count might be bigger.
        To quickly check, you can map a dict to all traces:
        ```
        # create a dict trace_id -> list of traces:
        traces = {}
        for t in segment:
            traces.setdefault(t.id, []).append(t)
        # check gaps / overlaps:
        for t_list in traces.values():
            if len(t_list) > 1:
                # trace has gaps or overlaps. You can merge, raise and so on...
        ```

    :param station: the optional Inventory `Inventory` object, resulting from the
        segment(s) StationXML stored in the DB. The inventory is used to remove the
        waveform instrumental response and convert its data in physical units: as such,
        it is most likely needed (see ObsPy doc for details). If None, the StationXML
        is not available (see download config, where the default is to download
        StationXML)

    :param event: an optional `Event` object, resulting from the segment(s) QuakeML
        stored in the DB. Note that basic and often sufficient event information is
        available in the Segment Metadata, e.g.:
        `segment[0].stats.segment_meta.event_magnitude`.
        If None, the QuakeML is not available (see download config, where the default
        is not to download QuakeXML)

    :param: config: an optional dictionary representing the configuration parameters
        accessible globally by all processed segments. The purpose of the `config`
        is to encourage decoupling of code and configuration for better and more
        maintainable code, avoiding, e.g., many similar processing functions differing
        by few hard-coded parameters. For a couple of simple parameters, a custom config
        is usually an overkill, and you can implement your parameters here

    :return: a row of the resulting table, as dict, pandas Series, or -
        if a single segment should produce several rows - a pandas DataFrame or a
        list/ tuple of the those object types.
        The dict / Series keys, or DataFrame column names will compose the column names
        (table header); you are not forced to always return the same type of object,
        as long as the column names are always the same.

        Returning None or nothing is also valid: in this case the segment will be
        silently skipped

        The output format file, if an output file is provided, will be inferred from
        the file extension. Supported formats are 'csv' and 'hdf'. For details, see:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_hdf.html

        Supported data types are int, float, bool, str and - with HDF - date times
        (see pandas `to_datetime`). `str` should be avoided when possible (e.g. use int
        ids to track the uniqueness of a row, not string). HDF are recommended because
        more lightweight and better at preserving data types. The only drawback is that
        string columns must be pre-allocated via `min_itemsize`. For instance, if you
        want to write the network name of the segment under the column `network`, you
        must provide `min_itemsize = {'network': 8 }` (assuming no network name will be
        longer as 8 characters)
    """
    # Assure the stream has only one trace by simply counting the traces in stream
    # (stream.get_gaps() does this more accurately, but it's slower)
    if len(segment) != 1:
        raise SkipSegment(f"{len(segment)} traces (probably gaps/overlaps)")

    if station is None:
        raise SkipSegment("no inventory provided")

    trace = segment[0]  # work with the (surely) one trace now
    segment_meta: SegmentMetadata = trace.stats.segment_metadata

    # discard saturated signals (according to the threshold set in the config file):
    amp_ratio = np.true_divide(np.nanmax(np.abs(trace.data)), 2**23)
    if amp_ratio >= config['amp_ratio_threshold']:
        raise SkipSegment('possibly saturated (amp. ratio exceeds)')

    # preprocess the trace: apply a bandpass filter (mag dependent), remove the
    # instrumental response and RETURN A TRACE IN ACCELERATION UNITS (m/s**2)
    # WARNING: this modifies the segment.stream() permanently!
    # If you want to preserve the original stream, store trace.copy() beforehand
    try:
        trace = bandpass_remresp(trace, station, segment_meta.event_magnitude, config)
    except (TypeError, ObsPyException, ValueError) as resp_error:
        raise SkipSegment("Error in 'bandpass_remresp': %s" % str(resp_error))

    spectra = signal_noise_spectra(segment, segment_meta.arrival_time, config)
    normal_f0, normal_df, normal_spe = spectra['Signal']
    noise_f0, noise_df, noise_spe = spectra['Noise']
    fcmin = mag2freq(segment_meta.event_magnitude)
    fcmax = config['bandpass']['freq_max']  # used in bandpass_remresp
    snr_min_max = snr(
        normal_spe,
        noise_spe,
        signals_form=config['sn_spectra']['type'],
        fmin=fcmin,
        fmax=fcmax,
        delta_signal=normal_df,
        delta_noise=noise_df
    )
    snr_min_1 = snr(
        normal_spe,
        noise_spe,
        signals_form=config['sn_spectra']['type'],
        fmin=fcmin,
        fmax=1,
        delta_signal=normal_df,
        delta_noise=noise_df
    )
    snr_1_10 = snr(
        normal_spe,
        noise_spe,
        signals_form=config['sn_spectra']['type'],
        fmin=1,
        fmax=10,
        delta_signal=normal_df,
        delta_noise=noise_df
    )
    snr_10_max = snr(
        normal_spe,
        noise_spe,
        signals_form=config['sn_spectra']['type'],
        fmin=10,
        fmax=fcmax,
        delta_signal=normal_df,
        delta_noise=noise_df
    )
    if snr_min_max < config['snr_threshold']:
        raise SkipSegment('low snr %f' % snr_min_max)

    # calculate cumulative

    cum_trace = cumsumsq(trace, normalize=True, copy=True)
    # Note above: copy=True prevent original trace from being modified
    # get times when cumulative reaches specific values/labels
    _cumlabels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    _cumtimes = (
        timeof(cum_trace, i) for i in np.searchsorted(cum_trace.data, _cumlabels)
    )
    cumtime = {c: t for c, t in zip(_cumlabels, _cumtimes)}

    # double event (heuristic algorithm to filter out malformed data)
    try:
        score, t_double, tt1, tt2 = get_multievent_sg(
            cum_trace,
            cumtime[0.05],
            cumtime[0.95],
            config['savitzky_golay'],
            config['multievent_thresholds']
        )
    except IndexError as _ierr:
        raise SkipSegment("Error in 'get_multievent_sg': %s" % str(_ierr))
    if score in {1, 3}:
        raise SkipSegment(
            'Double event detected %d %s %s %s' % (score, t_double, tt1, tt2)
        )

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
    meanoff = meanslice(
        trace_cut_vel,
        100,
        cumtime[0.05],
        trace_cut_vel.stats.endtime
    )

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = normal_f0 + np.arange(len(normal_spe)) * normal_df
    required_amplitudes = np.interp(
        np.log10(required_freqs),
        np.log10(ampspec_freqs),
        normal_spe
    ) / trace.stats.sampling_rate

    # compute synthetic WA
    trace_wa = synth_wood_anderson(trace.copy(), config)
    try:
        _argmax = np.nanargmax(np.abs(trace_wa.data))
    except ValueError as verr:
        raise SkipSegment('Unable to compute max WoodAnderson: ' + str(verr))
    t_WA = timeof(trace_wa, _argmax)
    maxWA = trace_wa.data[_argmax]

    # write stuff to csv / hdf:
    ret = {}

    ret['snr_fmin_fmax'] = snr_min_max
    ret['snr_fmin_1'] = snr_min_1
    ret['snr_1_10'] = snr_1_10
    ret['snr_10_fmax'] = snr_10_max

    # cumulative times:
    for _cumlabel in [0.05, 0.5, 0.95]:
        ret['cumtime__%.2f' % _cumlabel] = cumtime[_cumlabel].datetime

    ret['dist_deg'] = segment_meta.event_distance_deg        # dist
    ret['dist_km'] = segment_meta.event_distance_km  # dist_km
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
    ret['channel'] = segment_meta.channel_code
    ret['channel_component'] = segment_meta.orientation_code
    # event metadata:
    ret['ev_id'] = segment_meta.event_id
    ret['ev_lat'] = segment_meta.event_latitude
    ret['ev_lon'] = segment_meta.event_longitude
    ret['ev_dep'] = segment_meta.event_depth_km
    ret['ev_mag'] = segment_meta.event_magnitude
    ret['ev_mty'] = segment_meta.event_magnitude_type
    # station metadata:
    # ret['st_id'] = segment.station.id
    ret['st_name'] = segment_meta.station_code
    ret['st_net'] = segment_meta.network_code
    ret['st_lat'] = segment_meta.latitude
    ret['st_lon'] = segment_meta.longitude
    ret['st_ele'] = segment_meta.elevation
    ret['score'] = score
    ret['d2max'] = float(tt1)
    ret['offset'] = np.abs(meanoff/PGV)
    for freq, amp in zip(required_freqs, required_amplitudes):
        ret['f_%.5f' % freq] = float(amp)

    return ret


def bandpass_remresp(trace: Trace, inventory: Inventory, magnitude:float, config: dict):
    """
    Apply a pre-process on the given segment waveform by filtering the signal and
    removing the instrumental response, returning a new Trace in acceleration unit
    (meters/second**2)

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
    7. Remove the instrumental response. For info see:
       https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html

    IMPORTANT: This function modifies the segment stream in-place: further calls to
    `segment.stream()` will return the pre-processed stream. If needed, you
    can store the raw stream beforehand (`raw_trace=segment.stream().copy()`)

    :return: a Trace object
    """
    # define some parameters:
    bp_conf = config['bandpass']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(
        trace,
        mag2freq(magnitude),
        freq_max=bp_conf['freq_max'],
        max_nyquist_ratio=bp_conf['max_nyquist_ratio'],
        corners=bp_conf['corners'],
        copy=False
    )
    trace.remove_response(
        inventory=inventory,
        output='ACC',
        water_level=None,
        pre_filt=None
    )
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
    b = asmatrix(
        [[k**i for i in order_range] for k in range(-half_window, half_window+1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def get_multievent_sg(cum_trace: Trace, tmin, tmax, sg_params, multievent_thresholds):
    """
    Return the tuple (or a list of tuples, if the first argument is a stream) of the
    values (score, UTCDateTime of arrival)
    where scores is:
    0: no double event, 1: double event inside tmin_tmax, 2: double event after tmax,
    3: both double event previously defined are detected
    If score is 2 or 3, the second argument is the UTCDateTime denoting the occurrence
    of the first sample triggering the double event after tmax
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
    indices = np.where(
        second_derivs[1] >= multievent_thresholds['after_tmax_inpercent']
    )[0]
    if len(indices):
        result = 2

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the
    # duration
    deltatime = 0
    starttime = tmin
    endtime = None
    indices = np.where(
        second_derivs[0] >= multievent_thresholds['inside_tmin_tmax_inpercent']
    )[0]
    if len(indices) >= 2:
        idx0 = indices[0]
        starttime = timeof(traces[0], idx0)
        idx1 = indices[-1]
        endtime = timeof(traces[0], idx1)
        deltatime = endtime - starttime
        if deltatime >= multievent_thresholds['inside_tmin_tmax_insec']:
            result += 1

    return result, deltatime, starttime, endtime


def synth_wood_anderson(trace: Trace, config: dict):
    """Low-level function to calculate the synthetic wood-anderson of `trace` (which
    must be in acceleration units, see `bandpass_remresp`). The dict
    `config['simulate_wa']` must be implemented and houses the Wood-Anderson parameters:
    'sensitivity', 'zeros', 'poles' and 'gain'. Modifies the trace in place
    """
    config_wa = dict(config['paz_wa'])
    # parse complex string to complex numbers:
    zeros_parsed = map(complex, (c.replace(' ', '') for c in config_wa['zeros']))
    config_wa['zeros'] = list(zeros_parsed)
    poles_parsed = map(complex, (c.replace(' ', '') for c in config_wa['poles']))
    config_wa['poles'] = list(poles_parsed)
    # compute synthetic WA response. This modifies the trace in-place!

    # double integration (move to displacement):
    trace.integrate().integrate()

    return trace.simulate(paz_remove=None, paz_simulate=config_wa)


def signal_noise_spectra(segment: Stream, arrival_time: datetime, config: dict):
    """Compute the signal and noise spectra, as dict of strings mapped to tuples
    (x0, dx, y). Does not modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, frequencies)
    """
    arrival_time = (
        UTCDateTime(arrival_time) + config['sn_windows']['arrival_time_shift']
    )
    win_len = config['sn_windows']['signal_window']
    # assumes stream has only one trace:
    signal_trace, noise_trace = sn_split(segment[0], arrival_time, win_len)
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
    """
    Return the mean (ignoring NaNs) of the trace data, optionally slicing the trace
    first. If the trace number of points is lower than `nptmin`, returns NaN (numpy.nan)
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

    from pathlib import Path
    # ------------------------------------
    # Setup config: you can build your own dict of parameters or load it from a YAML
    # file as in the example below (change path according to your needs):
    import yaml
    config_path = Path(__file__).with_suffix('.yaml')
    config = yaml.safe_load(config_path.read_text())
    # get the database URL. Do NOT TYPE anywhere URLs with passwords (e.g. postgres), or
    # if you do, do not COMMIT the file and keep it local. A good solution is to read
    # the db URL used for downloading the data from its config. Example:
    download_path = Path(__file__).parent / 'download.yaml'
    dburl = yaml.safe_load(download_path.read_text())['dburl']
    # segments to process
    # For details, see {{ THE_SEGMENT_OBJECT_WIKI_URL_SEGMENT_SELECTION }}
    # The variable below can also be a list/numpy array of integers denoting the
    # database IDs of the segments to process (e.g., IDs read from a file)
    segments_selection = {
        'gap_score_percent': '[-50, 50]',
    }
    # output file
    outfile = '__enter_your_csv_or_hdf_path_here__'  # None, valid file str, or Path
    if outfile == '__enter_your_csv_or_hdf_path_here__':
        raise ValueError(
            'The module is not yet implemented to be run as script. '
            'Please open the file and edit the variables in the script '
            'section (e.g., config_file, outfile, dburl) at the end of the module'
        )
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

    from stream2segment.process import process

    # run imap or process here. Example with process (see function `main` at the top
    # of the module, that you can modify as you wish):
    process(
        main,
        dburl,
        segments_selection=segments_selection,
        config=config,
        outfile=outfile,
        append=append,
        writer_options=writer_options,
        logfile=logfile,
        verbose=verbose,
        multi_process=multiprocess,
        chunksize=chunksize
    )
