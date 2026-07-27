"""
Stream2segment processing module generating a parametric table
based on the two horizontal components of a segment

Modify `run()` and optionally `run_segment()` and execute `python <this_file_path>`

For a general overview on segment processing (applicable e.g., in custom code, Jupyter
Notebook), see
https://github.com/rizac/stream2segment/wiki/using-stream2segment-in-your-python-code

Implementation hints:
1. if you run this module with a lot of segments, we recommend to try to run
your code on a smaller dataset first (change temporarily the
segment selection in `run()`) and inspect the logfile.
2. For any exception raised that is not a bug but should simply discard the
current segment and continue with the next one, wrap the part of code affected in a
"try ... except" statement, and raise a `SkipSegment` with the error
message that you want to appear in the log file for later inspection
"""
from datetime import datetime
from pathlib import Path
from math import factorial  # for savitzky_golay function
import sys
import numpy as np
import yaml
from obspy import Trace, Stream, UTCDateTime
from obspy.core.inventory.inventory import Inventory
from obspy.core.event import Event
from obspy.core.util.obspy_types import ObsPyException
from stream2segment.process import SkipSegment, SegmentMetadata, process_segments
# functions to show how to import them:
from stream2segment.process.traces import (
    bandpass, cumsumsq, fft, ampspec, powspec, timeof, sn_split
)
from stream2segment.process.ndarrays import triangsmooth, snr


def run():
    """Generate a parametric table"""

    # get the database URL. Do NOT TYPE anywhere URLs with passwords (e.g. postgres), or
    # if you do, do not COMMIT the file and keep it local. By default, we assume it is
    # the 1st argument passed to this script (python <this_file_path> <dburl> <outfile>):
    dburl = sys.argv[1] if len(sys.argv) > 1 else ""
    if not dburl:
        raise ValueError(
            'Please provide a database url or the yaml file used for download the data,'
            'either as command line argument '
            f'(python {Path(__file__).name} <dburl> <outfile>), '
            'or modifying the code directly (variable `dburl` inside `def run()`)'
        )
    if Path(dburl).suffix.lower() in ('.yaml', '.yml'):
        # If the download config was passed as argument, read its db url:
        dburl = yaml.safe_load(Path(dburl).read_text())['db']

    # segments to process from the chosen Database
    # For details, see https://github.com/rizac/stream2segment/wiki/the-segment-object#segments-selection
    segments_selection = {
        'gap_score_percent': '[-50, 50]',
    }

    # output file. We assume it is the 2nd argument passed to this script
    # (python <this_file_path> <dburl> <outfile>):
    outfile = sys.argv[2] if len(sys.argv) > 2 else ""
    if not outfile:
        raise ValueError(
            'Please provide an output file path, either via the command line '
            f'(python {Path(__file__).name} <dburl> <outfile>), '
            'or modifying the code directly (variable `outfile` inside `def run()`)'
        )

    # Setup config setting a dict of parameters available to the processing function
    # (in the example below, we read the dict from a yaml file with the same name
    # as this module):
    config = yaml.safe_load(Path(__file__).with_suffix('.yaml').read_text())

    # execute `run_segment` on each segment selected from `dburl`:
    process_segments(
        run_segment,
        dburl,
        segments_selection=segments_selection,
        # Each segment is passed to the processing function as an ObsPy Stream
        # containing a single component (e.g. vertical). Set True to group all
        # available components (usually 3) into one Stream:
        group_components=True,
        config=config,
        outfile=outfile,
        # Append to existing table, if file exists (if False, overwrite file):
        append=False,
        # Csv or Hdf options ({} = no options. See pandas to_hdf or to_csv for details):
        writer_options={},
        # Set the log file path to track all skipped segment (SkipSegment exceptions).
        # Set to True to automatically create a log file in the same directory of your
        # output file. Set False or "" to ignore logging:
        logfile=True,
        # Show progressbar on the terminal and additional info:
        verbose=True,
        # Use parallel sub-processes to speed up the routine:
        multi_process=False,
        # Segment chunk size to load (None: let the program handle it):
        chunksize=None
    )


def run_segment(
    segment: Stream,
    station: Inventory | None,
    event: Event | None,
    config: dict
):
    """Run processing on a single segment, returning one or more rows of the final
    parametric table.

    IMPORTANT: When this function is called from `process_segments` or `map_segments`,
    any exception raised here will interrupt the whole routine. To interrupt only the
    current segment and continue with the next one, raise programmatically a
    `stream2segment.process.SkipSegment` with an optional message that will appear
    in the logfile, if configured. Example:
    ```
    if segment[0].stats.sampling_rate < 60:
        raise SkipSegment("segment sample rate too low")`
    ```

    :param segment: an ObsPy `Stream` object, a container of ObsPy `Trace`s each
        representing a Segment on the DB. Depending on the user's configuration ,
        this object contains only a single Trace (accessible via `segment[0]`), or
        all (usually three) components of a recorded waveform segment.

        For each Trace, the metadata stored in the DB is accessible via
        the `Trace.stats.segment_metadata` attribute. For details, see:
        https://github.com/rizac/stream2segment/wiki/the-segment-object

        Traces separated by gaps or overlaps are included as separate Trace objects,
        so the total Trace count might be bigger than expected. To quickly check:
        ```
        ids = [t.get_id() for t in segment]
        if len(ids) != len(set(ids)):  # ids are not unique
            # some trace has gaps or overlaps. You can merge, ignore, raise SkipSegment
        ```

    :param station: the optional `Inventory` object resulting from the
        segment(s) StationXML stored in the DB. The inventory is used to remove the
        waveform instrumental response and convert its data in physical units (see
        ObsPy doc for details). If None, the StationXML is not available due to error
        or because not explicitly downloaded (although the default is to download
        StationXML)

    :param event: an optional `Event` object, resulting from the segment(s) QuakeML
        stored in the DB. Note that basic and often sufficient event information is
        available in the Segment Metadata, e.g.:
        `segment[0].stats.segment_metadata.event_magnitude`.
        If None, the QuakeML is not available due to error or because not explicitly
        downloaded (which is the default)

    :param config: an optional dictionary representing the configuration parameters
        accessible globally by all processed segments. The purpose of the `config`
        is to encourage decoupling of code and configuration for better and more
        maintainable code, avoiding, e.g., many similar processing functions differing
        by few hard-coded parameters. For few of simple parameters, a custom config is
        usually an overkill, and you can implement your parameters directly in the code

    :return: a row of the resulting table, as dict, pandas Series, or -
        if a single segment should produce several rows - a pandas DataFrame or a
        list or tuple of those object types.
        The dict / Series keys, or DataFrame column names will compose the column names
        (table header); for any object type you return, the column names must be
        always the same.

        Not returning any object (or returning None) is also valid: in this case the
        segment will be silently skipped

        Note: When this function is called from `process_segments` with an output file,
        the file format will be inferred from the file extension.
        Supported formats are 'csv' and 'hdf'. For details, see:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_hdf.html

        Supported data types are int, float, bool, str and - with HDF - date times
        (see pandas `to_datetime`). `str` should be avoided when possible (e.g. use int
        ids to track the uniqueness of a row, not string). HDF are recommended because
        they are more lightweight and preserve the data types. The only drawback is that
        string columns must be pre-allocated via `min_itemsize`. For instance, if you
        want to write the network name of the segment under the column `network`, you
        must provide `min_itemsize = {'network': 8 }` (assuming no network name will be
        longer as 8 characters)
    """
    if station is None:
        raise SkipSegment("no station inventory provided")
    # Check the stream has only one trace (stream.get_gaps() does this more accurately,
    # but it's slower)
    ids = [t.get_id() for t in segment]
    if len(ids) != len(set(ids)):
       raise SkipSegment('gaps / overlaps in some component')
    # Get the two horizontal components:
    horizontal = [t for t in segment if t.stats.channel[-1] not in ("Z", "3")]
    if len(horizontal) != 2:
        raise SkipSegment(f'{len(horizontal)} horizontal component(s) found')
    outputs = []
    scores = []
    for trace in horizontal:
        segment_metadata = trace.stats.segment_metadata

        # discard saturated signals (according to the threshold set in the config file):
        amp_ratio = np.true_divide(np.nanmax(np.abs(trace.data)), 2**23)
        if amp_ratio >= config['amp_ratio_threshold']:
            raise SkipSegment('possibly saturated (amp. ratio exceeds)')

        # bandpass the trace, according to the event magnitude.
        # trace will be in acceleration units
        try:
            trace = bandpass_remresp(
                trace,
                station,
                segment_metadata.event_magnitude,
                config
            )
        except (TypeError, ObsPyException, ValueError) as resp_error:
            raise SkipSegment("Error in 'bandpass_remresp': %s" % str(resp_error))
        try:
            spectra = signal_noise_spectra(trace, segment_metadata.arrival_time, config)
        except (ValueError,) as spectra_error:
            raise SkipSegment("Error in 'signal_noise_spectra': %s" % str(spectra_error))

        normal_f0, normal_df, normal_spe = spectra['Signal']
        noise_f0, noise_df, noise_spe = spectra['Noise']

        fcmin = mag2freq(segment_metadata.event_magnitude)
        fcmax = config['bandpass']['freq_max']  # used in bandpass_remresp
        snr_ = snr(
            normal_spe,
            noise_spe,
            signals_form=config['sn_spectra']['type'],
            fmin=fcmin,
            fmax=fcmax,
            delta_signal=normal_df,
            delta_noise=noise_df
        )
        snr1_ = snr(
            normal_spe,
            noise_spe,
            signals_form=config['sn_spectra']['type'],
            fmin=fcmin,
            fmax=1,
            delta_signal=normal_df,
            delta_noise=noise_df
        )
        snr2_ = snr(
            normal_spe,
            noise_spe,
            signals_form=config['sn_spectra']['type'],
            fmin=1,
            fmax=10,
            delta_signal=normal_df,
            delta_noise=noise_df
        )
        snr3_ = snr(
            normal_spe,
            noise_spe,
            signals_form=config['sn_spectra']['type'],
            fmin=10,
            fmax=fcmax,
            delta_signal=normal_df,
            delta_noise=noise_df
        )
        if snr_ < config['snr_threshold']:
            raise SkipSegment('low snr %f (%s)' % (snr_, trace.meta.channel))

        # calculate cumulative
        cum_trace = cumsumsq(trace, normalize=True, copy=True)
        # Note above: copy=True prevent original trace from being modified
        # get times where cumulative reaches specific values/labels
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
            raise SkipSegment(f"Error in 'get_multievent_sg': {_ierr}")
        if score in {1, 3}:
            raise SkipSegment(f'Double event detected {score} {t_double} {tt1} {tt2}')
        scores.append(score)

        # calculate PGA and times of occurrence (t_PGA):
        # note: you can also provide tstart tend for slicing
        trace_cut = trace.slice(cumtime[0.05], cumtime[0.95])
        try:
            _argmax = np.nanargmax(np.abs(trace_cut.data))
        except ValueError as verr:
            raise SkipSegment('Unable to compute PGA: ' + str(verr))
        #t_PGA = timeof(trace_cut, _argmax)
        PGA = trace_cut.data[_argmax]

        # PGV:
        trace_cut_vel = trace_cut.copy()
        trace_cut_vel.integrate()
        try:
            _argmax = np.nanargmax(np.abs(trace_cut_vel.data))
        except ValueError as verr:
            raise SkipSegment('Unable to compute PGV: ' + str(verr))
        #t_PGV = timeof(trace_cut_vel, _argmax)
        PGV = trace_cut_vel.data[_argmax]
        meanoff = meanslice(trace_cut_vel, 100, cumtime[0.05], trace_cut_vel.stats.endtime)
        # calculates amplitudes at the frequency bins given in the config file:
        required_freqs = np.array(config['freqs_interp'])
        ampspec_freqs = normal_f0 + np.arange(len(normal_spe)) * normal_df
        required_amplitudes = np.interp(
            np.log10(required_freqs),
            np.log10(ampspec_freqs),
            normal_spe
        ) / trace.stats.sampling_rate

        #required_periods = np.array(config['resp_spec_periods'])
        #dt = 1.0 / trace_cut.meta.sampling_rate
        #response_spectrum = get_response_spectrum(trace_cut.data, dt, required_periods,
        #                                          units="m/s/s")[0]
        outputs.append({
            "channel": trace_cut.meta.channel,
            "FAS": {"f": required_freqs, "amp": required_amplitudes},
        #    "SA": {"T": required_periods, "amp": response_spectrum["Pseudo-Acceleration"]},
            "PGA": PGA,
            "PGV": PGV,
        #    "time-history": trace_cut.data,
        #    "dt": dt
        })
    final_output = {
        "PGA": np.sqrt(outputs[0]["PGA"] * outputs[1]["PGA"]) * 100.0, # Geometric mean -> to cm/s/s
        "PGV": np.sqrt(outputs[0]["PGV"] * outputs[1]["PGV"]) * 100.0, # Geometric mean -> to cm/s/s
       # "SA": np.sqrt(outputs[0]["SA"]["amp"] * outputs[1]["SA"]["amp"]), # Geometric mean
        "EAS": np.sqrt(0.5 * (outputs[0]["FAS"]["amp"] ** 2.0 +
                              outputs[1]["FAS"]["amp"] ** 2.0))  # Effective Amplitude Spectrum
    }
    # write stuff to csv / hdf:
    ref_trace = segment[0]  # which trace is irrelevant for the metadata we will need
    net = ref_trace.stats.network
    sta = ref_trace.stats.station
    loc = ref_trace.stats.location
    cha = ref_trace.stats.channel

    segment_meta = ref_trace.stats.segment_metadata

    station_id = f"{net}.{sta}.{loc}.{cha[:-1]}"
    score = "|".join("{:.4f}".format(score) for score in scores)
    wfid = f"{segment_meta.event_id}|{segment_meta.station_id}"
    repi = segment_meta.event_distance_km  # d2km(segment.event_distance_deg)
    rhypo = np.sqrt(repi ** 2.0 + segment_meta.event_depth_km ** 2.0)
    ret = {
        "wfid": wfid,
        "event_id": segment_meta.event_id,
        "event_time": segment_meta.event_time.isoformat(),
        "event_longitude": segment_meta.event_longitude,
        "event_latitude": segment_meta.event_latitude,
        "event_hypo_depth": segment_meta.event_depth_km,
        "event_preferred_mag": segment_meta.event_magnitude,
        "event_preferred_mag_type": segment_meta.event_magnitude_type,
        "repi": repi, "rhypo": rhypo,
        "station_id": station_id,
        "network": net,
        "station": sta,
        "location": loc,
        "channel": cha[:-1],
        "station_longitude": segment_meta.longitude,
        "station_latitude": segment_meta.latitude,
        "station_elevation": segment_meta.elevation,
        "score": score,
        "PGA": final_output["PGA"],
        "PGV": final_output["PGV"],
    }

    for freq, amp in zip(required_freqs, final_output["EAS"]):
        ret[f"EAS_{freq:.5f}"] = float(amp)

    return ret


def bandpass_remresp(trace: Trace, station: Inventory, magnitude: float, config: dict):
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
        Must be less than `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or its n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over an odd-sized window centered at
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
    b = np.asmatrix(
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
    """Return (score, duration, start_time, end_time) describing
    whether a possible double event was detected, where `score` can be
    :
    0: no double event
    1: double event inside tmin and tmax
    2: double event after tmax
    3: both double event previously defined (1 and 2) are detected

    `duration`, `start_time`, and `end_time` describe the interval exceeding
    the threshold within `[tmin, tmax]`. They are zero/`None` if no such
    interval is found.
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


def signal_noise_spectra(trace: Trace, arrival_time: datetime, config):
    """Compute the signal and noise spectra.
    Does not modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the
        tuples (f0, df, spectrum_values)
    """
    arrival_time = (
        UTCDateTime(arrival_time) + config['sn_windows']['arrival_time_shift']
    )
    win_len = config['sn_windows']['signal_window']
    # assumes stream has only one trace:
    signal_trace, noise_trace = sn_split(trace, arrival_time, win_len)
    x0_sig, df_sig, sig = _spectrum(signal_trace, config)
    x0_noi, df_noi, noi = _spectrum(noise_trace, config)
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def _spectrum(trace, config):
    """Calculate the spectrum of a trace. Returns the tuple (0, df, values),
    where 0=f0 and the spectrum values depend on the config dict parameters.
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
    """Return the mean (ignoring NaNs) of the trace data, optionally slicing the trace
    first. If the number of points in the trace is lower than `nptmin`, return numpy.nan
    """
    if starttime is not None or endtime is not None:
        trace = trace.slice(starttime, endtime)
    if trace.stats.npts < nptmin:
        return np.nan
    val = np.nanmean(trace.data)
    return val


if __name__ == "__main__":
    # Run this module as script executing run()
    run()