'''
Created on Feb 2, 2017

@author: riccardo
'''
from __future__ import print_function
from cStringIO import StringIO
import sys
import numpy as np
from obspy.core.stream import read
from stream2segment.utils import get_session
from stream2segment.io.db import models
from stream2segment.analysis.mseeds import get_gaps, dfreq
from stream2segment.analysis.mseeds import remove_response, get_gaps, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, get_multievent, snr
from stream2segment.download.utils import get_inventory_query
from stream2segment.utils.url import url_read
from stream2segment.io.utils import loads_inv, dumps_inv
from obspy.core.utcdatetime import UTCDateTime
from stream2segment.analysis import amp_spec, freqs
import pandas as pd
import csv
import click
from click.termui import progressbar


def getid(segment):
    return str(segment.channel.id) + "[%s,%s].dbId=%s" % (segment.start_time.isoformat(),
                                                          segment.end_time.isoformat(),
                                                          str(segment.id)) 


def get_inventory(segment, session=None, **kwargs):
    """raises tons of exceptions (see main). FIXME: write doc
    :param session: if **not** None but a valid sqlalchemy session object, then
    the inventory, if downloaded because not present, will be saveed to the db (compressed)
    """
    data = segment.channel.station.inventory_xml
    if not data:
        query_url = get_inventory_query(segment.channel.station)
        data = url_read(query_url, **kwargs)
        if session and data:
            segment.channel.station.inventory_xml = dumps_inv(data)
            session.commit()
        elif not data:
            raise ValueError("No data from server")
    return loads_inv(data)


def docalc(outcsvfile, sta_inv_required=True, sta_inv_save=False):
    inventories = {}
    sess = get_session()
    query = sess.query(models.Segment)
    seg_len = query.count()
    errlog = StringIO()
    sys.stdout.write("Processing, please wait\n")
    sys.stdout.write("Output file: '%s'\n" % outcsvfile)
    with open(outcsvfile, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with progressbar(length=seg_len) as pbar:
            for seg in query:  # http://stackoverflow.com/questions/1078383/sqlalchemy-difference-between-query-and-query-all-in-for-loops
                if sta_inv_required:
                    inv_obj = inventories.get(seg.channel.station.id,
                                              sess if sta_inv_save else None)
                    if inv_obj is None:
                        try:
                            inv_obj = get_inventory(seg)
                        except Exception as exc:
                            errlog.write("ERROR downloading inventory: %s\n" % str(exc))
                            continue
                        inventories[seg.channel.station.id] = inv_obj
                try:
                    array = process_segment(seg, inv_obj)
                    if array is not None:
                        csvwriter.writerow([str(seg.channel.id), seg.start_time.isoformat(),
                                            seg.end_time.isoformat()] + array.tolist())
                except Exception as exc:
                    errlog.write("ERROR processing segment '%s': %s\n" % (getid(seg), str(exc)))
                pbar.update(1)

    sys.stderr.write(errlog.getvalue())

station_inv_required = True  # True or False if station inventory must be used for calculations
station_inv_save = False  # if the above is False, it is ignored. Otherwise, tells whether we should save non-saved inventories
arrival_time_delay = 0
amp_ratio_threshold = 0.8
bandpass_freq_max = 20  # the max frequency, in Hz
bandpass_max_nyquist_ratio = 0.9  # the amount of freq_max to be taken. low-pass corner = max_nyquist_ratio * freq_max (defined above)
bandpass_corners = 2  # the corners
# the window length (in seconds). There will be two spectra, one calculated on
# AT-window_length (noisy signal), the other on AT+window_length ('normal' signal):
snr_window_length = 60  # snr_fixedwindow_in_sec
remove_response_water_level = 60
remove_response_output = 'ACC'  # or 'VEL', 'DISP'
taper_max_percentage = 0.05  # the taper percentage used when tapering (applied on any tapered object)
freqs_interp = [0.1, 0.106365, 0.113136, 0.120337, 0.127997, 0.136145, 0.144811, 0.154028,
                0.163833, 0.174261, 0.185354, 0.197152, 0.209701, 0.22305, 0.237248, 0.252349,
                0.268412, 0.285497, 0.30367, 0.323, 0.34356, 0.365429, 0.388689, 0.413431,
                0.439747, 0.467739, 0.497512, 0.52918, 0.562864, 0.598692, 0.636801, 0.677336,
                0.72045, 0.766309, 0.815088, 0.866971, 0.922156, 0.980855, 1.04329, 1.1097,
                1.18033, 1.25547, 1.33538, 1.42038, 1.5108, 1.60696, 1.70925, 1.81805, 1.93378,
                2.05687, 2.18779, 2.32705, 2.47518, 2.63273, 2.80031, 2.97856, 3.16816, 3.36982,
                3.58432, 3.81248, 4.05516, 4.31328, 4.58784, 4.87987, 5.19049, 5.52088, 5.8723,
                6.24609, 6.64368, 7.06657, 7.51638, 7.99483, 8.50372, 9.04501, 9.62076, 10.2332,
                10.8845, 11.5774, 12.3143, 13.0982, 13.9319, 14.8187, 15.762, 16.7653, 17.8324,
                18.9675, 20.1749, 21.4591, 22.825, 24.2779, 25.8233, 27.467, 29.2154, 31.075,
                33.0531, 35.157, 37.3949, 39.7752, 42.307, 45.]


def process_segment(seg, station_inventory=None):
    if not seg.data:
        raise ValueError('empty data')

    mseed = read(StringIO(seg.data))

    if get_gaps(mseed):
        raise ValueError('has gaps')

    if len(mseed) != 1:
        raise ValueError('more than one obspy.Trace')
        # raise ValueError("Mseed has more than one Trace")

    # work on the trace now. All functions will return Traces or scalars, which is better
    # so we can write them to database more easily
    mseed = mseed[0]

    ampratio = amp_ratio(mseed)
    if ampratio >= amp_ratio_threshold:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # convert to UTCDateTime for operations later:
    a_time = UTCDateTime(seg.arrival_time) + arrival_time_delay

    evt = seg.event
    mseed = bandpass(mseed, evt.magnitude, freq_max=bandpass_freq_max,
                     max_nyquist_ratio=bandpass_max_nyquist_ratio, corners=bandpass_corners)

    mseed_rem_resp = remove_response(mseed, station_inventory, output=remove_response_output,
                                     water_level=remove_response_water_level)

    # to calculate cumulative:
    # mseed_cum = cumsum(mseed_rem_resp)
    # and then:
    # t005, t010, t025, t050, t075, t90, t95 = cumtimes(mseed_cum, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    # then, for instance:
    # mseed_rem_resp_t05_t95 = mseed_rem_resp.slice(t05, t95)

    # t_PGA, PGA = maxabs(mseed_rem_resp, a_time, t95)  # if remove_response_output == 'ACC'
    # t_PGV, PGV = maxabs(mseed_rem_resp, a_time, t95)  # if remove_response_output = 'VEL'

    fft_rem_resp = fft(mseed_rem_resp, a_time, snr_window_length,
                       taper_max_percentage=taper_max_percentage)
    aspec = amp_spec(fft_rem_resp.data, True)

    return np.interp(freqs_interp, freqs(aspec, dfreq(fft_rem_resp)), aspec)


@click.command()
@click.argument('outfile')
def main(outfile):
    docalc(outfile, station_inv_required, station_inv_save)
    
    
if __name__ == '__main__':
    main()   # pylint: disable=E1120
