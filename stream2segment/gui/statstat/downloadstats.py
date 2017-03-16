'''
Created on Dec 21, 2016

@author: riccardo
'''
import os
import pandas as pd
import numpy as np
import jinja2
from cStringIO import StringIO
from stream2segment.utils import get_session
from stream2segment.io.db.models import Channel, Segment, Station
from stream2segment.io.utils import loads
from click import progressbar
from obspy.io.mseed.core import InternalMSEEDReadingError

import warnings
from collections import defaultdict
from obspy.core.stream import read

warnings.filterwarnings("ignore")


def get_stats():
    sess = get_session()
    # set the bounds for the map:
    minlat = minlon = maxlat = maxlon = None
    # defaultdicts of errors/ warnings. Basically the allow to be default
    # in two levels: defaultdict[key1][key2] += 1
    errors = defaultdict(lambda: defaultdict(int))
    warnings = defaultdict(lambda: defaultdict(int))
    # stats is a dict of channel (or station ids) mapped to a list [good, total]
    stats_df = pd.DataFrame(columns=['id', 'lat', 'lon'],
                            data=sess.query(Station.id, Station.latitude, Station.longitude).all())
    stats_df.set_index('id')
    stats_df['ok'] = 0.0
    stats_df['total'] = 0.0

    try:
        segments = sess.query(Segment)
        with progressbar(length=segments.count()) as bar:
            for seg in segments:
                bar.update(1)
                sta_id = seg.station.id
                data = seg.data
                percentgood = 0
                if data:
                    try:
                        mseed = read(StringIO(data))
                        total_pts = sum(trace.stats.npts for trace in mseed)
                        if total_pts > 0:
                            missing_samples = 0
                            for g in mseed.get_gaps():
                                missing = g[-1]
                                if missing < 0:
                                    warnings[ch_id]['negative missing points'] += 1
                                    missing = -missing
                                missing_samples += missing

                            percentgood = 1 - np.true_divide(missing_samples, total_pts)
                            if percentgood < 0:
                                warnings[ch_id]['missing points greater than total npts'] += 1
                                percentgood = 0

                            for t in mseed:
                                if t.stats.endtime <= t.stats.starttime:
                                    warnings[ch_id]['endtime lower or equal than starttime'] += 1
                                if t.stats.sampling_rate < cha_sample_rate:
                                    warnings[ch_id][('mseed s.rate (real): %d Hz, '
                                                     'channel s.rate (nominal) : %d Hz')
                                                    % (t.stats.sampling_rate, cha_sample_rate)] += 1
                    except InternalMSEEDReadingError as exc:
                        errors[ch_id][exc.__class__.__name__ + ": " + str(exc)] += 1

                count.loc[ch_id, evt_id] += 1
                if pd.isnull(sumsampleratios.loc[ch_id, evt_id]):
                    sumsampleratios.loc[ch_id, evt_id] = percentgood
                else:
                    sumsampleratios.loc[ch_id, evt_id] += percentgood

            # count[count == 0] = 1  # so that division below works
            # d1 = sumsampleratios / count
            d1 = sumsampleratios.T
            count = count.T

            gen = datagen(d1, count, tolatlon, warnings, errors)

            text = render(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "downloadstats.html"), {'stations': gen, 'minlat': minlat,
                                                  'minlon': minlon,
                                                  'maxlat': maxlat, 'maxlon': maxlon})

            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "data_aval_dec_2016.html"), 'w') as opn:
                opn.write(text)

    finally:
        sess.close()


def datagen(dfr, count_df, id2latlon, warnings, errors):
    for ch_id in dfr.columns:
        count = count_df[ch_id].sum(skipna=True)
        if ~pd.isnull(count) and ~pd.isnull(dfr[ch_id]).all():
            lat, lon = id2latlon[ch_id][0], id2latlon[ch_id][1]
            try:
                filt = count_df[ch_id] > 0
                value = get_mean(dfr[ch_id][filt] / count_df[ch_id][filt])
            except ZeroDivisionError:
                dfg = 9
            value = int(round(100*value))/100.0
            haserr = ch_id in errors
            haswarn = ch_id in warnings
            warn = "<br>".join("%s: %s mseed" % (k, str(v))
                               for k, v in warnings[ch_id].iteritems()) if ch_id in warnings else ""
            err = "<br>".join("%s: %s mseed" % (k, str(v))
                              for k, v in errors[ch_id].iteritems()) if haserr else ""
            yield {'lat': lat, 'name': ch_id, 'value':  value,
                   'lon': lon, 'fill_color': get_color(value, haswarn, haserr),
                   "count": count, "warn": warn, "err": err}


def render(tpl_path, args_dict):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(loader=jinja2.FileSystemLoader(path or './')).\
        get_template(filename).render(args_dict)


def get_mean(series):
    val = np.mean(series[~pd.isnull(series)])  # this works
    # nanmean = np.nanmean(series.values) # this doesn't .FIXME: why?
    
#     if val > 0 and val < 1:
#         print "M: " + str(val)
    return val

def get_color(val, haswarn, haserr):
#     if haserr:
#         return "#ff0000"
    formatstr = "#%02x%02xff" if haswarn else "#ff%02x%02x" if haserr else "#%02xff%02x"
    val = int(round(255 * (1-val)))
    return formatstr % (val, val)


if __name__ == '__main__':
    get_stats()
    pass