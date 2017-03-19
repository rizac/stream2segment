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
from stream2segment.process.utils import strof, dcname
import json

warnings.filterwarnings("ignore")


def get_stats(outfile):
    sess = get_session()
    # defaultdicts of errors/ warnings. Basically the allow to be default
    # in two levels: defaultdict[key1][key2] += 1
    errors = defaultdict(lambda: defaultdict(list))
    warnings = defaultdict(lambda: defaultdict(list))
    # stats is a dict of channel (or station ids) mapped to a list [good, total]
    stats_df = pd.DataFrame(columns=['id', 'lat', 'lon'],
                            data=sess.query(Station.id, Station.latitude, Station.longitude).all())
    stats_df.set_index('id', inplace=True)
    # http://stackoverflow.com/questions/10457584/redefining-the-index-in-a-pandas-dataframe-object
    stats_df.index.name = None
    stats_df['total'] = 0.0
    stats_df['dcen'] = ''

    try:
        segments = sess.query(Segment)
        length = segments.count()
        with progressbar(length=length) as bar:
            for seg in segments:
                bar.update(1)
                sta_id = seg.station.id
                data = seg.data
                stats_df.loc[sta_id, 'total'] += 1
                stats_df.loc[sta_id, 'dcen'] = dcname(seg.datacenter)
                warn, err = None, None
                if data is None:
                    warn = 'HTTP client/server error'
                elif not data:
                    warn = 'HTTP response: empty data (0 bytes received)'
                else:
                    try:
                        mseed = read(StringIO(data))
                        if len(mseed) > 1:  # do this check before cause get_gaps
                            # might take more time
                            gaps = mseed.get_gaps()
                            # From the docs: The returned list contains one item in the
                            # following form for each gap/overlap:
                            # [network, station, location, channel, starttime of the gap,
                            # end time of the gap, duration of the gap,
                            # number of missing samples]
                            if gaps:
                                if any(g[-1] < 0 for g in gaps):
                                    warn = 'Has overlaps'
                                else:
                                    warn = 'Has gaps'
                            else:
                                # this should never happen, for safety:
                                warn = 'No gaps but num. of traces > 1'
                        else:
                            # surely only one trace:
                            trace = mseed[0]
                            if trace.stats.endtime <= trace.stats.starttime:
                                warn = 'endtime <= starttime'
                            else:
                                srate_real = trace.stats.sampling_rate
                                srate_nominal = seg.channel.sample_rate
                                if srate_real != srate_nominal:
                                    warn = 'sample rate != channel sample rate'

                    except Exception as exc:
                        warn = None
                        err = exc.__class__.__name__ + ": " + str(exc)

                if warn:
                    if warn not in stats_df:
                        stats_df[warn] = 0
                    stats_df.loc[sta_id, warn] += 1
                    warnings[sta_id][warn].append(strof(seg))
                elif err:
                    if err not in stats_df:
                        stats_df[err] = 0
                    stats_df.loc[sta_id, err] += 1
                    errors[sta_id][err].append(strof(seg))

            # drop stations which did not have segments (note that we write when segments
            # are null or empty, so stations without segments should in principle be there for
            # some weird reason and we do not consider them):
            stats_df = stats_df[stats_df['total'] > 0]
            text = render(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "downloadstats.html"), {'stations': stats_df.to_json(orient='split'),
                                                  'warnerrs_labels': stats_df.columns.values.tolist()[4:],
                                                  'warnings': json.dumps(warnings),
                                                  'errors': json.dumps(errors),
                                                  'minlat': stats_df['lat'].min(),
                                                  'minlon': stats_df['lon'].min(),
                                                  'maxlat': stats_df['lat'].max(),
                                                  'maxlon': stats_df['lon'].max()})

            with open(outfile, 'w') as opn:
                opn.write(text)

    finally:
        sess.close()


# def datagen(dfr, count_df, id2latlon, warnings, errors):
#     for ch_id in dfr.columns:
#         count = count_df[ch_id].sum(skipna=True)
#         if ~pd.isnull(count) and ~pd.isnull(dfr[ch_id]).all():
#             lat, lon = id2latlon[ch_id][0], id2latlon[ch_id][1]
#             try:
#                 filt = count_df[ch_id] > 0
#                 value = get_mean(dfr[ch_id][filt] / count_df[ch_id][filt])
#             except ZeroDivisionError:
#                 dfg = 9
#             value = int(round(100*value))/100.0
#             haserr = ch_id in errors
#             haswarn = ch_id in warnings
#             warn = "<br>".join("%s: %s mseed" % (k, str(v))
#                                for k, v in warnings[ch_id].iteritems()) if ch_id in warnings else ""
#             err = "<br>".join("%s: %s mseed" % (k, str(v))
#                               for k, v in errors[ch_id].iteritems()) if haserr else ""
#             yield {'lat': lat, 'name': ch_id, 'value':  value,
#                    'lon': lon, 'fill_color': get_color(value, haswarn, haserr),
#                    "count": count, "warn": warn, "err": err}


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

# def get_color(val, haswarn, haserr):
# #     if haserr:
# #         return "#ff0000"
#     formatstr = "#%02x%02xff" if haswarn else "#ff%02x%02x" if haserr else "#%02xff%02x"
#     val = int(round(255 * (1-val)))
#     return formatstr % (val, val)


if __name__ == '__main__':
    get_stats('/Users/riccardo/work/gfz/data/dec_2016.1000.sqlite.stations.html')
    pass