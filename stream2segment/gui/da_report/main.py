'''
Creates a data availability report after download
Created on Dec 21, 2016

@author: riccardo
'''
import os
import pandas as pd
import jinja2
from cStringIO import StringIO
from stream2segment.utils import get_session, get_progressbar
from stream2segment.io.db.models import Channel, Segment, Station, DataCenter, Event
# from stream2segment.io.utils import loads
from click import progressbar
from obspy.io.mseed.core import InternalMSEEDReadingError

import warnings
from collections import defaultdict
from obspy.core.stream import read
from stream2segment.process.utils import dcname, segstr
import json
import concurrent.futures
from urlparse import urlparse
from sqlalchemy.orm import load_only
from datetime import datetime
from stream2segment.io.db.pd_sql_utils import withdata
from itertools import chain

warnings.filterwarnings("ignore")


def segquery(session):
    qry = session.query(Segment.channel_id, Segment.start_time,
                        Segment.end_time, Segment.id, Segment.data, Channel.sample_rate,
                        DataCenter.station_url).join(Segment.datacenter, Segment.channel)

    qry = session.query(Segment).options(load_only(Segment.id))
    # FIXME: remove!!
#     'event.latitude': "[47, 56]"
#   'event.longitude': "[5, 16]"
#   'event.time': "(1998-01-01T00:00:00, 2017-04-10T00:00:00)"
    qry = qry.join(Segment.event).filter((Event.latitude >=47) & (Event.latitude <=56) &
                                   (Event.longitude >=5) & (Event.longitude <=16) &
                                   (Event.time > datetime(1998,1,1)) &
                                   (Event.time < datetime(2017,4,1)))

    return qry


def process_segment(segment, max_gap_ovlap_ratio):
    """Returns stats for the given segment"""
#     seg_ch_id, seg_stime, seg_etime, seg_id, data, channel_sample_rate, datacenter_station_url =\
#         segqueryargs

    data = segment.data
    warn, err = None, None
    if not data:
        code = segment.download_status_code
        if code >= 400 and code < 500:
            warn = "4xx: client error"
        elif code >= 500:
            warn = "5xx: server error"
        elif code != 204:
            if code is None:
                warn = "Null code: unknown error"
            else:
                warn = "No data, response code != 204"
        else:
            warn = "204: No data"
    else:
        try:
            if segment.max_gap_ovlap_ratio >= max_gap_ovlap_ratio:
                warn = 'Max gap/overlap duration >= %f sampling period' % max_gap_ovlap_ratio
            else:
                mseed = read(StringIO(data))
                if len(mseed) > 1:  # do this check before cause get_gaps
                    warn = 'More than one trace (potential gaps/overlaps)'
#                     # might take more time
#                     gaps = segment.max_gap_ovlap_ratio >= mac_gap_ovlap_ratio
#                     # From the docs: The returned list contains one item in the
#                     # following form for each gap/overlap:
#                     # [network, station, location, channel, starttime of the gap,
#                     # end time of the gap, duration of the gap,
#                     # number of missing samples]
#                     if gaps:
#                         if any(g[-1] < 0 for g in gaps):
#                             warn = 'More than one trace'
#                         else:
#                             warn = 'Has gaps'
#                     else:
#                         # this should never happen, for safety:
#                         warn = 'No gaps but num. of traces > 1'
                else:
                    # surely only one trace:
                    trace = mseed[0]
                    if trace.stats.endtime <= trace.stats.starttime:
                        warn = 'endtime <= starttime'
                    else:
                        channel_sample_rate = segment.channel.sample_rate
                        srate_real = trace.stats.sampling_rate
                        srate_nominal = channel_sample_rate
                        if srate_real != srate_nominal:
                            warn = 'sample rate != channel sample rate'

        except Exception as exc:
            # some semgnet might raise a obspy.io.mseed.core.InternalMSEEDReadingError
            # however, catch a broad exception cause we will anyway print it and handle it
            warn = None
            err = exc.__class__.__name__ + ": " + str(exc)

    net, sta, loc, cha = segment.channel_id.split(".")
    seg_info_list = []
    if warn or err:
        # optimize data, as we might write a lot to the html. Do not repeat network and station
        # the rest of the optimization is done afterwards
        seg_info_list = [loc, cha, '', '', '', '', segment.id]
        sdate, stime = segment.start_time.isoformat().split('T')
        edate, etime = segment.end_time.isoformat().split('T')
        seg_info_list[2] = sdate
        seg_info_list[3] = stime
        seg_info_list[5] = etime
        if sdate != edate:
            seg_info_list[4] = edate

    return warn, err, net + "." + sta, urlparse(segment.datacenter.station_url).netloc,\
        seg_info_list


def create_da_html(sess, outfile, max_gap_ovlap_ratio=0.5, isterminal=False):
    """Creates an html file displaying interactive data availability statistics
    from a given session bound to a database with already downloaded data"""
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
    stats_df['total'] = 0
    stats_df['dcen'] = -1

    dc2idx = {}  # datacenter names to indices, this makes injected jinjs json dict lighter
    # providing at the end that we convert this dict to a list (see below datacenter_array)
    warn2colidx = {}  # warnings names to indices, same reason as above
    err2colidx = {}  # warnings names to indices, same reason as above

    seg_query = segquery(sess)

    unexpected_errs = 0

    with get_progressbar(isterminal, length=seg_query.count()) as pbar:
        for seg in seg_query:
            pbar.update(1)
            try:
                warn, err, sta_id, dc_name, seg_info_list = process_segment(seg,
                                                                            max_gap_ovlap_ratio)
                stats_df.loc[sta_id, 'total'] += 1
                if dc_name not in dc2idx:
                    dc2idx[dc_name] = len(dc2idx)
                stats_df.loc[sta_id, 'dcen'] = dc2idx[dc_name]
                if warn:
                    if warn not in warn2colidx:
                        stats_df[warn] = 0
                        warn2colidx[warn] = stats_df.columns.get_loc(warn)
                    stats_df.loc[sta_id, warn] += 1
                    warnings[sta_id][warn2colidx[warn]].append(seg_info_list)
                elif err:
                    if err not in err2colidx:
                        stats_df[err] = 0
                        err2colidx[err] = stats_df.columns.get_loc(err)
                    stats_df.loc[sta_id, err] += 1
                    errors[sta_id][err2colidx[err]].append(seg_info_list)
            except Exception as _:
                unexpected_errs += 1

    sess.close()  # for safety

    # drop stations which did not have segments (note that we write when segments
    # are null or empty, so stations without segments should in principle be there for
    # some weird reason and we do not consider them):
    stats_df = stats_df[stats_df['total'] > 0]

    # convert dtypes to integers except lat and lon
    # (seems that when doing += 1 it converts to float)
    # this saves size in the html
    stats_df[stats_df.columns[2:]] = stats_df[stats_df.columns[2:]].astype(int)

    # make datacenter array:
    dcen_array = [0] * len(dc2idx)
    for dc, idx in dc2idx.iteritems():
        dcen_array[idx] = str(dc)  # if k is unicode and this is py2, avoid u'whatever' in html

    text = render(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "da_report.html"), {'stations': stats_df.to_json(orient='split'),
                                      'datacenters': dcen_array,
                                      'warnerrs_labels': stats_df.columns.values.tolist()[4:],
                                      'warnings': json.dumps(warnings),
                                      'errors': json.dumps(errors),
                                      'minlat': stats_df['lat'].min(),
                                      'minlon': stats_df['lon'].min(),
                                      'maxlat': stats_df['lat'].max(),
                                      'maxlon': stats_df['lon'].max()})

    with open(outfile, 'w') as opn:
        opn.write(text)

    if isterminal and unexpected_errs:
        print "Warning: unexpected exceptions (%d segments skipped)" % unexpected_errs


def render(tpl_path, args_dict):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(loader=jinja2.FileSystemLoader(path or './')).\
        get_template(filename).render(args_dict)
