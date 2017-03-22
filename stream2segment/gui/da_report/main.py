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
from stream2segment.io.db.models import Channel, Segment, Station, DataCenter
from stream2segment.io.utils import loads
from click import progressbar
from obspy.io.mseed.core import InternalMSEEDReadingError

import warnings
from collections import defaultdict
from obspy.core.stream import read
from stream2segment.process.utils import dcname, segstr
import json
import concurrent.futures
from urlparse import urlparse

warnings.filterwarnings("ignore")


def segquery(session):
    return session.query(Segment.channel_id, Segment.start_time, Segment.end_time, Segment.id,
                         Segment.data, Channel.sample_rate,
                         DataCenter.station_query_url).join(Segment.datacenter, Segment.channel)


def process_segment(segqueryargs):
    """Returns stats for the given segment"""
    data = segqueryargs[4]
    channel_sample_rate = segqueryargs[5]
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
                    srate_nominal = channel_sample_rate
                    if srate_real != srate_nominal:
                        warn = 'sample rate != channel sample rate'

        except Exception as exc:
            # some semgnet might raise a obspy.io.mseed.core.InternalMSEEDReadingError
            # however, catch a broad exception cause we will anyway print it and handle it
            warn = None
            err = exc.__class__.__name__ + ": " + str(exc)
    return warn, err, segstr(*segqueryargs[:4]), ".".join(segqueryargs[0].split(".")[:2]),\
        urlparse(segqueryargs[6]).netloc


def create_da_html(sess, outfile, isterminal=False):
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
    pbar = get_progressbar(isterminal)
    unexpected_errs = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = (executor.submit(process_segment, segdata) for segdata in seg_query)
        with pbar(length=seg_query.count()) as pb:
            for future in concurrent.futures.as_completed(futures):
                pb.update(1)
                try:
                    warn, err, segid_str, sta_id, dc_name = future.result()
                    stats_df.loc[sta_id, 'total'] += 1
                    if dc_name not in dc2idx:
                        dc2idx[dc_name] = len(dc2idx)
                    stats_df.loc[sta_id, 'dcen'] = dc2idx[dc_name]
                    if warn:
                        if warn not in warn2colidx:
                            stats_df[warn] = 0
                            warn2colidx[warn] = stats_df.columns.get_loc(warn)
                        stats_df.loc[sta_id, warn] += 1
                        warnings[sta_id][warn2colidx[warn]].append(segid_str)
                    elif err:
                        if err not in err2colidx:
                            stats_df[err] = 0
                            err2colidx[err] = stats_df.columns.get_loc(err)
                        stats_df.loc[sta_id, err] += 1
                        errors[sta_id][err2colidx[err]].append(segid_str)
                except:
                    unexpected_errs += 1

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
