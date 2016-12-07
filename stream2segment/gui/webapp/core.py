'''
Created on Jul 31, 2016

@author: riccardo
'''
from stream2segment.utils import get_session
# from flask import 
from stream2segment.io.db.models import Segment, Processing, Event, Station, Channel,\
    DataCenter, Run, SegmentClassAssociation, Class
# from stream2segment.classification import class_labels_df

import numpy as np
# from numpy import interp
from stream2segment.analysis.mseeds import cumsum, env, bandpass, amp_ratio,\
    cumtimes, interpolate, dfreq
from stream2segment.io.dataseries import loads    
# from stream2segment.classification.handlabelling import ClassAnnotator
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing as kos
from itertools import izip
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
import yaml
from stream2segment.io.db.pd_sql_utils import get_cols, get_col_names, commit
from sqlalchemy.sql.sqltypes import Binary, DateTime
from stream2segment.analysis import amp_spec, freqs, interp as analysis_interp
from stream2segment.main import load_def_cfg


def _get_session(app):
    # maybe not nicest way to store the session, but we want to avoid flask-sqlalchemy
    # for such a simple app
    key = '__DBSESSION__'
    if not app.config.get(key, None):
        sess = get_session(app.config['DATABASE_URI'])
        app.config[key] = sess

    return app.config[key]


def get_ids(session):
    segs = session.query(Segment).all()
    segs = [seg.id for seg in segs]
    return {'segment_ids': segs}


def get_classes(session):
    clazzes = session.query(Class).all()
    ret = []
    colz = get_col_names(Class)
    for c in clazzes:
        row = {}
        for col in colz:
            row[col] = getattr(c, col)
        row['count'] = session.query(SegmentClassAssociation).\
            filter(SegmentClassAssociation.class_id == c.id).count()
        ret.append(row)
    return ret


def toggle_class_id(session, segment_id, class_id):

    elms = session.query(SegmentClassAssociation).\
        filter((SegmentClassAssociation.class_id == class_id) &
               (SegmentClassAssociation.segment_id == segment_id)).all()

    if elms:
        for elm in elms:
            session.delete(elm)
    else:
        sca = SegmentClassAssociation(class_id=class_id, segment_id=segment_id,
                                      class_id_hand_labelled=True)
        session. add(sca)

    commit(session)

    return {'classes': get_classes(session),
            'segment_class_ids': get_segment_classes(session, segment_id)}


def get_data(session, seg_id):
    seg = session.query(Segment).filter(Segment.id == seg_id).first()
#     stream = loads(seg.data) if seg else Stream(traces=[Trace(np.array([]),
#                                                               header={'id': 'Not found'})]*3)

    # FIXME: better query with relationships possible!
    segs = []
    if seg:
        segs = [seg]
        # get same segments same station:
        same_channels_rows = session.query(Segment).join(Channel).join(Station).\
            filter((Station.id == seg.channel.station.id) &
                   (Channel.location == seg.channel.location) &
                   (Segment.start_time == seg.start_time) &
                   (Segment.end_time == seg.end_time)).all()
        # get those with only third component different
        for sss in same_channels_rows:
            if sss.id != seg_id and sss.channel.channel[:2] == seg.channel.channel[:2]:
                segs.append(sss)

        segs += [None] * max(0, 3 - len(segs))  # pad with Nones

    # Get processings: we have currently one processings per segment, but we might have more
    # in the future. So take last one. this process is cumbersome we might use database query more
    # clearly and efficiently:
    pros = []
    for sss in segs:
        pro = None
        if sss is not None:
            for pro_ in sss.processings:
                if pro is None or pro.run.run_id < pro_.run.run_id:
                    pro = pro_
        pros.append(pro)

    # define interpolation values
    MAX_NUM_PTS_TIMESCALE = 1050
    MAX_NUM_PTS_FREQSCALE = 215

    filtered_stream = tostream(pros, 'mseed_rem_resp_savewindow')
    times, filtered_stream = interpolate(filtered_stream, MAX_NUM_PTS_TIMESCALE,
                                         align_if_stream=True, return_x_array=True)
    stream = tostream(segs, 'data')
    _, stream = interpolate(stream, times,
                            align_if_stream=True, return_x_array=True)

    cumulative_trace = tostream(pros[0], 'cum_rem_resp')
    cumulative_trace = interpolate(cumulative_trace, times)
    evlp_trace = env(filtered_stream[0])
    evlp_trace = interpolate(evlp_trace, times)

#     snr_stream = [[], []] if not pros[0] else \
#         [loads(pros[0].fft_rem_resp_until_atime), loads(pros[0].fft_rem_resp_t05_t95)]

    time_data = {'labels': tojson(np.round(times * 1000.0)), 'datasets': []}
    datasets = time_data['datasets']

    # create datasets for chart.js:
    title = segs[0].channel.id
    datasets.append(to_chart_dataset(stream[0].data, title))
    datasets.append(to_chart_dataset(filtered_stream[0].data, title + " (Rem.resp+filtered)"))
    datasets.append(to_chart_dataset(stream[1].data, stream[1].id))
    datasets.append(to_chart_dataset(filtered_stream[1].data, stream[1].id +
                                     " (Rem.resp+filtered)"))
    datasets.append(to_chart_dataset(stream[2].data, stream[2].id))
    datasets.append(to_chart_dataset(filtered_stream[2].data, stream[2].id +
                                     " (Rem.resp+filtered)"))

    # cumulative:
    datasets.append(to_chart_dataset(cumulative_trace.data, title +
                                     " (Cumulative Rem.resp+filtered)"))
    # envelope
    datasets.append(to_chart_dataset(evlp_trace.data, title +
                                     " (Envelope Rem.resp+filtered)"))

    noisy_trace, sig_trace = (Trace(data=np.array([])), Trace(data=np.array([]))) \
        if not pros[0] else \
            (loads(pros[0].fft_rem_resp_until_atime)[0], loads(pros[0].fft_rem_resp_t05_t95)[0])

    noisy_trace.data = amp_spec(noisy_trace.data, signal_is_fft=True)
    sig_trace.data = amp_spec(sig_trace.data, signal_is_fft=True)

    freqz = freqs(noisy_trace.data, dfreq(noisy_trace))

    # interpolate (less pts):
    newfreqz, noisy_trace.data = analysis_interp(MAX_NUM_PTS_FREQSCALE, freqz, noisy_trace.data)
    _, sig_trace.data = analysis_interp(newfreqz, freqz, sig_trace.data)
    freqz = newfreqz

    freqs_log = np.log10(freqz[1:])
    freq_data = {'datasets': []}
    datasets = freq_data['datasets']
    # smooth signal:
    bwd = 100
    noisy_amps = kos(noisy_trace.data, freqz, bandwidth=bwd)
    sig_amps = kos(sig_trace.data, freqz, bandwidth=bwd)
    datasets.append(to_chart_dataset(noisy_amps[1:], title +
                                     " (Noise Rem.resp+filtered)", freqs_log))
    datasets.append(to_chart_dataset(sig_amps[1:], title +
                                     " (Signal Rem.resp+filtered)", freqs_log))

    seg = segs[0]
    metadata = []
    metadata.append(["EVENT", ''])
    metadata.append(["magnitude", seg.event.magnitude])
    metadata.append(["CHANNEL", ''])
    metadata.append(["id", seg.channel.id])
    metadata.append(["sample_rate", seg.channel.sample_rate])
    for title, instance in [('SEGMENT', seg), ('PROCESSING', pros[0])]:
        metadata.append([title, ''])
        for c in instance.__table__.columns:
            if isinstance(c.type, Binary) or len(c.foreign_keys):
                # len(c.foreign_keys) tells if c is a fkey: found no nicer way to tell it
                continue
            val = getattr(instance, c.key)
            if isinstance(c.type, DateTime):
                val = None if val is None else UTCDateTime(val)
            metadata.append([c.key, val])

    # load config        
    config = yaml.safe_load(seg.run.config)

    # before addding config add custom data (processing):
    metadata.append(['arrival_time (+ config delay)', UTCDateTime(seg.arrival_time) +
                     config['processing']['arrival_time_delay']])

    metadata.append(['noise_fft_start', (UTCDateTime(seg.arrival_time) +
                     config['processing']['arrival_time_delay']) -
                     (UTCDateTime(pros[0].cum_t95) - UTCDateTime(pros[0].cum_t05))])

    # metadata.append(("", ""))

    # We use safe_dump now to avoid python types. However, fix the unicode issue. See here:
    # http://stackoverflow.com/questions/27518976/how-can-i-get-pyyaml-safe-load-to-handle-python-unicode-tag
    # commented out: ANY PYTHON SPECIFIC STUFF IS UNREDABLE. Modified read funtion
#     yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/unicode",
#                                     lambda loader, node: node.value)
    dicts = ['CONFIG', config]
    while len(dicts):
        title, dct = dicts.pop(0), dicts.pop(0)
        metadata.append([title, ""])
        for key, val in dct.iteritems():
            if isinstance(val, dict):
                dicts += [(title+"."+key).upper(), val]
                continue
            else:
                metadata.append([key, val])

    # metadata.append(["Config", seg.run.config])  # .replace("\n", "<br>")))

    # convert datetimes for moment.js:
    for val in metadata:
        if isinstance(val[1], UTCDateTime):
            momentjs_timestamp = round(val[1].timestamp * 1000)
            if "start_date" in val[0]:
                val[1] = "[__DATE__]%s" % str(momentjs_timestamp)
            else:
                val[1] = "[__TIME__]%s" % str(momentjs_timestamp)


    class_ids = get_segment_classes(session, segs[0].id) 

    return {'time_data': time_data, 'freq_data': freq_data, 'metadata': metadata,
            'class_ids': class_ids}


def get_segment_classes(session, segment_id):
    class_ids = session.query(SegmentClassAssociation).\
        filter(SegmentClassAssociation.segment_id == segment_id).all()

    return [c.class_id for c in class_ids]


def tostream(model_instances, attr):
    """
        Returns a Stream from each getattr(model_instance, attr), where model_instance is in turn
        each element of model_instances (an ORM model instance)

        Returns a Trace if model_instance is a single model instance
    """
    traces = []
    ret_stream = True
    if not hasattr(model_instances, "__iter__"):
        model_instances = [model_instances]
        ret_stream = False

    for mis in model_instances:
        if not mis:
            tra = Trace(np.array([]), header={'id': 'NotFound'})
        else:
            data = getattr(mis, attr)
            tra = loads(data).traces[0]
            if isinstance(tra, Stream) and len(tra) == 1:
                # usually it's a single trace Stream (implementation choice)
                tra = tra[0]

        traces.append(tra)

    return Stream(traces) if ret_stream else traces[0]


def to_chart_dataset(np_array_y, title=None, np_array_x=None):
    """Converts the array to a dataset dict usable for Chart.js
        IF title is NOT NONE. Otherwise, converts array to a json serializable list
        :param array: a numpy array
    """

    if np_array_y is None:
        array = []
    else:
        if np_array_x is not None:
            array = []
            for x, y in izip(np_array_x, np_array_y):
                array.append({'x': x.item(), 'y': y.item()})
        else:
            array = np_array_y.tolist()
    return {'label': title, 'data': array}


def tojson(array):
    """
        :return: array.tolist()
    """
    return array.tolist()


def to_chart_data(np_xvalues, chart_datasets_list):
    """Converts the array to a dataset dict usable for Chart.js
        IF title is NOT NONE. Otherwise, converts array to a json serializable list
        :param array: a numpy array
    """
    return {'labels': tojson(np_xvalues), 'datasets': chart_datasets_list}


# simple script that converted badly formatte config to safe_dumps / safe_loads configs

# if __name__ ==  "__main__":
#     cfg = load_def_cfg()
#     session = get_session(cfg['dburi'])
#     runs = session.query(Run).all()
#     import re
#     reg1 = re.compile("^dbpath\\:", re.MULTILINE)
#     reg2 = re.compile("^processing_args_dict\\:", re.MULTILINE)
#     for r in runs:
#         cfg = r.config
#         newconfig = reg1.sub("dburi: ", cfg)
#         newconfig = reg2.sub("processing: ", newconfig)
#         yaml.load(newconfig)
#         r.config = newconfig
#     commit(session)
    