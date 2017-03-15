'''
Created on Jul 31, 2016

@author: riccardo
'''
from stream2segment.io.db.pd_sql_utils import colnames, commit, withdata
from stream2segment.io.db.models import Segment, Class, Station, Channel, DataCenter, Event,\
    ClassLabelling
from stream2segment.utils.sqlevalexpr import query, get_columns
from stream2segment.gui.webapp import get_session
from stream2segment.gui.webapp.plots import View, jsontimestamp, set_spectra_config
from stream2segment.utils import evalexpr


NPTS_WIDE = 900
NPTS_SHORT = 900
# FIXME: automatic retrieve by means of Segment class relationships?
METADATA = [("", Segment), ("event", Event), ("channel", Channel),
            ("station", Station), ("datacenter", DataCenter)]


SETTINGS = {
    'spectra': {'arrivalTimeShift': 0, 'signalWindow': [5, 95]}
    }


def get_init_data(orderby, withdataonly):
    classes = get_classes()
    metadata = create_metadata_list(True if classes else None)
    return {'segment_ids': get_segment_ids(orderby=orderby, withdataonly=withdataonly),
            'classes': classes,
            'metadata': metadata}


def create_metadata_list(has_classes):
    ret = []
    exclude = set([Station.inventory_xml.key, Segment.data.key])
    for prefix, model in METADATA:
        colz = colnames(model, fkey=False)
        for c in colz:
            if c in exclude:
                continue
            column = getattr(model, c)
            try:
                typename = type2str(column.type.python_type)
            except:
                continue
            ret.append([("%s." % prefix) + c if prefix else c, typename])

    if has_classes:
        ret.insert(0, ['classes.id', type2str(Class.id.type.python_type) +  # @UndefinedVariable
                       ". Type 'unset' to match unlabeled segments only"])

    return ret


def type2str(python_type):
    return python_type.__name__


def get_segment_ids(filterdata=None, orderby=None, withdataonly=True):
    # first parse separately the classes.id 'unset'
    additional_atts = []
    if filterdata:
        val = filterdata.get('classes.id', None)
        if val and val.strip() in ('unset', '"unset"', "'unset'"):
            filterdata.pop('classes.id')
            additional_atts.append(~Segment.classes.any())  # @UndefinedVariable
    if withdataonly:
        additional_atts.append(withdata(Segment.data))
    # use group by to remove possibly duplicates, especially if we query classes if which is
    # a many to many relationship. For info see:
    # http://stackoverflow.com/questions/23786401/why-do-multiple-table-joins-produce-duplicate-rows
    qry = query(get_session(), Segment.id, filterdata, orderby,
                *additional_atts).group_by(Segment.id)
    return [seg[0] for seg in qry]


def get_classes():
    session = get_session()
    clazzes = session.query(Class).all()
    ret = []
    colz = list(colnames(Class))
    for c in clazzes:
        row = {}
        for col in colz:
            row[col] = getattr(c, col)
        row['count'] = session.query(ClassLabelling).\
            filter(ClassLabelling.class_id == c.id).count()
        ret.append(row)
    return ret


def toggle_class_id(segment_id, class_id):
    session = get_session()
    elms = session.query(ClassLabelling).\
        filter((ClassLabelling.class_id == class_id) &
               (ClassLabelling.segment_id == segment_id)).all()

    if elms:
        for elm in elms:
            session.delete(elm)
    else:
        sca = ClassLabelling(class_id=class_id, segment_id=segment_id, is_hand_labelled=True)
        session. add(sca)

    commit(session)

    # re-query the database to be sure:
    segm = session.query(Segment).filter(Segment.id == segment_id).first()
    return {'classes': get_classes(),
            'segment_class_ids': [] if not segm else [c.id for c in segm.classes]}


def get_segment_data(seg_id, filtered, zooms, indices, metadata_keys=None):
    session = get_session()
    segment = session.query(Segment).filter(Segment.id == seg_id).first()
    plots = []
    spectra_wdws = []
    if indices:
        view = View.get(segment, filtered)
        plots = [view[i] for i in indices]
        if set(indices).intersection([0, 1]):
            spectra_wdws = [sorted([jsontimestamp(x[0]), jsontimestamp(x[0]) + 1000*x[1]])
                            for x in view.get_spectra_windows()]

#     ret = [Plot.fromsegment(seg, zooms[0], NPTS_WIDE, filtered, copy=True)]
# 
#     for i, seg_ in enumerate(itercomponents(seg, session), 1):
#         if i > 2:
#             break
#         ret.append(Plot.fromsegment(seg_, zooms[i], NPTS_WIDE, False, False))
# 
#     try:
#         trace = Traces.get(seg, filtered, raise_on_warning=True)
#         cumtrace = cumsum(trace)
#         ret.append(Plot.fromtrace_spectra(trace, zooms[3], NPTS_SHORT, seg.arrival_time,
#                                           cumulative=cumtrace))
#         ret.append(Plot.fromtrace(cumtrace, zooms[4], NPTS_SHORT,
#                                   title=trace.get_id() + " - Cumulative"))
#     except Exception as exc:  # @UnusedVariable
#         ret.append(Plot())
#         ret.append(Plot())

    return {'plotData': [r.tojson(zooms[i], NPTS_WIDE) for i, r in enumerate(plots)],
            'spectra_windows': spectra_wdws,
            'metadata': get_columns(segment, metadata_keys) if metadata_keys else {}}


def config_spectra(dic):
    ret = {'arrival_time_shift': evalexpr._eval(dic['arrival_time_shift']),
           'signal_window': evalexpr._eval(dic['signal_window'])
           }
    set_spectra_config(**ret)
    return True
