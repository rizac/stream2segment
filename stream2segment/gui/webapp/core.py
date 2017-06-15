'''
Created on Jul 31, 2016

@author: riccardo
'''
from stream2segment.io.db.pd_sql_utils import colnames, commit
from stream2segment.io.db.models import Segment, Class, Station, Channel, DataCenter, Event,\
    ClassLabelling, Run
# from stream2segment.io.db.sqlevalexpr import query, get_columns
from stream2segment.gui.webapp import get_session
from stream2segment.gui.webapp.plotviews import ViewManager, jsontimestamp  #, set_spectra_config
# from stream2segment.utils import evalexpr
# from sqlalchemy.sql.expression import and_
from stream2segment.io.db.queries import query4gui, count as query_count
from itertools import chain, cycle, izip


NPTS_WIDE = 900
NPTS_SHORT = 900
# FIXME: automatic retrieve by means of Segment class relationships?


def get_segments(conditions, orderby, metadata, classes):
    classes = get_classes() if classes else []
    metadata = get_metadata() if metadata else []
    qry = query4gui(get_session(), conditions=conditions, orderby=orderby)
    return {'segment_ids': [seg[0] for seg in qry],
            'classes': classes,
            'metadata': metadata}


def get_metadata(seg_id=None):
    '''Returns a list of tuples (column, column_type) if segment is None or
    (column, column_value) if segment is not None. In the first case, `column_type` is the
    string representation of the column python type (str, datetime,...), in the latter,
    it is the value of `segment` for that column'''
    session = get_session()
    if seg_id is not None:
        segment = session.query(Segment).filter(Segment.id == seg_id).first()  # FIXME: handle when not found!
        if not segment:
            return []

    METADATA = [("", Segment), ("event", Event), ("channel", Channel),
                ("station", Station), ("datacenter", DataCenter), ('run', Run)]

    def type2str(python_type):
        '''returns the str representation of a python type'''
        return python_type.__name__

    ret = []
    exclude = set([Station.inventory_xml.key, Segment.data.key])
    for prefix, model in METADATA:
        colnamez = colnames(model, fkey=False)
        for colname in colnamez:
            if colname in exclude:
                continue
            column = getattr(model, colname)
            try:
                if segment is not None:
                    if model is not Segment:
                        value = getattr(getattr(segment, prefix), colname)
                    else:
                        value = getattr(segment, colname)
                    try:
                        value = value.isoformat()
                    except AttributeError:
                        if value is None:
                            value = 'null'
                        else:
                            value = str(value)
                else:
                    value = type2str(column.type.python_type)
            except:
                continue
            ret.append([("%s." % prefix) + colname if prefix else colname, value])

    ret.insert(0, ['has_data', type2str(bool)])
    if query_count(session, Class.id) > 0:
        ret.insert(0, ['classes.id', type2str(Class.id.type.python_type)  # @UndefinedVariable
                       ])
        ret.insert(0, ['classes', type2str(str) +  # @UndefinedVariable
                       ": type either 'any' or 'none' (match segments with at least one class "
                       "set, or no class set, respectively)"])

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
        session.add(sca)

    commit(session)

    # re-query the database to be sure:
    return {'classes': get_classes(),
            'segment_class_ids': get_classes(segment_id)}


def set_classes(config):
    classes = config.get('class_labels', [])
    if not classes:
        return
    session = get_session()
    # do not add already added classes:
    clazzes = {c.label: c for c in session.query(Class)}
    for label, description in classes.iteritems():
        if label in clazzes and clazzes[label].description != description:
            cla = clazzes[label]
            cla.description = description  # update
        elif label not in clazzes:
            cla = Class(label=label, description=description)
        session.add(cla)
    session.commit()
    h = 9


def get_classes(seg_id=None):
    '''Write doc!! what does this return???'''
    session = get_session()
    if seg_id is not None:
        segment = session.query(Segment).filter(Segment.id == seg_id).first()
        return [] if not segment else [c.id for c in segment.classes]
    clazzes = session.query(Class).all()
    ret = []
    colz = list(colnames(Class))
    for c in clazzes:
        row = {}
        for col in colz:
            row[col] = getattr(c, col)
        row['count'] = session.query(ClassLabelling).\
            filter(ClassLabelling.class_id == c.id).count()  # FIX COUNT AS FUNC COUNT
        ret.append(row)
    return ret


def get_segment_data(seg_id, plotmanager, plot_indices, all_components, filtered, zooms,
                     metadata=False, classes=False):
    session = get_session()
    # segment = session.query(Segment).filter(Segment.id == seg_id).first()
    plots = []
    if not zooms:
        zooms = []
    spectra_wdws = []
    if plot_indices:
        zooms = chain(zooms, cycle([None]))  # set zooms to None if length is not enough
        get_plots_func = plotmanager.getfplots if filtered else plotmanager.getplots
        plots = get_plots_func(session, seg_id, all_components, *plot_indices)
#         if set(indices).intersection([0, 1]):  # if I am trying to update either spectra or
#             # base trace, also send spectra time windows:
#             spectra_wdws = [sorted([jsontimestamp(x[0]), jsontimestamp(x[0]) + 1000*x[1]])
#                             for x in plotmanager.get_spectra_windows(session, seg_id)]

    return {'plots': [p.tojson(z, NPTS_WIDE) for p, z in izip(plots, zooms)],
            # 'spectra_windows': spectra_wdws,
            # 'conf': plotmanager.config,
            'metadata': [] if not metadata else get_metadata(seg_id),
            'classes': [] if not classes else get_classes(seg_id)}  # get_columns(segment, metadata_keys) if metadata_keys else []}
#             'metadata': get_columns(segment, metadata_keys) if metadata_keys else {}}


# def config_spectra(dic):
#     ret = {'arrival_time_shift': evalexpr._eval(dic['arrival_time_shift']),
#            'signal_window': evalexpr._eval(dic['signal_window'])
#            }
#     set_spectra_config(**ret)
#     return True
