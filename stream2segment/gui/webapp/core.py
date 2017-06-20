'''
Created on Jul 31, 2016

@author: riccardo
'''
from stream2segment.io.db.pd_sql_utils import colnames, commit
from stream2segment.io.db.models import Segment, Class, Station, Channel, DataCenter, Event,\
    ClassLabelling, Run
# from stream2segment.io.db.sqlevalexpr import query, get_columns
# from stream2segment.gui.webapp import get_session
# from stream2segment.gui.webapp.plotviews import PlotsCache, jsontimestamp  #, set_spectra_config
# from stream2segment.utils import evalexpr
# from sqlalchemy.sql.expression import and_
from stream2segment.io.db.queries import query4gui, count as query_count
from itertools import chain, cycle, izip
from stream2segment.gui.webapp.plotviews import jsontimestamp
from datetime import datetime


NPTS_WIDE = 900
NPTS_SHORT = 900
# FIXME: automatic retrieve by means of Segment class relationships?


def get_segments(session, conditions, orderby, metadata, classes):
    classes = get_classes(session) if classes else []
    _metadata = []
    if metadata:
        _metadata = [[n, t, conditions.get(n, '')] for n, t in get_metadata(session)]
    # parse the orderby if it has a minus at the end it's descending:
    oby = orderby if not orderby else \
        [(k, "asc") if not k[-1] == '-' else (k[:-1], "desc") for k in orderby]
    qry = query4gui(session, conditions=conditions, orderby=oby)
    return {'segment_ids': [seg[0] for seg in qry],
            'classes': classes,
            'metadata': _metadata}


def get_metadata(session, seg_id=None):
    '''Returns a list of tuples (column, column_type) if segment is None or
    (column, column_value) if segment is not None. In the first case, `column_type` is the
    string representation of the column python type (str, datetime,...), in the latter,
    it is the value of `segment` for that column'''
    if seg_id is not None:
        segment = session.query(Segment).filter(Segment.id == seg_id).first()  # FIXME: handle when not found!
        if not segment:
            return []
    else:
        segment = None

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

    if segment is not None:
        ret.insert(0, ['classes.id', [c.id for c in segment.classes]  # @UndefinedVariable
                       ])
    else:
        ret.insert(0, ['has_data', type2str(bool)])
        if query_count(session, Class.id) > 0:
            ret.insert(0, ['classes.id', type2str(Class.id.type.python_type)  # @UndefinedVariable
                           ])
            ret.insert(0, ['classes', type2str(str) +  # @UndefinedVariable
                           ": type either 'any' or 'none' (match segments with at least one class "
                           "set, or no class set, respectively)"])

    return ret


def toggle_class_id(session, segment_id, class_id):
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
    return {'classes': get_classes(session),
            'segment_class_ids': get_classes(session, segment_id)}


def set_classes(session, config):
    classes = config.get('class_labels', [])
    if not classes:
        return
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


def get_classes(session, seg_id=None):
    '''Write doc!! what does this return???'''
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


def get_segment_data(session, seg_id, plotmanager, plot_indices, all_components, filtered, zooms,
                     metadata=False, classes=False, warnings=False, sn_wdws=False):
    # segment = session.query(Segment).filter(Segment.id == seg_id).first()
    plots = []
    zooms_ = parse_zooms(zooms)
    sn_windows = []
    if plot_indices:
        get_plots_func = plotmanager.getfplots if filtered else plotmanager.getplots
        plots = get_plots_func(session, seg_id, plot_indices, all_components)
        try:
            # return always sn_windows, as we already calculated them
            sn_windows = [sorted([jsontimestamp(x[0]), jsontimestamp(x[1])])
                          for x in plotmanager.get_sn_windows(seg_id, filtered)]
        except Exception:
            sn_windows = []
#         if set(indices).intersection([0, 1]):  # if I am trying to update either spectra or
#             # base trace, also send spectra time windows:
#             spectra_wdws = [sorted([jsontimestamp(x[0]), jsontimestamp(x[0]) + 1000*x[1]])
#                             for x in plotmanager.get_spectra_windows(session, seg_id)]

    return {'plots': [p.tojson(z, NPTS_WIDE) for p, z in izip(plots, zooms_)],
            'sn_windows': sn_windows,
            'warnings': [] if not warnings else plotmanager.get_warnings(seg_id),
            'metadata': [] if not metadata else get_metadata(session, seg_id),
            'classes': [] if not classes else get_classes(session, seg_id)}  # get_columns(segment, metadata_keys) if metadata_keys else []}
#             'metadata': get_columns(segment, metadata_keys) if metadata_keys else {}}


def parse_zooms(zooms):
    '''parses the zoom received from the frontend. Basically, if any zoom is a string,
    tries to parse it to datetime
    :param zooms: a list of 2-element tuples, or None's. The elements of the tuple can be number,
    Nones or strings (in datetime format)
    :return: an iterator over zooms. Uses itertools cycle so that this method can be safely used
    with izip never estinguishing it
    '''
    if not zooms:
        zooms = []
    for z in zooms:
        if z is None:
            continue
        for i in xrange(len(z)):
            if z[i] is not None:
                try:
                    try:
                        z[i] = float(z[i])
                    except ValueError:
                        str_ = (z[i][:-1] if z[i][-1] == 'Z' else z[i]).replace('T', ' ')
                        z[i] = jsontimestamp(datetime.strptime(str_, '%Y-%m-%d %H:%M:%S.%f'))
                except:
                    z[i] = None  # fixme: how to handle???
    return chain(zooms, cycle([None]))  # set zooms to None if length is not enough

# def config_spectra(dic):
#     ret = {'arrival_time_shift': evalexpr._eval(dic['arrival_time_shift']),
#            'signal_window': evalexpr._eval(dic['signal_window'])
#            }
#     set_spectra_config(**ret)
#     return True
