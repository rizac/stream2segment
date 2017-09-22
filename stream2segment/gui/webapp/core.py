'''
Core functionalities for the GUI web application

:date: Jul 31, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, zip

import re
from itertools import cycle

# from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from stream2segment.io.db.pd_sql_utils import colnames
from stream2segment.io.db.models import Segment, Class, Station, Channel, DataCenter, Event,\
    ClassLabelling, Run
from stream2segment.io.db.queries import query4gui, count as query_count
from stream2segment.gui.webapp.plotviews import jsontimestamp
# from stream2segment.io.db import sqlevalexpr
from stream2segment.utils.resources import yaml_load_doc, get_templates_fpath


NPTS_WIDE = 900  # FIXME: automatic retrieve by means of Segment class relationships?
NPTS_SHORT = 900  # FIXME: see above


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
    '''Returns a list of tuples (column, column_type) if `seg_id` is None or
    (column, column_value) if segment is not None. In the first case, `column_type` is the
    string representation of the column python type (str, datetime,...), in the latter,
    it is the value of `segment` for that column'''
    if seg_id is not None:
        segment = session.query(Segment).filter(Segment.id == seg_id).first()
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
    # exclude columns with byte data or too long text data which would make no sense to show:
    excluded_colnames = set([Station.inventory_xml.key, Segment.data.key, Run.log.key,
                             Run.config.key])
    # if segment is None, return hybrid attributes, too.
    # Use a dict of pairs: (model -> list of hybrid attributes to be shown)
    # The type of the hybrid attribute will be later inferred by sqlalchemy:
    included_colnames = {} if segment is not None else \
        {Segment: [Segment.has_data.key],  # @UndefinedVariable
         Station: [Station.has_inventory.key]}  # @UndefinedVariable

    for prefix, model in METADATA:
        colnamez = list(colnames(model, fkey=False))
        colnamez.extend(included_colnames.get(model, []))
        for colname in colnamez:
            if colname in excluded_colnames:
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
            except Exception as _:
                continue
            ret.append([("%s." % prefix) + colname if prefix else colname, value])

    if segment is None:  # add fields for selecting. These fields do not need to be set
        # if a segment is provided (e.g., classes for a segment is returned if requested via
        # a separate response key, and has_data does not need to be shown in the infos for a
        # segment)
        if query_count(session, Class.id) > 0:
            ret.insert(0, ['classes.id', type2str(Class.id.type.python_type)  # @UndefinedVariable
                           ])
            ret.insert(0, ['classes', type2str(str) +  # @UndefinedVariable
                           ": type either any: segments with at least one class, "
                           "or none: segments with no class"])

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

    try:
        session.commit()
    except SQLAlchemyError as _:
        session.rollback()

    # re-query the database to be sure:
    return {'classes': get_classes(session),
            'segment_class_ids': get_classes(session, segment_id)}


def set_classes(session, config):
    classes = config.get('class_labels', [])
    if not classes:
        return
    # do not add already added classes:
    clazzes = {c.label: c for c in session.query(Class)}
    for label, description in classes.items():
        cla = None
        if label in clazzes and clazzes[label].description != description:
            cla = clazzes[label]
            cla.description = description  # update
        elif label not in clazzes:
            cla = Class(label=label, description=description)
        if cla is not None:
            session.add(cla)
    session.commit()


def get_classes(session, seg_id=None):
    '''If seg_id is not None, returns a list of the segment class ids.
    Otherwise, a list of dicts where each dict is a db row in the form
    {table_column: row_value}. The dict will have also a "count" attribute
    denoting how many segments have that class set'''
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


def get_segment_data(session, seg_id, plotmanager, plot_indices, all_components, preprocessed,
                     zooms, metadata=False, classes=False, warnings=False, sn_wdws=False):
    """Returns the segment data, depending on the arguments
    :param session: a flask sql-alchemy session object
    :param seg_id: integer denoting the segment id
    :param plotmanager: a PlotManager object, storing all plots data and sn/windows data
    :param plot_indices: a list of plots to be calculated from the given `plotmanager` (which caches
    its plot for performance speed)
    :param all_components: boolean, whether or not the `plotmanager` should give all components for
    the main plot (plot representing the given segment's data, whose plot index is currently 0).
    Ignored if 0 is not in `plot_indices`
    :param preprocessed: boolean, whether or not the `plotmanager` should calculate the plots on
    the pre-processing function defined in the config (if any), or on the raw obspy Stream
    :param zooms: a list of **all plots** defined in the plotmanager, or None.
    Each element is either None, or a tuple of [xmin, xmax] values (xmin and xmax can be both None,
    to conform python slicing behaviour). Thus, the length of `zooms` most likely differs from
    that of `plot_indices`. the zooms of interest are, roughly speaking,
    [zooms[i] for i in plot_indices] (if zoom is not None)
    :param metadata: boolean, whether or not to return a list of the segment metadata. The list
    is a list of tuples ('column', value). A list is used to preserve order for client-side
    javascript parsing
    :param classes: boolean, whether to return the integers classes ids (if any) of the given
    segment
    :param warnings: boolean, whether to return the given warnings for the given segment. the
    warnings include: segment with gaps, inventory error (if inventory is required according to
    the config), and sn windows calculation error (e.g., bad values given from the config or the
    gui). The warnings is a list of (currently) at most 3 string elements
    :param sn_wdws: boolean, whether to returns the sn windows calculated according to the
    config values. The returned list is a 2-element list, where each element is in turn a
    2-element numeric list: [noise_window_start, noise_window_end],
    [signal_window_start, signal_window_end]
    """
    # segment = session.query(Segment).filter(Segment.id == seg_id).first()
    plots = []
    zooms_ = parse_zooms(zooms, plot_indices)
    sn_windows = []
    if sn_wdws:
        if sn_wdws['signal_window']:
            try:
                sn_wdws = {'signal_window': parse_array(sn_wdws['signal_window'], float),
                           'arrival_time_shift': float(sn_wdws['arrival_time_shift'])}
            except Exception:
                pass
        # set_sn_windows(self, session, a_time_shift, signal_window):
        plotmanager.update_config(sn_windows=sn_wdws)

    if plot_indices:
        plots = plotmanager.getplots(session, seg_id, plot_indices, preprocessed, all_components)
        try:
            # return always sn_windows, as we already calculated them. IT is better
            # to call this method AFTER get_plots_func defined above
            sn_windows = [sorted([jsontimestamp(x[0]), jsontimestamp(x[1])])
                          for x in plotmanager.get_cache(session, seg_id, 'sn_windows',
                                                         preprocessed, [])]
        except Exception:
            sn_windows = []

    return {'plots': [p.tojson(z, NPTS_WIDE) for p, z in zip(plots, zooms_)],
            'sn_windows': sn_windows,
            'warnings': [] if not warnings else plotmanager.get_warnings(seg_id, preprocessed),
            'metadata': [] if not metadata else get_metadata(session, seg_id),
            'classes': [] if not classes else get_classes(session, seg_id)}


def parse_array(str_array, parsefunc=None, try_return_scalar=True):
    '''splits str_array into elements, and apply func on each element
    :param str_array: a valid string array, with or without square brackets. Leading and
    trailing spaces will be ignored (str split is applied twice if the string has square
    brackets). The separation characters are the comma surrounded by zero or more spaces, or
    a one or more spaces. E.g. "  [1 ,3  ]", "[1,3]"
    '''
    # str_array should always be a string... just in case it's already a parsable value
    # (e.g., parsefunc = float and str-array = 5.6), then try to parse it first:
    if parsefunc is not None and try_return_scalar:
        try:
            return parsefunc(str_array)
        except Exception:
            pass
    d = str_array.strip()
    if d[0] == '[' and d[-1] == ']':
        d = d[1:-1].strip()
    _ = re.split("(?:\\s*,\\s*|\\s+)", d)
    if parsefunc is not None:
        _ = list(map(parsefunc, _))
    return _[0] if try_return_scalar and len(_) == 1 else _


def parse_zooms(zooms, plot_indices):
    '''parses the zoom received from the frontend. Basically, if any zoom is a string,
    tries to parse it to datetime
    :param zooms: a list of 2-element tuples, or None's. The elements of the tuple can be number,
    Nones or strings (in datetime format)
    :return: an iterator over zooms. Uses itertools cycle so that this method can be safely used
    with izip never estinguishing it
    '''
    if not zooms or not plot_indices:
        zooms = cycle([None, None])  # to be safe in iterations
    _zooms = []
    for plot_index in plot_indices:
        try:
            z = zooms[plot_index]
        except (IndexError, TypeError):
            z = [None, None]
        _zooms.append(z)
    return _zooms  # set zooms to None if length is not enough


def get_doc(key, plotmanager):
    '''returns the doc from the given key:
    :param plotmanager: the plotmanager. Used if key is 'preprocessfunc' (see below)
    :param key: 'preprocessfunc' (the doc will be the python doc implemented from the user)
    'sn_windows' (the doc will be parsed by the gui.yaml file implemented in resources folder),
    'segment_select' (the doc for the segment selection popup div)
    '''
    if key == 'preprocessfunc':
        ret = plotmanager.get_preprocessfunc_doc
    elif key == 'sn_windows':
        ret = yaml_load_doc(get_templates_fpath("gui.yaml"), "sn_windows")
    elif key == 'segment_select':
        ret = yaml_load_doc(get_templates_fpath("gui.yaml"), "segment_select")

    if not ret:
        ret = "error: documentation N/A"
    return ret.strip()
