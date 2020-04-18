'''
Db utitlties for the 'show' command

Created on 16 Apr 2020

@author: riccardo
'''
import os
import numpy as np
from sqlalchemy import func

# from flask import g
from stream2segment.process.db import (Segment, Class, ClassLabelling,
                                       Station, Download,
                                       get_session as _getsess)
from stream2segment.io.db.sqlevalexpr import exprquery, Inspector
from stream2segment.utils import secure_dburl
# import atexit

# rationale: Remember that from the GUI the user can navigate back and forward
# with the button, not by typing segment ids (there is the selection <form>
# for that). So, when th GUI shows up we need a fast way to know the number
# of total segments, and later load somehow efficiently the 'next' or 'previous'
# queried segment.
# Loading once all segment ids into SEG_IDS below might be the best solutions,
# but for huge database is inefficient. So when the page shows up,
# we query only the number N of segments to show via `get_segments_count`, and
# we set SEG_IDS as a numpy array of N NaNs (the method should be relatively
# fast as it issues  an SQL count). After that, When querying for a segment
# at given index (position), if SEG_IDS[index] is NaN, a block of
# SEG_QUERY_BLOCK segment ids (only ids, to speed up things) is loaded
# into SEG_IDS at the right position, and then with the segment id we can
# query the desired Segment.
SEG_IDS = []  # numpy array of segments ids (for better storage): filled with NaNs,
# populated on demand witht the block below:
SEG_QUERY_BLOCK = 50

_session = None  # pylint: disable=invalid-name


def init(app, dbpath):
    '''Initializes the database. this method must be called after the Flask
    app has been created nd before using it
    '''
    # https://nestedsoftware.com/2018/06/11/flask-and-sqlalchemy-without-the-flask-sqlalchemy-extension-3cf8.34704.html
    # and
    # https://flask.palletsprojects.com/en/1.1.x/appcontext/#storing-data

    # we store _dbpath globally
    global _session  # pylint: disable=global-statement, invalid-name
    _session = _getsess(dbpath, scoped=True)

    # we add a listener that whn a request is ended, the session should be
    # removed (see get_session below)
    @app.teardown_appcontext
    def close_db(error):
        """Closes the database again at the end of the request."""
        # if hasattr(g, 'session'):
        #     g.session.remove()
        _session.remove()


def get_session():
    '''Returns a sqlalchemy scoped session for interacting with the database'''
    # (see init above)
    return _session


def get_db_url(safe=True):
    '''Returns the db url (with password hidden, if present in the url)'''
    return secure_dburl(str(get_session().bind.engine.url))


def get_segments_count(conditions):
    '''Returns the number of segments to show (int) according to the given
    `conditions` (dict of selection expressions usually resulting from the
    'segment_select' parameter in the YAML config)
    '''
    session = get_session()
    num_segments = _query4gui(session.query(func.count(Segment.id)), conditions).scalar()
    if num_segments > 0:
        global SEG_IDS  # pylint: disable=global-statement
        SEG_IDS = np.full(num_segments, np.nan)
    return num_segments


def _query4gui(what2query, conditions, orderby=None):
    return exprquery(what2query, conditions=conditions, orderby=orderby)


def get_segment_id(seg_index, conditions):
    '''Returns the segment id (int) at a given index (position) in the GUI'''
    if np.isnan(SEG_IDS[seg_index]):
        # segment id not queryed yet: load chunks of segment ids:
        # Note that this is the best compromise between
        # 1) Querying by index, limiting by 1 and keeping track of the
        # offset: FAST at startup, TOO SLOW for each segment request
        # 2) Load all ids at once at the beginning: TOO SLOW at startup, FAST for each
        # segment request
        # (fast and slow refer to a remote db with 10millions row without config
        # and pyfile)
        limit = SEG_QUERY_BLOCK
        offset = int(seg_index / float(SEG_QUERY_BLOCK)) * SEG_QUERY_BLOCK
        limit = min(len(SEG_IDS) - offset, SEG_QUERY_BLOCK)
        segids = get_segment_ids(conditions,
                                 offset=offset, limit=limit)
        SEG_IDS[offset:offset+limit] = segids
    return int(SEG_IDS[seg_index])


def get_segment_ids(conditions, limit=50, offset=0):
    '''Fetches from the database a block of segments ids and returns them
    as list of integers'''
    session = get_session()
    # querying all segment ids is faster later when selecting a segment
    orderby = [('event.time', 'desc'), ('event_distance_deg', 'asc'),
               ('id', 'asc')]
    return [_[0] for _ in _query4gui(session.query(Segment.id),
                                     conditions, orderby).limit(limit).offset(offset)]


def get_segment(segment_id):  # , cols2load=None):
    '''Returns the segment identified by id `segment_id`'''
    # Rationale: we would like to use some sort of cache, because displaying
    # several plots might be time consuming. We ended up focusing on simplicity
    # (avoid everything more complex, we already tried and it's a pain):
    # cache a segment at a time. When a new segment is requested,
    # discard the previously cached (if any), and cache the new segment,
    # so that when e.g., a user chooses a different
    # user defined plot in the GUI (or checks/unchecks the 'preprocessing'
    # button) we do not need to reload the segment from the DB, neither we
    # need to re-open the segment stream from the Bytes sequence.
    # Now, we might user the session identity_map (https://stackoverflow.com/a/48988010)
    # but it seems that the identity_map will always be empty, as it is
    # cleared automatically at the end of each method using a db session
    # (i.e., the method sqlalchemy.orm.state.InstanceState._cleanup is called)
    # Thus, we avoid any form of caching, also in account of the fact that
    # most applications suggest to release the session after each request
    # (which is what we implemented): just use query.get which is implemented
    # to avoid querying the db if the instance is in the session (as we saw,
    # this should be never happen, but just in case). If you want to
    # implement some caching, use a different storage for a Segment instance
    # and add it back to the session (session.add)

    session = get_session()
    # for obj in session.identity_map.values():
    #     if isinstance(obj, Segment) and obj.id == segment_id:
    #         break
    # else:  # no break found (see above)
    #     # clear the session and release all connections, we have in any case
    #     # to query the database:
    #     session.remove()
    seg = session.query(Segment).get(segment_id)
    # session.add(seg)
    return seg


# @atexit.register
# def remove_session():
#     get_session().remove()
    # print('Db session removed')


def get_classes(segment_id=None):
    '''
    If `segment_id` is given (int not None), returns a list of classes ids
    (integers) denoting the classes associated to the given segment, or the
    empty list if no classes are set.

    Otherwise (`segment_id`=None), returns a list of classes. Each class is
    returned as dict with keys 'id', 'label' and 'count'. The returned list
    has the form:
    ```
    [
        ... ,
        {
         'id': (int)
         'labe;': (str)
         'count': (int) (number of segments labelled with this label)
        },
        ...
    ]
    ```
    Note that 'id' and 'label' might change depending on the ORM implementation
    '''
    if segment_id is not None:
        segment = get_segment(segment_id)
        return [] if not segment else sorted(c.id for c in segment.classes)

    session = get_session()
    colnames = [Class.id.key, Class.label.key, 'count']
    # Note isouter which produces a left outer join, important when we have no class labellings
    # (i.e. third column all zeros) otherwise with a normal join we would have no results
    data = session.query(Class.id, Class.label, func.count(ClassLabelling.id).label(colnames[-1])).\
        join(ClassLabelling, ClassLabelling.class_id == Class.id, isouter=True).group_by(Class.id).\
        order_by(Class.id)
    return [{name: val for name, val in zip(colnames, d)} for d in data]


def get_metadata(segment_id=None):
    '''Returns a list of tuples (column, column_type) if `segment_id` is None or
    (column, column_value) if segment is not None. In the first case, `column_type` is the
    string representation of the column python type (str, datetime,...), in the latter,
    it is the value of `segment` for that column'''
    excluded_colnames = set([Station.inventory_xml, Segment.data, Download.log,
                             Download.config, Download.errors, Download.warnings,
                             Download.program_version, Class.description])

    segment = None
    if segment_id is not None:
        # exclude all classes attributes (returned in get_classes):
        excluded_colnames |= {Class.id, Class.label}
        segment = get_segment(segment_id)
        if not segment:
            return []

    insp = Inspector(segment or Segment)
    attnames = insp.attnames(Inspector.PKEY | Inspector.QATT | Inspector.REL | Inspector.COL,
                             sort=True, deep=True, exclude=excluded_colnames)
    if segment_id is not None:
        # return a list of (attribute name, attribute value)
        return [(_, insp.attval(_)) for _ in attnames]
    # return a list of (attribute name, str(attribute type))
    return [(_, getattr(insp.atttype(_), "__name__"))
            for _ in attnames if insp.atttype(_) is not None]
