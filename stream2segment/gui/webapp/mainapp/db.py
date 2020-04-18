'''
Db utitlties for the 'show' command

Created on 16 Apr 2020

@author: riccardo
'''
import os
import numpy as np
from sqlalchemy import func

from flask import g
from stream2segment.process.db import (Segment, Class, ClassLabelling,
                                       Station, Download,
                                       get_session as _getsess)
from stream2segment.io.db.sqlevalexpr import exprquery, Inspector
from stream2segment.utils import secure_dburl
import atexit


SEG_IDS = []  # numpy array of segments ids (for better storage): filled with NaNs,
# populated on demand witht the block below:
SEG_QUERY_BLOCK = 50


# @_raiseifreturnsexception
# def classmeth_preprocessed_stream(self, preproc_func, config):
#     if not hasattr(self, '_preprocessed_stream'):
#         tmpstream = getattr(self, "_stream", None)
#         try:
#             if tmpstream is not None:
#                 self.stream(True)  # reload, so we pass to the preprocess func
#                 # the REAL stream (might have been modified)
#             self._preprocessed_stream = preproc_func(self, config)
#         except Exception as exc:
#             self._preprocessed_stream = exc
#         finally:
#             if tmpstream is not None:
#                 self._stream = tmpstream
#
#     return self._preprocessed_stream
#
#
# # https://nestedsoftware.com/2018/06/11/flask-and-sqlalchemy-without-the-flask-sqlalchemy-extension-3cf8.34704.html
# Segment.preprocessed_stream = classmeth_preprocessed_stream


_dbpath = None

def init(app, dbpath):
    # https://nestedsoftware.com/2018/06/11/flask-and-sqlalchemy-without-the-flask-sqlalchemy-extension-3cf8.34704.html
    # and
    # https://flask.palletsprojects.com/en/1.1.x/appcontext/#storing-data

    # we store _dbpath globally
    global _dbpath
    _dbpath = dbpath
    
    # we add a listener that whn a request is ended, the session should be
    # removed (see get_session below)
    @app.teardown_appcontext
    def close_db(error):
        """Closes the database again at the end of the request."""
        if hasattr(g, 'session'):
            g.session.remove()


def get_session():
    
    # (see init above)
    if not hasattr(g, 'session'):
        g.session = _getsess(_dbpath, scoped=True)
    return g.session


def get_db_url(safe=True):
    return secure_dburl(str(_dbpath))


def get_segments_count(conditions):
    session = get_session()
    num_segments = _query4gui(session.query(func.count(Segment.id)), conditions).scalar()
    if num_segments > 0:
        global SEG_IDS  # pylint: disable=global-statement
        SEG_IDS = np.full(num_segments, np.nan)
    return num_segments


def _query4gui(what2query, conditions, orderby=None):
    return exprquery(what2query, conditions=conditions, orderby=orderby)


def get_segment_id(seg_index, conditions):  # get_segment_select()
    session = get_session()
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
