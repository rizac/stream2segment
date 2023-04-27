"""
Db utilities for the 'show' command

Created on 16 Apr 2020

@author: riccardo
"""

from sqlalchemy import func
from datetime import datetime

from stream2segment.io.db import secure_dburl
from stream2segment.io.db.inspection import attnames, get_related_models
from stream2segment.process.db.models import (Segment, Station, get_classlabels)
from stream2segment.process.db.sqlevalexpr import exprquery, get_pytype, get_sqltype

# import atexit

_session = None  # noqa


def init(app, session):
    """Initialize the database. this method must be called after the Flask
    app has been created nd before using it.

    :param session: a SQLAlchemy SCOPED session
    """
    # https://nestedsoftware.com/2018/06/11/flask-and-sqlalchemy-without-the-flask-sqlalchemy-extension-3cf8.34704.html
    # and
    # https://flask.palletsprojects.com/en/1.1.x/appcontext/#storing-data

    global _session  # noqa
    _session = session

    # we add a listener that whn a request is ended, the session should be
    # removed (see get_session below)
    @app.teardown_appcontext
    def close_db(error):
        """Close the database again at the end of the request."""
        # if hasattr(g, 'session'):
        #     g.session.remove()
        get_session().remove()


def get_session():
    """Return a SQLAlchemy scoped session for interacting with the database"""
    # (see init above)
    return _session


def get_db_url(safe=True):
    """Return the db url (with password hidden, if present in the url)"""
    return secure_dburl(str(get_session().bind.engine.url))


def get_segments_count(conditions):
    """Compute the number of segments to show (int) according to the given
    `conditions` and stores it in the global varibale
    _segments_count

    :param conditions: dict of selection expressions usually resulting from the
        'segments_selection' parameter in the YAML config)
    """
    session = get_session()
    return _query4gui(session.query(func.count(Segment.id)),
                                    conditions).scalar()


def _query4gui(what2query, conditions, orderby=None):
    return exprquery(what2query, conditions=conditions, orderby=orderby)


def get_segment_id(seg_index, segment_count, conditions):
    """Fetch from the database a block of segments ids and returns them
    as list of integers"""
    session = get_session()
    # NOTE: sort by id only, is way FASTER:
    orderby = [('id', 'desc')]
    if seg_index > segment_count / 2.0:
        # The search might still take a lot if we want to select last elements:
        # reverse the sort order and the seg_index:
        orderby = [('id', 'asc')]
        seg_index = segment_count - seg_index - 1
    return _query4gui(session.query(Segment.id), conditions, orderby).\
        offset(seg_index).limit(1).one()[0]


def get_segment(segment_id):  # , cols2load=None):
    """Return the segment identified by id `segment_id`"""
    # We might want to cache segments, so that when e.g., a user chooses only a
    # different custom plot in the GUI (or checks/unchecks the 'preprocessing'
    # button) we do not need to reload the segment from the DB, neither we
    # need to re-open the segment stream from the Bytes sequence.
    # Now, we might use the session identity_map (https://stackoverflow.com/a/48988010)
    # but it seems that the identity_map will always be empty, as it is
    # cleared automatically at the end of each method using a db session
    # (i.e., the method sqlalchemy.orm.state.InstanceState._cleanup is called)
    # Thus, let's avoid any form of caching, also in account
    # of the fact that most Flask example suggest to release the db session after
    # each request (see `init` function above) and we
    # just use query.get (see below) which avoids querying the db if the
    # segment instance is already in the session. As we saw,
    # this should never happen, but just in case.

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
    """If `segment_id` is given (int not None), returns a list of classes ids
    (integers) denoting the classes associated to the given segment, or the
    empty list if no classes are set.

    Otherwise (`segment_id`=None), returns a list of classes. Each class is
    returned as dict with keys 'id', 'label' and 'count'. The returned list
    has the form:
    ```
    [
        ... ,
        {
         'id': int
         'label': str,
         'description': str
         'count': int (number of segments labelled with this label)
        },
        ...
    ]
    ```
    """
    if segment_id is not None:
        segment = get_segment(segment_id)
        return [] if not segment else sorted(c.id for c in segment.classes)

    return get_classlabels(get_session(), segments=True)


def get_metadata(segment_id=None):
    """Return a list of tuples (column, column_type) if `segment_id` is None or
    (column, column_value) if segment is not None. In the first case,
    `column_type` is the string representation of the column python type
    (str, datetime,...), in the latter, it is the value of `segment` for that
    column
    """
    segment = None
    if segment_id is not None:
        segment = get_segment(segment_id)
        if not segment:
            return []

    # map every related model to a function that will show only some of their attributes.
    # NOTE: this dict dictates also the ORDER of the related models
    related_models_attrs = {
        'event': lambda attr: attr not in {'contributor', 'contributor_id',
                                           'mag_author'},
        'station': lambda attr: attr != Station.inventory_xml.key,
        'channel': lambda attr: True,
        'datacenter': lambda attr: attr in {'id', 'dataselect_url'},
        'download': lambda attr: attr in {'id', 'run_time'}
    }
    related_models = get_related_models(Segment)
    seg_simple_att_names = _attnames(Segment, lambda attr: attr != Segment.data.key)
    if segment:
        # we have an instance, it is for showing data on the GUI. So:
        metadata = [(_, _jsonify(getattr(segment, _))) for _ in seg_simple_att_names]
        for relation_name, attr_filter_func in related_models_attrs.items():
            related_model = related_models[relation_name]
            related_model_attrs = _attnames(related_model, attr_filter_func)
            related_inst = getattr(segment, relation_name)
            metadata.extend((relation_name + '.' + a,
                             _jsonify(getattr(related_inst, a)))
                            for a in related_model_attrs)
    else:
        # we have a model (instance class), it is for selecting data on the GUI. So:
        metadata = [(_, _get_pytype(Segment, _)) for _ in seg_simple_att_names]
        for relation_name, attr_filter_func in related_models_attrs.items():
            related_model = related_models[relation_name]
            related_model_attrs = _attnames(related_model, attr_filter_func)
            metadata.extend((relation_name + '.' + a, _get_pytype(related_model, a))
                            for a in related_model_attrs)
        # remove None's (unknown type, no selection possible):
        metadata = [_ for _ in metadata if _[1] is not None]

    return metadata


def _attnames(model, filter_func=None):
    """Return a sorted list of (queriable) attributes defined on the model with optional
    filter function `func(att_name): -> bool`
    """
    # return non foreign key columns or queryable attributes only:
    att_itr = attnames(model, fkey=False, qatt=True, rel=False)
    return sorted(_ for _ in att_itr if filter_func is None or filter_func(_))


def _get_pytype(model, attrname):
    sqltype = get_sqltype(getattr(model, attrname))
    pytype = None if sqltype is None else get_pytype(sqltype)
    return None if pytype is None else str(pytype.__name__)


def _jsonify(obj):
    """flask converts datetime(s) to Fri, 08 Sep 2017 05:03:10 GMT, let's keep UTC"""
    return obj.isoformat(sep=' ') if isinstance(obj, datetime) else obj
