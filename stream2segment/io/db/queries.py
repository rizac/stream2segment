'''
Module implementing all relevant sql-alchemy queries of this program.
This allows to have a single place where all queries are
implemented to make easier potential changes in the ORM models or performance
optimizations

:date: Jun 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from sqlalchemy import func, distinct
from sqlalchemy.orm import load_only
from sqlalchemy.orm.util import aliased
from sqlalchemy.sql.expression import or_, and_
from sqlalchemy.orm.session import object_session

from stream2segment.io.db.models import Segment, Station, DataCenter, Channel
from stream2segment.io.db.sqlevalexpr import exprquery


def query4process(session, conditions={}):
    '''Returns a query yielding the the segments ids (and their stations ids) for the processing.

    :param session: the sql-alchemy session
    :param condition: a dict of segment attribute names mapped to a select expression, each
    identifying a filter (sql WHERE clause). See `:ref:sqlevalexpr.py`. Can be empty (no filter)

    :return: a query yielding the tuples: ```(Segment.id, Segment.station.id)```
    '''
    # Note: without the join below, rows would be duplicated
    qry = session.query(Segment.id, Station.id).join(Segment.station).order_by(Station.id)
    # Now parse selection:
    if conditions:
        # parse user defined conditions (as dict of key:value <=> "column": "expr")
        qry = exprquery(qry, conditions=conditions, orderby=None, distinct=True)
    return qry


def query4gui(session, conditions, orderby=None):
    '''Returns a query yielding the segments ids for the visualization in the GUI according to
    `conditions` and `orderby`, sorted by default (if orderby is None) by segment's event.time
    (descending) and then segment's event_distance_deg (ascending)

    :param session: the sql-alchemy session
    :param condition: a dict of segment attribute names mapped to a select expression, each
    identifying a filter (sql WHERE clause). See `:ref:sqlevalexpr.py`. Can be empty (no filter)
    :param orderby: if None, defaults to segment's event.time (descending) and then
    segment's event_distance_deg (ascending). Otherwise, a list of tuples, where the first
    tuple element is a segment attribute (in string format) and the second element is either 'asc'
    (ascending) or 'desc' (descending)
    :return: a query yielding the tuples: ```(Segment.id)```
    '''
    if orderby is None:
        orderby = [('event.time', 'desc'), ('event_distance_deg', 'asc')]
    return exprquery(session.query(Segment.id), conditions=conditions, orderby=orderby,
                     distinct=True)


def getallcomponents(session, seg_id):
    '''Returns a query yielding the segments ids which represent all the components of the
    same waveform segment identified by `seg_id`

    :param session: the sql-alchemy session
    :return: a query yielding the tuples: ```(Segment.id)```
    '''
    # let's do two queries, maybe not so efficient, but we didn't find a
    # way to work with relationships on aliased objects. Remember that this
    # query is launched once at each "visualize next segment" button click, so perfs loss
    # should be negligible
    segment = session.query(Channel.station_id,
                            Channel.location,
                            Channel.channel,
                            Segment.event_id
                            ).join(Segment.channel).filter(Segment.id == seg_id).first()

    # so the condition compare components with event instead of time range
    conditions = [Channel.station_id == segment[0],  # assures network + station
                  Channel.location == segment[1],  # assures location
                  Channel.channel.like(segment[2][:-1] + "_"),
                  Segment.event_id == segment[3]]   # @UndefinedVariable

    return session.query(Segment.id).join(Segment.channel).\
        filter(or_(Segment.id == seg_id, and_(*conditions)))


def query4inventorydownload(session):
    '''Returns a query yielding the stations which do not have inventories xml
    and have at least one segment with data.

    :param session: the sql-alchemy session
    :return: a query yielding the tuples:
    ```(Station.id, Station.network, Station.station, DataCenter.station_url,
        Station.start_time, Station.end_time)```
    '''
    return session.query(Station.id, Station.network, Station.station, DataCenter.station_url,
                         Station.start_time, Station.end_time).join(Station.datacenter).\
        filter((~Station.has_inventory) &
               (Station.segments.any(Segment.has_data)))  # @UndefinedVariable
