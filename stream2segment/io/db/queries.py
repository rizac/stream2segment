'''
Module to store all relevant queries done to the db in 
this program. This allows to have a single point where all queries are
implemented to make easier potential changes in the ORM models or performance
optimizations

:date: Jun 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from sqlalchemy import func, distinct
from sqlalchemy.orm import load_only
from stream2segment.io.db.models import Segment, Station, DataCenter, Channel
from stream2segment.io.db.sqlevalexpr import exprquery
from sqlalchemy.orm.util import aliased
from sqlalchemy.sql.expression import or_, and_
from sqlalchemy.orm.session import object_session


def query4process(session, conditions={}):
    # segement selection, build query:
    # Base query: query segments and station id Note: without the join below, rows would be
    # duplicated
    qry = session.query(Segment.id, Station.id).join(Segment.station).order_by(Station.id)
    # Now parse selection:
    if conditions:
        # parse user defined conditions (as dict of key:value <=> "column": "expr")
        qry = exprquery(qry, conditions=conditions, orderby=None, distinct=True)
    return qry


def query4gui(session, conditions, orderby=None):
    '''Returns a query yielding the one-element tuples (segid1,), ... (segidN,)
    according to `conditions`. orderby is None by default meaning the segments
    will be ordered by event.time (descending) and then event_distance_deg (ascending)'''
    if orderby is None:
        orderby = [('event.time', 'desc'), ('event_distance_deg', 'asc')]
    return exprquery(session.query(Segment.id), conditions=conditions, orderby=orderby,
                     distinct=True)


def getallcomponents(session, seg_id):
    '''Returns a query yielding the one-element tuples (segid1,), ... (segidN,)
    where segid1, segidN are the same channel originating from the same event but
    on all different channel components'''
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
    return session.query(Station.id, Station.network, Station.station, DataCenter.station_url,
                         Station.start_time, Station.end_time).join(Station.datacenter).\
            filter((~Station.has_inventory) &
                   (Station.segments.any(Segment.has_data)))  # @UndefinedVariable


# def count(session, model_or_column):
#     return session.query(func.count(model_or_column)).scalar()
# 
# def querycount(session, model_or_columns, filter=None):
#     query = session.query(func.count(*)).filter(rumorClass.exchangeDataState == state)
