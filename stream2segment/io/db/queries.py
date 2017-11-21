'''
Module implementing all relevant sql-alchemy queries of this program.
This allows to have a single place where all queries are
implemented to make easier potential changes in the ORM models or performance
optimizations

:date: Jun 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from itertools import chain

from sqlalchemy import func, distinct
from sqlalchemy.orm import load_only
from sqlalchemy.orm.util import aliased
from sqlalchemy.sql.expression import or_, and_, func, case, text
from sqlalchemy.orm.session import object_session
from sqlalchemy.sql.elements import literal_column
from sqlalchemy.orm import configure_mappers

from stream2segment.io.db.models import Segment, Station, DataCenter, Channel, Event
from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.download.utils import custom_download_codes
from collections import OrderedDict


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
    '''Returns a query yielding the segments ids for the visualization in the GUI (processing)
    according to `conditions` and `orderby`, sorted by default (if orderby is None) by
    segment's event.time (descending) and then segment's event_distance_deg (ascending)

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


def query4dreport(session, **binexprs2count):
    '''Returns a query yielding the segments ids for the visualization in the GUI (download
    report)
    ''' 
    # We should get something along the lines of:
    # SELECT data_centers.id AS data_centers_id,
    #     stations.id AS stations_id,
    #     count(segments.id) as csi,
    #     count(segments.sample_rate != channels.sample_rate) as segid
    # FROM
    # data_centers
    # JOIN stations ON data_centers.id = stations.datacenter_id
    # JOIN channels on channels.station_id = stations.id
    # JOIN segments on segments.channel_id = channels.id
    # GROUP BY stations.id

    def countif(key, binexpr):
        NULL = literal_column("NULL")
        return func.count(case([(binexpr, Segment.id)], else_=NULL)).label(key)

    qry = session.query(DataCenter.id.label('dc_id'),  # @UndefinedVariable
                        Station.id.label('station_id'),
                        Station.latitude.label('lat'),
                        Station.longitude.label('lon'),
                        func.count(Segment.id).label('num_segments'),
                        *[countif(k, v) for k, v in binexprs2count.items()])

    # ok seems that back referenced relationships are instantiated only after the first query is
    # made:
    # https://stackoverflow.com/questions/14921777/backref-class-attribute
    # workaround:
    configure_mappers()

    return qry.join(DataCenter.stations,  # @UndefinedVariable
                    Station.channels,  # @UndefinedVariable
                    Channel.segments,  # @UndefinedVariable
                    ).group_by(DataCenter.id, Station.id)


def querystationinfo4dreport(session, station_id, **binexprs2count):
    '''Returns a query yielding the segments ids for the visualization in the GUI (download
    report)
    ''' 
    # We should get something along the lines of:
    # SELECT data_centers.id AS data_centers_id,
    #     stations.id AS stations_id,
    #     count(segments.id) as csi,
    #     count(segments.sample_rate != channels.sample_rate) as segid
    # FROM
    # data_centers
    # JOIN stations ON data_centers.id = stations.datacenter_id
    # JOIN channels on channels.station_id = stations.id
    # JOIN segments on segments.channel_id = channels.id
    # GROUP BY stations.id

    def countif(key, binexpr):
        YES = text("'&#10004;'")
        NO = text("''")
        return case([(binexpr, YES)], else_=NO).label(key)

    qry = session.query(Segment.id, Segment.seed_identifier,
                        Segment.event_id,
                        Segment.start_time, Segment.end_time,
                        *[countif(k, v) for k, v in binexprs2count.items()])

    # ok seems that back referenced relatioships are instanitated only after the first query is
    # made:
    # https://stackoverflow.com/questions/14921777/backref-class-attribute
    # workaround:
    configure_mappers()

    return qry.join(Segment.event, Segment.station
                    ).filter(Station.id == station_id).order_by(Event.time.desc(),
                                                                Segment.event_distance_deg.asc())


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
