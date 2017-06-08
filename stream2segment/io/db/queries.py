'''
Module to store all relevant queries done to the db in 
this program. This allows to have a single point where all queries are
implemented to make easier potential changes in the ORM models or performance
optimizations

Created on Jun 8, 2017

@author: riccardo
'''
from sqlalchemy import func, distinct
from sqlalchemy.orm import load_only
from stream2segment.io.db.models import Segment, Station, DataCenter
from stream2segment.io.db.sqlevalexpr import exprquery


def getquery4process(session, conditions, *seg_atts):
    '''
    seg_atts: what has to be loaded
    '''
    # segement selection, build query:
    # Base query: query segments and station id Note: without the join below, rows would be
    # duplicated
    segs_staids = session.query(Segment, Station.id).join(Segment.station).\
        options(load_only(*seg_atts)).\
        order_by(Station.id)  # @UndefinedVariable
    # Now parse selection:
    if conditions:
        # parse user defined conditions (as dict of key:value <=> "column": "expr")
        segs_staids = exprquery(segs_staids, conditions=conditions, orderby=None, distinct=True)
    return segs_staids


def query4gui(session, conditions, orderby):
    return exprquery(session.query(Segment.id), conditions=conditions, orderby=orderby,
                     distinct=True)


def query4inventorydownload(session):
    return session.query(Station.id, Station.network, Station.station, DataCenter.station_url,
                         Station.start_time, Station.end_time).join(Station.datacenter).\
            filter((~Station.has_inventory) &
                   (Station.segments.any(Segment.has_data)))  # @UndefinedVariable
