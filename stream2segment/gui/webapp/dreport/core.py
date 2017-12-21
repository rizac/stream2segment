'''
Core functionalities for the GUI web application (download report)

:date: Oct 12, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, zip

import re
from itertools import cycle, chain
from collections import OrderedDict

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import load_only
from sqlalchemy import func, distinct
from sqlalchemy.orm import load_only
from sqlalchemy.orm.util import aliased
from sqlalchemy.sql.expression import or_, and_, func, case, text
from sqlalchemy.orm.session import object_session
from sqlalchemy.sql.elements import literal_column
from sqlalchemy.orm import configure_mappers

from stream2segment.io.db.pdsql import colnames
from stream2segment.io.db.models import Segment, Class, Station, Channel, DataCenter, Event,\
    ClassLabelling, Download
from stream2segment.utils.resources import yaml_load_doc, get_templates_fpath

from stream2segment.download.utils import custom_download_codes


def _getlabels(max_gap_overlap=(-0.5, 0.5)):
    urlexc, mseedexc, time_err, time_warn = custom_download_codes()
    c_empty = Segment.data.isnot(None) & (func.length(Segment.data) == 0)
    # sql between includes endpoints
    no_gaps = Segment.maxgap_numsamples.between(max_gap_overlap[0], max_gap_overlap[1])
    c_data = Segment.has_data == True  # @IgnorePep8
    c_gaps = c_data & ~no_gaps
    c_srate_mismatch = c_data & no_gaps & (Segment.sample_rate != Channel.sample_rate)
    return OrderedDict([['no code', (True, Segment.download_code.is_(None))],
                        ['url error', (True, Segment.download_code == urlexc)],
                        ['mseed error', (True, Segment.download_code == mseedexc)],
                        ['4xx HTTP code', (True, (Segment.download_code >= 400) &
                                                 (Segment.download_code < 500))],
                        ['5xx HTTP code', (True, Segment.download_code >= 500)],
                        ['empty data', (True, c_empty & ~
                                        (Segment.download_code == time_err))],
                        ['gaps/overlaps', (True, c_gaps)],
                        ['sample rate mismatch (channel vs. data)', (False, c_srate_mismatch)],
                        ['data completely out of request\'s time span',
                         (True, (Segment.download_code == time_err))],
                        ['data partially out of request\'s time span',
                         (False, (Segment.download_code == time_warn))]
                        ])


def selectablelabels():
    return [(k, v[0], 0) for k, v in _getlabels().items()]


def binexprs2count():
    return OrderedDict([(k, v[1]) for k, v in _getlabels().items()])


def get_data(session):
    binexprs2count_ = binexprs2count()
    query = query4dreport(session, **binexprs2count_)
    return query.all()


def get_station_data(session, station_id, selectedLabels):
    lbls = _getlabels()
    binexprs = {}
    for key, val in selectedLabels:
        if not val:
            continue
        binexprs[key] = lbls[key][1]

    query = querystationinfo4dreport(session, station_id, **binexprs)
    return query.all()


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

    qry = session.query(Segment.id, Segment.seed_id,
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
