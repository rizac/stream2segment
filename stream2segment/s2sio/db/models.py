'''
Created on Jul 15, 2016

@author: riccardo
'''
# from sqlalchemy import engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    Binary,
    # Numeric,
    # event,
    # CheckConstraint,
    # BigInteger,
    UniqueConstraint,
)
import datetime
# from sqlalchemy.sql.sqltypes import BigInteger, BLOB
# from sqlalchemy.sql.schema import ForeignKey

_Base = declarative_base()


class Base(_Base):

    __abstract__ = True

    @classmethod
    def get_col_names(cls):
        return cls.get_cols().keys()

    @classmethod
    def get_cols(cls):
        return cls.__table__.columns


class Run(Base):
    """The runs"""

    __tablename__ = "runs"

    id = Column(DateTime, primary_key=True, default=datetime.datetime.utcnow)
    log = Column(String)
    warnings = Column(Integer)
    errors = Column(Integer)
    segments_found = Column(Integer)
    segments_written = Column(Integer)
    segments_skipped = Column(Integer)
    config = Column(String)
    program_version = Column(String)


class Class(Base):  # pylint: disable=no-init
    """A Building.
    """
    __tablename__ = 'classes'

    id = Column(Integer, primary_key=True)
    label = Column(String)
    description = Column(String)


class Event(Base):
    """Events"""

    __tablename__ = "events"

    id = Column(String, primary_key=True)
    time = Column(DateTime)
    latitude = Column(Float)
    longitude = Column(Float)
    depth_km = Column(Float)
    author = Column(String)
    catalog = Column(String)
    contributor = Column(String)
    contributor_id = Column(String)
    mag_type = Column(String)
    magnitude = Column(Float)
    mag_author = Column(String)
    event_location_name = Column(String)


# Network|Station|Latitude|Longitude|Elevation|SiteName|StartTime|EndTime
def sta_pkey_default(context):
    return context.current_parameters['network'] + "." + context.current_parameters['station']


class Station(Base):
    """Stations"""

    __tablename__ = "stations"

    id = Column(String, primary_key=True, default=sta_pkey_default, onupdate=sta_pkey_default)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    inventory_xml = Column(Binary)
    __table_args__ = (
                      UniqueConstraint('network', 'station', name='net_sta_uc'),
                     )


def cha_pkey_default(context):
    return context.current_parameters['station_id'] + "." + \
        context.current_parameters['location'] + "." + context.current_parameters['channel']


class Channel(Base):
    """Channels"""

    __tablename__ = "channels"

    id = Column(String, primary_key=True, default=cha_pkey_default, onupdate=cha_pkey_default)
    station_id = Column(String, ForeignKey("stations.id"), nullable=False)
    location = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    depth = Column(Float)
    azimuth = Column(Float)
    dip = Column(Float)
    sensor_description = Column(String)
    scale = Column(Float)
    scale_freq = Column(Float)
    scale_units = Column(String)
    sample_rate = Column(Float)
    __table_args__ = (
                      UniqueConstraint('station_id', 'location', 'channel',
                                       name='net_sta_loc_cha_uc'),
                     )


# def seg_pkey_default(context):
#     return hash((context.current_parameters['channel_id'],
#                  context.current_parameters['start_time'].isoformat(),
#                  context.current_parameters['end_time'].isoformat()))


class Segment(Base):
    """Segments"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True)  # , default=seg_pkey_default, onupdate=seg_pkey_default)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    event_distance_deg = Column(Float)
    data = Column(Binary)
    start_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    query_url = Column(String)
    run_id = Column(DateTime, ForeignKey("runs.id"), nullable=False)
    __table_args__ = (
                      UniqueConstraint('channel_id', 'start_time', 'end_time',
                                       name='net_sta_loc_cha_stime_etime_uc'),
                     )


class Processing(Base):

    __tablename__ = "processing"

    segment_id = Column(Integer, ForeignKey("segments.id"), primary_key=True)
    acc_t05_t95 = Column(Binary)
    fft_acc_t05_t95 = Column(Binary)
    fft_acc_until_t05 = Column(Binary)
    wood_anderson = Column(Binary)
    cumulative = Column(Binary)
    pga = Column(Float)
    pgv = Column(Float)
    pwa = Column(Float)
    t_pga = Column(DateTime)
    t_pgv = Column(DateTime)
    t_pwa = Column(DateTime)
    cum_t05 = Column(DateTime)
    cum_t10 = Column(DateTime)
    cum_t25 = Column(DateTime)
    cum_t50 = Column(DateTime)
    cum_t75 = Column(DateTime)
    cum_t90 = Column(DateTime)
    cum_t95 = Column(DateTime)
    snr = Column(Float)
    snr_t05_t95 = Column(Float)
    snr_t10_t90 = Column(Float)
    amplitude_ratio = Column(Float)
    double_event_result = Column(Integer)


class SegmentClass(Base):

    __tablename__ = "segment_classes"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    class_id_hand_labelled = Column(Boolean)
