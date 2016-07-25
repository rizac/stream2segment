'''
Created on Jul 15, 2016

@author: riccardo
'''
from pandas import to_datetime, to_numeric
# from sqlalchemy import engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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
from stream2segment.s2sio.db.pd_sql_utils import get_col_names, get_cols # , df_to_table_rows
# from sqlalchemy.sql.sqltypes import BigInteger, BLOB
# from sqlalchemy.sql.schema import ForeignKey

_Base = declarative_base()


class Base(_Base):

    __abstract__ = True

    @classmethod
    def get_cols(cls):
        return get_cols(cls)

    @classmethod
    def get_col_names(cls):
        return get_col_names(cls)


class FDSNBase(Base):
    """Base class which translates a FDSN query. See Event, Station and Channel subclasses"""
    __abstract__ = True

    @classmethod
    def rename_cols(cls, fdsn_query_df, *args, **kwargs):
        col_mapping = cls.get_col_mapping(fdsn_query_df, *args, **kwargs)
        return fdsn_query_df.rename(columns=col_mapping)  # [col_mapping.values()]

    @classmethod
    def get_col_mapping(cls, dataframe, *args, **kwargs):
        raise NotImplementedError("Subclass of FDSNBase does not overrides `get_col_mapping`")


class Run(Base):
    """The runs"""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)  # auto increment should be set to True
    run_time = Column(DateTime, unique=True, default=datetime.datetime.utcnow)
    log = Column(String)
    warnings = Column(Integer)
    errors = Column(Integer)
    # segments_found = Column(Integer)
    # segments_written = Column(Integer)
    # segments_skipped = Column(Integer)
    config = Column(String)
    program_version = Column(String)


def dc_datasel_default(context):
    return context.current_parameters['station_query_url'].replace("/station", "/dataselect")


class DataCenter(Base):
    """DataCenters"""

    __tablename__ = "data_centers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_query_url = Column(String, nullable=False)
    dataselect_query_url = Column(String, default=dc_datasel_default, onupdate=dc_datasel_default)

    segments = relationship("Segment", backref="data_centers")
    stations = relationship("Station", backref="data_centers")

    __table_args__ = (
                      UniqueConstraint('station_query_url', 'dataselect_query_url',
                                       name='sta_data_uc'),
                     )


class Event(FDSNBase):
    """Events"""

    __tablename__ = "events"

    id = Column(String, primary_key=True, autoincrement=False)
    time = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_km = Column(Float, nullable=False)
    author = Column(String)
    catalog = Column(String)
    contributor = Column(String)
    contributor_id = Column(String)
    mag_type = Column(String)
    magnitude = Column(Float, nullable=False)
    mag_author = Column(String)
    event_location_name = Column(String)

    segments = relationship("Segment", backref="events")

    @classmethod
    def get_col_mapping(cls, dataframe):
        # "+" operator is weird on dataframe.columns object. Although not used here convert to list:
        dfcols = dataframe.columns.values.tolist()
        mycols = cls.get_col_names()
        return {cold: cnew for cold, cnew in zip(dfcols, mycols)}


def sta_pkey_default(context):
    return context.current_parameters['network'] + "." + context.current_parameters['station']


class Station(FDSNBase):
    """Stations"""

    __tablename__ = "stations"

    id = Column(String, primary_key=True, default=sta_pkey_default, onupdate=sta_pkey_default)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    inventory_xml = Column(Binary)

    __table_args__ = (
                      UniqueConstraint('network', 'station', name='net_sta_uc'),
                     )

    channels = relationship("Channel", backref="stations")

    @classmethod
    def get_col_mapping(cls, dataframe, level):
        # "+" operator is weird on dataframe.columns object, convert to list:
        dfcols = dataframe.columns.values.tolist()
        mycols = cls.get_col_names()
        if level == 'channel':
            # these are the dataframe expected columns for a station (level=channel) query:
            #  #Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|
            #  SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
            # Some of them are for the Channel table, so select them:
            dfcols = dfcols[:2] + dfcols[4:7] + dfcols[-2:]
            # and now this table columns mapping (by name, so we can safely add any new column at
            # any index):
            mycols = [Station.network.key, Station.station.key, Station.latitude.key,
                      Station.longitude.key, Station.elevation.key, Station.start_time.key,
                      Station.end_time.key]
        elif level == 'station':
            # these are the dataframe columns for a station (level=channel) query:
            #  #Network|Station|Latitude|Longitude|Elevation|SiteName|StartTime|EndTime
            # set this table columns mapping (by name, so we can safely add any new column at any
            # index):
            mycols = [Station.network.key, Station.station.key, Station.latitude.key,
                      Station.longitude.key, Station.elevation.key, Station.site_name.key,
                      Station.start_time.key, Station.end_time.key]
        else:
            raise ValueError("'%s.get_col_mapping' error: expected 'channel' or 'station' as "
                             "'level' argument value, found '%s'" % str(level))

        return {cold: cnew for cold, cnew in zip(dfcols, mycols)}


def cha_pkey_default(context):
    return context.current_parameters['station_id'] + "." + \
        context.current_parameters['location'] + "." + context.current_parameters['channel']


class Channel(FDSNBase):
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
    sample_rate = Column(Float, nullable=False)

    __table_args__ = (
                      UniqueConstraint('station_id', 'location', 'channel',
                                       name='net_sta_loc_cha_uc'),
                     )

    segments = relationship("Segment", backref="channels")

    @classmethod
    def get_col_mapping(cls, dataframe):
        # "+" operator is weird on dataframe.columns object, convert to list:
        dfcols = dataframe.columns.values.tolist()
        mycols = cls.get_col_names()
        # these are the dataframe columns for a station (level=channel) query:
        #  #Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|
        #  SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
        # Some of them are for the Station table, so select them:
        dfcols = dfcols[2:4] + dfcols[7:15]
        # and now this table columns mapping (by name, so we can safely add any new column at
        # any index):
        mycols = [Channel.location.key, Channel.channel.key, Channel.depth.key, Channel.azimuth.key,
                  Channel.dip.key, Channel.sensor_description.key, Channel.scale.key,
                  Channel.scale_freq.key, Channel.scale_units.key, Channel.sample_rate.key]
        return {cold: cnew for cold, cnew in zip(dfcols, mycols)}


class SegmentClassAssociation(Base):

    __tablename__ = "segment_class_asociations"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    class_id_hand_labelled = Column(Boolean)

    __table_args__ = (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)


class Class(Base):  # pylint: disable=no-init
    """A Building.
    """
    __tablename__ = 'classes'

    id = Column(Integer, primary_key=True, autoincrement=False)
    label = Column(String)
    description = Column(String)

#     segments = relationship(
#         "Segment",
#         secondary=SegmentClassAssociation,
#         back_populates="classes")


class Segment(Base):
    """Segments"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True)  # , default=seg_pkey_default, onupdate=seg_pkey_default)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    event_distance_deg = Column(Float, nullable=False)
    data = Column(Binary, nullable=False)
    start_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

    processings = relationship("Processing", backref="segments")
#     classes = relationship(
#         "Class",
#         secondary=SegmentClassAssociation,
#         back_populates="segments")

    __table_args__ = (
                      UniqueConstraint('channel_id', 'start_time', 'end_time',
                                       name='net_sta_loc_cha_stime_etime_uc'),
                     )


class Processing(Base):

    __tablename__ = "processing"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"))
    run_id = Column(Integer, ForeignKey("runs.id"))
    mseed_rem_resp_savewindow = Column(Binary)
    fft_rem_resp_t05_t95 = Column(Binary)
    fft_rem_resp_until_atime = Column(Binary)
    wood_anderson_savewindow = Column(Binary)
    cumulative = Column(Binary)
    pga_atime_t95 = Column(Float)
    pgv_atime_t95 = Column(Float)
    pwa_atime_t95 = Column(Float)
    t_pga_atime_t95 = Column(DateTime)
    t_pgv_atime_t95 = Column(DateTime)
    t_pwa_atime_t95 = Column(DateTime)
    cum_t05 = Column(DateTime)
    cum_t10 = Column(DateTime)
    cum_t25 = Column(DateTime)
    cum_t50 = Column(DateTime)
    cum_t75 = Column(DateTime)
    cum_t90 = Column(DateTime)
    cum_t95 = Column(DateTime)
    snr_rem_resp_fixedwindow = Column(Float)
    snr_rem_resp_t05_t95 = Column(Float)
    snr_rem_resp_t10_t90 = Column(Float)
    amplitude_ratio = Column(Float)
    is_saturated = Column(Boolean)
    has_gaps = Column(Boolean)
    double_event_result = Column(Integer)
    secondary_event_time = Column(DateTime)
    coda_tmax = Column(DateTime)
    coda_length_sec = Column(Float)

    __table_args__ = (UniqueConstraint('segment_id', 'run_id', name='seg_run_uc'),)



# FIXME: implement runs datetime server side, and run test to see it's utc!
# FIXME: implement relations, and test joins
# fixme: implement DataFrame write, and test it
# fixme: implement ondelete and oncascade when possible
# FIXME: many to one with respect to segments table!!
