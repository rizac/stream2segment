"""
s2s database ORM
"""
# :date: Jul 15, 2016
import gzip
import sqlite3
from math import pi
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    SmallInteger,
    LargeBinary,
    UniqueConstraint,
    event,
    TypeDecorator,
    Index
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.expression import literal, case, select, func  # or_ , and_
from stream2segment.io.db import sqlalchemy_version

try:
    from sqlalchemy.orm import declarative_base  # v1.4+
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base  # v<1.4


if sqlalchemy_version < 2:  # https://stackoverflow.com/a/75634238
    __sa_select__ = select

    def select(*entities, **kw):
        """backward compatible select"""
        return __sa_select__(list(entities), **kw)

    __sa_case__ = case

    def case(*entities, **kw):
        """backward compatible select"""
        return __sa_case__(list(entities), **kw)  # noqa


Base = declarative_base()


class CompressedBinary(TypeDecorator):
    """Custom column type for compressed XML (QuakeML and StationXML)"""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else gzip.compress(value, compresslevel=9)

    def process_result_value(self, value, dialect):
        return None if value is None else gzip.decompress(value)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Turn foreign keys ON for SQLite. For info see:
    https://stackoverflow.com/a/13719230

    :param dbapi_connection:
    :param connection_record:
    :return:
    """
    # play well with other DB backends:
    if type(dbapi_connection) is sqlite3.Connection:  # @UndefinedVariable
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

on_del_upd_cascade = {
    'onupdate': 'CASCADE',
    'ondelete': 'CASCADE'
}

class DownloadRun(Base):  # noqa
    """Model representing the executed downloads"""
    __tablename__ = 'download_run'

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, server_default=func.now())  # FIXME: needed?
    s2s_version = Column(String)
    log = Column(String)
    summary = Column(String)
    config = Column(String)


class WebService(Base):
    """Model representing a web service (e.g., event web service)"""
    __tablename__ = 'webservice'

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint('url', name='unique_url'),)


class Event(Base):  # noqa
    """Model representing a seismic Event"""
    __tablename__ = 'event'

    id = Column(Integer, primary_key=True, autoincrement=True)
    webservice_id = Column(Integer, ForeignKey(WebService.id), nullable=False)
    eventid = Column(String, nullable=False)
    time = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    depth_km = Column(Float, nullable=False)
    # author = Column(String)
    catalog = Column(String, nullable=False)
    # contributor = Column(String)
    # contributor_id = Column(String)
    mag_type = Column(String)
    magnitude = Column(Float, nullable=False, index=True)
    # mag_author = Column(String)
    # event_location_name = Column(String)
    # event_type = Column(String)

    __table_args__ = (
        UniqueConstraint('catalog', 'eventid', name='ws_eventid_uc'),
    )  # <- tuple


class QuakeML(Base):
    """Model representing a Waveform segment"""
    __tablename__ = 'quakeml'

    id = Column(
        Integer,
        ForeignKey(Event.id, **on_del_upd_cascade),
        primary_key=True,
        nullable=False
    )
    data = Column(CompressedBinary, nullable=False)


class StationXML(Base):
    """Model representing a StationXML data"""
    __tablename__ = 'stationxml'

    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(CompressedBinary, nullable=False)


class Channel(Base):
    """Model representing a Station"""
    __tablename__ = 'channel'

    id = Column(Integer, primary_key=True, autoincrement=True)
    # webservice_id = Column(Integer, ForeignKey(WebService.id), nullable=False)
    stationxml_id = Column(
        Integer,
        ForeignKey(StationXML.id, ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True
    )
    network_code = Column(String(8), nullable=False, index=True)
    station_code = Column(String(8), nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    elevation = Column(Float)
    # site_name = Column(String)
    # start_time = Column(DateTime, nullable=False)  # = channel start time
    # end_time = Column(DateTime)  # = channel end time
    location_code = Column(String(8), nullable=False)
    band_code = Column(String(1), nullable=False)
    instrument_code = Column(String(1), nullable=False)
    orientation_code = Column(String(1), nullable=False)
    depth = Column(Float)
    azimuth = Column(Float)
    dip = Column(Float)
    # sensor_description = Column(String)
    # scale = Column(Float)
    # scale_freq = Column(Float)
    # scale_units = Column(String)
    sample_rate = Column(Float, nullable=False)

    __table_args__ = (
        Index(
            'channel_without_orientation__index',
            'network_code',
            'station_code',
            'location_code',
            'instrument_code',
            'band_code'
        ),
        UniqueConstraint(
            'network_code',
            'station_code',
            'location_code',
            'band_code',
            'instrument_code',
            'orientation_code',
            #'webservice_id',
            name='unique_channel'
        ),
    )

    # hybrid properties:

    @hybrid_property
    def network_station_code(self):
        return f"{self.network_code}.{self.station_code}"

    @network_station_code.expression
    def network_station_code(cls):
        return func.concat(cls.network_code, literal("."), cls.station_code)

    @hybrid_property
    def channel_code(self):
        return f"{self.band_code}{self.instrument_code}{self.orientation_code}"

    @channel_code.expression
    def channel_code(cls):
        return func.concat(cls.band_code, cls.instrument_code, cls.orientation_code)


class Segment(Base):
    """Model representing a Downloaded segment"""
    __tablename__ = 'segment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey(Event.id, **on_del_upd_cascade), nullable=False
    )
    channel_id = Column(
        Integer, ForeignKey(Channel.id, **on_del_upd_cascade), nullable=False
    )
    webservice_id = Column(Integer, ForeignKey(WebService.id), nullable=False)

    event_distance_km = Column(SmallInteger, nullable=False, index=True)
    noise_window_s = Column(SmallInteger, nullable=False)  # duration (in s) of start time relative to arrival_time (often < 0)  # noqa
    signal_window_s = Column(SmallInteger, nullable=False)  # duration (in s) of end time relative to arrival_time (often > 0)  # noqa
    gap_score_percent = Column(SmallInteger, nullable=False)

    __table_args__ = (
        Index("okseg_event_channel_index", "event_id", "channel_id"),
        UniqueConstraint('event_id', 'channel_id', name='unique_event_channel'),
    )

    @hybrid_property
    def event_distance_deg(self):
        return self.event_distance_km / (2.0 * 6371 * pi / 360.0)

    @event_distance_deg.expression
    def event_distance_deg(cls):
        return cls.event_distance_km / (2.0 * 6371 * pi / 360.0)

    @hybrid_property
    def duration_s(self):
        return self.signal_window_s - self.noise_window_s

    @duration_s.expression
    def duration_s(cls):
        return cls.signal_window_s - cls.noise_window_s


class MiniSeed(Base):
    """Model representing a Waveform segment"""
    __tablename__ = 'miniseed'

    id = Column(
        Integer,
        ForeignKey(Segment.id, **on_del_upd_cascade),
        primary_key=True,
        nullable=False
    )
    data = Column(LargeBinary, nullable=False)


class SkippedSegment(Base):
    """
    Model representing a segment with no data (204 Http response,
    miniSEED data error, time out of range)
    """
    __tablename__ = 'skipped_segment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey(Event.id, **on_del_upd_cascade), nullable=False
    )
    channel_id = Column(
        Integer, ForeignKey(Channel.id, **on_del_upd_cascade), nullable=False
    )
    download_code = Column(SmallInteger)

    __table_args__ = (
        Index("skipseg_event_channel_index", "event_id", "channel_id"),
        UniqueConstraint('event_id', 'channel_id', name='unique_event_channel'),
    )

class ClassLabel(Base):
    """Model representing a segment class label"""
    __tablename__ = 'class_label'

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String, nullable=False)
    description = Column(String)

    __table_args__ = (  # noqa
        UniqueConstraint('label', name='unique_label'),
    )


class ClassLabeling(Base):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""
    __tablename__ = 'class_labeling'

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(
        Integer, ForeignKey(Segment.id, **on_del_upd_cascade), nullable=False
    )
    class_label_id = Column(
        Integer,
        ForeignKey(ClassLabel.id, ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False
    )
    is_hand_labelled = Column(Boolean, server_default="1")
    annotator = Column(String)

    __table_args__ = (
        UniqueConstraint(
            'segment_id', 'class_label_id', name='unique_segment_label'
        ),
    )
