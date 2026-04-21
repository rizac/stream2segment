"""
s2s database ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import gzip
import sqlite3
from math import pi
from sqlalchemy import (
    Column,
    ForeignKey as SqlAlchemyForeignKey,  # we override it (see below)
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
from sqlalchemy.ext.hybrid import hybrid_property  # , hybrid_method
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import relationship, backref, deferred, load_only
from sqlalchemy.sql.expression import literal, case, select, func  # or_ , and_

# from stream2segment.io import Fdsnws
from stream2segment.io.db import sqlalchemy_version
# from stream2segment.io.db.sqlconstructs import concat, deg2km, duration_sec

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


class _Base:
    """Abstract base class for a Stream2segment ORM Model"""  # FIXME REMOVE!

    def __str__(self):
        """Return a meaningful string representation (with info on loaded
        columns and related objects)"""
        cls = self.__class__
        ret = [str(cls.__name__)]
        # provide a meaningful str representation, but show only loaded
        # attributes (https://stackoverflow.com/a/261191)
        mapper = inspect(cls)
        me_dict = self.__dict__
        loaded_cols, unloaded_cols = 0, 0
        idx = 1
        maxchar = 10
        ret.append('')
        for c in mapper.columns.keys():
            if c in me_dict:
                val = me_dict[c]
                cut_str = ''
                if hasattr(val, "__len__") and len(val) > maxchar:
                    elm = 'characters' if isinstance(val, str) else 'elements'
                    cut_str = ', %d %s, showing first %d only' % \
                              (len(val), elm, maxchar)
                    val = val[:maxchar]
                ret.append("  %s: %s (%s%s)" % (
                    c, str(val), str(val.__class__.__name__), cut_str))
                loaded_cols += 1
            else:
                ret.append("  %s" % c)
                unloaded_cols += 1
        ret[idx] = ' attributes (%d of %d loaded):' % (
            loaded_cols, loaded_cols + unloaded_cols)
        idx = len(ret)
        ret.append('')
        loaded_rels, unloaded_rels = 0, 0
        for r in mapper.relationships.keys():
            if r in me_dict:
                ret.append("  %s: `%s` object" %
                           (r, str(me_dict[r].__class__.__name__)))
                loaded_rels += 1
            else:
                ret.append("  %s" % r)
                unloaded_rels += 1
        ret[idx] = ' related_objects (%d of %d loaded):' % \
                   (loaded_rels, loaded_rels + unloaded_rels)
        return "\n".join(ret)


Base = declarative_base(cls=_Base)


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


def ForeignKey(*pos, **kwa):
    """Override the ForeignKey defined in SqlAlchemy by providing default
    `onupdate='CASCADE'` and `ondelete='CASCADE'` if the two keyword argument
    are missing in `kwa`. If this behavior needs to be modified for some column
    in the future, just provide the arguments in the constructor as one would
    do with sqlalchemy ForeignKey class E.g.:
    col = Column(..., ForeignKey(..., onupdate='SET NULL',...), nullable=False)
    """
    if 'onupdate' not in kwa:
        kwa['onupdate'] = 'CASCADE'
    if 'ondelete' not in kwa:
        kwa['ondelete'] = 'CASCADE'
    return SqlAlchemyForeignKey(*pos, **kwa)


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

    id = Column(Integer, ForeignKey(Event.id), primary_key=True)
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
    webservice_id = Column(Integer, ForeignKey(WebService.id), nullable=False)
    stationxml_id = Column(Integer, ForeignKey(StationXML.id), nullable=True)
    network_code = Column(String(8), nullable=False, index=True)
    station_code = Column(String(8), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    # site_name = Column(String)
    start_time = Column(DateTime, nullable=False)  # = channel start time
    end_time = Column(DateTime)  # = channel end time
    location_code = Column(String(8), nullable=False, index=True)
    band_code = Column(String(1), nullable=False, index=True)
    instrument_code = Column(String(1), nullable=False, index=True)
    orientation_code = Column(String(1), nullable=False, index=True)
    depth = Column(Float)
    azimuth = Column(Float)
    dip = Column(Float)
    # sensor_description = Column(String)
    scale = Column(Float)
    scale_freq = Column(Float)
    scale_units = Column(String)
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
            'start_time',
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


# MINISEED_READ_ERROR_CODE = -2  FIXME check where used and remove


class MiniSeed(Base):
    """Model representing a Waveform segment"""
    __tablename__ = 'miniseed'

    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(LargeBinary, nullable=False)


class Segment(Base):
    """Model representing a Downloaded segment"""
    __tablename__ = 'segment'

    id = Column(Integer, ForeignKey(MiniSeed.id), primary_key=True)
    event_id = Column(Integer, ForeignKey(Event.id), nullable=False)
    webservice_id = Column(Integer, ForeignKey(WebService.id), nullable=False)
    channel_id = Column(Integer, ForeignKey(Channel.id), nullable=False)
    event_distance_km = Column(SmallInteger, nullable=False, index=True)
    noise_window_s = Column(SmallInteger, nullable=False)  # duration (in s) of start time relative to arrival_time (often < 0)
    signal_window_s = Column(SmallInteger, nullable=False)  # duration (in s) of end time relative to arrival_time (often > 0)
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



class SkippedSegment(Base):
    """
    Model representing a segment with no data (204 Http response,
    miniSEED data error, time out of range)
    """
    __tablename__ = 'skipped_segment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey(Event.id), nullable=False)
    channel_id = Column(Integer, ForeignKey(Channel.id), nullable=False)
    download_code = Column(SmallInteger)

    __table_args__ = (
        Index("skipseg_event_channel_index", "event_id", "channel_id"),
        UniqueConstraint('event_id', 'channel_id', name='unique_event_channel'),
    )

class ClassLabel(Base):
    """Model representing a segment class label"""
    __tablename__ = 'class_label'

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    description = deferred(Column(String))

    __table_args__ = (  # noqa
        UniqueConstraint('label', name='unique_label'),
    )


class ClassLabeling(Base):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""
    __tablename__ = 'class_labeling'

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey(Segment.id), nullable=False)
    class_label_id = Column(Integer, ForeignKey(ClassLabel.id), nullable=False)
    is_hand_labelled = Column(Boolean, server_default="1")
    annotator = Column(String)

    __table_args__ = (
        UniqueConstraint(
            'segment_id', 'class_label_id', name='unique_segment_label'
        ),
    )
