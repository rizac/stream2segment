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
    LargeBinary,
    UniqueConstraint,
    event, TypeDecorator)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.hybrid import hybrid_property  # , hybrid_method
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import relationship, backref, deferred, aliased, load_only, \
    selectinload
from sqlalchemy.sql.expression import text, case, select, func  # or_ , and_

# from stream2segment.io import Fdsnws
from stream2segment.io.db import sqlalchemy_version
from stream2segment.io.db.sqlconstructs import concat, deg2km, duration_sec

try:
    from sqlalchemy.ext.declarative import declarative_base  # v<1.4
except ImportError:
    from sqlalchemy.orm import declarative_base  # v1.4+


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
    """Abstract base class for a Stream2segment ORM Model"""

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

    # info = relationship("DownloadRunInfo", uselist=False)


class DownloadRunInfo(Base):
    """
    Model representing ting the download run info. Keep separated as it contains
    relatively big data seldom used in query and never in joins
    """

    __tablename__ = 'download_run_info'

    id = Column(Integer, ForeignKey(DownloadRun.id), primary_key=True)
    s2s_version = Column(String)
    log = Column(String)
    summary = Column(String)
    config = Column(String)


class WebService(Base):
    """Model representing a web service (e.g., event web service)"""
    __tablename__ = 'webservice'

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint('url', name='url_uc'),)


class Event(Base):  # noqa
    """Model representing a seismic Event"""
    __tablename__ = 'event'

    id = Column(Integer, primary_key=True, autoincrement=True)
    webservice_id = Column(
        Integer, ForeignKey(WebService.id), index=True, nullable=True
    )
    eventid = Column(String, nullable=False)
    time = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
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

    quakeml = relationship("QuakeML", uselist=False)  # One-to-one

    @property
    def quakeml_data(self):
        return None if self.quakeml is None else self.quakeml.data

    web_service = relationship(WebService)

    @property
    def url(self):
        return f'{self.web_service.url}?eventid={str(self.eventid)}'

    __table_args__ = (
        UniqueConstraint('catalog', 'eventid', name='ws_eventid_uc'),
    )  # <- tuple


class QuakeML(Base):
    """Model representing a Waveform segment"""
    __tablename__ = 'quakeml'

    id = Column(Integer, ForeignKey(Event.id), primary_key=True)
    data = Column(CompressedBinary, nullable=False)



# def check_datacenter_urls_fdsn(target):  # FIXME REMOVE
#     """Check for datacenter URLs. To be used as argument for sqlalchemy.listen or
#     listen_to (see implementation in this program), e.g.
#     ```
#     @event.listens_for(DataCenter, 'before_insert', check_datacenter_urls_fdsn)
#     @event.listens_for(DataCenter, 'before_update', check_datacenter_urls_fdsn)
#     ```
#     or
#     `event.listens_for(DataCenter, 'before_insert')(check_datacenter_urls_fdsn)`
#     For info on validation see:
#     https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf
#     """
#     # Note: we thought about using validators, but we ended up with infinite
#     # recursion loops
#     fdsn = Fdsnws(target.station_url if target.dataselect_url is None
#                   else target.dataselect_url)
#     target.station_url = fdsn.url(Fdsnws.STATION)
#     target.dataselect_url = fdsn.url(Fdsnws.DATASEL)



class StationXML(Base):
    """Model representing a StationXML data"""
    __tablename__ = 'stationxml'

    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(CompressedBinary, nullable=False)


class Channel(Base):
    """Model representing a Station"""
    __tablename__ = 'channel'

    id = Column(Integer, primary_key=True, autoincrement=True)
    webservice_id = Column(
        Integer, ForeignKey(WebService.id), nullable=False, index=True
    )
    stationxml_id = Column(
        Integer, ForeignKey(StationXML.id), nullable=True, index=True
    )
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
    depth = Column(Float, index=True)
    azimuth = Column(Float)
    dip = Column(Float)
    # sensor_description = Column(String)
    scale = Column(Float)
    scale_freq = Column(Float)
    scale_units = Column(String)
    sample_rate = Column(Float, nullable=False, index=True)

    web_service = relationship(WebService)

    @property
    def url(self):
        params = "&".join([
            f"net={self.network_code}",
            f"sta={self.station_code}",
            f"loc={self.location_code}",
            f"cha={self.channel_code}"
        ])
        return f"{self.webservice.url}?{params}&level=channel"

    # __table_args__ = (
    #     UniqueConstraint('webservice_id', 'network_code', 'station_code', name='channel_uc'),
    # )

    # hybrid properties:

    @hybrid_property
    def network_station_code(self):
        return f"{self.network}.{self.station}"

    @network_station_code.expression
    def network_station_code(cls):  # noqa
        """Return the station code, i.e. self.network + '.' + self.station"""
        return concat(cls.network_code, text("'.'"), cls.station_code). \
            label('network_station_code')

    @hybrid_property
    def channel_code(self):
        return f'{self.band_code}{self.instrument_code}{self.orientation_code}'

    @channel_code.expression
    def channel_code(cls):  # noqa
        """Return the channel code"""
        return concat(cls.band_code, cls.instrument_code, cls.orientation_code).\
            label('channel_code')


# MINISEED_READ_ERROR_CODE = -2  FIXME check where used and remove


class Segment(Base):
    """Model representing a Downloaded segment"""
    __tablename__ = 'segment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey(Event.id), nullable=False, index=True)
    webservice_id = deferred(Column(
        Integer, ForeignKey(WebService.id), nullable=False, index=True
    ))
    channel_id = Column(Integer, ForeignKey(Channel.id), nullable=False, index=True)
    download_run_id = deferred(Column(
        Integer, ForeignKey(DownloadRun.id), nullable=False, index=True
    ))
    event_distance_deg = Column(Float, nullable=False)
    # download_code = Column(Integer, index=True)
    # start_time = Column(DateTime)
    # arrival_time = Column(DateTime, nullable=False)
    # end_time = Column(DateTime)
    # arrival_time_numsamples = Column(Integer, nullable=False)
    # sample_rate = Column(Float)
    noise_window_sec = Column(Float)  # duration (in s) of saved data until arrival_time
    signal_window_sec = Column(Float)  # duration (in s) of saved data from arrival_time
    maxgap_numsamples = Column(Float)
    # request_start = deferred(Column(DateTime, nullable=False))
    # request_end = deferred(Column(DateTime, nullable=False))
    # queryauth = deferred(Column(Boolean, nullable=False, server_default="0"))
    # has_data = Column(Boolean, nullable=False, server_default="0", index=True)

    miniseed = relationship("MiniSeed", uselist=False)  # One-to-one

    @property
    def miniseed_data(self):
        return None if self.miniseed is None else self.miniseed.data

    webservice = relationship(WebService)
    channel = relationship(Channel, backref=backref("segments", lazy="dynamic"))

    @property
    def url(self):
        """Return the full URL that can be used to (re)download the Segment
        waveform data in miniSEED format (For details, see GET request here:
        https://www.fdsn.org/webservices/fdsnws-dataselect-1.1.pdf)
        """
        net, sta = self.channel.network_code, self.channel.station_code
        loc, cha = self.channel.location_code, self.channel.channel_code
        start = self.start_time.isoformat('T')
        end = self.end_time.isoformat('T')
        return (f"{self.webservice.url}?"
                f"net={net}&sta={sta}&loc={loc}&cha={cha}&start={start}&end={end}")

    @property
    def stationxml_url(self):
        params = "&".join([
            f"net={self.channel.network_code}",
            f"sta={self.channel.station_code}"
        ])
        return f"{self.channel.webservice.url}?{params}&level=response"

    event = relationship(Event, backref=backref("segments", lazy="dynamic"))

    # `classes` below is kind-of private, because exposing it in selection expression is
    # complex (it is the only many-to-many relationship) and also in most case redundant,
    # as users is generally interested to have the labels only (see `self.classlabels`):
    # classes = relationship("ClassLabel",  lazy='dynamic',  # viewonly=True,
    #                        secondary="class_labeling",
    #                        backref=backref("segments", lazy="dynamic"))

    download_run = relationship(DownloadRun)

    # relationships (implement here only those shared by download+process):
    # stationxml = relationship(StationXML, backref=backref("segments", lazy="dynamic"))

    # Relationship spanning 3 tables (https://stackoverflow.com/a/17583437)
    # stationxml = relationship(StationXML,
    #                         # `secondary` must be table name in metadata:
    #                         secondary=Channel,
    #                         primaryjoin="Segment.channel_id == Channel.id",
    #                         secondaryjoin="StationXML.id == Channel.stationxml_id",
    #                         uselist=False,
    #                         # the following two params are set in order to make this
    #                         # relationship work in v 1 and 2, but no idea why due to
    #                         # the lack of clarity in sqlalchemy docs
    #                         viewonly=True,
    #                         sync_backref=False,
    #                         backref=backref("segments", lazy="dynamic"))

    __table_args__ = (
        UniqueConstraint('channel_id', 'event_id', name='chaid_evtid_uc'),
    )

    @property
    def siblings(self):
        # Get the SQLAlchemy session managing this instance
        session = self.dbsession

        # If no session is attached, or channel is not set (we need its attributes),
        # return an empty query that matches nothing (safe fallback).
        if not session or not self.channel or self.channel.orientation_code not in {'N', 'Z', 'E', '1', '2', '3'}:
            return session.query(Segment).filter(False)  # noqa

        return (
            session.query(Segment)
            .options(selectinload(Segment.miniseed))
            .join(Channel)
            .filter(
                Segment.id != self.id,                         # exclude self  # noqa
                Segment.channel_id == self.channel_id,         # noqa
                Segment.event_id == self.event_id,
                Channel.location_code == self.channel.location_code,
                Channel.band_code == self.channel.band_code,
                Channel.instrument_code == self.channel.instrument_code
            )
        )



    @hybrid_property
    def event_distance_km(self):
        return self.event_distance_deg * (2.0 * 6371 * pi / 360.0)

    @event_distance_km.expression
    def event_distance_km(cls):  # pylint:disable=no-self-argument
        return deg2km(cls.event_distance_deg)

    @hybrid_property
    def duration_sec(self):
        try:
            return (self.end_time - self.start_time).total_seconds()
        except TypeError:  # some None(s)
            return None

    @duration_sec.expression
    def duration_sec(cls):  # pylint:disable=no-self-argument
        return duration_sec(cls.start_time, cls.end_time)

    @hybrid_property
    def classlabels_count(self):
        return self.classes.count()  # len(self.classes) > 0

    @classlabels_count.expression
    def classlabels_count(cls):  # noqa
        return select(func.count(ClassLabeling.id)).\
            where(ClassLabeling.segment_id == cls.id).\
            label('classlabels_count')

    @property
    def classlabels(self):
        """Return a sorted list of strings denoting the class labels assigned to this
        segment"""
        return sorted(_.label for _ in self.classes.options(load_only(ClassLabel.label)))


class MiniSeed(Base):
    """Model representing a Waveform segment"""
    __tablename__ = 'mini_seed'

    id = Column(Integer, ForeignKey(Segment.id), primary_key=True)
    data = Column(LargeBinary, nullable=False)


class FailedDownloadedSegment(Base):
    """
    Model representing a failed Downloaded segment (no data, server / client error, miniseed error,
    timeout)
    """
    __tablename__ = 'failed_downloaded_segment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey(Event.id), nullable=False, index=True)
    # webservice_id = deferred(Column(
    #     Integer, ForeignKey("WebService.id"), nullable=False, index=True
    # ))
    channel_id = Column(Integer, ForeignKey(Channel.id), nullable=False, index=True)
    download_run_id = deferred(Column(
        Integer, ForeignKey(DownloadRun.id), nullable=False, index=True
    ))
    download_code = Column(Integer, index=True)

    __table_args__ = (
        UniqueConstraint('channel_id', 'event_id', name='chaid_evtid_uc'),
    )

class ClassLabel(Base):
    """Model representing a segment class label"""
    __tablename__ = 'class_label'

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    description = deferred(Column(String))

    __table_args__ = (  # noqa
        UniqueConstraint('label', name='class_label_name_uc'),
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
        UniqueConstraint('segment_id', 'class_label_id', name='seg_class_uc'),
    )
