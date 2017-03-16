'''
Created on Jul 15, 2016

@author: riccardo
'''
import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, deferred
from sqlalchemy import (
    Column,
    ForeignKey as SqlAlchemyForeignKey,  # we override it (see below)
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
    event)
from sqlalchemy.sql.expression import func
from sqlalchemy.orm.mapper import validates
# from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect

_Base = declarative_base()

from sqlalchemy.engine import Engine  # @IgnorePep8
import sqlite3  # @IgnorePep8


# http://stackoverflow.com/questions/13712381/how-to-turn-on-pragma-foreign-keys-on-in-sqlalchemy-migration-script-or-conf
# for setting foreign keys in sqlite:
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if type(dbapi_connection) is sqlite3.Connection:  # play well with other DB backends
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(_Base):

    __abstract__ = True

    def __str__(self):
        cls = self.__class__
        ret = [cls.__name__ + ":"]
        insp = inspect(cls)
        for colname, col in insp.columns.items():
            typ = col.type
            typ_str = str(typ)
            try:
                val = "(not shown)" if type(typ) == Binary else str(getattr(self, colname))
            except Exception as exc:
                val = "(not shown: %s)" % str(exc)
            ret.append("%s %s: %s" % (colname, typ_str, val))
        for relationship in insp.relationships.keys():
            ret.append("%s: relationship" % relationship)
        return "\n".join(ret)


def ForeignKey(*pos, **kwa):
    """Overrides the ForeignKey defined in SqlAlchemy by providing default
    `onupdate='CASCADE'` and `ondelete='CASCADE'` if the two keyword argument are missing in `kwa`.
    As all Foreign keys defined here have nullable=False, this seem to be a reasonable choice.
    If this behavior needs to be modified for some column in the future,
    just provide the arguments in the constructor as one would do with sqlalchemy ForeignKey class
    E.g.: column = Column(..., ForeignKey(..., onupdate='SET NULL',...), nullable=True)
    """
    if 'onupdate' not in kwa:
        kwa['onupdate'] = 'CASCADE'
    if 'ondelete' not in kwa:
        kwa['ondelete'] = 'CASCADE'
    return SqlAlchemyForeignKey(*pos, **kwa)


class Run(Base):
    """The runs"""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)  # pylint:disable=invalid-name
    # run_time below has server_default as `func.now()`. This issues a CURRENT TIMESTAMP
    # on the SQL side. That's ok, BUT the column CANNOT BE UNIQUE!!
    # the CURRENT TIMESTAMP is evaluated once at the beginning of an SQL Statement,
    # so two references in the same session will result in the same value
    # (https://www.ibm.com/developerworks/community/blogs/SQLTips4DB2LUW/entry/current_timestamp?lang=en)
    # If we need to make a datetime unique, then either specify
    # 1) default=datetime.datetime.utcnow() BUT NO server_default (the latter seems to have
    # priority if both are provided)
    # 2) or don't make the column unique (what we did)
    run_time = Column(DateTime, server_default=func.now())
    log = deferred(Column(String))
    warnings = Column(Integer, server_default="0", default=0)
    errors = Column(Integer, server_default="0", default=0)
    # segments_found = Column(Integer)
    # segments_written = Column(Integer)
    # segments_skipped = Column(Integer)
    config = deferred(Column(String))
    program_version = Column(String)


# def dc_datasel_default(context):
#     return context.current_parameters['station_query_url'].replace("/station", "/dataselect")


class DataCenter(Base):
    """DataCenters"""

    __tablename__ = "data_centers"

    id = Column(Integer, primary_key=True, autoincrement=True)  # pylint:disable=invalid-name
    station_query_url = Column(String, nullable=False)  # if you change attr, see BELOW!
    dataselect_query_url = Column(String, nullable=False)  # , default=dc_datasel_default, onupdate=dc_datasel_default)

    # segments = relationship("Segment", backref="data_centers")
    # stations = relationship("Station", backref="data_centers")

    __table_args__ = (
                      UniqueConstraint('station_query_url', 'dataselect_query_url',
                                       name='sta_data_uc'),
                     )


# standard decorator style for event listener. Note: we cannot implement validators otherwise
# we have an infinite recursion loop!
@event.listens_for(DataCenter, 'before_insert')
@event.listens_for(DataCenter, 'before_update')
def receive_before_update(mapper, connection, target):
    """listen for the 'before_update' event. For info on validation see:
     https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf
    """
    if target.station_query_url is not None and \
            '/station/' in target.station_query_url and target.dataselect_query_url is None:
        target.dataselect_query_url = target.station_query_url.replace("/station/", "/dataselect/")
    elif target.dataselect_query_url is not None and \
            '/dataselect/' in target.dataselect_query_url and target.station_query_url is None:
        target.station_query_url = target.dataselect_query_url.replace("/dataselect/", "/station/")


class Event(Base):
    """Events"""

    __tablename__ = "events"

    id = Column(String, primary_key=True, autoincrement=False,
                nullable=False)  # pylint:disable=invalid-name
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

    # segments = relationship("Segment", back_populates="event")


# def sta_pkey_default(context):
#     return context.current_parameters['network'] + "." + context.current_parameters['station']


class Station(Base):
    """Stations"""

    __tablename__ = "stations"

    id = Column(String, primary_key=True)  # , default=sta_pkey_default, onupdate=sta_pkey_default)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    inventory_xml = deferred(Column(Binary))  # lazy load: only upon direct access

    __table_args__ = (
                      UniqueConstraint('network', 'station', name='net_sta_uc'),
                     )

    # http://stackoverflow.com/questions/33708219/how-to-handle-sqlalchemy-onupdate-when-current-context-is-empty
    # we prefer over event.listen_for cause the function is inside the class (clearer)
    @validates('network', 'station')
    def update_id(self, key, value):
        vals = (value, self.station) if key == 'network' else (self.network, value)
        if all(i is not None for i in vals):
            self.id = "%s.%s" % vals
        return value

    datacenter = relationship("DataCenter", backref=backref("stations", lazy="dynamic"))


# @event.listens_for(Station.network, 'set')
# def update_id_from_network(target, value, oldvalue, initiator):
#     if target.station:
#         target.id = value + "." + target.station
# 
# 
# @event.listens_for(Station.station, 'set')
# def update_id_from_station(target, value, oldvalue, initiator):
#     target.id = target


# def cha_pkey_default(context):
#     return context.current_parameters['station_id'] + "." + \
#         context.current_parameters['location'] + "." + context.current_parameters['channel']


class Channel(Base):
    """Channels"""

    __tablename__ = "channels"

    id = Column(String, primary_key=True)  # , default=cha_pkey_default, onupdate=cha_pkey_default)
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

    # http://stackoverflow.com/questions/33708219/how-to-handle-sqlalchemy-onupdate-when-current-context-is-empty
    # we prefer over event.listen_for cause the function is inside the class (clearer)
    @validates('station_id', 'location', 'channel')
    def update_id(self, key, value):
        vals = (value, self.location, self.channel) if key == 'station_id' else \
                (self.station_id, value, self.channel) if key == 'location' else \
                (self.station_id, self.location, value)
        if all(i is not None for i in vals):
            self.id = "%s.%s.%s" % vals
        return value

    station = relationship("Station", backref=backref("channels", lazy="dynamic"))


class Segment(Base):
    """The Segments table"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True)  # , default=seg_pkey_default, onupdate=seg_pkey_default)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    event_distance_deg = Column(Float, nullable=False)
    data = deferred(Column(Binary))  # lazy load only upon access
    start_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

    event = relationship("Event", backref=backref("segments", lazy="dynamic"))
    channel = relationship("Channel", backref=backref("segments", lazy="dynamic"))
    datacenter = relationship("DataCenter", backref=backref("segments", lazy="dynamic"))
    run = relationship("Run", backref=backref("segments", lazy="dynamic"))
    # http://stackoverflow.com/questions/17580649/sqlalchemy-relationships-across-multiple-tables
    # this method will work better, as the ORM can also handle
    # eager loading with this one.
    station = relationship("Station", secondary="channels",  # <-  must be table name in metadata
                           primaryjoin="Segment.channel_id == Channel.id",
                           secondaryjoin="Station.id == Channel.station_id",
                           viewonly=True, uselist=False,
                           backref=backref("segments", lazy="dynamic"))
    classes = relationship("Class", lazy='dynamic',
                           secondary="class_labellings",  # <-  must be table name in metadata
                           viewonly=True, backref=backref("segments", lazy="dynamic"))

    __table_args__ = (
                      UniqueConstraint('channel_id', 'start_time', 'end_time',
                                       name='net_sta_loc_cha_stime_etime_uc'),
                     )


class Class(Base):  # pylint: disable=no-init
    """A class label"""
    __tablename__ = 'classes'

    id = Column(Integer, primary_key=True)
    label = Column(String)
    description = Column(String)


class ClassLabelling(Base):

    __tablename__ = "class_labellings"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    is_hand_labelled = Column(Boolean, server_default="1")  # Note: "TRUE" fails in sqlite!

    __table_args__ = (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)
