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
    LargeBinary,
    # Numeric,
    # event,
    # CheckConstraint,
    # BigInteger,
    UniqueConstraint,
    event)
from sqlalchemy.sql.expression import func, text
from sqlalchemy.orm.mapper import validates
# from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method

_Base = declarative_base()

from sqlalchemy.engine import Engine  # @IgnorePep8
import sqlite3  # @IgnorePep8


def withdata(model_column):
    """Returns a filter argument for returning instances with values of
    `model_column` NOT *empty* nor *null*. `model_column` type must be STRING or BLOB
    :param model_column: A valid column name, e.g. an attribute Column defined in some
    sqlalchemy orm model class (e.g., 'User.data'). **The type of the column must be STRING or
    BLOB**, otherwise result is undefined. For instance, numeric column with zero as value
    are *not* empty (as the sql length function applied to numeric returns the number of
    bytes)
    :example:
    ```
    # given a table User, return empty or none via "~"
    session.query(User.id).filter(~withdata(User.data)).all()

    # return "valid" columns:
    session.query(User.id).filter(withdata(User.data)).all()
    ```
    """
    return (model_column.isnot(None)) & (func.length(model_column) > 0)


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
                val = "(not shown)" if type(typ) == LargeBinary else str(getattr(self, colname))
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
    warnings = Column(Integer, server_default=text('0'))  # , default=0)
    errors = Column(Integer, server_default=text('0'))  # , default=0)
    # segments_found = Column(Integer)
    # segments_written = Column(Integer)
    # segments_skipped = Column(Integer)
    config = deferred(Column(String))
    program_version = Column(String)


# def dc_datasel_default(context):
#     return context.current_parameters['station_url'].replace("/station", "/dataselect")


class Event(Base):
    """Events"""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)  # pylint:disable=invalid-name
    webservice_id = Column(Integer, ForeignKey("web_services.id"), nullable=False)
    eventid = Column(String, nullable=False)
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

    __table_args__ = (
                      UniqueConstraint('webservice_id', 'eventid', name='ws_eventid_uc'),
                     )
    # segments = relationship("Segment", back_populates="event")


# the class below might be linked to a station_url and dataselect_url, and be a FDSNWSService
# table to whom events, stations and dataselect are all linked. However,
# Whereas IRIS provides a single service for the three of them, EIDA provides only station and
# dataselect. Moreover, EIDA has several datacenters (federated) whereas IRIS has just one (centralized)
# The program will have thus a "service" parameter which can be set to "eida" or "iris" and 
# by means of the routing service we then do our filtering of networks and stations accordingly
class WebService(Base):
    """event fdsn service"""
    __tablename__ = "web_services"

    id = Column(Integer, primary_key=True, autoincrement=True)  # pylint:disable=invalid-name
    name = Column(String)
    type = Column(String)
    url = Column(String, nullable=False)  # if you change attr, see BELOW!

    # segments = relationship("Segment", backref="data_centers")
    # stations = relationship("Station", backref="data_centers")

    __table_args__ = (
                      UniqueConstraint('url', name='url_uc'),
                     )


class DataCenter(Base):
    """DataCenters"""

    __tablename__ = "data_centers"

    id = Column(Integer, primary_key=True, autoincrement=True)  # pylint:disable=invalid-name
    station_url = Column(String, nullable=False)  # if you change attr, see BELOW!
    dataselect_url = Column(String, nullable=False)  # , default=dc_datasel_default, onupdate=dc_datasel_default)

    # segments = relationship("Segment", backref="data_centers")
    # stations = relationship("Station", backref="data_centers")

    @hybrid_property
    def netloc(self):
        """Returns the network location of the current datacenter, following urlparse.netloc
        This property is only intended for test purposes
        """
        # we implement our custom function to avoid useless imports
        address = self.station_url if self.station_url else self.dataselect_url
        if not address:
            return ''
        address_ = address.lower()
        start = 0
        if address_.startswith("http://"):
            start = 7
        elif address_.startswith("https://"):
            start = 8
        end = address.find("/", start)
        if end == -1:
            end = None
        return address[start:end]

    __table_args__ = (
                      UniqueConstraint('station_url', 'dataselect_url',
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
    if target.station_url is not None and target.dataselect_url is None:
        target.dataselect_url = dc_get_other_service_url(target.station_url)
    elif target.dataselect_url is not None and target.station_url is None:
        target.station_url = dc_get_other_service_url(target.dataselect_url)


def dc_get_other_service_url(url):
    """Returns the dataselect service if url denotes a datacenter station service url,
    otherwise the station service. If dc_url has nor "/station/" neither "/dataselect/" in its
    string, a ValueError is raised"""
    if '/station/' in url:
        return url.replace("/station/", "/dataselect/")
    elif '/dataselect/' in url:
        return url.replace("/dataselect/", "/station/")
    raise ValueError("url does not contain neither '/dataselect/' nor '/station/'")


class Station(Base):
    """Stations"""

    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, autoincrement=True)  # , default=sta_pkey_default, onupdate=sta_pkey_default)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = deferred(Column(Float))
    site_name = deferred(Column(String))
    start_time = Column(DateTime, nullable=False)
    end_time = deferred(Column(DateTime))
    inventory_xml = deferred(Column(LargeBinary))  # lazy load: only upon direct access

    @hybrid_property
    def has_inventory(self):
        return bool(self.inventory_xml)

    @has_inventory.expression
    def has_inventory(cls):  # @NoSelf
        return withdata(cls.inventory_xml)

    __table_args__ = (
                      UniqueConstraint('network', 'station', 'start_time', name='net_sta_stime_uc'),
                     )

    # http://stackoverflow.com/questions/33708219/how-to-handle-sqlalchemy-onupdate-when-current-context-is-empty
    # we prefer over event.listen_for cause the function is inside the class (clearer)
#     @validates('network', 'station', 'start_time')
#     def update_id(self, key, value):
#         vals = (value, self.station, self.start_time) if key == 'network' else \
#             (self.network, value, self.start_time) if key == 'station' else \
#             (self.network, self.station, value)
#         id_ = get_station_id(*vals)
#         if id_ is not None:
#             self.id = id_
#         return value

    datacenter = relationship("DataCenter", backref=backref("stations", lazy="dynamic"))


# def get_station_id(network, station, starttime):
#     pack = [network, station, starttime]
#     if any(i is not None for i in pack):
#         return None
#     stime = starttime.isoformat() if hasattr(starttime, 'isoformat') else starttime
#     pack[-1] = stime[:-9] if stime.endswith("T00:00:00") else stime
#     return ".".join(pack)

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

    id = Column(Integer, primary_key=True, autoincrement=True)  # , default=cha_pkey_default, onupdate=cha_pkey_default)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)
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
#     @validates('station_id', 'location', 'channel')
#     def update_id(self, key, value):
#         vals = (value, self.location, self.channel) if key == 'station_id' else \
#                 (self.station_id, value, self.channel) if key == 'location' else \
#                 (self.station_id, self.location, value)
#         if all(i is not None for i in vals):
#             self.id = "%s.%s.%s" % vals
#         return value
#
    station = relationship("Station", backref=backref("channels", lazy="dynamic"))


class Segment(Base):
    """The Segments table"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True)  # , default=seg_pkey_default, onupdate=seg_pkey_default)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    seed_identifier = Column(String)
    event_distance_deg = Column(Float, nullable=False)
    data = deferred(Column(LargeBinary))  # lazy load only upon access
    download_status_code = Column(Integer, nullable=True)
    start_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    sample_rate = Column(Float)
    max_gap_ovlap_ratio = Column(Float)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

#     @hybrid_property
#     def seedid(self):
#         if self.seed_identifier:
#             return self.seed_identifier
#         else:
#             return ".".join([x if x else "" for x in
#                              [self.station.network, self.station.station, self.channel.location,
#                               self.channel.channel]])

    # DEFINE HTBRID PROPERTIES. ACTUALY, WE ARE JUST INTERESTED IN HYBRID CLASSMETHODS FOR
    # QUERYING, BUT IT SEEMS THERE IS NO WAY TO DEFINE THEM WITHOUT DEFINING THE INSTANCE METHOD
    @hybrid_property
    def has_data(self):
        return bool(self.data)

    @has_data.expression
    def has_data(cls):  # @NoSelf
        return withdata(cls.data)

    @hybrid_method
    def has_class(self, *ids):
        if not ids:
            return self.classes.count() > 0
        else:
            _ids = set(ids)
            return any(c.id in _ids for c in self.classes)

    @has_class.expression
    def has_class(cls, *ids):  # @NoSelf
        any_ = cls.classes.any
        if not ids:
            return any_()
        else:
            return any_(Class.id.isin_(ids))

    @hybrid_property
    def strid(self):
        return self.seed_identifier or \
            ".".join([self.station.network, self.station.station,
                      self.channel.location, self.channel.channel])

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
#     hl_classes = relationship("Class", lazy='dynamic',
#                               secondary="class_labellings",  # <-  must be table name in metadata
#                               viewonly=True, backref=backref("segments", lazy="dynamic"))
#     cl_classes = relationship("Class", lazy='dynamic',
#                               secondary="class_labellings",  # <-  must be table name in metadata
#                               viewonly=True, backref=backref("segments", lazy="dynamic"))

    __table_args__ = (
                      UniqueConstraint('channel_id', 'start_time', 'end_time',
                                       name='chaid_stime_etime_uc'),
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
    annotator = Column(String, nullable=True)

    __table_args__ = (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)
