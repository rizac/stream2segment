'''
Models for the ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

import re
import sqlite3

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
from sqlalchemy.sql.expression import func, text, case, null, select, join, alias, exists, and_
# from sqlalchemy.orm.mapper import validates
# from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.util import aliased

_Base = declarative_base()

from sqlalchemy.engine import Engine  # @IgnorePep8


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
    # play well with other DB backends:
    if type(dbapi_connection) is sqlite3.Connection:  # @UndefinedVariable
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(_Base):

    __abstract__ = True

    def __str__(self):
        cls = self.__class__
        ret = [cls.__name__ + ":"]
        insp = inspect(cls)
        for colname, col in insp.columns.items():  # Note: returns a list, if perfs are a concern,
            # we should iterate over the underlying insp.columns._data (but's cumbersome)
            typ = col.type
            typ_str = str(typ)
            try:
                val = "(not shown)" if type(typ) == LargeBinary else str(getattr(self, colname))
            except Exception as exc:
                val = "(not shown: %s)" % str(exc)
            ret.append("%s %s: %s" % (colname, typ_str, val))
        for relationship in insp.relationships.keys():  # see note above on insp.columns.items()
            ret.append("%s: relationship" % relationship)
        return "\n".join(ret)


def ForeignKey(*pos, **kwa):
    """Overrides the ForeignKey defined in SqlAlchemy by providing default
    `onupdate='CASCADE'` and `ondelete='CASCADE'` if the two keyword argument are missing in `kwa`.
    If this behavior needs to be modified for some column in the future,
    just provide the arguments in the constructor as one would do with sqlalchemy ForeignKey class
    E.g.: column = Column(..., ForeignKey(..., onupdate='SET NULL',...), nullable=False)
    """
    if 'onupdate' not in kwa:
        kwa['onupdate'] = 'CASCADE'
    if 'ondelete' not in kwa:
        kwa['ondelete'] = 'CASCADE'
    return SqlAlchemyForeignKey(*pos, **kwa)


class Download(Base):
    """The downloads executed"""

    __tablename__ = "downloads"

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


class WebService(Base):
    """web service (e.g., event web service)"""
    __tablename__ = "web_services"

    id = Column(Integer, primary_key=True, autoincrement=True)  # pylint:disable=invalid-name
    name = Column(String)  # optional
    type = Column(String)  # e.g.: event, station, dataselect (currently only event is used)
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
    dataselect_url = Column(String, nullable=False)
    organization_name = Column(String)

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
    if (target.station_url is None) != (target.dataselect_url is None):
        normalizedfdsn = fdsn_urls(target.station_url if target.dataselect_url is None else
                                   target.dataselect_url)
        if normalizedfdsn:
            target.station_url = normalizedfdsn[0]
            target.dataselect_url = normalizedfdsn[1]


def fdsn_urls(url):
    '''Returns the strings tuple (station_url, dataselect_url) by parsing url as a fdsn url
    (https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf). Returns None if url
    is not parsable as fdsn url.
    '''
    services = ['station', 'dataselect']
    _ = re.match("^(.*/fdsnws)/(?P<service>.*?)/(?P<majorversion>.*?)(|/.*)$", url)
    if _:
        try:
            if _.group('service') in services:
                remaining = _.group(4)
                if not remaining or remaining == '/' or remaining == '/query?':
                    remaining = "/query"
                return "%s/%s/%s%s" % (_.group(1), services[0], _.group(3), remaining), \
                    "%s/%s/%s%s" % (_.group(1), services[1], _.group(3), remaining)
        except IndexError:
            pass
    return None


class Station(Base):
    """Stations"""

    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, autoincrement=True)
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

    datacenter = relationship("DataCenter", backref=backref("stations", lazy="dynamic"))


class Channel(Base):
    """Channels"""

    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, autoincrement=True)
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

    @hybrid_property
    def band_code(self):
        '''returns the first letter of the channel field, or None if the latter has not length 3'''
        return self.channel[0] if len(self.channel) == 3 else None

    @band_code.expression
    def band_code(cls):  # @NoSelf
        '''returns the sql expression returning the first letter of the channel field,
        or NULL if the latter has not length 3'''
        # return an sql expression matching the last char or None if not three letter channel
        return case([(func.length(cls.channel) == 3, func.left(cls.channel, 1))], else_=null())

    @hybrid_property
    def instrument_code(self):
        '''returns the second letter of the channel field, or None if the latter has not length 3'''
        return self.channel[1] if len(self.channel) == 3 else None

    @instrument_code.expression
    def instrument_code(cls):  # @NoSelf
        '''returns the sql expression returning the second letter of the channel field,
        or NULL if the latter has not length 3'''
        # return an sql expression matching the last char or None if not three letter channel
        return case([(func.length(cls.channel) == 3, func.substr(cls.channel, 2, 1))], else_=null())

    @hybrid_property
    def orientation_code(self):
        '''returns the third letter of the channel field, or None if the latter has not length 3'''
        return self.channel[2] if len(self.channel) == 3 else None

    @orientation_code.expression
    def orientation_code(cls):  # @NoSelf
        '''returns the sql expression returning the third letter of the channel field,
        or NULL if the latter has not length 3'''
        # return an sql expression matching the last char or None if not three letter channel
        return case([(func.length(cls.channel) == 3, func.right(cls.channel, 1))], else_=null())

    __table_args__ = (
                      UniqueConstraint('station_id', 'location', 'channel',
                                       name='net_sta_loc_cha_uc'),
                     )

    station = relationship("Station", backref=backref("channels", lazy="dynamic"))


class Segment(Base):
    """The Segments table"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    data_identifier = Column(String)
    event_distance_deg = Column(Float, nullable=False)
    data = Column(LargeBinary)
    download_status_code = Column(Integer)
    start_time = Column(DateTime)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    sample_rate = Column(Float)
    max_gap_overlap_ratio = Column(Float)
    download_id = Column(Integer, ForeignKey("downloads.id"), nullable=False)
    request_start = Column(DateTime, nullable=False)
    request_end = Column(DateTime, nullable=False)

    # DEFINE HYBRID PROPERTIES. ACTUALY, WE ARE JUST INTERESTED IN HYBRID CLASSMETHODS FOR
    # QUERYING, BUT IT SEEMS THERE IS NO WAY TO DEFINE THEM WITHOUT DEFINING THE INSTANCE METHOD
    @hybrid_property
    def has_data(self):
        return bool(self.data)

    @has_data.expression
    def has_data(cls):  # @NoSelf
        return withdata(cls.data)

    @hybrid_method
    def has_class(self, *ids):  # this is used only for testing purposes. See test_db
        if not ids:
            return self.classes.count() > 0
        else:
            _ids = set(ids)
            return any(c.id in _ids for c in self.classes)

    @has_class.expression
    def has_class(cls, *ids):  # @NoSelf  # this is used only for testing purposes. See test_db
        any_ = cls.classes.any
        if not ids:
            return any_()
        else:
            return any_(Class.id.isin_(ids))

    @hybrid_property
    def seed_identifier(self):
        return self.data_identifier or \
            ".".join([self.station.network, self.station.station,
                      self.channel.location, self.channel.channel])

    @seed_identifier.expression
    def seed_identifier(cls):  # @NoSelf
        '''returns data_identifier if the latter is not None, else net.sta.loc.cha by querying the
        relative channel and station'''
        # Needed note: To know what we are doing in 'sel' below, please look:
        # http://docs.sqlalchemy.org/en/latest/orm/extensions/hybrid.html#correlated-subquery-relationship-hybrid
        # Notes
        # - we use limit(1) cause we might get more than one
        # result. Regardless of why it happens (because we don't join or apply a distinct?)
        # it is relevant for us to get the first result which has the requested
        # network+station and location + channel strings
        # - the label(...) at the end makes all the difference. The doc is, as always, unclear
        # http://docs.sqlalchemy.org/en/latest/core/sqlelement.html#sqlalchemy.sql.expression.label
        dot = text("'.'")
        sel = select([func.concat(Station.network, dot, Station.station, dot,
                                  Channel.location, dot, Channel.channel)]).\
            where((Channel.id == cls.channel_id) & (Station.id == Channel.station_id)).limit(1).\
            label('seedidentifier')
        return case([(cls.data_identifier.isnot(None), cls.data_identifier)],
                    else_=sel)

    event = relationship("Event", backref=backref("segments", lazy="dynamic"))
    channel = relationship("Channel", backref=backref("segments", lazy="dynamic"))
    datacenter = relationship("DataCenter", backref=backref("segments", lazy="dynamic"))
    download = relationship("Download", backref=backref("segments", lazy="dynamic"))

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
                      UniqueConstraint('channel_id', 'event_id',
                                       name='chaid_evtid_uc'),
                     )


class Class(Base):
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
    annotator = Column(String)

    __table_args__ = (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)
