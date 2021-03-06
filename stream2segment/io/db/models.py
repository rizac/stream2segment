"""
s2s database ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

import re
import os
from datetime import datetime
import sqlite3
from math import pi
# we could simply import urlparse from stream2segment.utils, but we want this module
# to be standalone. thus:
try:  # py3:
    from urllib.parse import urlparse
except ImportError:  # py2
    from urlparse import urlparse  # noqa

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
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
# from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property  # , hybrid_method
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import relationship, backref, deferred  # , load_only
# from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.session import object_session
# from sqlalchemy.orm.util import aliased
from sqlalchemy.sql.expression import (func, text, case, select,
                                       FunctionElement,
                                       # null, join, alias, exists, and_
                                       )


_Base = declarative_base()


##############################################################################
# Register non-standard SQL functions
# (Make the following SQL functions work with both SQLite and Postgres. If you
# add support for new databases, you should modify the code below. For info:
# http://docs.sqlalchemy.org/en/latest/core/compiler.html#further-examples
##############################################################################

# function `strpos`

class strpos(FunctionElement):
    name = 'strpos'
    type = Integer()


@compiles(strpos)
def standard_strpos(element, compiler, **kw):
    """delegates strpos to the strpos db function"""
    return compiler.visit_function(element)


@compiles(strpos, 'sqlite')
def sqlite_strpos(element, compiler, **kw):
    return "instr(%s)" % compiler.process(element.clauses)
    # return func.instr(compiler.process(element.clauses))


# function `concat`

class concat(FunctionElement):
    name = 'concat'
    type = String()


@compiles(concat)
def standard_concat(element, compiler, **kw):
    return compiler.visit_function(element)


@compiles(concat, 'sqlite')
def sqlite_concat(element, compiler, **kw):
    return " || ".join(compiler.process(c) for c in element.clauses)


# two utility functions to return the timestamp from a datetime
def _duration_sqlite(start, end):
    """Return the time in seconds since 1970 as floating point for of the
    specified argument (a datetime in sqlite format)
    """
    # note: sqlite time format is bizarre. They have %s: timestamp in SECONDS
    # since 1970, %f seconds only (with 3 decimal digits WTF?) and %S: seconds
    # part (integer). Thus to have a floating point value with 3 decimal digits
    # we should return:
    # ```
    # round(strftime('%s',{}) + strftime('%f',{}) - strftime('%S',{}), 3)".\
    #   format(dtime)
    # ```
    # However, for performance reasons we think it's sufficient to return the
    # seconds, thus we keep it more simple with the use round at the end to
    # coerce to float with 3 decimal digits, for safety (yes, round in sqlite
    # returns a float) and avoid integer divisions when needed but proper
    # floating point arithmentic
    return ("round(strftime('%s',{1})+strftime('%f',{1})-strftime('%S',{1}) - "
            "(strftime('%s',{0})+strftime('%f',{0})-strftime('%S',{0})), 3)").\
        format(start, end)


def _duration_postgres(start, end):
    """Return the time in seconds since 1970 as floating point for of the
    specified argument (a datetime in postgres format)
    """
    # Note: we use round at the end to coerce to float with 3 decimal digits,
    # for safety and avoid integer divisions when needed but proper floating
    # point arithmentic
    return "round(EXTRACT(EPOCH FROM ({1}-{0}))::numeric, 3)".format(start,
                                                                     end)


# function `duration_sec`

class duration_sec(FunctionElement):
    name = 'duration_sec'
    type = Float()


@compiles(duration_sec)
def standard_duration_sec(element, compiler, **kw):
    starttime, endtime = [compiler.process(c) for c in element.clauses]
    return _duration_postgres(starttime, endtime)


@compiles(duration_sec, 'sqlite')
def sqlite_duration_sec(element, compiler, **kw):
    starttime, endtime = [compiler.process(c) for c in element.clauses]
    return _duration_sqlite(starttime, endtime)


# function `missing_data_sec`

class missing_data_sec(FunctionElement):
    name = 'missing_data_sec'
    type = Float()


@compiles(missing_data_sec)
def standard_missing_data_sec(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "({1}) - ({0})".format(_duration_postgres(start, end),
                                  _duration_postgres(request_start, request_end))


@compiles(missing_data_sec, 'sqlite')
def sqlite_missing_data_sec(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "({1}) - ({0})".format(_duration_sqlite(start, end),
                                  _duration_sqlite(request_start, request_end))


# function `missing_data_ratio`

class missing_data_ratio(FunctionElement):
    name = 'missing_data_ratio'
    type = Float()


@compiles(missing_data_ratio)
def standard_missing_data_ratio(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "1.0 - (({0}) / ({1}))".format(_duration_postgres(start, end),
                                          _duration_postgres(request_start, request_end))


@compiles(missing_data_ratio, 'sqlite')
def sqlite_missing_data_ratio(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "1.0 - (({0}) / ({1}))".format(_duration_sqlite(start, end),
                                          _duration_sqlite(request_start, request_end))


# function `deg2km`

class deg2km(FunctionElement):
    name = 'deg2km'
    type = Float()


@compiles(deg2km)
def standard_deg2km(element, compiler, **kw):
    deg = compiler.process(list(element.clauses)[0])
    return "%s * (2.0 * 6371 * 3.14159265359 / 360.0)" % deg


# function `substr`

class substr(FunctionElement):
    name = 'substr'
    type = String()


@compiles(substr)
def standard_substr(element, compiler, **kw):
    clauses = list(element.clauses)
    column = compiler.process(clauses[0])
    start = compiler.process(clauses[1])
    leng = compiler.process(clauses[2])
    return "substr(%s, %s, %s)" % (column, start, leng)


# =================================
# end of non-standard SQL functions
# =================================


def withdata(model_column):
    """Return a filter argument for returning instances with values of
    `model_column` NOT *empty* nor *null*. `model_column` type must be STRING
    or BLOB. Examples:
    ```
    # given a table User, return empty or none via "~"
    session.query(User.id).filter(~withdata(User.data)).all()

    # return "valid" columns:
    session.query(User.id).filter(withdata(User.data)).all()
    ```

    :param model_column: A valid column name, e.g. an attribute Column defined
        in some SQL-Alchemy orm model class (e.g., 'User.data'). **The type of
        the column must be STRING or BLOB**, otherwise result is undefined.
        For instance, numeric column with zero as value are *not* empty (as
        the SQL length function applied to numeric returns the number of bytes)
    """
    return (model_column.isnot(None)) & (func.length(model_column) > 0)


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


class Base(_Base):
    """Abstract base class for a Stream2segment ORM Model"""

    __abstract__ = True

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


class Download(Base):  # pylint: disable=too-few-public-methods
    """Model representing the executed downloads"""

    __tablename__ = "downloads"

    id = Column(Integer, primary_key=True, autoincrement=True)  # noqa
    # run_time below has server_default as `func.now()`. This issues a CURRENT
    # TIMESTAMP on the SQL side. That's ok, BUT the column CANNOT BE UNIQUE!!
    # the CURRENT TIMESTAMP is evaluated once at the beginning of an SQL
    # Statement, so two references in the same session will result in the same
    # value. If we need to make a datetime unique, then either specify
    # 1) default=datetime.datetime.utcnow() BUT NO server_default (the latter
    # seems to have priority if both are provided)
    # 2) or don't make the column unique (what we did)
    run_time = Column(DateTime, server_default=func.now())
    log = deferred(Column(String))  # lazy load: only upon direct access
    warnings = Column(Integer, server_default=text('0'))  # , default=0)
    errors = Column(Integer, server_default=text('0'))  # , default=0)
    # segments_found = Column(Integer)
    # segments_written = Column(Integer)
    # segments_skipped = Column(Integer)
    config = deferred(Column(String))
    program_version = Column(String)


class Event(Base):  # pylint: disable=too-few-public-methods
    """Model representing a seismic Event"""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)  # noqa
    webservice_id = Column(Integer, ForeignKey("web_services.id"),
                           nullable=False)
    event_id = Column(String, nullable=False)
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

    __table_args__ = (UniqueConstraint('webservice_id', 'event_id',
                                       name='ws_eventid_uc'),)
    # segments = relationship("Segment", back_populates="event")


class WebService(Base):
    """Model representing a web service (e.g., event web service)"""

    __tablename__ = "web_services"

    id = Column(Integer, primary_key=True, autoincrement=True)  # noqa
    name = Column(String)  # optional
    type = Column(String)  # e.g.: event, station, dataselect
    url = Column(String, nullable=False)  # if you change attr, see BELOW!

    # segments = relationship("Segment", backref="data_centers")
    # stations = relationship("Station", backref="data_centers")

    __table_args__ = (UniqueConstraint('url', name='url_uc'),)


class DataCenter(Base):
    """Model representing a Data center (data provider, e.g. EIDA Node)"""

    __tablename__ = "data_centers"

    id = Column(Integer, primary_key=True, autoincrement=True)  # noqa
    station_url = Column(String, nullable=False)
    dataselect_url = Column(String, nullable=False)
    organization_name = Column(String)

    __table_args__ = (UniqueConstraint('station_url', 'dataselect_url',
                                       name='sta_data_uc'),)


@event.listens_for(DataCenter, 'before_insert')
@event.listens_for(DataCenter, 'before_update')
def receive_before_update(mapper, connection, target):
    """listen for the 'before_update' event to update missing DataCenter URLs.
     For info on validation see:
     https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf
    """
    # Note: we though about using validators but we ended up with infinite
    # recursion loops
    if (target.station_url is None) != (target.dataselect_url is None):
        try:
            fdsn = Fdsnws(target.station_url if target.dataselect_url is None
                          else target.dataselect_url)
            target.station_url = fdsn.url(Fdsnws.STATION)
            target.dataselect_url = fdsn.url(Fdsnws.DATASEL)
        except ValueError:
            pass


class Fdsnws(object):
    """Fdsn w(eb) s(ervice) URL normalizer. Gets any URL, checks its
    correctness and allows retrieving all other FDSN URLs easily and safely.
    Example:
    ```
    fdsn = Fdsnws(url)
    station_query_url = fdsn.url(Fdsnws.STATION)
    dataselect_query_url = fdsn.url(Fdsnws.DATASEL)
    dataselect_queryauth_url = fdsn.url(Fdsnws.DATASEL, method=Fdsnws.QUERYAUTH)
    ```
    """
    # equals to the string 'station', used in urls for identifying the FDSN
    # station service:
    STATION = 'station'
    # equals to the string 'dataselect', used in urls for identifying the FDSN
    # data service:
    DATASEL = 'dataselect'
    # equals to the string 'event', used in urls for identifying the FDSN event
    # service:
    EVENT = 'event'
    # equals to the string 'query', used in urls for identifying the FDSN
    # service query method:
    QUERY = 'query'
    # equals to the string 'queryauth', used in urls for identifying the FDSN
    # service query method (with authentication):
    QUERYAUTH = 'queryauth'
    # equals to the string 'auth', used  (by EIDA only?) in urls for querying
    # username and password with provided token:
    AUTH = 'auth'
    # equals to the string 'version', used in urls for identifying the FDSN
    # service query method:
    VERSION = 'version'
    # equals to the string 'application.wadl', used in urls for identifying the
    # FDSN service application wadl method:
    APPLWADL = 'application.wadl'

    def __init__(self, url):
        """Initialize a Fdsnws object from a FDSN URL

        :param url: string denoting the Fdsn web service url
            Example of valid urls (the scheme 'https://' might be omitted
            and will default to 'http://'. An ending '/' or '?' will be ignored
            if present):
            https://www.mysite.org/fdsnws/<station>/<majorversion>
            http://www.mysite.org/fdsnws/<station>/<majorversion>/<method>
        """
        # do not use urlparse as we should import from stream2segment.url for
        # py2 compatibility but this will cause circular imports:

        obj = urlparse(url)
        if not obj.scheme:
            obj = urlparse('http://' + url)
        if not obj.netloc:
            raise ValueError('no domain specified or invalid scheme, '
                             'check typos')

        self.site = "%s://%s" % (obj.scheme, obj.netloc)

        pth = obj.path
        #  urlparse has already removed query char '?' and params and fragment
        # from the path. Now check the latter:
        reg = re.match("^(?:/.+)*/fdsnws/(?P<service>[^/]+)/"
                       "(?P<majorversion>[^/]+)(?P<method>.*)$",
                       pth)
        try:
            self.service = reg.group('service')
            self.majorversion = reg.group('majorversion')
            method = reg.group('method')

            if self.service not in [self.STATION, self.DATASEL, self.EVENT]:
                raise ValueError("Invalid <service> '%s' in '%s'" %
                                 (self.service, pth))
            try:
                float(self.majorversion)
            except ValueError:
                raise ValueError("Invalid <majorversion> '%s' in '%s'" %
                                 (self.majorversion, pth))
            if method not in ('', '/'):
                method = method[1:] if method[0] == '/' else method
                method = method[:-1] if len(method) > 1 and method[-1] == '/' \
                    else method
                if method not in ['', self.QUERY, self.QUERYAUTH, self.AUTH,
                                  self.VERSION, self.APPLWADL]:
                    raise ValueError("Invalid method '%s' in '%s'" %
                                     (method, pth))
        except ValueError:
            raise
        except Exception:
            raise ValueError("Invalid FDSN URL '%s': it should be "
                             "'[site]/fdsnws/<service>/<majorversion>', "
                             "check potential typos" % str(url))

    def url(self, service=None, majorversion=None, method=None):
        """Build a new url from this object url. Arguments which are 'None'
        will default to this object's url passed in the constructor. The
        returned URL denotes the base url (with no query parameter and no
        trailing '?' or '/') in order to build queries to a FDSN web service

        :param service: None or one of this class static attributes:
            `STATION`, `DATASEL`, `EVENT`
        :param majorversion: None or numeric value or string parsable to number
            denoting the service major version. Defaults to 1 when None
            `STATION`, `DATASEL`, `EVENT`
        :param method: None or one of the class static attributes
            `QUERY` (the default when None), `QUERYAUTH`, `VERSION`, `AUTH`
            or `APPLWADL`
        """
        return "%s/fdsnws/%s/%s/%s" % (self.site, service or self.service,
                                       str(majorversion or self.majorversion),
                                       method or self.QUERY)

    def __str__(self):
        return self.url('<service>', None, '<method>')


class Station(Base):
    """Model representing a Station"""

    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"),
                           nullable=False)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    inventory_xml = Column(LargeBinary)

    @hybrid_property
    def has_inventory(self):
        return bool(self.inventory_xml)

    @has_inventory.expression
    def has_inventory(cls):  # pylint:disable=no-self-argument
        return withdata(cls.inventory_xml)

    @hybrid_property
    def netsta_code(self):
        return "%s.%s" % (self.network, self.station)

    @netsta_code.expression
    def netsta_code(cls):  # pylint:disable=no-self-argument
        """Return the station code, i.e. self.network + '.' + self.station"""
        dot = text("'.'")
        return concat(Station.network, dot, Station.station).\
            label('networkstationcode')

    __table_args__ = (UniqueConstraint('network', 'station', 'start_time',
                                       name='net_sta_stime_uc'),)

    datacenter = relationship("DataCenter",
                              backref=backref("stations", lazy="dynamic"))


class Channel(Base):
    """Model representing a Channel"""

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
        """Return the first letter of the channel field"""
        return self.channel[0:1]  # if len(self.channel) == 3 else None

    @band_code.expression
    def band_code(cls):  # pylint:disable=no-self-argument
        """Return the sql expression returning the first letter of the channel
        field"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return substr(cls.channel, 1, 1)

    @hybrid_property
    def instrument_code(self):
        """Return the second letter of the channel field"""
        return self.channel[1:2]  # if len(self.channel) == 3 else None

    @instrument_code.expression
    def instrument_code(cls):  # pylint:disable=no-self-argument
        """Return the sql expression returning the second letter of the channel
        field"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return substr(cls.channel, 2, 1)

    @hybrid_property
    def band_instrument_code(self):
        """Return the first two letters of the channel field. Useful when we
        want to get the same record on different orientations/components"""
        return self.channel[0:2]  # if len(self.channel) == 3 else None

    @band_instrument_code.expression
    def band_instrument_code(cls):  # pylint:disable=no-self-argument
        """Return the sql expression returning the first two letters of the
        channel field. Useful for queries where we want to get the same record
        on different orientations/components"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return substr(cls.channel, 1, 2)

    @hybrid_property
    def orientation_code(self):
        """Return the third letter of the channel field"""
        return self.channel[2:3]  # if len(self.channel) == 3 else None

    @orientation_code.expression
    def orientation_code(cls):  # pylint:disable=no-self-argument
        """Return the sql expression returning the third letter of the channel
        field"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return substr(cls.channel, 3, 1)

    __table_args__ = (UniqueConstraint('station_id', 'location', 'channel',
                                       name='net_sta_loc_cha_uc'),)

    station = relationship("Station",
                           backref=backref("channels", lazy="dynamic"))


class Segment(Base):
    """Model representing a Waveform segment"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"),
                           nullable=False)
    data_seed_id = Column(String)
    event_distance_deg = Column(Float, nullable=False)
    data = Column(LargeBinary)
    download_code = Column(Integer)
    start_time = Column(DateTime)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    sample_rate = Column(Float)
    maxgap_numsamples = Column(Float)
    download_id = Column(Integer, ForeignKey("downloads.id"), nullable=False)
    request_start = Column(DateTime, nullable=False)
    request_end = Column(DateTime, nullable=False)
    queryauth = Column(Boolean, nullable=False,
                       server_default="0")  # note: null fails in sqlite!

    # DEFINE HYBRID PROPERTIES WITH RELATIVE SELECTION EXPRESSIONS FOR QUERYING
    # THE DB:

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
    def missing_data_sec(self):
        try:
            return (self.request_end - self.request_start).total_seconds() - \
                (self.end_time - self.start_time).total_seconds()
        except TypeError:  # some None(s)
            return None

    @missing_data_sec.expression
    def missing_data_sec(cls):  # pylint:disable=no-self-argument
        return missing_data_sec(cls.start_time, cls.end_time,
                                cls.request_start, cls.request_end)

    @hybrid_property
    def missing_data_ratio(self):
        try:
            ratio = ((self.end_time - self.start_time).total_seconds() /
                     (self.request_end - self.request_start).total_seconds())
            return 1.0 - ratio
        except TypeError:  # some None's
            return None

    @missing_data_ratio.expression
    def missing_data_ratio(cls):  # pylint:disable=no-self-argument
        return missing_data_ratio(cls.start_time, cls.end_time,
                                  cls.request_start, cls.request_end)

    @hybrid_property
    def has_data(self):
        return bool(self.data)

    @has_data.expression
    def has_data(cls):  # pylint:disable=no-self-argument
        return withdata(cls.data)

    @hybrid_property
    def has_class(self):
        return len(self.classes) > 0

    @has_class.expression
    def has_class(cls):  # pylint:disable=no-self-argument
        return cls.classes.any()

    def sds_path(self, root='.'):
        """Return a string representing the seiscomp data structure (sds) path
        where to store the given segment or any data associated with it.
        The returned path has no extension (to be supplied by the user)
        and has the following format (e_id=event id, y=year, d=day of year):
        <root>/<e_id>/<net>/<sta>/<loc>/<cha>.D/<net>.<sta>.<loc>.<cha>.<y>.<d>
        For info see:
        https://www.seiscomp3.org/doc/applications/slarchive/SDS.html
        """
        # year > N > S > L > C.D > segments > N.S.L.C.year.day.event_id.mseed
        seg_dtime = self.request_start  # note that start_time might be None
        year = seg_dtime.year
        net, sta = self.station.network, self.station.station
        loc, cha = self.channel.location, self.channel.channel
        # day is in [1, 366], padded with zeroes:
        day = '%03d' % ((seg_dtime - datetime(year, 1, 1)).days + 1)
        eid = self.event_id
        return os.path.join(root,
                            str(eid), str(year), net, sta, loc, cha + ".D",
                            '.'.join((net, sta, loc, cha, str(year), day)))

    def del_classes(self, *ids_or_labels, **kwargs):
        """Delete segment classes

        :param ids_or_labels: list of int (denoting class ids) or str (denoting
            class label)
        """
        self.edit_classes('del', *ids_or_labels, **kwargs)

    def set_classes(self, *ids_or_labels, **kwargs):
        """Set segment classes, replacing old ones, if any

        :param ids_or_labels: list of int (denoting class ids) or str (denoting
            class label)
        :param kwargs: py2 compatible keyword arguments (PEP 3102): currently
            supported is 'annotator' (str, default: None) and 'auto_commit'
            (bool, default: True). If `annotator` is not None, the class
            assignment is saved as hand labelled
        """
        self.edit_classes('set', *ids_or_labels, **kwargs)

    def add_classes(self, *ids_or_labels, **kwargs):
        """Add segment classes, keeping old ones, if any

        :param ids_or_labels: list of int (denoting class ids) or str (denoting
            class label)
        :param kwargs: py2 compatible keyword arguments (PEP 3102): currently
            supported is 'annotator' (str, default: None) and 'auto_commit'
            (bool, default: True). If `annotator` is not None, the class
            assignment is saved as hand labelled
        """
        self.edit_classes('add', *ids_or_labels, **kwargs)

    def edit_classes(self, mode, *ids_or_labels, **kwargs):
        """ Edit segment classes

        :param mode: either 'add' 'set' or 'del'
        :param ids_or_labels: list of int (denoting class ids) or str (denoting
            class label)
        :param kwargs: py2 compatible keyword arguments (PEP 3102): currently
            supported is 'annotator' (str, default: None) and 'auto_commit'
            (bool, default: True). If `annotator` is not None, the class
            assignment is saved as hand labelled
        """
        auto_commit = kwargs.get('auto_commit', True)
        annotator = kwargs.get('annotator', None)
        sess = object_session(self)
        needs_commit = False
        ids = set(ids_or_labels)
        labels = set(_ for _ in ids if type(_) in (bytes, str))
        ids -= labels
        if mode == 'set':
            self.classes[:] = []
        else:
            classes = list(self.classes)

        if mode == 'del':
            for cla in classes:
                if cla.id in ids or cla.label in labels:
                    self.classes.remove(cla)
                    needs_commit = True
            ids = labels = set()  # do not add anything
        elif mode == 'add':
            for cla in classes:
                if cla.id in ids:
                    # already set, remove it and don't add it again:
                    ids.remove(cla.id)
                if cla.label in labels:
                    # already set, remove it and don't add it again:
                    labels.remove(cla.label)
        elif mode != 'set':
            raise TypeError("`mode` argument needs to be in "
                            "('add', 'del', 'set'), '%s' supplied" % str(mode))

        if ids or labels:
            # filter on ids, or None:
            flt1 = None if not ids else Class.id.in_(ids)
            # filter on labels, or None:
            flt2 = None if not labels else Class.label.in_(labels)
            flt = flt1 if flt2 is None else flt2 if flt1 is None else \
                (flt1 | flt2)
            classids2add = [_[0] for _ in sess.query(Class.id).filter(flt)]
            if classids2add:
                needs_commit = True
                sess.add_all((ClassLabelling(class_id=cid,
                                             segment_id=self.id,
                                             annotator=annotator,
                                             is_hand_labelled=annotator is not None)
                              for cid in classids2add))

        if needs_commit and auto_commit:
            try:
                sess.commit()
            except SQLAlchemyError as _:
                sess.rollback()
                raise

    def get_siblings(self, parent=None, colname=None):
        """Return an SQL-Alchemy query yielding all siblings of this segment
        according to `parent`.

        :param parent: str or None (default: None). Any of the following:
            - `None`: return all db segments of the same recorded event, on the
               other channel components / orientations
            - `stationname`: return all db segments of the same station, where
               a station is the tuple (newtwork code, station code)
            - `networkname`: return all db segments of the same network code.
            - `datacenter`, `event`, `station`, `channel`: return all db
               segments from the associated foreign key (a station in this case
               is the tuple (newtwork code, station code, start_time). If
               `colname` is missing/None, providing any of these arguments is
               equivalent to access the `segments` attribute on the referenced
               Object: `segment.query_siblings('station').all()` equals to
               `segment.station.segments.all()`

        :param colname: str or None (default:None). If None, yield Segment
            objects. Otherwise yield only the given Segment attributes, as one
            element tuple (e.g. 'id' will yield `(id1,), (id2,), ... `). In the
            latter case no Segment is stored in the session's identity_map,
            meaning that a (sort-of) cache mechanism will not be used, but also
            that less memory will be consumed (session.expunge_all() will clear
            the cache in case)
        """
        # FIXME: Currently this method might be improved. The colname attribute
        # does not give huge benefits (one element tuples are hard to digest)
        # we might improved it by returning always Segments (which is what most
        # users want), loading first all ids in memory efficient numpy array,
        # and then querying ny id and yielding each Segment. Then, every N
        # yields, expunge_all() and at the end re-add the current segment to
        # the session)
        session = object_session(self)
        qry = session.query(Segment if colname is None
                            else getattr(Segment, colname))

        if parent is None:
            qry = qry.join(Segment.channel).\
                filter((Segment.event_id == self.event_id) &
                       (Channel.station_id == self.channel.station_id) &
                       (Channel.location == self.channel.location) &
                       (Channel.band_instrument_code ==
                        self.channel.band_instrument_code))
        elif parent == 'stationname':
            qry = qry.join(Segment.channel, Channel.station).\
                filter((Station.network == self.channel.station.network) &
                       (Station.station == self.channel.station.station))
        elif parent == 'networkname':
            qry = qry.join(Segment.channel, Channel.station).\
                filter((Station.network == self.channel.station.network))
        elif parent == 'station':
            qry = qry.join(Segment.channel).\
                filter((Channel.station_id == self.channel.station_id))
        else:
            try:
                qry = qry.filter(getattr(Segment, parent + '_id') ==
                                 getattr(self, parent + '_id'))
            except AttributeError:
                raise TypeError("invalid 'parent' argument '%s'" % parent)
        return qry.filter(Segment.id != self.id)

    @hybrid_property
    def seed_id(self):
        try:
            return self.data_seed_id or \
                ".".join([self.station.network, self.station.station,
                          self.channel.location, self.channel.channel])
        except (TypeError, AttributeError):
            return None

    @seed_id.expression
    def seed_id(cls):  # pylint:disable=no-self-argument
        """Return data_seed_id if the latter is not None, else net.sta.loc.cha
        by querying the relative channel and station"""
        # Needed note: To know what we are doing in 'sel' below, please look:
        # http://docs.sqlalchemy.org/en/latest/orm/extensions/hybrid.html#correlated-subquery-relationship-hybrid
        # Notes
        # - we use limit(1) cause we might get more than one result. Regardless
        #   of why it happens (because we don't join or apply a distinct?) it
        #   is relevant for us to get the first result which has the requested
        #   network+station and location + channel strings
        # - the label(...) at the end makes all the difference. The doc is, as
        #   it often happens, convoluted:
        #   http://docs.sqlalchemy.org/en/latest/core/sqlelement.html#sqlalchemy.sql.expression.label
        dot = text("'.'")
        sel = select([concat(Station.network, dot, Station.station, dot,
                             Channel.location, dot, Channel.channel)]).\
            where((Channel.id == cls.channel_id) &
                  (Station.id == Channel.station_id)).\
            limit(1).label('seedidentifier')
        return case([(cls.data_seed_id.isnot(None), cls.data_seed_id)],
                    else_=sel)

    # Relationships:

    event = relationship("Event", backref=backref("segments",
                                                  lazy="dynamic"))
    channel = relationship("Channel", backref=backref("segments",
                                                      lazy="dynamic"))
    # Relationship spanning 3 tables (https://stackoverflow.com/a/17583437)
    station = relationship("Station",
                           # `secondary` must be table name in metadata:
                           secondary="channels",
                           primaryjoin="Segment.channel_id == Channel.id",
                           secondaryjoin="Station.id == Channel.station_id",
                           uselist=False,  # viewonly=True,
                           backref=backref("segments", lazy="dynamic"))
    classes = relationship("Class",  # lazy='dynamic', viewonly=True,
                           # `secondary` must be table name in metadata:
                           secondary="class_labellings",
                           backref=backref("segments", lazy="dynamic"))
    datacenter = relationship("DataCenter", backref=backref("segments",
                                                            lazy="dynamic"))
    download = relationship("Download", backref=backref("segments",
                                                        lazy="dynamic"))

    __table_args__ = (UniqueConstraint('channel_id', 'event_id',
                                       name='chaid_evtid_uc'),)


class Class(Base):
    """Model representing a segment class label"""

    __tablename__ = 'classes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    description = Column(String)

    __table_args__ = (UniqueConstraint('label', name='class_label_uc'),)


class ClassLabelling(Base):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""

    __tablename__ = "class_labellings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    is_hand_labelled = Column(Boolean,
                              server_default="1")  # "TRUE" fails in sqlite!
    annotator = Column(String)

    __table_args__ = (UniqueConstraint('segment_id', 'class_id',
                                       name='seg_class_uc'),)
