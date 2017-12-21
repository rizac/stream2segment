'''
Models for the ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

import re
from datetime import datetime
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
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.sql.expression import func, text, case, null, select, join, alias, exists, and_
# from sqlalchemy.orm.mapper import validates
# from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.util import aliased
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm.strategy_options import load_only
from sqlalchemy.orm.session import object_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import InstrumentedAttribute

_Base = declarative_base()

from sqlalchemy.engine import Engine  # @IgnorePep8


# making standard functions. These functions are not standard across sql dialects (sqlite and postgres)
# and thus need to be standardized here
# For info see: http://docs.sqlalchemy.org/en/latest/core/compiler.html#further-examples
class strpos(FunctionElement):
    name = 'strpos'
    type = Integer()


@compiles(strpos)
def standard_strpos(element, compiler, **kw):
    '''delegates strpos to the strpos db function'''
    return compiler.visit_function(element)  # func.strpos(compiler.process(element.clauses))


@compiles(strpos, 'sqlite')
def sqlite_strpos(element, compiler, **kw):
    return "instr(%s)" % compiler.process(element.clauses)
    # return func.instr(compiler.process(element.clauses))


class concat(FunctionElement):
    name = 'concat'
    type = String()


@compiles(concat)
def standard_concat(element, compiler, **kw):
    return compiler.visit_function(element)  # func.strpos(compiler.process(element.clauses))


@compiles(concat, 'sqlite')
def sqlite_concat(element, compiler, **kw):
    return " || ".join(compiler.process(c) for c in element.clauses)


# two utility functions to return the timestamp from a datetime
def _duration_sqlite(start, end):
    '''returns the time in seconds since 1970 as floating point for of the
    soecified argument (a datetime in sqlite format)'''
    # note: sqlite is bizarre: they have %s: timestamp in SECONDS since 1970, %f seconds only
    # (with three decimal digits) and %S: seconds part (integer). Thus to have a floating point
    # value with 3 decimal digits we should do:
    # return "round(strftime('%s',{}) + strftime('%f',{}) - strftime('%S',{}), 3)".format(dtime)
    # However, for performance reasons we think it's sufficient to return the seconds, thus
    # we keep it more simple with the use round at the end
    # to coerce to float with 3 decimal digits, for safety (yes, round in sqlite returns a float)
    # and avoid integer divisions when needed but proper floating point arithmentic
    return ("round(strftime('%s',{1})+strftime('%f',{1})-strftime('%S',{1}) - "
            "(strftime('%s',{0})+strftime('%f',{0})-strftime('%S',{0})), 3)").format(start, end)


def _duration_postgres(start, end):
    '''returns the time in seconds since 1970 as floating point for of the
    soecified argument (a datetime in postgres format)'''
    # Note: we use round at the end
    # to coerce to float with 3 decimal digits, for safety
    # and avoid integer divisions when needed but proper floating point arithmentic
    return "round(EXTRACT(EPOCH FROM ({1}-{0}))::numeric, 3)".format(start, end)


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


class missing_data_sec(FunctionElement):
    name = 'missing_data_sec'
    type = Float()


@compiles(missing_data_sec)
def standard_missing_data_sec(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c) for c in element.clauses]
    return "({1}) - ({0})".format(_duration_postgres(start, end),
                                  _duration_postgres(request_start, request_end))


@compiles(missing_data_sec, 'sqlite')
def sqlite_missing_data_sec(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c) for c in element.clauses]
    return "({1}) - ({0})".format(_duration_sqlite(start, end),
                                  _duration_sqlite(request_start, request_end))


class missing_data_ratio(FunctionElement):
    name = 'missing_data_ratio'
    type = Float()


@compiles(missing_data_ratio)
def standard_missing_data_ratio(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c) for c in element.clauses]
    return "1.0 - (({0}) / ({1}))".format(_duration_postgres(start, end),
                                          _duration_postgres(request_start, request_end))


@compiles(missing_data_ratio, 'sqlite')
def sqlite_missing_data_ratio(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c) for c in element.clauses]
    return "1.0 - (({0}) / ({1}))".format(_duration_sqlite(start, end),
                                          _duration_sqlite(request_start, request_end))


class deg2km(FunctionElement):
    name = 'deg2km'
    type = Float()


@compiles(deg2km)
def standard_deg2km(element, compiler, **kw):
    deg = compiler.process(list(element.clauses)[0])
    return "%s * (2.0 * 6371 * 3.14159265359 / 360.0)" % deg


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
    '''abstract base class. It just provides a normalized each model's string representation
    and ease the addition of common methods in the future, if any'''

    __abstract__ = True

    def __str__(self):
        cls = self.__class__
        ret = [str(cls.__name__)]
        # provide a meaningful str representation, but show only loaded attributes
        # (https://stackoverflow.com/questions/258775/how-to-find-out-if-a-lazy-relation-isnt-loaded-yet-with-sqlalchemy)
        mapper = inspect(cls)
        me_dict = self.__dict__
        loaded_cols, unloaded_cols = 0, 0
        idx = 1
        STRBYTES_MAXCHAR = 30
        ret.append('')
        for c in mapper.columns.keys():
            if c in me_dict:
                val = me_dict[c]
                if type(val) == str and len(val) > STRBYTES_MAXCHAR:
                    val = val[:STRBYTES_MAXCHAR] + \
                        " ... (showing first %d characters only)" % STRBYTES_MAXCHAR
                elif type(val) == bytes and len(val) > STRBYTES_MAXCHAR:
                    val = val[:STRBYTES_MAXCHAR] + \
                        b" ... (showing first %d characters only)" % STRBYTES_MAXCHAR
                elif type(val) == datetime:
                    val = val.isoformat()
                else:
                    val = str(val)
                ret.append("  %s: %s" % (c, val))
                loaded_cols += 1
            else:
                ret.append("  %s" % c)
                unloaded_cols += 1
        ret[idx] = ' columns (%d of %d loaded):' % (loaded_cols, loaded_cols + unloaded_cols)
        idx = len(ret)
        ret.append('')
        loaded_rels, unloaded_rels = 0, 0
        for r in mapper.relationships.keys():
            if r in me_dict:
                ret.append("  %s: `%s` object" % (r, str(me_dict[r].__class__.__name__)))
                loaded_rels += 1
            else:
                ret.append("  %s" % r)
                unloaded_rels += 1
        ret[idx] = ' relationships (%d of %d loaded):' % (loaded_rels, loaded_rels + unloaded_rels)
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

    __table_args__ = (
                      UniqueConstraint('webservice_id', 'event_id', name='ws_eventid_uc'),
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

    __table_args__ = (
                      UniqueConstraint('station_url', 'dataselect_url',
                                       name='sta_data_uc'),
                     )


# global var defining the substring to search for returning the network location


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
        return self.channel[0:1]  # if len(self.channel) == 3 else None

    @band_code.expression
    def band_code(cls):  # @NoSelf
        '''returns the sql expression returning the first letter of the channel field,
        or NULL if the latter has not length 3'''
        # return an sql expression matching the last char or None if not three letter channel
        return substr(cls.channel, 1, 1)
        # return case([(func.length(cls.channel) == 3, func.substr(cls.channel, 1, 1))], else_=null())

    @hybrid_property
    def instrument_code(self):
        '''returns the second letter of the channel field, or None if the latter has not length 3'''
        return self.channel[1:2]  # if len(self.channel) == 3 else None

    @instrument_code.expression
    def instrument_code(cls):  # @NoSelf
        '''returns the sql expression returning the second letter of the channel field,
        or NULL if the latter has not length 3'''
        # return an sql expression matching the last char or None if not three letter channel
        return substr(cls.channel, 2, 1)
        # return case([(func.length(cls.channel) == 3, func.substr(cls.channel, 2, 1))], else_=null())

    @hybrid_property
    def orientation_code(self):
        '''returns the third letter of the channel field, or None if the latter has not length 3'''
        return self.channel[2:3]  # if len(self.channel) == 3 else None

    @orientation_code.expression
    def orientation_code(cls):  # @NoSelf
        '''returns the sql expression returning the third letter of the channel field,
        or NULL if the latter has not length 3'''
        # return an sql expression matching the last char or None if not three letter channel
        return substr(cls.channel, 3, 1)
        #return case([(func.length(cls.channel) == 3, func.substr(cls.channel, 3, 1))], else_=null())

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

    # DEFINE HYBRID PROPERTIES. ACTUALY, WE ARE JUST INTERESTED IN HYBRID CLASSMETHODS FOR
    # QUERYING, BUT IT SEEMS THERE IS NO WAY TO DEFINE THEM WITHOUT DEFINING THE INSTANCE METHOD
    @hybrid_property
    def event_distance_km(self):
        return self.event_distance_deg * (2.0 * 6371 * 3.14159265359 / 360.0)

    @event_distance_km.expression
    def event_distance_km(cls):  # @NoSelf
        return deg2km(cls.event_distance_deg)

    @hybrid_property
    def duration_sec(self):
        try:
            return (self.end_time - self.start_time).total_seconds()
        except TypeError:  # some None's
            return None

    @duration_sec.expression
    def duration_sec(cls):  # @NoSelf
        return duration_sec(cls.start_time, cls.end_time)

    @hybrid_property
    def missing_data_sec(self):
        try:
            return (self.request_end - self.request_start).total_seconds() - \
                (self.end_time - self.start_time).total_seconds()
        except TypeError:  # some None's
            return None

    @missing_data_sec.expression
    def missing_data_sec(cls):  # @NoSelf
        return missing_data_sec(cls.start_time, cls.end_time, cls.request_start, cls.request_end)

    @hybrid_property
    def missing_data_ratio(self):
        try:
            return 1.0 - ((self.end_time - self.start_time).total_seconds() /
                          (self.request_end - self.request_start).total_seconds())
        except TypeError:  # some None's
            return None

    @missing_data_ratio.expression
    def missing_data_ratio(cls):  # @NoSelf
        return missing_data_ratio(cls.start_time, cls.end_time, cls.request_start, cls.request_end)

    @hybrid_property
    def has_data(self):
        return bool(self.data)

    @has_data.expression
    def has_data(cls):  # @NoSelf
        return withdata(cls.data)

    @hybrid_property
    def has_class(self):
        return len(self.classes) > 0

    @has_class.expression
    def has_class(cls):  # @NoSelf
        return cls.classes.any()

    def get(self, *columns):  # DEPRECATED: used for testing
        '''Gets the values in the relative columns'''
        qry = object_session(self).query(*columns)  # .select_from(self.__class__)
        jointables = set(c.class_ for c in columns)
        if jointables:
            joins = []
            model = self.__class__
            for r in self.__mapper__.relationships.values():
                if r.mapper.class_ in jointables:
                    joins.append(getattr(model, r.key))
            if joins:
                qry = qry.join(*joins)
        metadata = qry.filter(Segment.id == self.id).all()
        if len(metadata) == 1:
            return metadata[0]
        else:
            return metadata

    def del_classes(self, *ids_or_labels):
        self.edit_classes(*ids_or_labels, mode='del')

    def set_classes(self, *ids_or_labels, annotator=None):
        self.edit_classes(*ids_or_labels, mode='set', annotator=annotator)

    def add_classes(self, *ids_or_labels, annotator=None):
        self.edit_classes(*ids_or_labels, mode='add', annotator=annotator)

    def edit_classes(self, *ids_or_labels, mode, auto_commit=True, annotator=None):
        """:param mode: either 'add' 'set' or 'del'"""
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
            for c in classes:
                if c.id in ids or c.label in labels:
                    self.classes.remove(c)
                    needs_commit = True
            ids = labels = set()  # do not add anything
        elif mode == 'add':
            for c in classes:
                if c.id in ids:
                    ids.remove(c.id)  # already set, remove it and don't add it again
                if c.label in labels:
                    labels.remove(c.label)  # already set, remove it and don't add it again
        elif mode != 'set':
            raise TypeError("`mode` argument needs to be in ('add', 'del', 'set'), "
                            "'%s' supplied" % str(mode))

        if ids or labels:
            flt1 = None if not ids else Class.id.in_(ids)  # filter on ids, or None
            flt2 = None if not labels else Class.label.in_(labels)  # filter on labels, or None
            flt = flt1 if flt2 is None else flt2 if flt1 is None else (flt1 | flt2)
            classids2add = [_[0] for _ in sess.query(Class.id).filter(flt)]
            if classids2add:
                needs_commit = True
                sess.add_all((ClassLabelling(class_id=cid,
                                             segment_id=self.id, annotator=annotator,
                                             is_hand_labelled=annotator is not None)
                              for cid in classids2add))

        if needs_commit and auto_commit:
            try:
                sess.commit()
            except SQLAlchemyError as _:
                sess.rollback()
                raise

    @hybrid_property
    def seed_id(self):
        try:
            return self.data_seed_id or \
                ".".join([self.station.network, self.station.station,
                          self.channel.location, self.channel.channel])
        except (TypeError, AttributeError):
            return None

    @seed_id.expression
    def seed_id(cls):  # @NoSelf
        '''returns data_seed_id if the latter is not None, else net.sta.loc.cha by querying the
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
        sel = select([concat(Station.network, dot, Station.station, dot,
                             Channel.location, dot, Channel.channel)]).\
            where((Channel.id == cls.channel_id) & (Station.id == Channel.station_id)).limit(1).\
            label('seedidentifier')
        return case([(cls.data_seed_id.isnot(None), cls.data_seed_id)], else_=sel)

    event = relationship("Event", backref=backref("segments", lazy="dynamic"))
    channel = relationship("Channel", backref=backref("segments", lazy="dynamic"))
    # http://stackoverflow.com/questions/17580649/sqlalchemy-relationships-across-multiple-tables
    # this method will work better, as the ORM can also handle
    # eager loading with this one.
    station = relationship("Station", secondary="channels",  # <-  must be table name in metadata
                           primaryjoin="Segment.channel_id == Channel.id",
                           secondaryjoin="Station.id == Channel.station_id",
                           uselist=False,  # viewonly=True,
                           backref=backref("segments", lazy="dynamic"))
    classes = relationship("Class",  # lazy='dynamic', viewonly=True,
                           secondary="class_labellings",  # <-  must be table name in metadata
                           backref=backref("segments", lazy="dynamic"))
    datacenter = relationship("DataCenter", backref=backref("segments", lazy="dynamic"))
    download = relationship("Download", backref=backref("segments", lazy="dynamic"))

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

    __table_args__ = (UniqueConstraint('label', name='class_label_uc'),)


class ClassLabelling(Base):

    __tablename__ = "class_labellings"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    is_hand_labelled = Column(Boolean, server_default="1")  # Note: "TRUE" fails in sqlite!
    annotator = Column(String)

    __table_args__ = (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)
