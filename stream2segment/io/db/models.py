"""
s2s database ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

import sqlite3

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
    event)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property  # , hybrid_method
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import relationship, backref, deferred  # , load_only

from sqlalchemy.sql.expression import (func, text)

from stream2segment.io import Fdsnws


class Base:
    """Abstract base class for a Stream2segment ORM Model"""

    @declared_attr
    def __tablename__(cls):
        chars = [cls.__name__[0].lower()]
        for char in cls.__name__[1:]:
            charl = char.lower()
            if charl != char:
                chars.append('_')
            chars.append(charl)
        chars.append('es' if chars[-1] == 's' else 's')
        return ''.join(chars)

    @declared_attr
    def id(cls):
        return Column('id', Integer, primary_key=True, autoincrement=True)  # noqa

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


class Download(Base):  # pylint: disable=too-few-public-methods
    """Model representing the executed downloads"""

    # Column(Integer, primary_key=True, autoincrement=True)  # noqa

    # run_time below has server_default as `func.now()`. This issues a CURRENT
    # TIMESTAMP on the SQL side. That's ok, BUT the column CANNOT BE UNIQUE!!
    # the CURRENT TIMESTAMP is evaluated once at the beginning of an SQL
    # Statement, so two references in the same session will result in the same
    # value. If we need to make a datetime unique, then either specify
    # 1) default=datetime.datetime.utcnow() BUT NO server_default (the latter
    # seems to have priority if both are provided)
    # 2) or don't make the column unique (what we did)
    run_time = Column(DateTime, server_default=func.now())

    @declared_attr
    def log(cls):
        return deferred(Column(String))  # lazy load: only upon direct access

    warnings = Column(Integer, server_default=text('0'))  # , default=0)
    errors = Column(Integer, server_default=text('0'))  # , default=0)

    @declared_attr
    def config(cls):
        return deferred(Column(String))

    program_version = Column(String)


class Event(Base):  # pylint: disable=too-few-public-methods
    """Model representing a seismic Event"""

    # id = Column(Integer, primary_key=True, autoincrement=True)  # noqa

    @declared_attr
    def webservice_id(cls):
        return Column(Integer, ForeignKey("web_services.id"), nullable=False)

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
    event_type = Column(String)
    quakeml = Column(LargeBinary, nullable=True)

    @property
    def url(self):
        return self.webservice.url + '?eventid=%s' % str(self.event_id)

    @declared_attr
    def webservice(cls):
        return relationship("WebService", backref=backref("events", lazy="dynamic"))

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return UniqueConstraint('webservice_id', 'event_id',
                                name='ws_eventid_uc'),  # <- tuple


class WebService(Base):
    """Model representing a web service (e.g., event web service)"""
    # NOTE: This class currently implements an FDSN event web service only.
    # The name was left general in case in the future we want to merge
    # DataCenter with this model. This is also why the intention of the
    # 'type' column below, which currently has only 'event' implemented
    # (no 'station' or 'dataselect')

    # id = Column(Integer, primary_key=True, autoincrement=True)  # noqa
    name = Column(String)
    type = Column(String)  # e.g. event. See comment above
    url = Column(String, nullable=False)  # if you change attr, see BELOW!

    # segments = relationship("Segment", backref="data_centers")
    # stations = relationship("Station", backref="data_centers")

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return UniqueConstraint('url', name='url_uc'),  # <- tuple


class DataCenter(Base):
    """Model representing a Data center (data provider, e.g. EIDA Node)"""

    # id = Column(Integer, primary_key=True, autoincrement=True)  # noqa
    station_url = Column(String, nullable=False)
    dataselect_url = Column(String, nullable=False)
    organization_name = Column(String)  # e.g. EIDA (I guess?)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return UniqueConstraint('station_url', 'dataselect_url',
                                name='sta_data_uc'),  # <- tuple


def check_datacenter_urls_fdsn(mapper, connection, target):
    """Check for datacenter URLs. To be used as argument for sqlalchemy.listen or
    listen_to (see implementation in this program), e.g.
    ```
    @event.listens_for(DataCenter, 'before_insert', check_datacenter_urls_fdsn)
    @event.listens_for(DataCenter, 'before_update', check_datacenter_urls_fdsn)
    ```
    or
    `event.listens_for(DataCenter, 'before_insert')(check_datacenter_urls_fdsn)`
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
            # the idea here is to populate a missing field, not to raise...
            # however, raising might be a better solution but should be done not
            # only when either field is None
            pass


class Station(Base):
    """Model representing a Station"""

    # id = Column(Integer, primary_key=True, autoincrement=True)
    @declared_attr
    def datacenter_id(cls):
        return Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    inventory_xml = Column(LargeBinary)

    @property
    def url(self):
        qry_str = 'net=%s&sta=%s&start=%s' % \
                  (self.network, self.station, self.start_time.isoformat('T'))
        return self.datacenter.station_url + '?%s' % qry_str

    @hybrid_property
    def has_inventory(self):
        return bool(self.inventory_xml)

    @has_inventory.expression
    def has_inventory(cls):  # pylint:disable=no-self-argument
        return withdata(cls.inventory_xml)

    # relationships (implement here only those shared by download+process):
    @declared_attr
    def datacenter(cls):
        return relationship("DataCenter", backref=backref("stations", lazy="dynamic"))

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('network', 'station', 'start_time',
                                 name='net_sta_stime_uc'),)


class Channel(Base):
    """Model representing a Channel"""

    # id = Column(Integer, primary_key=True, autoincrement=True)
    @declared_attr
    def station_id(cls):
        return Column(Integer, ForeignKey("stations.id"), nullable=False)
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

    # relationships:
    # relationships (implement here only those shared by download+process):
    @declared_attr
    def station(cls):
        return relationship("Station", backref=backref("channels", lazy="dynamic"))

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('station_id', 'location', 'channel',
                                 name='net_sta_loc_cha_uc'),)


MINISEED_READ_ERROR_CODE = -2


class Segment(Base):
    """Model representing a Waveform segment"""

    # id = Column(Integer, primary_key=True, autoincrement=True)

    @declared_attr
    def event_id(cls):
        return Column(Integer, ForeignKey("events.id"), nullable=False)

    @declared_attr
    def channel_id(cls):
        return Column(Integer, ForeignKey("channels.id"), nullable=False)

    @declared_attr
    def datacenter_id(cls):
        return Column(Integer, ForeignKey("data_centers.id"), nullable=False)

    data_seed_id = Column(String)
    event_distance_deg = Column(Float, nullable=False)
    data = Column(LargeBinary)
    download_code = Column(Integer)
    start_time = Column(DateTime)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    sample_rate = Column(Float)
    maxgap_numsamples = Column(Float)

    @declared_attr
    def download_id(cls):
        return Column(Integer, ForeignKey("downloads.id"), nullable=False)

    request_start = Column(DateTime, nullable=False)
    request_end = Column(DateTime, nullable=False)
    queryauth = Column(Boolean, nullable=False,
                       server_default="0")  # note: null fails in sqlite!

    @property
    def url(self):
        """Return the full URL that can be used to (re)download the Segment
        waveform data in miniSEED format (For details, see GET request here:
        https://www.fdsn.org/webservices/fdsnws-dataselect-1.1.pdf)
        """
        net, sta = self.station.network, self.station.station
        loc, cha = self.channel.location, self.channel.channel
        qry_str = 'net=%s&sta=%s&loc=%s&cha=%s&start=%s&end=%s' % \
                  (net, sta, loc, cha, self.request_start.isoformat('T'),
                   self.request_end.isoformat('T'))
        return self.datacenter.dataselect_url + '?%s' % qry_str

    @hybrid_property
    def has_data(self):
        return bool(self.data)

    @has_data.expression
    def has_data(cls):  # pylint:disable=no-self-argument
        return withdata(cls.data)

    @hybrid_property
    def has_valid_data(self):
        return bool(self.data) and self.download_code is not None and \
               self.download_code != MINISEED_READ_ERROR_CODE

    @has_valid_data.expression
    def has_valid_data(cls):  # pylint:disable=no-self-argument
        # download code should never be None. However, for safety, != None
        # checks also that the server HTTP status code is an integer properly
        # set. Note that by checking cls.download_code == 200 is not sufficient
        # as there are custom codes set during download
        return withdata(cls.data) & \
               cls.download_code.isnot(None) & \
               (cls.download_code != MINISEED_READ_ERROR_CODE)

    # relationships (implement here only those shared by download+process):
    @declared_attr
    def station(cls):
        # Relationship spanning 3 tables (https://stackoverflow.com/a/17583437)
        return relationship("Station",
                            # `secondary` must be table name in metadata:
                            secondary="channels",
                            primaryjoin="Segment.channel_id == Channel.id",
                            secondaryjoin="Station.id == Channel.station_id",
                            uselist=False,
                            # the following two params are set in order to make this
                            # relationship work in v 1 and 2, but no idea why due to
                            # the lack of clarity in sqlalchemy docs
                            viewonly=True,
                            sync_backref=False,
                            backref=backref("segments", lazy="dynamic"))

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('channel_id', 'event_id', name='chaid_evtid_uc'),)


class Class(Base):
    """Model representing a segment class label"""

    # id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    description = Column(String)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('label', name='class_label_uc'),)


class ClassLabelling(Base):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""

    # Column(Integer, primary_key=True, autoincrement=True)
    @declared_attr
    def segment_id(cls):
        return Column(Integer, ForeignKey("segments.id"), nullable=False)

    @declared_attr
    def class_id(cls):
        return Column(Integer, ForeignKey("classes.id"), nullable=False)

    is_hand_labelled = Column(Boolean,
                              server_default="1")  # "TRUE" fails in sqlite!
    annotator = Column(String)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)
