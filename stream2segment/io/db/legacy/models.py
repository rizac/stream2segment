"""
s2s database ORM (legacy, v<=4)
"""
import math
# Jul 15, 2016
import sqlite3
import gzip
import zipfile
import zlib
import bz2
from io import BytesIO

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
    event,
    cast
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import deferred

from sqlalchemy.sql.expression import (func, text)
try:
    from sqlalchemy.orm import declarative_base  # v1.4+
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base  # v<1.4


Base = declarative_base()


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


class Download(Base):
    """Model representing the executed downloads"""
    __tablename__ = 'downloads'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
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
    config = deferred(Column(String))
    program_version = Column(String)


class Event(Base):
    """Model representing a seismic Event"""
    __tablename__ = 'events'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
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
    event_type = Column(String)

    # @property
    # def url(self):
    #     return self.webservice.url + '?eventid=%s' % str(self.event_id)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint(
            'webservice_id', 'event_id', name='ws_eventid_uc'),
        )


class WebService(Base):
    """Model representing a web service (e.g., event web service)"""
    __tablename__ = 'web_services'

    # NOTE: This class currently implements an FDSN event web service only.
    # The name was left general in case in the future we want to merge
    # DataCenter with this model. This is also why the intention of the
    # 'type' column below, which currently has only 'event' implemented
    # (no 'station' or 'dataselect')

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    type = Column(String)  # e.g. event. See comment above
    url = Column(String, nullable=False)  # if you change attr, see BELOW!

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('url', name='url_uc'),)


class DataCenter(Base):
    """Model representing a Data center (data provider, e.g. EIDA Node)"""
    __tablename__ = 'data_centers'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    station_url = Column(String, nullable=False)
    dataselect_url = Column(String, nullable=False)
    organization_name = Column(String)  # e.g. EIDA (I guess?)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (
            UniqueConstraint(
                'station_url', 'dataselect_url', name='sta_data_uc'),
        )


class StationXML(Base):
    """Model representing a Station"""
    __tablename__ = 'stations'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    network_code = Column('network', String, nullable=False)
    station_code = Column('station', String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    inventory_xml = Column(LargeBinary)

    # @property
    # def url(self):
    #     qry_str = 'net=%s&sta=%s&start=%s' % \
    #               (self.network_code, self.station_code, self.start_time.isoformat('T'))
    #     return self.datacenter.station_url + '?%s' % qry_str

    # @hybrid_property
    # def has_inventory(self):
    #     return bool(self.inventory_xml)
    #
    # @has_inventory.expression
    # def has_inventory(cls):  # pylint:disable=no-self-argument
    #     return withdata(cls.inventory_xml)

    @property
    def data(self):  # noqa
        """Return the station inventory. See `Segment.inventory` for details"""
        # inventory is lazy loaded. The output of the loading process
        # (or the Exception raised, if any) is stored in the self._inventory
        # attribute. When querying the inventory a further time, the stored value
        # is returned, or raised (if it is an Exception)
        return decompress(self.inventory_xml)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (
            UniqueConstraint(
                'network', 'station', 'start_time', name='net_sta_stime_uc'
            ),
        )


def decompress(bytestr):
    """Decompress `bytestr` (a sequence of bytes) trying to guess the compression
    format. If no guess can be made, returns bytestr. Otherwise, returns the
    de-compressed sequence of bytes. Raises IOError, zipfile.BadZipfile, zlib.error if
    compression is detected but did not work. Note that this might happen if
    (accidentally) the sequence of bytes is not compressed but starts with bytes
    denoting a compression type. Thus function caller should not necessarily raise
    exceptions if this function does, but try to read `bytestr` as if it was not
    compressed
    """
    # check if the data is compressed (https://stackoverflow.com/a/19127748):
    if bytestr.startswith(b"\x1f\x8b\x08"):  # gzip
        # raises IOError in case
        with gzip.GzipFile(mode='rb', fileobj=BytesIO(bytestr)) as gzip_obj:
            bytestr = gzip_obj.read()
    elif bytestr.startswith(b"\x42\x5a\x68"):  # bz2
        bytestr = bz2.decompress(bytestr)  # raises IOError in case
    elif bytestr.startswith(b"\x50\x4b\x03\x04"):  # zip
        # raises zipfile.BadZipfile in case
        with zipfile.ZipFile(BytesIO(bytestr), 'r') as zip_obj:
            namelist = zip_obj.namelist()
            if len(namelist) != 1:
                raise ValueError("Found zipped content with %d archives, "
                                 "can only uncompress single archive "
                                 "content" % len(namelist))
            bytestr = zip_obj.read(namelist[0])
    else:
        barray = bytearray(bytestr[:2])  # py 2+3 https://stackoverflow.com/a/41843740
        byte1 = barray[0]
        byte2 = barray[1]
        if (byte1 * 256 + byte2) % 31 == 0 and (byte1 & 143) == 8:  # zlib. 143=int('10001111', 2)
            bytestr = zlib.decompress(bytestr)  # raises zlib.error in case
    return bytestr


class Channel(Base):
    """Model representing a Channel"""
    __tablename__ = 'channels'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)
    location_code = Column('location', String, nullable=False)
    channel_code = Column('channel', String, nullable=False)
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
        return self.channel_code[0:1]  # if len(self.channel) == 3 else None

    @band_code.expression
    def band_code(cls):
        """Return the sql expression returning the first letter of the channel
        field"""
        # return a sql expression matching the last char or None if not three
        # letter channel
        return func.substr(cls.channel_code, 1, 1)

    @hybrid_property
    def instrument_code(self):
        """Return the second letter of the channel field"""
        return self.channel_code[1:2]  # if len(self.channel) == 3 else None

    @instrument_code.expression
    def instrument_code(cls):
        """Return the sql expression returning the second letter of the channel
        field"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return func.substr(cls.channel_code, 2, 1)

    @hybrid_property
    def band_instrument_code(self):
        """Return the first two letters of the channel field. Useful when we
        want to get the same record on different orientations/components"""
        return self.channel_code[0:2]

    @band_instrument_code.expression
    def band_instrument_code(cls):
        """Return the sql expression returning the first two letters of the
        channel field. Useful for queries where we want to get the same record
        on different orientations/components"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return func.substr(cls.channel_code, 1, 2)

    @hybrid_property
    def orientation_code(self):
        """Return the third letter of the channel field"""
        return self.channel_code[2:3]

    @orientation_code.expression
    def orientation_code(cls):
        """Return the sql expression returning the third letter of the channel
        field"""
        # return an sql expression matching the last char or None if not three
        # letter channel
        return func.substr(cls.channel_code, 3, 1)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (
            UniqueConstraint(
                'station_id', 'location', 'channel', name='net_sta_loc_cha_uc'
            ),
        )


MINISEED_READ_ERROR_CODE = -2


class Segment(Base):
    """Model representing a Waveform segment"""
    __tablename__ = 'segments'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
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
    queryauth = Column(Boolean, nullable=False, server_default="0")  # note: null fails in sqlite!  # noqa

    # @property
    # def url(self):
    #     """Return the full URL that can be used to (re)download the Segment
    #     waveform data in miniSEED format (For details, see GET request here:
    #     https://www.fdsn.org/webservices/fdsnws-dataselect-1.1.pdf)
    #     """
    #     net, sta = self.station.network, self.station.station
    #     loc, cha = self.channel.location, self.channel.channel
    #     qry_str = 'net=%s&sta=%s&loc=%s&cha=%s&start=%s&end=%s' % \
    #               (net, sta, loc, cha, self.request_start.isoformat('T'),
    #                self.request_end.isoformat('T'))
    #     return self.datacenter.dataselect_url + '?%s' % qry_str

    @hybrid_property
    def event_distance_km(self):
        return int(round(self.event_distance_deg * self._DEG2KM))

    @event_distance_km.expression
    def event_distance_km(cls):
        return cast(func.round(cls.event_distance_deg * cls._DEG2KM), Integer)

    _DEG2KM = 2.0 * 6371 * math.pi / 360.0

    @hybrid_property
    def noise_window_s(self):
        return int(round((self.arrival_time - self.start_time).total_seconds()))

    @noise_window_s.expression
    def noise_window_s(cls):
        return cast(
            func.round(duration_sec(cls.start_time, cls.arrival_time)),
            Integer
        )

    @hybrid_property
    def signal_window_s(self):
        return int(round((self.end_time - self.arrival_time).total_seconds()))

    @signal_window_s.expression
    def signal_window_s(cls):
        return cast(
            func.round(duration_sec(cls.arrival_time, cls.end_time)),
            Integer
        )

    @hybrid_property
    def gap_score_percent(self):
        try:
            return int(round(self.maxgap_numsamples * 100))
        except TypeError:  # some None(s)
            return None

    @gap_score_percent.expression
    def gap_score_percent(cls):
        return cast(func.round(cls.maxgap_numsamples * 100), Integer)

    # @hybrid_property
    # def has_data(self):
    #     return bool(self.data)
    #
    # @has_data.expression
    # def has_data(cls):
    #     return withdata(cls.data)

    @hybrid_property
    def has_valid_data(self):
        return bool(self.data) and self.download_code is not None and \
               self.download_code != MINISEED_READ_ERROR_CODE

    @has_valid_data.expression
    def has_valid_data(cls):
        # download code should never be None. However, for safety, != None
        # checks also that the server HTTP status code is an integer properly
        # set. Note that by checking cls.download_code == 200 is not sufficient
        # as there are custom codes set during download
        return (
            withdata(cls.data) &
            cls.download_code.isnot(None) &
            (cls.download_code != MINISEED_READ_ERROR_CODE)
        )

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (
            UniqueConstraint('channel_id', 'event_id', name='chaid_evtid_uc'),
        )


class duration_sec(FunctionElement):
    name = 'duration_sec'
    type = Float()
    inherit_cache = True


@compiles(duration_sec)
def standard_duration_sec(element, compiler, **kw):
    starttime, endtime = [compiler.process(c) for c in element.clauses]
    return _duration_postgres(starttime, endtime)


@compiles(duration_sec, 'sqlite')
def sqlite_duration_sec(element, compiler, **kw):
    starttime, endtime = [compiler.process(c) for c in element.clauses]
    return _duration_sqlite(starttime, endtime)


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

class Class(Base):
    """Model representing a segment class label"""

    __tablename__ = 'classes'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    description = Column(String)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (UniqueConstraint('label', name='class_label_uc'),)


class ClassLabelling(Base):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""
    __tablename__ = 'class_labellings'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    is_hand_labelled = Column(Boolean, server_default="1")  # "TRUE" fails in sqlite!
    annotator = Column(String)

    @declared_attr
    def __table_args__(cls):  # noqa  # https://stackoverflow.com/a/43993950
        return (
            UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),
        )
