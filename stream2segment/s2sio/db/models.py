'''
Created on Jul 15, 2016

@author: riccardo
'''
from pandas import to_datetime, to_numeric
# from sqlalchemy import engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    ForeignKey,
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
import datetime
from sqlalchemy.orm.mapper import validates
# from stream2segment.s2sio.db.pd_sql_utils import get_col_names, get_cols
# from sqlalchemy.sql.sqltypes import BigInteger, BLOB
# from sqlalchemy.sql.schema import ForeignKey


class attrdict(dict):
    """A dict-like object whose keys can be accessed also as attributes
    ```
    d = attrdict(a=9, b='g')
    d['a'] == d.a  # True
    d['b'] == d.b  # True
    ```
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


_Base = declarative_base()


class Base(_Base):

    __abstract__ = True

#     @classmethod
#     def get_cols(cls):
#         return get_cols(cls)

    @classmethod
    def get_col_names(cls, primary_keys=True, foreign_keys=True):
        """Returns the column names, i.e. those attributes reflecting a db table column
        Note that this method is also implemented in pd_sql_utils but we copy it here for
        decoupling the two modules
        """
        if primary_keys and foreign_keys:
            return cls.__table__.columns.keys()
        elif not foreign_keys:
            fkeys = set((fk.parent for fk in cls.__table__.foreign_keys))
            return [k for k, c in cls.__table__.columns.items()
                    if (primary_keys or not c.primary_key) and
                    (foreign_keys or c not in fkeys)]
        else:
            return [k for k, c in cls.__table__.columns.items() if not c.primary_key]

    def __str__(self):
        return ",\n".join("%s=%s" % (str(c), str(getattr(self, c))) for c in self.get_col_names())

    def copy(self):
        """Returns a copy of this object. Modifications made on attributes of the copy will not
        affect this instance"""
        return self.__class__(**{c: getattr(self, c) for c in self.get_col_names()})

    def toattrdict(self, primary_keys=False, foreign_keys=False):
        """Returns a dict-like object representing this instance. For convenience, the returned dict
        keys can be also accessed as attributes
        :param primary_keys: boolean (False by default): whether or not primary keys should be
        copied. If False, the returned dict won't have attributes and keys corresponding to this
        instance primary keys
        :param foreign_keys: boolean (False by default): whether or not foreign keys should be
        copied. If False, the returned dict won't have attributes and keys corresponding to this
        instance foreign keys.
        :Example:
        ```
        Event(id='4', latitude=67).toattrdict().latitude # returns 60
        Event(id='4', latitude=67).toattrdict()['latitude'] # same as above, it's a dict object
        Event(id='4', latitude=67).toattrdict().latitude # AttributeError: id primary key
        Event(id='4', latitude=67, primary_keys=True).toattrdict().id # returns '4'
        Event(**Event(latitude='4').toattrdict())  # returns a copy of the inner event
        ```
        """
        # see:
        # http://stackoverflow.com/questions/2441796/how-to-discover-table-properties-from-sqlalchemy-mapped-object
        fkeys = set([]) if foreign_keys is True else set((fk.parent
                                                          for fk in self.__table__.foreign_keys))
        # note below: we removed the option "autocommit" cause all columns seem to have that
        # attribute set to True
        return attrdict(**{k: getattr(self, k) for k, c in self.__class__.__table__.columns.items()
                           if (primary_keys or not c.primary_key) and
                           (foreign_keys or c not in fkeys)})

    def putindict(self, colnames=None):
        if colnames is None:
            colnames = self.get_col_names()
        return attrdict(**{k: getattr(self, k) for k in colnames})


class Run(Base):
    """The runs"""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)  # pylint:disable=invalid-name
    run_time = Column(DateTime, unique=True, default=datetime.datetime.utcnow)
    log = Column(String)
    warnings = Column(Integer)
    errors = Column(Integer)
    # segments_found = Column(Integer)
    # segments_written = Column(Integer)
    # segments_skipped = Column(Integer)
    config = Column(String)
    program_version = Column(String)


def dc_datasel_default(context):
    return context.current_parameters['station_query_url'].replace("/station", "/dataselect")


class DataCenter(Base):
    """DataCenters"""

    __tablename__ = "data_centers"

    id = Column(Integer, primary_key=True, autoincrement=True)  # pylint:disable=invalid-name
    station_query_url = Column(String, nullable=False)
    dataselect_query_url = Column(String, default=dc_datasel_default, onupdate=dc_datasel_default)

    # segments = relationship("Segment", backref="data_centers")
    # stations = relationship("Station", backref="data_centers")

    __table_args__ = (
                      UniqueConstraint('station_query_url', 'dataselect_query_url',
                                       name='sta_data_uc'),
                     )


class Event(Base):
    """Events"""

    __tablename__ = "events"

    id = Column(String, primary_key=True, autoincrement=False)  # pylint:disable=invalid-name
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
    # datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    network = Column(String, nullable=False)
    station = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    site_name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    inventory_xml = Column(Binary)

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

    # datacenter = relationship("DataCenter", backref="stations")


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

    station = relationship("Station", backref="channels")


class SegmentClassAssociation(Base):

    __tablename__ = "segment_class_associations"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    class_id_hand_labelled = Column(Boolean)

    __table_args__ = (UniqueConstraint('segment_id', 'class_id', name='seg_class_uc'),)


class Class(Base):  # pylint: disable=no-init
    """A Building.
    """
    __tablename__ = 'classes'

    id = Column(Integer, primary_key=True, autoincrement=False)
    label = Column(String)
    description = Column(String)

#     segments = relationship(
#         "Segment",
#         secondary=SegmentClassAssociation,
#         back_populates="classes")


class Segment(Base):
    """Segments"""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True)  # , default=seg_pkey_default, onupdate=seg_pkey_default)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    datacenter_id = Column(Integer, ForeignKey("data_centers.id"), nullable=False)
    event_distance_deg = Column(Float, nullable=False)
    data = Column(Binary, nullable=False)
    start_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

    event = relationship("Event", backref="segments")
    channel = relationship("Channel", backref="segments")
    datacenter = relationship("DataCenter", backref="segments")
    run = relationship("Run", backref="segments")

#    processings = relationship("Processing", backref="segments")

#     classes = relationship(
#         "Class",
#         secondary=SegmentClassAssociation,
#         back_populates="segments")

    __table_args__ = (
                      UniqueConstraint('channel_id', 'start_time', 'end_time',
                                       name='net_sta_loc_cha_stime_etime_uc'),
                     )


class Processing(Base):

    __tablename__ = "processing"

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey("segments.id"))
    run_id = Column(Integer, ForeignKey("runs.id"))
    mseed_rem_resp_savewindow = Column(Binary)
    fft_rem_resp_t05_t95 = Column(Binary)
    fft_rem_resp_until_atime = Column(Binary)
    wood_anderson_savewindow = Column(Binary)
    cum_rem_resp = Column(Binary)
    pga_atime_t95 = Column(Float)
    pgv_atime_t95 = Column(Float)
    pwa_atime_t95 = Column(Float)
    t_pga_atime_t95 = Column(DateTime)
    t_pgv_atime_t95 = Column(DateTime)
    t_pwa_atime_t95 = Column(DateTime)
    cum_t05 = Column(DateTime)
    cum_t10 = Column(DateTime)
    cum_t25 = Column(DateTime)
    cum_t50 = Column(DateTime)
    cum_t75 = Column(DateTime)
    cum_t90 = Column(DateTime)
    cum_t95 = Column(DateTime)
    snr_rem_resp_fixedwindow = Column(Float)
    snr_rem_resp_t05_t95 = Column(Float)
    snr_rem_resp_t10_t90 = Column(Float)
    amplitude_ratio = Column(Float)
    is_saturated = Column(Boolean)
    has_gaps = Column(Boolean)
    double_event_result = Column(Integer)
    secondary_event_time = Column(DateTime)
    coda_start_time = Column(DateTime)  # the coda start time
    coda_slope = Column(Float)  # slope of the regression line
    # coda_intercept : float  # intercept of the regression line
    coda_r_value = Column(Float)  # correlation coefficient
    coda_is_ok = Column(Boolean)

    segment = relationship("Segment", backref="processings")
    run = relationship("Run", backref="processings")

    __table_args__ = (UniqueConstraint('segment_id', 'run_id', name='seg_run_uc'),)



# FIXME: implement runs datetime server side, and run test to see it's utc!
# FIXME: test joins with relations
# fixme: implement DataFrame write, and test it
# fixme: implement ondelete and oncascade when possible
# FIXME: many to one with respect to segments table!!
