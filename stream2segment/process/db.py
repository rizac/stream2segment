"""
s2s process database ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

import os
from datetime import datetime
from math import pi
from io import BytesIO
import gzip
import zipfile
import zlib
import bz2

from sqlalchemy import event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm.session import object_session
from sqlalchemy.sql.expression import func, text, case, select
from obspy.core.stream import _read  # noqa
from obspy.core.inventory.inventory import read_inventory

from stream2segment.io.db.sqlconstructs import missing_data_ratio, missing_data_sec, \
    duration_sec, deg2km, concat, substr
from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.process import SkipSegment
from stream2segment.io.db import models, get_session  # noqa
# (get_session should be imported from here in user code)


Base = declarative_base(cls=models.Base)


class Download(Base, models.Download):  # pylint: disable=too-few-public-methods
    """Model representing the executed downloads"""
    pass


class Event(Base, models.Event):  # pylint: disable=too-few-public-methods
    """Model representing a seismic Event"""
    pass


class WebService(Base, models.WebService):
    """Model representing a web service (e.g., event web service)"""
    pass


class DataCenter(Base, models.DataCenter):
    """Model representing a Data center (data provider, e.g. EIDA Node)"""
    pass


# listen for insertion and updates and check Datacenter URLS (the call below
# is the same as decorating check_datacenter_urls_fdsn with '@event.listens_for'):
event.listens_for(DataCenter, 'before_insert')(models.check_datacenter_urls_fdsn)
event.listens_for(DataCenter, 'before_update')(models.check_datacenter_urls_fdsn)


class Class(Base, models.Class):
    """Model representing a segment class label"""
    pass


class ClassLabelling(Base, models.ClassLabelling):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""
    pass


class Channel(Base, models.Channel):
    """Model representing a Channel"""

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


class Station(Base, models.Station):
    """Model representing a Station"""

    @hybrid_property
    def netsta_code(self):
        return "%s.%s" % (self.network, self.station)

    @netsta_code.expression
    def netsta_code(cls):  # pylint:disable=no-self-argument
        """Return the station code, i.e. self.network + '.' + self.station"""
        dot = text("'.'")
        return concat(Station.network, dot, Station.station). \
            label('networkstationcode')


    def inventory(self, reload=False):
        """Return the inventory from self (a segment class)"""
        # inventory is lazy loaded. The output of the loading process
        # (or the Exception raised, if any) is stored in the self._inventory
        # attribute. When querying the inventory a further time, the stored value
        # is returned, or raised (if it is an Exception)
        inventory = getattr(self, "_inventory", None)
        if reload and inventory is not None:
            inventory = None
        if inventory is None:
            try:
                inventory = self._inventory = get_inventory(self)
            except Exception as exc:   # pylint: disable=broad-except
                inventory = self._inventory = \
                    SkipSegment("Station inventory (xml) error: %s" %
                                (str(exc) or str(exc.__class__.__name__)))
        if isinstance(inventory, Exception):
            raise inventory
        return inventory


def get_inventory(station):
    """Return the inventory object for the given station.
    Raises :class:`SkipSegment` if inventory data is empty
    """
    data = station.inventory_xml
    if not data:
        raise SkipSegment('no data')
    return get_inventory_from_bytes(data)


def get_inventory_from_bytes(bytestr):
    """Return the inventory object given an input bytes sequence representing an
    inventory (xml) from, e.g., downloaded data
    :param bytestr: the sequence of bytes. It can be compressed with any of the function
    defined when saving the byte string (gzip, bz and so on). The method will first try
    to de-compress data. Then, the de-compressed data (if de-compression does not fail)
    or the data passed as argument will be passed to ObsPy `read_inventory`

    :return: an `class: obspy.core.inventory.inventory.Inventory` object
    """
    try:
        bytestr = decompress(bytestr)
    except(IOError, zipfile.BadZipfile, zlib.error) as _:
        pass  # try anyway to open the file (who knows)
    return read_inventory(BytesIO(bytestr), format="STATIONXML")


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


class Segment(Base, models.Segment):
    """Model representing a Waveform segment"""

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

    def inventory(self, reload=False):
        return self.station.inventory(reload)

    def dbsession(self):
        return object_session(self)

    def siblings(self, parent=None, conditions=None, colname=None):
        """Return a SQLAlchemy query yielding the siblings of this segments
        according to `parent`. Refer to the method Segment.get_siblings in
        :module:`models.py`.

        :parent: a string identifying the parent whereby perform a selection
        :conditions: a dict of strings mapped to string expressions to be
            evaluated, and select a subset of siblings. None (the defaults) means:
            empty dict (no additional slection condition)
        """
        sblngs = self.get_siblings(parent, colname=colname)  # returns a Segment object
        if conditions:
            sblngs = exprquery(sblngs, conditions, orderby=None)
        return sblngs

    def stream(self, reload=False):
        """Return the stream from self (a segment class)"""
        # stream is lazy loaded. The output of the loading process
        # (or the Exception raised, if any) is stored in the self._stream attribute.
        # When querying the stream a further time, the stored value is returned,
        # or raised (if it is an Exception)
        stream = getattr(self, "_stream", None)
        if reload and stream is not None:
            stream = None
        if stream is None:
            try:
                stream = self._stream = get_stream(self)
            except Exception as exc:  # pylint: disable=broad-except
                stream = self._stream = \
                    SkipSegment("MiniSeed error: %s" %
                                (str(exc) or str(exc.__class__.__name__)))

        if isinstance(stream, Exception):
            raise stream
        return stream

    # Relationships:

    event = relationship("Event", backref=backref("segments",
                                                  lazy="dynamic"))
    channel = relationship("Channel", backref=backref("segments",
                                                      lazy="dynamic"))
    # (station relationship is implemented in superclass, as it's needed for download)
    classes = relationship("Class",  # lazy='dynamic', viewonly=True,
                           # `secondary` must be table name in metadata:
                           secondary="class_labellings",
                           backref=backref("segments", lazy="dynamic"))
    datacenter = relationship("DataCenter", backref=backref("segments",
                                                            lazy="dynamic"))
    download = relationship("Download", backref=backref("segments",
                                                        lazy="dynamic"))


def get_stream(segment, format="MSEED", headonly=False, **kwargs):  # noqa
    """Return a Stream object relative to the given segment. The optional
    arguments are the same than `obspy.core.stream.read` (excepts than "format"
    defaults to "MSEED")

    :param segment: a model ORM instance representing a Segment (waveform data
        db row)
    :param format: string, optional (default "MSEED"). Format of the file to
        read. See ObsPy `Supported Formats`_ section below for a list of
        supported formats. If `format` is set to ``None`` it will be
        automatically detected which results in a slightly slower reading. If a
        format is specified, no further format checking is done.
    :param headonly: bool, optional (dafult: False). If set to ``True``, read
        only the data header. This is most useful for scanning available meta
        information of huge data sets
    :param kwargs: Additional keyword arguments passed to the underlying
        waveform reader method.
    """
    data = segment.data
    if not data:
        raise SkipSegment('no data')
    # Do not call `obspy.core.stream.read` because, when passed a BytesIO, if
    # it fails reading it stores the bytes data to a temporary file and
    # re-tries by reading the file. This is a useless and time-consuming
    # behavior in our case: `data` is directly downloaded from the data-center:
    # if we fail we should raise immediately. To do that, we call
    # ``obspy.core.stream._read`, which is what `obspy.core.stream.read` does
    # internally. Note that calling _read might require some attention as
    # "private" methods might change across versions. Also, FYI, the source
    # function which does the real job is "obspy.io.mseed.core._read_mseed"
    try:
        return _read(BytesIO(data), format, headonly, **kwargs)
    except Exception as terr:
        raise SkipSegment(str(terr))
