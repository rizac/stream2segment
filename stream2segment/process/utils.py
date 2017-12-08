'''
Module documentation to be implemented

:date: Sep 19, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# provide some imports to let python3 syntax work also in python 2.7+ effortless. 
# Any of the defaults import below can be safely removed if python2+
# compatibility is not needed

# standard python imports (must be the first import)
from __future__ import absolute_import, division, print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      super, zip)
from future.utils import viewitems

from collections import OrderedDict
from io import BytesIO

from sqlalchemy.orm.session import object_session
from sqlalchemy.orm.exc import UnmappedInstanceError
from sqlalchemy.exc import SQLAlchemyError
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import _read, Stream
from obspy.core.trace import Trace

from stream2segment.io.utils import loads_inv, dumps_inv
from stream2segment.utils.url import urlread
from stream2segment.utils import urljoin
from stream2segment.io.db.models import Segment, Station, Channel
from stream2segment.math.traces import cumsum, cumtimes


def raiseifreturnsexception(func):
    '''decorator that makes a function raise the returned exception, if any
    (otherwise no-op, and the function value is returned as it is)'''
    def wrapping(*args, **kwargs):
        ret = func(*args, **kwargs)
        if isinstance(ret, Exception):
            raise ret
        return ret
    return wrapping


class gui(object):
    """decorators for the processing.py file for making function displayed on the gui
    (Graphical User Interface)"""

    @staticmethod
    def preprocess(func):
        '''decorator that adds the attribute func._s2s_att = "gui.preprocess"'''
        func._s2s_att = "gui.preprocess"
        return func

    @staticmethod
    def customplot(func):
        '''decorator that adds the attribute func._s2s_att = "gui.customplot"'''
        func._s2s_att = "gui.customplot"
        return func

    @staticmethod
    def sideplot(func):
        '''decorator that adds the attribute func._s2s_att = "gui.sideplot"'''
        func._s2s_att = "gui.sideplot"
        return func


def segmentclass4process(func):
    """decorator that temporarily adds to a Segment obspy methods for processing"""

    @raiseifreturnsexception
    def stream(self):
        stream = getattr(self, "_stream", None)
        if stream is None:
            try:
                stream = self._stream = get_stream(self)
            except Exception as exc:
                stream = self._stream = Exception("MiniSeed error: %s" %
                                                  (str(exc) or str(exc.__class__.__name__)))

        return stream

    @raiseifreturnsexception
    def inventory(self):
        inventory = getattr(self, "_inventory", None)
        if inventory is None:
            try:
                save_station_inventory = self._config['save_inventory']
            except:
                save_station_inventory = False
            try:
                inventory = self._inventory = get_inventory(self.station, save_station_inventory)
            except Exception as exc:
                inventory = self._inventory = Exception("Station inventory (xml) error: %s" %
                                                        (str(exc) or str(exc.__class__.__name__)))
        return inventory

    def sn_windows(self):
        '''returns the tuples (start, end), (start, end) where the first list is the signal
        window, and the second is the noise window. All elements are UtcDateTime's
        '''
        # No cache for this variable, as we might modify the segment stream in-place thus it's
        # hard to know when a recalculation is needed (this is particularly important when
        # bounds relative to the cumulative sum are given, if an interval was given there would be
        # no problem)
        return get_sn_windows(self._config, self.arrival_time, self.stream())

    def _query_to_other_orientations(self, *query_args):
        return self.dbsession().query(*query_args).join(Segment.channel).\
                filter((Segment.id != self.id) & (Segment.event_id == self.event_id) &
                       (Channel.station_id == self.channel.station_id) &
                       (Channel.location == self.channel.location) &
                       (Channel.band_code == self.channel.band_code) &
                       (Channel.instrument_code == self.channel.instrument_code))

    def segments_on_other_orientations(self):
        seg_other_orientations = getattr(self, "_other_orientations", None)
        if seg_other_orientations is None:
            segs = self._query_on_other_orientations(Segment).all()
            seg_other_orientations = self._other_orientations = segs
            # assign also to other segments:y
            for seg in segs:
                seg._other_orientations = [s for s in segs if s.id != seg.id] + [self]

        return seg_other_orientations

    def wrapper(*args, **kwargs):
        already_wrapped = hasattr(Segment, "_config")
        if not already_wrapped:
            Segment._config = {}
            Segment.stream = stream
            Segment.inventory = inventory
            Segment.sn_windows = sn_windows
            Segment.segments_on_other_orientations = segments_on_other_orientations
            Segment.dbsession = lambda self: object_session(self)
            Segment._query_to_other_orientations = _query_to_other_orientations

        try:
            return func(*args, **kwargs)
        finally:
            if not already_wrapped:
                # delete attached attributes:
                del Segment._config
                del Segment.stream
                del Segment.inventory
                del Segment.sn_windows
                del Segment.dbsession
                del Segment.segments_on_other_orientations
                del Segment._query_to_other_orientations

    return wrapper


def get_sn_windows(config, a_time, stream):
    '''Returns the spectra windows from a given arguments. Used by `_spectra`
    :return the tuple (start, end), (start, end) where all arguments are `UTCDateTime`s
    and the first tuple refers to the noisy window, the latter to the signal window
    '''
    if 'sn_windows' not in config:
        raise TypeError("'sn_windows' not defined in config")
    if 'arrival_time_shift' not in config['sn_windows']:
        raise TypeError("'arrival_time_shift' not defined in config['sn_windows']")
    if 'signal_window' not in config['sn_windows']:
        raise TypeError("'signal_window' not defined in config['sn_windows']")

    if len(stream) != 1:
        raise ValueError(("Unable to get sn-windows: %d traces in stream "
                         "(possible gaps/overlaps)") % len(stream))

    a_time = UTCDateTime(a_time) + config['sn_windows']['arrival_time_shift']
    # Note above: UTCDateTime +float considers the latter in seconds
    # we use UTcDateTime for consistency as the package functions
    # work with that object type
    try:
        cum0, cum1 = config['sn_windows']['signal_window']
        t0, t1 = cumtimes(cumsum(stream[0]), cum0, cum1)
        nsy, sig = (a_time - (t1-t0), a_time), (t0, t1)
    except TypeError:  # not a tuple/list? then it's a scalar:
        shift = config['sn_windows']['signal_window']
        nsy, sig = (a_time-shift, a_time), (a_time, a_time+shift)
    # note: returns always tuples as they cannot be modified by the user (safer)
    return sig, nsy


def get_stream(segment, format="MSEED", headonly=False, **kwargs):  # @ReservedAssignment
    """Returns a Stream object relative to the given segment. The optional arguments are the same
    than `obspy.core.stream.read` (excepts than "format" defaults to "MSEED")
    :param segment: a model ORM instance representing a Segment (waveform data db row)
    :param format: string, optional (default "MSEED"). Format of the file to read. See obspy
        `Supported Formats`_ section below for a list of supported
        formats. If format is set to ``None`` it will be automatically detected which
        results in a slightly slower reading. If a format is specified, no
        further format checking is done.
    :param headonly: bool, optional (dafult: False). If set to ``True``, read only the data
        header. This is most useful for scanning available meta information of huge data sets
    :param kwargs: Additional keyword arguments passed to the underlying
        waveform reader method.
    """
    data = segment.data
    if not data:
        raise ValueError('no data')
    # Do not call `obspy.core.stream.read` because, when passed a BytesIO, if it fails reading
    # it stores the bytes data to a temporary file and re-tries by reading the file.
    # This is a useless and time-consuming behavior in our case: `data` is directly
    # downloaded from the data-center: if we fail we should raise immediately. To do that,
    # we call ``obspy.core.stream._read`, which is what `obspy.core.stream.read` does internally.
    # Note that: calling _read might require some attention as "private" methods might change
    # across versions. Also, FYI, the source function which does the real job is
    # "obspy.io.mseed.core._read_mseed"
    try:
        return _read(BytesIO(data), format, headonly, **kwargs)
    except Exception as terr:
        # As some exceptions break the processing of all remaining segments, wrap here errors
        # and raise a ValueError which breaks only current segment in case
        raise ValueError(str(terr))


def get_inventory(station, saveinventory=True, **urlread_kwargs):
    """Gets the inventory object for the given station, downloading it and saving it
    if not data is empty/None.
    Raises any utils.url.URLException for any url related errors, ValueError if inventory data
    is empty
    """
    data = station.inventory_xml
    if not data:
        data = download_inventory(station, **urlread_kwargs)
        # saving the inventory must NOT prevent the continuation.
        # If we are here we have non-empty data:
        if saveinventory and data:
            try:
                save_inventory(station, data)
            except Exception as _:
                pass  # FIXME: how to handle it?
    if not data:
        raise ValueError('no data')
    return loads_inv(data)  # convert from bytes into obspy object


def download_inventory(station, **urlread_kwargs):
    '''downloads the given inventory. Raises utils.url.URLException on error'''
    query_url = get_inventory_url(station)
    return urlread(query_url, **urlread_kwargs)[0]


def get_inventory_url(station):
    return urljoin(station.datacenter.station_url, station=station.station,
                   network=station.network, level='response')


def save_inventory(station, downloaded_bytes_data):
    """Saves the inventory. Raises SqlAlchemyError if station's session is None,
    or (most likely IntegrityError) on failure
    """
    station.inventory_xml = dumps_inv(downloaded_bytes_data)
    try:
        object_session(station).commit()
    except UnmappedInstanceError:
        raise
    except SQLAlchemyError:
        object_session(station).rollback()
        raise


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        super(LimitedSizeDict, self).__init__(*args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        super(LimitedSizeDict, self).__setitem__(key, value)
        self._check_size_limit()

    def update(self, *args, **kwargs):  # python2 compatibility (python3 calls __setitem__)
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, got %d" % len(args))
            other = dict(args[0])
            for key in other:
                super(LimitedSizeDict, self).__setitem__(key, other[key])
        for key in kwargs:
            super(LimitedSizeDict, self).__setitem__(key, kwargs[key])
        self._check_size_limit()

    def setdefault(self, key, value=None):  # python2 compatibility (python3 calls __setitem__)
        if key not in self:
            self[key] = value
        return self[key]

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self._popitem_size_limit()

    def _popitem_size_limit(self):
        return self.popitem(last=False)


class InventoryCache(LimitedSizeDict):

    def __init__(self, size_limit=30):
        super(InventoryCache, self).__init__(size_limit=size_limit)
        self._segid2staid = dict()

    def __setitem__(self, segment, inventory_or_exception):
        if inventory_or_exception is None:
            return
        super(InventoryCache, self).__setitem__(segment.station.id, inventory_or_exception)
        self._segid2staid[segment.id] = segment.station.id

    def __getitem__(self, segment_id):
        inventory = None
        staid = self._segid2staid.get(segment_id, None)
        if staid is not None:
            inventory = super(InventoryCache, self).get(staid, None)
            if inventory is None:  # expired, remove the key:
                self._segid2staid.pop(segment_id)
        return inventory
