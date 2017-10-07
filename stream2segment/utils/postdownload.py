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
from sqlalchemy.exc import SQLAlchemyError
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import _read

from stream2segment.io.utils import loads_inv, dumps_inv
from stream2segment.utils.url import urlread
from stream2segment.utils import urljoin
from stream2segment.io.db.models import Segment, Station
from stream2segment.analysis.mseeds import cumsum, cumtimes
from sqlalchemy.orm.exc import UnmappedInstanceError


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
    "decorators for the processing.py file"

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


class SegmentWrapper(object):

    def __init__(self, config={}):
        self.__config = config

    def reinit(self, session, segment_id, stream=None, inventory=None,
               **kwargs):
        # Mandatory attributes:
        self.__session = session
        self.__segment_id = segment_id
        self.__segment = None
        self.__stream = stream
        self.__inv = inventory
        self.__sn_windows = None
        # Custom attributes. If they are used or not inside the methods, is up to the
        # implementation (for the moment, none of them is used unless it overrides one of the
        # previous mandatory attributes):
        for name, value in viewitems(kwargs):
            setattr(self, "__%s" % name, value)
        return self

    @property
    @raiseifreturnsexception
    def __getseg(self):
        if self.__segment is None:
            try:
                self.__segment = self.__session.query(Segment).\
                    filter(Segment.id == self.__segment_id).one()
            except Exception as exc:
                self.__segment = exc
        return self.__segment

    @raiseifreturnsexception
    def stream(self):
        if self.__stream is None:
            try:
                self.__stream = get_stream(self)
            except Exception as exc:
                self.__stream = Exception("MiniSeed error: %s" %
                                          (str(exc) or str(exc.__class__.__name__)))

        return self.__stream

    @raiseifreturnsexception
    def inventory(self):
        if self.__inv is None:
            try:
                save_station_inventory = self.__config['save_inventory']
            except:
                save_station_inventory = False
            try:
                self.__inv = get_inventory(self.station, save_station_inventory)
            except Exception as exc:
                self.__inv = Exception("Station inventory (xml) error: %s" %
                                       (str(exc) or str(exc.__class__.__name__)))
        return self.__inv

    @raiseifreturnsexception
    def timewindow(self, wtype):
        '''returns [start, end] as UtcDateTime's representing the signal or noise window
        :param wtype: either 'signal', 's', 'noise' or 'n'
        '''
        if wtype in ('signal', 'noise'):
            wtype = wtype[0]
        wtype = wtype.lower()
        if wtype not in ('s', 'n'):
            return TypeError(("segment.timewindow() got wrong argument '%s'. "
                              "Please provide 'signal', 's', 'noise' or 'n'") % str(wtype))

        if self.__sn_windows is None:
            try:
                self.__sn_windows = get_sn_windows(self.__config, self.arrival_time, self.stream())
            except Exception as exc:
                self._sn__windows = [exc, exc]  # hack to return an exception regardless of wtype
        return self.__sn_windows[1 if wtype == 'n' else 0]

    def __getattr__(self, name):
        return getattr(self._SegmentWrapper__getseg, name)


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
        nsy, sig = [a_time - (t1-t0), a_time], [t0, t1]
    except TypeError:  # not a tuple/list? then it's a scalar:
        shift = config['sn_windows']['signal_window']
        nsy, sig = [a_time-shift, a_time], [a_time, a_time+shift]
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
    except TypeError as terr:
        # as type errors should generally be raised in case of errors (see obspy source code),
        # and type errors break the processing of all remaining segments,
        # we do not want this behaviour. Thus raise a ValueError with the same message
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
