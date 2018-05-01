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
# from builtins import (ascii, chr, dict, filter, hex, input,
#                       int, map, next, oct, open, pow, range, round,
#                       super, zip)

from io import BytesIO
from contextlib import contextmanager
from itertools import chain, repeat

from sqlalchemy.orm.session import object_session
from sqlalchemy.orm.exc import UnmappedInstanceError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import load_only
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import _read

from stream2segment.io.utils import loads_inv, dumps_inv
from stream2segment.utils.url import urlread
from stream2segment.utils import urljoin
from stream2segment.io.db.models import Segment, Channel, Class
from stream2segment.process.math.traces import cumsumsq, cumtimes
from stream2segment.utils.inputargs import S2SArgument
from stream2segment.io.db.sqlevalexpr import exprquery


def getseg(session, segment_id, cols2load=None):
    '''Returns the segment identified by id `segment_id` by querying the session and,
    if not found, by querying the database
    :param cols2load: if the db has to be queried, specifies a list of columns to load. E.g.:
    `cols2load=[Segment.id]`
    '''
    seg = session.query(Segment).get(segment_id)
    if seg:
        return seg
    query = session.query(Segment).filter(Segment.id == segment_id)
    if cols2load:
        query = query.options(load_only(*cols2load))
    return query.first()


def raiseifreturnsexception(func):
    '''decorator that makes a function raise the returned exception, if any
    (otherwise no-op, and the function value is returned as it is)'''
    def wrapping(*args, **kwargs):
        '''wrapping function which raises if the returned value is an exception'''
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
        func._s2s_att = "gui.preprocess"  # pylint: disable=protected-access
        return func

    @staticmethod
    def customplot(func):  # DEPRECATED: backward compatibility
        '''decorator that adds the attribute func._s2s_att = "gui.customplot"'''
        func._s2s_att = "gui.customplot"  # pylint: disable=protected-access
        return gui.plot('b')(func)

    @staticmethod
    def sideplot(func):  # DEPRECATED: backward compatibility
        '''decorator that adds the attribute func._s2s_att = "gui.sideplot"'''
        return gui.plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})(func)

    @staticmethod
    def plot(*args, **kwargs):
        '''decorator that adds the attribute func._s2s_att = "gui.plot" and the given properties
        :param kwargs: 'position' ('b' for bottom, the default, or 'r' for right), 'xaxis', 'yaxis'
        (both dict of plotly axis properties, default: None, i.e. empty dict.
        For info on axis, see: https://plot.ly/python/axes/)
        '''
        position = kwargs.get('position', 'b')
        xaxis = kwargs.get('xaxis', None)
        yaxis = kwargs.get('yaxis', None)

        # Note: we want to allow @decorator, @decorator() and @decorator(position='b',...)
        # Solution along the lines of what found here:
        # https://stackoverflow.com/questions/3931627/how-to-build-a-decorator-with-optional-parameters
        # First define decorator wrapper:
        def decorator(func):
            '''sets the attributes on the function in order to make it recognizable as gui func'''
            func._s2s_att = 'gui.plot'  # pylint: disable=protected-access
            func._s2s_position = position  # pylint: disable=protected-access
            func._s2s_xaxis = xaxis or {}  # pylint: disable=protected-access
            func._s2s_yaxis = yaxis or {}  # pylint: disable=protected-access
            return func

        if len(args) == 1 and hasattr(args[0], '__call__') and not kwargs:
            # we called @gui.plot (with no arguments nor brackets)
            return decorator(args[0])

        # now we pay back: we have to parse args, as we might have called the
        # decorator with positional arguments...
        if len(args) > 3:
            raise SyntaxError('@gui.plot: 0 to 3 positional arguments expected, '
                              '%d received' % len(args))

        if len(args) >= 1:
            position = args[0]
        if len(args) >= 2:
            xaxis = args[1]
        if len(args) == 3:
            yaxis = args[2]

        return decorator

    @staticmethod
    def get_func_attrs(func):
        '''returns the function attributes for a function decorated with this class decorators:
        attname, position, xaxis, yaxis
        check for attname first: if empty string, the function is not a gui decorated function
        '''
        return getattr(func, '_s2s_att', ''), \
            getattr(func, '_s2s_position', 'b'), \
            getattr(func, '_s2s_xaxis', {}), \
            getattr(func, '_s2s_yaxis', {})


@contextmanager
def enhancesegmentclass(config_dict=None, overwrite_config=False):
    """contextmanager to be used in a with statement with a custom config dict.
    The contextmanager temporarily adds to a Segment obspy methods for processing.

    You can use this contextmanager in nested with statements, the methods are not added twice
    and will be removed after the last contextmanager (the first issued) will exit

    Usage:

    ```
    with enhancesegmentclass(config_dict):
        ... code here ...
        # now Segment class is enhanced anymore (class methods and attributes added):
        # segment.stream()
        # segment.siblings()
        # segment.inventory()
        # segment.sn_windows()  # if config_dict has the properly configured keys
    # now Segment class is not enhanced anymore (class methods and attributes removed)
    ```

    :param config_dict: the configuration dictionary. Usually from a yaml file
    :param overwrite_config: if True, the new config overrides the previously set
        `Segment._config` one, if any
    """

    already_enhanced = hasattr(Segment, "_config")
    if already_enhanced:
        if overwrite_config:
            Segment._config = config_dict or {}
        yield
    else:
        @raiseifreturnsexception
        def stream(self):
            '''returns the stream from self (a segment class)'''
            stream = getattr(self, "_stream", None)
            if stream is None:
                try:
                    stream = self._stream = get_stream(self)
                except Exception as exc:  # pylint: disable=broad-except
                    stream = self._stream = Exception("MiniSeed error: %s" %
                                                      (str(exc) or str(exc.__class__.__name__)))

            return stream

        @raiseifreturnsexception
        def inventory(self):
            '''returns the inventory from self (a segment class)'''
            inventory = getattr(self, "_inventory", None)
            if inventory is None:
                save_station_inventory = self._config.get('save_inventory', False)
                try:
                    inventory = self._inventory = get_inventory(self.station,
                                                                save_station_inventory)
                except Exception as exc:   # pylint: disable=broad-except
                    inventory = self._inventory = \
                        Exception("Station inventory (xml) error: %s" %
                                  (str(exc) or str(exc.__class__.__name__)))
            return inventory

        def sn_windows(self):
            '''returns the tuples (start, end), (start, end) where the first list is the signal
            window, and the second is the noise window. All elements are UtcDateTime's
            '''
            # No cache for this variable, as we might modify the segment stream in-place thus it's
            # hard to know when a recalculation is needed (this is particularly important when
            # bounds relative to the cumulative sum are given, if an interval was given there would
            # be no problem)
            return get_sn_windows(self._config, self.arrival_time, self.stream())

        def siblings(self, parent=None, colname=None):
            '''returns a query yielding the siblings of this segments according to `parent`
            Refer to the method Segment.get_siblings in models.py. Note that colname will not
            be exposed to the public thorug the processing templates help'''
            sblngs = self.get_siblings(parent, colname)
            conditions = self._config.get('segment_select', {})
            if conditions:
                sblngs = exprquery(sblngs, conditions, orderby=None, distinct=True)
            return sblngs

        Segment._config = config_dict or {}
        Segment.dbsession = lambda self: object_session(self)
        Segment.stream = stream
        Segment.inventory = inventory
        Segment.sn_windows = sn_windows
        Segment.siblings = siblings
        try:
            yield
        finally:
            # delete attached attributes:
            del Segment._config
            del Segment.stream
            del Segment.inventory
            del Segment.sn_windows
            del Segment.dbsession
            del Segment.siblings


def get_sn_windows(config, a_time, stream):
    '''Returns the spectra windows from a given arguments. Used by `_spectra`
    :return the tuple (start, end), (start, end) where all arguments are `UTCDateTime`s
    and the first tuple refers to the noisy window, the latter to the signal window
    '''
    # Use S2SArgument class to raise pre-formatted exception messages from our validation callbacks:
    name = 'sn_windows'
    snw_dic = S2SArgument(name).getfrom(config)
    atime_shift = S2SArgument('arrival_time_shift').getfrom(snw_dic, callback=float)

    def sw_callback(sn_windows):
        '''callback to parse sn_windows, which should be either a float or a iterable of two
        floats. Being called by S2SArgument, any exception will be caught and raised with a
        BadArgument exception properly formatted with the argument name provided and the
        exception
        '''
        try:
            cum0, cum1 = sn_windows
            if cum0 < 0 or cum0 > 1 or cum1 < 0 or cum1 > 1:
                raise ValueError('elements provided in signal window must be in [0, 1]')
            return float(cum0), float(cum1)
        except TypeError:  # not a tuple/list? then it's a scalar:
            return float(sn_windows)
    s_windows = S2SArgument('signal_window').getfrom(snw_dic, callback=sw_callback)

    if len(stream) != 1:
        raise ValueError(("Unable to get sn-windows: %d traces in stream "
                         "(possible gaps/overlaps)") % len(stream))

    a_time = UTCDateTime(a_time) + atime_shift
    # Note above: UTCDateTime +float considers the latter in seconds
    # we use UTcDateTime for consistency as the package functions
    # work with that object type
    if hasattr(s_windows, '__len__'):
        cum0, cum1 = s_windows
        trim_trace = stream[0].copy().trim(starttime=a_time)
        times = cumtimes(cumsumsq(trim_trace, normalize=False), cum0, cum1)
        nsy, sig = (a_time - (times[1]-times[0]), a_time), (times[0], times[1])
    else:
        nsy, sig = (a_time-s_windows, a_time), (a_time, a_time+s_windows)
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
    '''joins the attributes of `station` to build its inventory url (including the
    query arguments) and returns the url.
    :param station: a models.Station object
    '''
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


def set_classes(session, config, commit=True):
    '''Reads the 'class_labels' dict from the current configuration and stores the
    relative classes on the db, if such  a dict is found. The dict should be in the form
    `label:description`.
    Further addition to the config dict will add new classes based on their
    labels, or update existing classes description (if the labels is found on the db).
    Deletion is not possible from the config'''
    config_classes = config.get('class_labels', [])
    if not config_classes:
        return
    # do not add already added config_classes:
    needscommit = True  # flag telling if we need commit
    # seems odd but googling I could not find a better way to infer it from the session
    db_classes = {c.label: c for c in session.query(Class)}
    for label, description in config_classes.items():
        if label in db_classes and db_classes[label].description != description:
            db_classes[label].description = description  # update
            needscommit = True
        elif label not in db_classes:
            session.add(Class(label=label, description=description))
            needscommit = True
    if commit and needscommit:
        session.commit()


def get_slices(array, chunksize):
    '''Divides len(array)
    by `chunksize` yielding the array slices until exaustion.
    If `array` is an integer, it denotes the length of the array and the tuples (start, end)
    will be yielded.
    This method intelligently re-arranges the (start, end) indices in order to optimize the
    number of iterations yielded. ``
    '''
    if hasattr(array, '__len__'):
        total = len(array)  # == array.shape[0] in case of numpy arrays
    else:
        total = array
        array = None
    rem = total % chunksize
    quot = int(total / chunksize)
    if rem == 0:
        iterable = repeat(chunksize, quot)
    elif quot > rem:
        iterable = chain(repeat(chunksize+1, rem), repeat(chunksize, quot-rem))
    else:
        iterable = chain(repeat(chunksize, quot), [rem])
    start = end = 0
    for chunk in iterable:
        start = end
        end = start + chunk
        yield array[start:end] if array is not None else (start, end)
