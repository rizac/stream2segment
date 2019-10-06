'''
ORM for processing. Enhances Segment and Station class with several methods.

For functions called from within the processing rotuine (e.g., see the classmeth_*
functions) remember that raising ValueError will block only the currently processed segment,
all other exceptions will terminat ethe whole subroutine

Created on 26 Mar 2019

@author: riccardo
'''
from io import BytesIO

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import _read
# from sqlalchemy import event

from stream2segment.io.utils import loads_inv
from stream2segment.utils import _get_session
from stream2segment.io.db.models import (Segment, Station, Base, object_session,
                                         Class, Event, Channel, DataCenter)
from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.process.math.traces import cumsumsq, timeswhere


def get_session(dburl, scoped=False):
    '''Returns an SQLALchemy session object for **processing** downloaded Segments'''
    # We want to enhance the Segment class with new (processing-related) methods.
    # We might write a new class Segment2 extending 'models.Segment' (as usual) but SQLAlchemy
    # is not designed to handle conditional ORMs like this, and also this implementation
    # might introduce bugs due to the 'wrong' Segment class used.
    # We need thus to programmatically toggle on/off the new methods:
    # What we found here is maybe not the cleanest solution, but it's currently the only one
    # with no overhead: simply attach methods to the Segment class. Call manually
    # '_toggle_enhance_segment(False) to detach the methods when and if you need to.
    _toggle_enhance_segment(True)
    return _get_session(dburl, Base, scoped)


def _toggle_enhance_segment(value):
    if value == hasattr(Segment, '_stream'):
        return
    if value:
        Station.inventory = classmeth_inventory
        Segment.stream = classmeth_stream
        Segment.inventory = lambda self: self.station.inventory()
        Segment.dbsession = lambda self: object_session(self)  # pylint: disable=unnecessary-lambda
        Segment.sn_windows = classmeth_sn_windows
        Segment.siblings = classmeth_siblings
    else:
        del Station.inventory
        del Station._inventory
        del Segment.stream
        del Segment._stream
        del Segment.inventory
        del Segment.dbsession
        del Segment.sn_windows
        del Segment.siblings


def configure_classes(session, update_dict, commit=True):
    '''Configure the Classes of the database related to the given session

    :param update_dict: a dictionary of string (class labels) mapped to the class
    description
    '''
    if not update_dict:
        return
    # do not add already added config_classes:
    needscommit = True  # flag telling if we need commit
    # seems odd but googling I could not find a better way to infer it from the session
    db_classes = {c.label: c for c in session.query(Class)}
    for label, description in update_dict.items():
        if label in db_classes and db_classes[label].description != description:
            db_classes[label].description = description  # update
            needscommit = True
        elif label not in db_classes:
            session.add(Class(label=label, description=description))
            needscommit = True
    if commit and needscommit:
        session.commit()


def _raiseifreturnsexception(func):
    '''decorator that makes a function raise the returned exception, if any
    (otherwise no-op, and the function value is returned as it is)'''
    def wrapping(*args, **kwargs):
        '''wrapping function which raises if the returned value is an exception'''
        ret = func(*args, **kwargs)
        if isinstance(ret, Exception):
            raise ret
        return ret
    return wrapping


@_raiseifreturnsexception
def classmeth_inventory(self):
    '''returns the inventory from self (a segment class)'''
    # inventory is lazy loaded. The output of the loading process
    # (or the Exception raised, if any) is stored in the self._inventory attribute.
    # When querying the inventory a further time, the stored value is returned,
    # or raised (if it is an Exception)
    inventory = getattr(self, "_inventory", None)
    if inventory is None:
        try:
            inventory = self._inventory = get_inventory(self)
        except Exception as exc:   # pylint: disable=broad-except
            inventory = self._inventory = \
                ValueError("Station inventory (xml) error: %s" %
                           (str(exc) or str(exc.__class__.__name__)))
    return inventory


def get_inventory(station):
    """Returns the inventory object for the given station.
    Raises ValueError if inventory data is empty
    """
    data = station.inventory_xml
    if not data:
        raise ValueError('no data')
    return loads_inv(data)


@_raiseifreturnsexception
def classmeth_stream(self):
    '''returns the stream from self (a segment class)'''
    # stream is lazy loaded. The output of the loading process
    # (or the Exception raised, if any) is stored in the self._stream attribute.
    # When querying the stream a further time, the stored value is returned,
    # or raised (if it is an Exception)
    stream = getattr(self, "_stream", None)
    if stream is None:
        try:
            stream = self._stream = get_stream(self)
        except Exception as exc:  # pylint: disable=broad-except
            stream = self._stream = \
                ValueError("MiniSeed error: %s" %
                           (str(exc) or str(exc.__class__.__name__)))

    return stream


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


def classmeth_sn_windows(self, win_length, atime_shift=0):
    '''Computes time windows from the arguments.

    :param win_length: float or 2-element tuple. If float, it is the window length,
        in seconds. If two element tuple, denotes the start and end time of the window
        length, relative to the segment waveform cumulative sum of the signal after the
        arrival time. Thus if win_len = [0.05, 0.95], the segment waveform will be cut
        and the cumulative sum C of the waveform after arrival time will be calculated.
        The time where C reaches its 95% minus the time where C reaches its %% will
        denote the window length

    :return the tuple (start, end), (start, end) where all arguments are `UTCDateTime`s
        and the first tuple refers to the noisy window, the latter to the signal window
    '''
    s_windows = parse_sn_windows(win_length)
    stream_ = self.stream()

    if len(stream_) != 1:
        raise ValueError(("Unable to get sn-windows: %d traces in stream "
                          "(possible gaps/overlaps)") % len(stream_))

    a_time = UTCDateTime(self.arrival_time) + atime_shift
    # Note above: UTCDateTime +float considers the latter in seconds
    # we use UTcDateTime for consistency as the package functions
    # work with that object type
    if hasattr(s_windows, '__len__'):
        cum0, cum1 = s_windows
        trim_trace = stream_[0].copy().trim(starttime=a_time)
        times = timeswhere(cumsumsq(trim_trace, copy=False, normalize=True), cum0, cum1)
        nsy, sig = (a_time - (times[1]-times[0]), a_time), (times[0], times[1])
    else:
        nsy, sig = (a_time-s_windows, a_time), (a_time, a_time+s_windows)
    # note: returns always tuples as they cannot be modified by the user (safer)
    return sig, nsy


def parse_sn_windows(window):
    '''Returns the argument parsed to float (or with all its elements parsed to float).

    :param window: either a string float or a iterable of two float elements
        (by floats we mean also float parsable strings)
    '''
    try:
        try:
            cum0, cum1 = window
            if cum0 < 0 or cum0 > 1 or cum1 < 0 or cum1 > 1:
                raise ValueError('elements must be both in [0, 1]')
            return float(cum0), float(cum1)
        except TypeError:  # not a tuple/list? then it's a scalar:
            return float(window)
    except Exception as exc:
        raise Exception('S/N Window error: %s' % str(exc))


def classmeth_siblings(self, parent=None, conditions=None):
    '''returns a SQLAlchemy query yielding the siblings of this segments according to `parent`
    Refer to the method Segment.get_siblings in :module:`models.py`.

    :parent: a string identifying the parent whereby perform a selection
    :conditions: a dict of strings mapped to string expressions to be evaluated, and select
        a subset of siblings. None (the defaults) means: empty dict (no additional slection
        condition)
    '''
    sblngs = self.get_siblings(parent, colname=None)  # returns a Segment object
    if conditions:
        sblngs = exprquery(sblngs, conditions, orderby=None)
    return sblngs
