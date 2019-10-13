'''
ORM for processing. Enhances Segment and Station class with several methods.

For functions called from within the processing rotuine (e.g., see the classmeth_*
functions) remember that raising ValueError will block only the currently processed segment,
all other exceptions will terminat ethe whole subroutine

Created on 26 Mar 2019

@author: riccardo
'''
from io import BytesIO

import numpy as np

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import _read
# from sqlalchemy import event

from stream2segment.io.utils import loads_inv
from stream2segment.utils import _get_session
from stream2segment.io.db.models import (Segment, Station, Base, object_session,
                                         Class, Event, Channel, DataCenter)
from stream2segment.io.db.sqlevalexpr import exprquery
from stream2segment.process.math.ndarrays import cumsumsq
from stream2segment.process.math.traces import timeof


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
        Segment.siblings = classmeth_siblings
    else:
        del Station.inventory
        del Station._inventory
        del Segment.stream
        del Segment._stream
        del Segment.inventory
        del Segment.dbsession
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
