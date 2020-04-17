'''
Object Relationa Models (ORM) for processing.

**This is the module to use FOR IMPORTING database stuff when processing
downloaded segments**. Example:

```
from stream2segment.process.db import get_session, Station, Segment

sess = get_session(dburl)

sess.query(Segment.id).join(Segment.station).filter(...).all()
```

Created on 26 Mar 2019

@author: riccardo
'''

# IMPORTANT DEVELOPER NOTES:
# For any of those custom methods (e.g., see the classmeth_* functions) remember,
# when raising, that from the stream2segment processing routine,
# ValueError will block only the currently processed segment,
# all other exceptions will terminate the whole routine

from io import BytesIO

from obspy.core.stream import _read
# from sqlalchemy import event

from stream2segment.io.utils import loads_inv
from stream2segment.utils import _get_session
from stream2segment.io.db.models import (Segment, Station, Base, object_session,
                                         Class, ClassLabelling, Download,
                                         Event, Channel, DataCenter,
                                         WebService)
from stream2segment.io.db.sqlevalexpr import exprquery


def get_session(dburl, scoped=False, **engine_args):
    """
    Create and returns an sql alchemy session for IO db operations aiming to
    **process downloaded**

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param scoped: boolean (False by default) if the session must be scoped session
    :param engine_args: optional keyword argument values for the
        `create_engine` method. E.g.:
        ```
        _get_session(dbpath, db_url, connect_args={'connect_timeout': 10})
        ```
        other options are e.g., encoding='latin1', echo=True etcetera
    """
    # We want to enhance the Segment class with new (processing-related) obspy
    # methods.
    # Solution 1: we might do it already in the io.db.models module, but there we do not
    # want to keep things clean and avoid importing anything so that the module
    # can be portable (in case) to new projects.
    # Solution 2: we might then write here a new class Segment2 extending
    # 'models.Segment' but SQLAlchemy
    # is not designed to handle conditional ORMs like this, and also this implementation
    # might introduce bugs due to the 'wrong' Segment class used.
    # We need thus to programmatically toggle on/off the new methods:
    # What we found here is maybe not the cleanest solution, but it's currently the only one
    # with no overhead: simply attach methods to the Segment class. Call manually
    # '_toggle_enhance_segment(False) to detach the methods when and if you need to
    # (it should never be the case)
    _toggle_enhance_segment(True)
    return _get_session(dburl, Base, scoped, **engine_args)


def _toggle_enhance_segment(value):
    if value == hasattr(Segment, 'dbsession'):
        return

    if value:
        Station.inventory = classmeth_inventory
        Segment.stream = classmeth_stream
        Segment.inventory = lambda self, *a, **kw: self.station.inventory(*a, **kw)
        Segment.dbsession = lambda self: object_session(self)  # pylint: disable=unnecessary-lambda
        Segment.siblings = classmeth_siblings
    else:
        del Station.inventory
        # del Station._inventory
        del Segment.stream
        # del Segment._stream
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
def classmeth_inventory(self, reload=False):
    '''returns the inventory from self (a segment class)'''
    # inventory is lazy loaded. The output of the loading process
    # (or the Exception raised, if any) is stored in the self._inventory attribute.
    # When querying the inventory a further time, the stored value is returned,
    # or raised (if it is an Exception)
    inventory = getattr(self, "_inventory", None)
    if reload and inventory is not None:
        inventory = None
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
def classmeth_stream(self, reload=False):
    '''returns the stream from self (a segment class)'''
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


def classmeth_siblings(self, parent=None, conditions=None, colname=None):
    '''returns a SQLAlchemy query yielding the siblings of this segments according to `parent`
    Refer to the method Segment.get_siblings in :module:`models.py`.

    :parent: a string identifying the parent whereby perform a selection
    :conditions: a dict of strings mapped to string expressions to be evaluated, and select
        a subset of siblings. None (the defaults) means: empty dict (no additional slection
        condition)
    '''
    sblngs = self.get_siblings(parent, colname=colname)  # returns a Segment object
    if conditions:
        sblngs = exprquery(sblngs, conditions, orderby=None)
    return sblngs
