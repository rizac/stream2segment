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

# from sqlalchemy import event
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from obspy.core.stream import _read

from stream2segment.io.utils import loads_inv
from stream2segment.io.db import get_session as _get_session
from stream2segment.io.db.models import (Segment, Station, Base, object_session,
                                         Class, ClassLabelling, Download,
                                         Event, Channel, DataCenter,
                                         WebService)
from stream2segment.io.db.sqlevalexpr import exprquery


def get_session(dburl, scoped=False, **engine_args):
    """
    Create and returns an sql alchemy session for IO db operations aiming to
    **process downloaded data**

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param scoped: boolean (False by default) if the session must be scoped session
    :param engine_args: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dburl, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
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
    return _get_session(dburl, Base, scoped, create_all=False, **engine_args)


def _toggle_enhance_segment(value):
    if value == hasattr(Segment, 'dbsession'):
        return

    if value:
        Station.inventory = classmeth_inventory
        Segment.stream = classmeth_stream
        Segment.inventory = lambda self, *a, **kw: self.station.inventory(*a, **kw)
        Segment.dbsession = lambda self: object_session(self)  # noqa
        Segment.siblings = classmeth_siblings
    else:
        del Station.inventory
        # del Station._inventory
        del Segment.stream
        # del Segment._stream
        del Segment.inventory
        del Segment.dbsession
        del Segment.siblings


def configure_classes(session, *, add, rename, delete, commit=True):
    """Configure the Classes of the database related to the given session

    :param add: Class labels to add as a Dict[str, str]. The dict keys are
        the new class labels, the dict values are the label description
    :param rename: Class labels to rename as Dict[str, Sequence[str]]
        The dict keys are the old class labels, and the dict values are
        a 2-element sequence (e.g., list/tuple) denoting the new class label
        and the new description. The latter can be None (= do not modify
        the description, just change the label)
    :param delete: Class labels to delete, as Squence[str] denoting the class
        labels to delete
    """
    db_classes = {c.label: c for c in session.query(Class)}
    if add:
        for label, description in add.items():
            if label in db_classes:  # unique constraint
                continue
            class_label = Class(label=label, description=description)
            session.add(class_label)
            db_classes[label] = class_label

    if rename:
        for label, (new_label, new_description) in rename.items():
            if label not in db_classes:  # unique constraint
                continue
            db_classes[label].label = new_label
            if new_description is not None:
                db_classes[label].description = new_description

    if delete:
        for label in delete:
            if label in db_classes:
                session.delete(db_classes[label])

    if commit:
        try:
            session.commit()
        except SQLAlchemyError as sqlerr:
            session.rollback()
            raise


def get_classes(session, include_counts=True):
    """Return a list of classes on the database of the given `session`. Each
    class is returned as dict with keys 'id', 'label' and 'description':
    ```
    [
        ... ,
        {
         'id': int
         'label': str,
         'description': str
         'count': int (number of segments labelled with this label)
        },
        ...
    ]
    ```
    :param include_counts: boolean (True by default). Whether to include
        the 'count' in each dict. Set to False if you don;t need the information
         as the function might be faster
    """
    if not include_counts:
        return [
            {'id': c.id, 'label': c.label, 'description': c.description}
            for c in session.query(Class)
        ]

    colnames = [Class.id.key, Class.label.key, Class.description.key, 'count']
    # compose the query step by step:
    query = session.query(Class.id, Class.label, Class.description,
                          func.count(ClassLabelling.id).label(colnames[-1]))
    # Join class labellings to get how many segments per class:
    # Note: `isouter` below, which produces a left outer join, is important
    # when we have no class labellings (i.e. third column all zeros) otherwise
    # with a normal join we would have no results
    query = query.join(ClassLabelling,
                       ClassLabelling.class_id == Class.id, isouter=True)
    # group by class id:
    query = query.group_by(Class.id).order_by(Class.id)
    return [{name: val for name, val in zip(colnames, d)} for d in query]





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
