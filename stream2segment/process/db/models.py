"""
s2s process database ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

import os
from datetime import datetime
from io import BytesIO, StringIO

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.session import object_session
from sqlalchemy.sql.expression import func
from obspy.core.event import read_events, Catalog
from obspy.core.stream import Stream, _read  # noqa
from obspy.core.inventory.inventory import read_inventory, Inventory

from stream2segment.io.db import models


class SkipSegment(Exception):
    """Stream2segment exception indicating a segment processing error that should
    resume to the next segment without interrupting the whole routine
    """
    pass  # (we can also pass an exception in the __init__, superclass converts it)


# setup Event enhanced features:

def get_catalog(event: models.Event, format_="QUAKEML", **kwargs) -> Catalog:
    """Return the Catalog object for the given event. The Obspy event
    object is accessible as catalog[0], or catalog.events[0]
    """
    data = event.data
    if not data:
        raise SkipSegment('no data')
    try:
        return read_events(BytesIO(data), format=format_, **kwargs)
    except Exception as exc:
        raise SkipSegment("Catalog (QuakeML) error: %s" %
                          (str(exc) or str(exc.__class__.__name__)))


models.Event.catalog = property(get_catalog)


# setup Station enhanced features:

def get_inventory(segment: models.Segment, format_="QUAKEML", **kwargs) -> Inventory:
    """Return the inventory object for the given station.
    Raises :class:`SkipSegment` if inventory data is empty
    """
    data = segment.stationxml_data
    if not data:
        raise SkipSegment('no data')
    try:
        return read_inventory(BytesIO(data), format="STATIONXML")
    except Exception as exc:
        raise SkipSegment("Inventory (StationXML) error: %s" %
                          (str(exc) or str(exc.__class__.__name__)))


models.Segment.inventory = property(get_inventory)


def inventory2xml(inventory: Inventory, validate=False) -> bytes:
    """Return the given Inventory as `bytes` in StationXML format"""
    buffer = BytesIO()
    inventory.write(buffer, format='stationxml', validate=validate)
    return buffer.getvalue()


def inventory2text(inventory: Inventory, validate=False) -> str:
    """ Return the given Inventory as `str` in text format (equivalent to
    FDSN format=text)"""
    buffer = StringIO()
    inventory.write(buffer, format='stationtxt', validate=validate)
    return buffer.getvalue()


# setup Segment enhanced features:

def get_stream(segment: models.Segment,  format="MSEED", headonly=False, **kwargs) -> Stream:  # noqa
    """Return the stream object for the given segment.
    Raises :class:`SkipSegment` if inventory data is empty
    """
    data = segment.miniseed_data
    if not data:
        raise SkipSegment('no data')
    try:
        return _read(BytesIO(data), format, headonly, **kwargs)
    except Exception as exc:
        raise SkipSegment("Stream error: %s" %
                          (str(exc) or str(exc.__class__.__name__)))


models.Segment.stream = property(get_stream)


def sds_path(segment: models.Segment, root='.'):
    """Return a string representing the SeisComP data structure
    which can be used as path to store the segment miniSEED:

    `root/EID/Year/NET/STA/CHAN.D/NET.STA.LOC.CHAN.TYPE.YEAR.DAY`

    where `root` is the optional argument, EID is the database unique id of
    the event (integer), and all other fields are defined here:
    https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html

    :param segment: the Segment instance
    :param root: Optional (defaults to '.' when missing). The root path of this
        segment file (first argument of `os.path.join`)
    """
    # year > N > S > L > C.D > segments > N.S.L.C.year.day.event_id.mseed
    seg_dtime = segment.start_time  # note that start_time might be None
    year = seg_dtime.year
    net, sta = segment.channel.network, segment.channel.station
    loc, cha = segment.channel.location, segment.channel.channel
    # day is in [1, 366], padded with zeroes:
    day = '%03d' % ((seg_dtime - datetime(year, 1, 1)).days + 1)
    eid = segment.event_id
    typ = 'D'
    return os.path.join(root,
                        str(eid), str(year), net, sta, loc, cha + "." + typ,
                        '.'.join((net, sta, loc, cha, typ, str(year), day)))


models.Segment.sds_path = property(sds_path)


def dbsession(self):
    """Return the database session to which this object is attached. Use with care:
    the session is for advanced users who need full freedom to interact with
    the database.
    For an introduction, see: https://docs.sqlalchemy.org/en/latest/orm/session.html
    """
    return object_session(self)


models.Segment.dbsession = property(dbsession)


def label(
        segment: Segment,  # noqa
        *ids_or_labels,
        session=None,
        commit=True,
        remove=False,
        annotator=None
):
    """Add / removes class label(s) to this segment

    :param ids_or_labels: variable-length argument of the unique IDs
        (int) or labels (str) of the class labels to be added/removed to this segment.
        Ids or labels already set (if remove=False) or unset (remove=True) are safely
        ignored.
        WARNING: an empty `ids_or_labels` with remove=True will REMOVE ALL CLASS LABELS
    :param session: the db session. If None, gets the current segment session
        (raises SQLAlchemyException if this segment is detached from a session)
    :param commit: boolean (default: True) denoting if any change
        should be saved to the database (flush pending changes and commit
        the current transaction).
        Advanced users can set this parameter to False to manage the
        transaction manually and eventually call `segment.dbsession.commit()`
        when needed
    :param annotator: (str, default: None). The annotator assigning the labelling.
        Ignored if remove=True and for all labels that are already added to this segment.
        A None annotator should mean that the label assignment is the result of a
        classifier prediction and not human inspection: providing an
        annotator (not None) will set the `is_hand_labelled` property of the Class
        labelling to True
    :param remove: boolean (default False) telling if the supplied ids and labels should
        be removed (default: False, add them).
        WARNING: an empty `ids_or_labels` with remove=True will REMOVE ALL CLASS LABELS
    :raise: :class:`sqlalchemy.exc.SQLAlchemyError` if a commit error occurs.
        For info see: https://docs.sqlalchemy.org/en/latest/orm/session_basics.html
    """
    if not session:
        session = segment.dbsession
    if not session:
        raise SQLAlchemyError(
            'No session specified and Segment is detached'
        )
    ClassLabel = models.ClassLabel  # noqa
    needs_commit = False
    if ids_or_labels or remove:
        label_ids = {v for v in ids_or_labels if isinstance(v, int)}
        label_names = [v for v in ids_or_labels if isinstance(v, str)]
        if label_names:
            label_ids |= set(session.query(ClassLabel.id).filter(
                ClassLabel.label.in_(label_names)))

        if not label_ids and remove:
            label_ids = set(session.query(ClassLabel.id))  # get all ids

        if not label_ids:
            return needs_commit

        class_labels = segment.classlabels
        if remove:
            class_labels.filter(ClassLabel.id.in_(label_ids)). \
                delete(synchronize_session='fetch')
            needs_commit = True
        else:
            label_ids -= set(_.id for _ in segment.classlabels)
            if not label_ids:
                return needs_commit
            for c_lbl in session.query(ClassLabel).filter(ClassLabel.id.in_(label_ids)):
                class_labels.append(c_lbl)
                needs_commit = True

            if annotator:
                ClassLabeling = models.ClassLabeling  # noqa
                for c_lbling in session.query(ClassLabeling).filter(
                        ClassLabeling.class_label_id.in_(label_ids) &
                        (ClassLabeling.segment_id == segment.id)
                ):
                    c_lbling.annotator = annotator
                    needs_commit = True

    if needs_commit and commit:
        session.commit()
        needs_commit = False

    return needs_commit


models.Segment.label = label


def get_classlabels(session, segments_count=False):
    """Return a list of class labels in a dict form:
    ```
    {
     'id': int
     'label': str,
     'description': str
     'segments': int (number of segments labelled with this object, or 0)
    }
    ```
    :param session: the database session. See `get_session` for info
    :param segments_count: bool. If true, each class label dict will
        also provide in the 'segments' key the number of segments labelled
        with the given class label. When false (the default) no counting is
        performed (it might be time consuming) the 'segments' key
        value is 0
    """
    ClassLabel = models.ClassLabel  # noqa
    ClassLabeling = models.ClassLabeling  # noqa
    colnames = [
        ClassLabel.id.key, "label", ClassLabel.description.key, 'segments'
    ]

    if not segments_count:
        return [
            {
                colnames[0]: c.id,
                colnames[1]: c.label,
                colnames[2]: c.description,
                colnames[3]: 0
            } for c in session.query(ClassLabel)
        ]

    # compose the query step by step:
    query = session.query(
        ClassLabel.id, ClassLabel.label, ClassLabel.description,
        func.count(ClassLabeling.id).label(colnames[-1])
    )
    # Join class labellings to get how many segments per class:
    # Note: `isouter` below, which produces a left outer join, is important
    # when we have no class labellings (i.e. third column all zeros) otherwise
    # with a normal join we would have no results
    query = query.join(
        ClassLabeling, ClassLabeling.class_id == ClassLabel.id, isouter=True
    )
    # group by class id:
    query = query.group_by(ClassLabel.id).order_by(ClassLabel.id)
    return [{name: val for name, val in zip(colnames, d)} for d in query]

