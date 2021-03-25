"""
Database management functions
"""
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError


from stream2segment.io import inputvalidation
from stream2segment.io.db import close_session
from stream2segment.io.db.models import get_classlabels
from stream2segment.download.db import Class, Download, Segment
from stream2segment.io.inputvalidation import validate_param, valid_session


def classlabels(dburl, *, add, rename, delete):
    """Configure the class labels of the database related to the database
    of the given session. Return a dict of class labels (mapped to their
    description) in the db after the operation

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
    session = inputvalidation.validate_param("dburl", dburl,
                                             inputvalidation.valid_session,
                                             for_process=False, scoped=False)
    configure_classlabels(session, add=add, rename=rename, delete=delete)
    return get_classlabels(session, Class, include_counts=False)


def configure_classlabels(session, *, add, rename, delete, commit=True):
    """Configure the class labels of the database related to the database
    of the given session. Lower level function than `classlabels`, accepts
    a `session` object and optional `commit` (to be performed later if needed)

    :param add: Class labels to add as a Dict[str, str]. The dict keys are
        the new class labels, the dict values are the label description
    :param rename: Class labels to rename as Dict[str, Sequence[str]]
        The dict keys are the old class labels, and the dict values are
        a 2-element sequence (e.g., list/tuple) denoting the new class label
        and the new description. The latter can be None (= do not modify
        the description, just change the label)
    :param delete: Class labels to delete, as Squence[str] denoting the class
        labels to delete
    :param commit: boolean (default True) whether to commit (save changes
        to the database). If True and the commit fails, the session is
        rolled back before raising
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


def ddrop(dburl, download_ids, confirm_func=input):
    """Drop data from the database by download id(s). Drops also all segments.

    :param confirm_func: a function accepting a single argument and that
        should return "y" to proceed, or any other value to stop. It defaults
        to the builtin `input` function, to be used from the command line interface
        Set to None to disable the confirmation (but do it at your own risk)

    :return: None if prompt is True and the user decided not to drop via user
        input, otherwise a dict of deleted download ids mapped to either:
        - an int (the number of segments deleted)
        - an exception (if the download id could not be deleted)
    """
    ret = {}
    session = validate_param('dburl', dburl, valid_session)
    try:
        ids = [_[0] for _ in
               session.query(Download.id).filter(Download.id.in_(download_ids))]
        if not ids:
            return ret
        if confirm_func is not None:
            segs = session.query(func.count(Segment.id)).\
                filter(Segment.download_id.in_(ids)).scalar()
            val = confirm_func('Do you want to delete %d download execution(s) '
                                '(id=%s) and the associated %d segment(s) from the '
                                'database [y|n]?' % (len(ids), str(ids), segs))
            if val.lower().strip() != 'y':
                return None

        for did in ids:
            ret[did] = session.query(func.count(Segment.id)).\
                filter(Segment.download_id == did).scalar()
            try:
                session.query(Download).filter(Download.id == did).delete()
                session.commit()
                # be sure about how many segments we deleted:
                ret[did] -= session.query(func.count(Segment.id)).\
                    filter(Segment.download_id == did).scalar()
            except Exception as exc:
                session.rollback()
                ret[did] = exc
        return ret
    finally:
        close_session(session, True)