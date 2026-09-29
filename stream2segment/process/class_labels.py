"""class labels utilities"""
import os
import numbers

from sqlalchemy import Engine, insert, select, delete
from sqlalchemy.exc import IntegrityError

from stream2segment.io.db.pdsql import get_col_max
from stream2segment.process import SegmentMetadata
from stream2segment.process.segments_selection import is_legacy_db


def add_class_label(db: Engine, label: str, description: str = "") -> bool:
    """Add a new class label"""
    return _edit_class_label(db, label, description or "")


def delete_class_label(db: Engine, label_or_id: str | int) -> bool:
    """Delete a new class label. WARNING: use with caution: this might also delete
    all associated class labeling, losing information of which segment was associated
    to the deleted class label
    """
    return _edit_class_label(db, label_or_id, None)


def _edit_class_label(
    db: Engine, label: str, description_or_none_to_delete: str | None = ""
) -> bool:
    """Add or remove a new class label"""
    if is_legacy_db(db):
        from stream2segment.io.db.legacy.models import ClassLabel
    else:
        from stream2segment.io.db.models import ClassLabel

    class_id = None
    if description_or_none_to_delete is not None:  # adding class
        class_id = get_col_max(db, ClassLabel.id) + 1

    with db.begin() as conn:  # noqa
        try:
            with conn.begin_nested():
                if class_id is not None:  # adding class
                    conn.execute(
                        insert(ClassLabel).values(
                            id=class_id,
                            label=label,
                            description=str(description_or_none_to_delete)
                        )
                    )
                elif isinstance(label, numbers.Integral):
                    conn.execute(
                        delete(ClassLabel).where(
                            ClassLabel.id == label,
                        )
                    )
                else:
                    conn.execute(
                        delete(ClassLabel).where(
                            ClassLabel.label == label,
                        )
                    )
                return True
        except IntegrityError:
            pass

        # return if the class is set:
        if isinstance(label, numbers.Integral):
            stmt = select(ClassLabel.id).where((ClassLabel.id == label))
        else:
            stmt = select(ClassLabel.id).where((ClassLabel.label == label))
        return (conn.execute(stmt).scalar_one_or_none() is None) == class_id is None


def get_class_labels(
    db: Engine, segment: SegmentMetadata | int | None
) -> dict[int, str]:
    """Return a dict of class label ids mapped to the relative label. If `segment` is
    None, return all classes stored in the db. Otherwise, returns only classes labels
    added to the given segment
    """
    if is_legacy_db(db):
        from stream2segment.io.db.legacy.models import ClassLabel, ClassLabeling
    else:
        from stream2segment.io.db.models import ClassLabel, ClassLabeling

    stmt = select(ClassLabel.id, ClassLabel.label)
    if segment is not None:
        stmt = stmt.join(ClassLabeling, ClassLabeling.class_label_id == ClassLabel.id)
        stmt = stmt.where(ClassLabeling.segment_id == getattr(segment, 'id', segment))

    with db.connect() as conn:
        return {_[0]: _[1] for _ in conn.execute(stmt).fetchall()}


def set_class_labeling(
    db: Engine,
    segment: SegmentMetadata | int,
    class_label_or_id: int | str,
    annotator: str = "_auto_"
) -> bool:
    """Add a class labeling to the given segment

    :return: True if the given class labeling was added, False otherwise
    """
    if annotator == '_auto_':
        try:
            import getpass
            annotator = str(getpass.getuser())
            if len(annotator) > 2:
                annotator = annotator[0] + '*' * (len(annotator) - 2) + annotator[-1]
            else:
                annotator = 'user id ' + str(os.getuid())
        except Exception:  # noqa
            annotator = ''

    return _edit_class_labeling(db, segment, class_label_or_id, annotator=annotator)


def delete_class_labeling(
    db: Engine, segment: SegmentMetadata | int, class_label_or_id: int | str
) -> bool:
    """Delete the given class labeling associated to the given segment.

    :return: True if the given class labeling was deleted, False otherwise
    """
    return _edit_class_labeling(db, segment, class_label_or_id)


def _edit_class_labeling(
    db: Engine, segment: SegmentMetadata | int, class_label_or_id: int | str, **kwargs
) -> bool:
    """Add or delete the given class to the given segment class labeling"""
    if is_legacy_db(db):
        from stream2segment.io.db.legacy.models import ClassLabeling, ClassLabel
    else:
        from stream2segment.io.db.models import ClassLabeling, ClassLabel

    class_id = class_label_or_id
    if isinstance(class_label_or_id, str):
        stmt = select(ClassLabel.id).where(ClassLabel.label == class_label_or_id)
        with db.connect() as conn:
            class_id = conn.execute(stmt).scalar_one_or_none()
        if class_id is None:
            raise ValueError(f"No class label associated with id {class_label_or_id}")

    seg_id = getattr(segment, 'id', segment)

    add = len(kwargs) > 0
    with db.begin() as conn:  # noqa
        try:
            with conn.begin_nested():
                if add:
                    conn.execute(
                        insert(ClassLabeling).values(
                            segment_id=seg_id,
                            class_label_id=class_id,
                            **kwargs  # e.g. annotator=...
                        )
                    )
                else:
                    conn.execute(
                        delete(ClassLabeling).where(
                            ClassLabeling.segment_id == seg_id,
                            ClassLabeling.class_label_id == class_id,
                        )
                    )
                return True
        except IntegrityError:
            pass

        # we got integrity error, return whether the element is aded / deleted:
        stmt = select(ClassLabeling.id).where(
            (ClassLabeling.class_label_id == class_id) &
            (ClassLabeling.segment_id == seg_id)
        )

        # add: element must NOT be None. Not add (del): element must be None. So:
        return (conn.execute(stmt).scalar_one_or_none() is not None) == add
