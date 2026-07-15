# from stream2segment.process.segments_selection import build_where_clause
from stream2segment.process.main import (
    process_segments, map_segments, get_segments, SkipSegment,
    get_default_segments_selection, SegmentMetadata, suppress_printouts
)
# from stream2segment.process import traces


# def get_classlabels(db):
#     """Yields the Python objects representing each class label stored on the given db.
#     The object main attributes are `label`, `description`, '`id` and `segments`, which
#     can be used to yield the Segments assigned to the given class label.
#
#     :param db: the database URL, as string, or a `session` object already created from
#         an given URL (see :func:`get_session`). URLs must be given in this format:
#         https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls.
#         NOTE: if `db` is a string, a db session is opened and closed just before this
#         function returns: afterwards, attributes returning related db objects (e.g.,
#         `segments`) might not be accessible anymore
#     """
#     sess = get_session(db) if isinstance(db, str) else db
#     try:
#         yield from sess.query(Class)
#     finally:
#         if sess is not db:  # we created the session here, close it before returning
#             sess.close()

