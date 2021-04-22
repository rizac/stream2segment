from stream2segment.process.db.sqlevalexpr import exprquery
from stream2segment.process.main import process, s2smap
from stream2segment.process.db import get_session


def get_segments(dburl, conditions, orderby=None):
    return exprquery(get_session)


class SkipSegment(Exception):
    """Stream2segment exception indicating a segment processing error that should
    resume to the next segment without interrupting the whole routine
    """
    pass  # (we can also pass an exception in the __init__, superclass converts it)
