from stream2segment.utils import _get_session
from stream2segment.io.db.models import Base

def get_session(dburl, scoped=False):
    return _get_session(dburl, Base, scoped)
