from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine


class SessionScope(object):
    """
        Class handling sqlalchemy sessions. Initialize this object with an sqlalchemy engine
        and then 
        Call self.session() to get an sqlalchemy session, or use a with statement which handles
        commit and rollback when exiting:
        with self.session_scope() as session:
            ... do something with the session ...
        NOTE: the with statement sets also the returned session as class attribute
        self._open_session, so that subclasses might use it like this:

        def do_something(self, ...):
            if self._open_session:
    """
    def __init__(self, sql_alchemy_engine_or_dburl):
        if isinstance(sql_alchemy_engine_or_dburl, Engine):
            self.engine = sql_alchemy_engine_or_dburl
            self.db_uri = self.engine.engine.url
        else:
            self.db_uri = sql_alchemy_engine_or_dburl
            self.engine = create_engine(sql_alchemy_engine_or_dburl)
        self._open_session = None

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.session()
        self._open_session = session
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
            self._open_session = None

    def session(self):
        return Session(self.engine)
