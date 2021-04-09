from stream2segment.io.db import get_session as _get_session, DbNotFound, is_sqlite


def get_session(dburl, scoped=False, **engine_kwargs):
    """Returns an SqlAlchemy session object for downloading data"""
    try:
        sess = _get_session(dburl, scoped, check_db_existence=not is_sqlite(dburl),
                            **engine_kwargs)
    except DbNotFound as dbnf:
        raise ValueError('%s. Did you create the database first?' % str(dbnf))

    # Note: this creates the SCHEMA, not the database
    # the import below is in the function because slightly time consuming:
    from stream2segment.download.db.models import Base
    try:
        Base.metadata.create_all(sess.get_bind())
    except Exception as exc:
        raise ValueError('Error creating tables. Possible reason: tables created '
                         'with an older version or with a different program '
                         '(original error: %s)' % str(exc))
    return sess


