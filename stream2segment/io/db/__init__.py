def get_session(dbpath, base=None, scoped=False, create_all=True, **engine_args):
    """
    Create an SQLAlchemy session for IO database operations

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param base: a declarative base. If None, defaults to the default declarative base
        used in this package for downloading
    :param scoped: boolean (False by default) if the session must be scoped session
    :param create_all: If True (the default), this method will issue queries that
        first check for the existence of each individual table, and if not found
        will issue the CREATE statements
    :param engine_args: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
    """
    # import stuff here in case they are time consuming, to avoid
    # taking time to load other module stuff
    from sqlalchemy.orm.scoping import scoped_session
    from sqlalchemy.engine import create_engine
    from sqlalchemy.orm.session import sessionmaker
    from stream2segment.io.db.models import Base

    if base is None:
        base = Base  # default declarative base, withour obspy methods

    # init the session:
    engine = create_engine(dbpath, **engine_args)
    if create_all:
        base.metadata.create_all(engine)  # @UndefinedVariable

    if not scoped:
        # create a configured "Session" class
        session = sessionmaker(bind=engine)
        # create a Session
        return session()

    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)


