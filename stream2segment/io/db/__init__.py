from contextlib import contextmanager

from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker


def get_session(dbpath, scoped=False, **engine_args):
    """Create an SQLAlchemy session for IO database operations

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param scoped: boolean (False by default) if the session must be scoped session
    :param engine_args: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
    """
    engine = create_engine(dbpath, **engine_args)
    session_factory = sessionmaker(bind=engine)

    if not scoped:
        # create a Session
        return session_factory()

    return scoped_session(session_factory)


def database_exists(url_or_engine):
    """Return iuf the database exists. Works for Postgres, MySQL, SQLite.

    :param url_or_engine: SQLAlchemy engine or string denoting a database URL.
    """
    # Check if database exist, works for mysql, postgres, sqlite.
    # Info here: https://stackoverflow.com/a/3670000
    # (We could use the package sqlalchemy-utils, but we decided
    # not to import it to avoid overhead for just one function)
    text = 'SELECT 1'
    with _engine(url_or_engine) as engine:
        try:
            result = engine.execute(text)
            result.close()
            return True
        except (ProgrammingError, OperationalError):
            return False


# def create_database(url_or_engine, encoding='utf8', template='template1'):
#     """Issue the appropriate CREATE DATABASE statement. Supported backends:
#     postgres and sqlite( in this latter case is no-op, i.e. it does nothing
#     as Sqlite creates a database automatically)
#
#     :param url_or_engine: SQLAlchemy engine or string denoting a database URL.
#     :param encoding: The encoding to create the database as. Default: 'utf8'
#     :param template: str, default='template1'
#         The name of the template from which to create the new database. At the
#         moment only supported by PostgreSQL driver.
#
#     To create a database, you can pass a simple URL that would have
#     been passed to ``create_engine``. ::
#
#         create_database('postgresql://postgres@localhost/name')
#
#     You may also pass the url from an existing engine. ::
#
#         create_database(engine)
#     """
#     with _engine(url_or_engine) as engine:
#         url = str(engine.url)
#         if url.startswith('postgres'):
#             if '/' not in url:
#                 raise ValueError('Wrong or Missing database name')
#             url_base, dbname = url[:url.rfind('/')], url[:url.rfind('/') + 1:]
#             if '"' in dbname:
#                 raise ValueError('" not allowed in dbname')
#
#             conn = engine.connect()
#             # https://stackoverflow.com/a/8977109
#             conn.execute("commit")
#             if not template:
#                 raise ValueError('No template provided template')
#             if '"' in template:
#                 raise ValueError('Invalid template name %s' % template)
#             # in postgres, single quote are used to denote strings,
#             # double quote are used to delimit an identifier
#             text = "CREATE DATABASE \"{0}\" ENCODING '{1}' TEMPLATE \"{2}\"".format(
#                 dbname,
#                 encoding,
#                 template
#             )
#             conn.execute(text)
#             conn.close()


@contextmanager
def _engine(url_or_engine):
    engine = url_or_engine
    engine_needs_disposal = False
    if isinstance(url_or_engine, str):
        engine_needs_disposal = True
        engine = create_engine(url_or_engine)
    try:
        yield engine
    finally:
        if engine_needs_disposal:
            engine.dispose()


# def get_session(dbpath, base=None, scoped=False, create_all=True, **engine_args):
#     """
#     Create an SQLAlchemy session for IO database operations
#
#     :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
#     :param base: a declarative base. If None, defaults to the default declarative base
#         used in this package for downloading
#     :param scoped: boolean (False by default) if the session must be scoped session
#     :param create_all: If True (the default), this method will issue queries that
#         first check for the existence of each individual table, and if not found
#         will issue the CREATE statements
#     :param engine_args: optional keyword argument values for the
#         `create_engine` method. E.g., let's provide two engine arguments,
#         `echo` and `connect_args`:
#         ```
#         get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
#         ```
#         For info see:
#         https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
#     """
#     # import stuff here in case they are time consuming, to avoid
#     # taking time to load other module stuff
#     from sqlalchemy.orm.scoping import scoped_session
#     from sqlalchemy.engine import create_engine
#     from sqlalchemy.orm.session import sessionmaker
#     from stream2segment.io.db(dot)models import Base
#
#     if base is None:
#         base = Base  # default declarative base, withour obspy methods
#
#     # init the session:
#     engine = create_engine(dbpath, **engine_args)
#     if create_all:
#         base.metadata.create_all(engine)  # @UndefinedVariable
#
#     if not scoped:
#         # create a configured "Session" class
#         session = sessionmaker(bind=engine)
#         # create a Session
#         return session()
#
#     session_factory = sessionmaker(bind=engine)
#     return scoped_session(session_factory)


