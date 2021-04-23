import re
from contextlib import contextmanager

from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker


def get_session(dbpath, scoped=False, check_db_existence=True, **engine_args):
    """Create an SQLAlchemy session for IO database operations

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param scoped: boolean (False by default) if the session must be scoped session
    :param check_db_existence: True by default, will raise a :class:`DbNotFound` if the
        database does not exist
    :param engine_args: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
    """
    if not isinstance(dbpath, str):
        raise TypeError('string required, %s found' % str(type(dbpath)))

    try:
        engine = create_engine(dbpath, **engine_args)
    except (SQLAlchemyError, ValueError) as _:
        # ValueError: 'postgresql://4:a6gfds' (cannot create port)
        raise ValueError('Cannot create a db engine. Possible reason: '
                         'the URL is not well formed or contains typos '
                         '(original error: %s)' % str(_))

    if check_db_existence:
        # the only case when we don't care if the database exists is when
        # we have sqlite and we are downloading. Thus
        if not database_exists(engine):
            raise DbNotFound(dbpath)

    session_factory = sessionmaker(bind=engine)

    if not scoped:
        # create a Session
        return session_factory()

    return scoped_session(session_factory)


class DbNotFound(ValueError):
    """DbNotFound are exception raised when the database could not be found. this
    happens basically when either the db does not exist, or any entry (user, password,
    host) is wrong. E.g.: this connects to an engine but might raise this exception:
    postgresql://<user>:<password>@<host>.gfz-potsdam.de/me"
    whereas this does not even raise this exception and fails when creating an engine:
    wrong_dialect_and_driver://<user>:<password>@<host>.gfz-potsdam.de/me"
    """
    def __init__(self, dburl):
        super().__init__(dburl)

    @property
    def dburl(self):
        return self.args[0]

    def __str__(self):
        # Warning: if you change the message below, check also the message raised in
        # check also stream2segment.download.db::valid_session
        # that relies upon this:
        return 'Database "%s" not accessible. Possible reason: wrong user/password/host ' \
               'in the URL, or the db does not exist' % get_dbname(self.dburl)


def is_sqlite(dburl):
    return isinstance(dburl, str) and dburl.lower().startswith('sqlite')


def is_postgres(dburl):
    return isinstance(dburl, str) and dburl.lower().startswith('postgres')


def get_dbname(dburl):
    return dburl[dburl.rfind('/') + 1:]


def database_exists(url_or_engine):
    """Return true if the database exists. Works for Postgres, MySQL, SQLite.

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
        except (ProgrammingError, OperationalError) as exc:
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


def close_session(session, dispose_engine=True):
    """Close the SQLAlchemy session
    https://docs.sqlalchemy.org/en/13/orm/session_basics.html#closing
    and the underline engine accessible via `session.get_bind()`
    https://docs.sqlalchemy.org/en/14/core/connections.html?highlight=dispose#engine-disposal
    unless `dispose_engine` is False (default: True).

    :param session: a SQLAlchemy session
    :param dispose_engine: boolean (default True when missing) close also the
        underlying engine
    :return: True if all required operation(s) where performed with no exceptions,
        False otherwise
    """
    ret = True
    try:
        session.close()
        # close_all_sessions()
    except Exception:
        ret = False
    if dispose_engine:
        try:
            session.get_bind().dispose()
        except Exception:
            ret = False
    return ret


def secure_dburl(dburl):
    """Return a printable database name by removing passwords, if any

    :param dburl: database path as string in the format:
        dialect+driver://username:password@host:port/database
        For info see:
        http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
    """
    return re.sub(r"://(.*?):(.*)@", r"://\1:***@", dburl)