import re
import os
from contextlib import contextmanager

from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy import text, __version__ as __sa_version__


sqlalchemy_version = float(".".join(__sa_version__.split('.')[0:2]))  # https://stackoverflow.com/a/75634238

# IMPORTS to be called from the codebase to fix sqlalchemy 1.x vs 2.x changes:

if sqlalchemy_version >= 2:
    from sqlalchemy.orm import declarative_base  # noqa
else:
    from sqlalchemy.ext.declarative import declarative_base  # noqa


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
        # set max timeout if not set
        if is_postgres(dbpath):
            timeout = 30  # in seconds
            engine_args.setdefault('connect_args', {})
            engine_args['connect_args'].setdefault('connect_timeout', timeout)
        engine = create_engine(dbpath, **engine_args)
    except (SQLAlchemyError, ValueError) as _:
        # ValueError: 'postgresql://4:a6gfds' (cannot create port)
        raise ValueError('Cannot create a db engine. Possible reason: '
                         'the URL is not well formed or contains typos '
                         '(original error: %s)' % str(_))

    if check_db_existence:
        # (the only case when we don't care if the database exists is when
        #  we have sqlite and we are downloading)
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
        # `stream2segment.download.db::valid_session` that relies upon it
        return 'Database not accessible. Possible reason: wrong user/password/host ' \
               'in the URL, timeout (do you use VPN?) or the db does not exist'


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
    # We adopt a quick and dirt solution from https://stackoverflow.com/a/3670000
    # slightly modified because although they claimed it does, it doesn't work for sqlite
    # (a db is created if it does not exist). For a more sophisticated solution, see:
    # https://sqlalchemy-utils.readthedocs.io/en/latest/_modules/sqlalchemy_utils/functions/database.html#database_exists

    # Is it sqlite?
    url_ = get_url(url_or_engine)
    if is_sqlite(url_):
        return os.path.isfile(_extract_file_path(url_))

    with _engine(url_or_engine) as engine:
        try:
            with engine.begin() as conn:
                conn.execute(text('SELECT 1'))
                return True
        except (ProgrammingError, OperationalError) as _:
            return False


def _extract_file_path(sqlite_url):
    return os.path.abspath(sqlite_url[10:])  # remove sqlite:///


def get_url(url_or_engine):
    """Return the URL from the given argument (if already url, return the argument)
    """
    if isinstance(url_or_engine, str):
        return url_or_engine
    return str(url_or_engine.url)


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