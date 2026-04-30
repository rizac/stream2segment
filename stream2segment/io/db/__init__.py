import re
import os
from os.path import isabs, abspath, join
from typing import Literal

from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError
from sqlalchemy.engine import create_engine as sa_create_engine
from sqlalchemy import inspect
from sqlalchemy import text, __version__ as __sa_version__


sqlalchemy_version = float(".".join(__sa_version__.split('.')[0:2]))  # https://stackoverflow.com/a/75634238

# IMPORTS to be called from the codebase to fix sqlalchemy 1.x vs 2.x changes:

# if sqlalchemy_version >= 2:
#     from sqlalchemy.orm import declarative_base  # noqa
# else:
#     from sqlalchemy.ext.declarative import declarative_base  # noqa


def create_engine(dbpath: str, check_db_existence=True, **kwargs):
    """
    Wrapper around SQLAlchemy create_engine. Returns an Engine for IO DB operations

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param check_db_existence: True by default, will raise a :class:`DbNotFound` if the
        database does not exist
    :param kwargs: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
    """
    try:
        # set max timeout if not set
        if is_postgres(dbpath):
            timeout = 20  # in seconds
            kwargs.setdefault('connect_args', {})
            kwargs['connect_args'].setdefault('connect_timeout', timeout)
        engine = sa_create_engine(dbpath, **kwargs)
    except (SQLAlchemyError, ValueError) as _:
        # ValueError: 'postgresql://4:a6gfds' (cannot create port)
        raise ValueError('Cannot create a db engine. Possible reason: '
                         'the URL is not well formed or contains typos '
                         '(original error: %s)' % str(_))

    if check_db_existence:
        # (the only case when we don't care if the database exists is when
        #  we have sqlite, and we are downloading)
        if not database_exists(engine):
            raise DbNotFound(dbpath)

    return engine


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
    def dburl(self):  # FIXME USED? OVERLY VERBOSE CLASS RIGHT?
        return self.args[0]

    def __str__(self):
        # Warning: if you change the message below, check also the message raised in
        # `stream2segment.download.db::valid_session` that relies upon it
        return 'Database not accessible. Possible reason: wrong user/password/host ' \
               'in the URL, timeout (do you use VPN?) or the db does not exist'


sqlite_prefix = "sqlite:///"

sqlite_in_memory_path = ":memory:"


def is_sqlite(dburl: str):
    return dburl.lower().startswith(sqlite_prefix)


postgres_prefix = "postgres://"

def is_postgres(dburl: str):
    return dburl.lower().startswith(postgres_prefix)


def get_dbname(dburl: str):
    return dburl[dburl.rfind('/') + 1:]


def database_exists(engine):
    """Return true if the database exists. Works for Postgres, MySQL, SQLite.

    :param engine: SQLAlchemy engine or string denoting a database URL.
    """
    # We adopt a quick and dirt solution from https://stackoverflow.com/a/3670000
    # slightly modified because although they claimed it does, it doesn't work for sqlite
    # (a db is created if it does not exist). For a more sophisticated solution, see:
    # https://sqlalchemy-utils.readthedocs.io/en/latest/_modules/sqlalchemy_utils/functions/database.html#database_exists

    db_url = str(engine.url)
    if is_sqlite(db_url):
        file_path = db_url.removeprefix(sqlite_prefix)
        if file_path != sqlite_in_memory_path and not os.path.isfile(file_path):
            return False

    try:
        with engine.begin() as conn:
            conn.execute(text('SELECT 1'))
            return True
    except (ProgrammingError, OperationalError) as _:
        return False


def resolve_db_path(db_url: str, rel_dir_path=None) -> str:
    """
    Resolve SQLite URL to absolute path relative to rel_dir_path.
    Non-SQLite URLs are returned unchanged, as well as SQLite URLs with absolute path.
    If the SQLite URL contains a relative path, it is resolved relative
    to `rel_dir_path` or, if the latter is None, to the current working directory.

    :param db_url: database URL.
    :param rel_dir_path: base directory used to resolve relative SQLite paths.
    """
    if is_sqlite(db_url):
        file_path = db_url.removeprefix(sqlite_prefix)
        if file_path == sqlite_in_memory_path or isabs(file_path):
            return db_url
        elif rel_dir_path is None:
            return sqlite_prefix + abspath(file_path)
        else:
            return sqlite_prefix + abspath(join(rel_dir_path, file_path))

    return db_url


def secure_dburl(dburl):
    """Return a printable database name by removing passwords, if any

    :param dburl: database path as string in the format:
        dialect+driver://username:password@host:port/database
        For info see:
        http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
    """
    return re.sub(r"://(.*?):(.*)@", r"://\1:***@", dburl)


def s2s_db_version(engine) -> Literal[-1, 4, 5]:
    from stream2segment.io.db.models import MiniSeed

    inspector = inspect(engine)
    if inspector.has_table(MiniSeed.__tablename__) or not inspector.get_table_names():
        return 5
    return 4
