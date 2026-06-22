import re
import os
from os.path import isabs, abspath, join
from typing import Literal

from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError
from sqlalchemy.engine import Engine, create_engine as sa_create_engine
from sqlalchemy import text, __version__ as __sa_version__, inspect

# legacy sqlalchemy version, just in case (https://stackoverflow.com/a/75634238)
sqlalchemy_version = float(".".join(__sa_version__.split('.')[0:2]))


def create_engine(dbpath: str, check_db_existence=True, **kwargs) -> Engine:
    """
    Wrapper around SQLAlchemy create_engine. Returns an Engine for IO DB operations

    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param check_db_existence: True by default, will raise a :class:`DbNotFound` if the
        database does not exist
    :param kwargs: optional keyword argument values for the
        `create_engine` method. E.g., providing `echo` and `connect_args`:
        ```
        create_engine(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        <https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args>
    """
    try:
        # set max timeout if not set
        if is_postgres(dbpath):
            timeout = 10  # in seconds
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
            raise DbNotFound(
               'Database not accessible. Possible reason: wrong user/password/host '
               'in the URL, timeout (do you use VPN?) or the db does not exist'
            )

    return engine


def close_engine(engine: Engine):
    """close the engine, this function is implemented for easy patching in tests"""
    if engine is not None:
        engine.dispose()


class DbNotFound(Exception):
    pass

sqlite_prefix = "sqlite:///"


def is_sqlite (db_url: str, in_memory: bool | None = None) -> bool:
    if db_url.lower().startswith(sqlite_prefix):
        if in_memory is None:
            return True
        else:
            return (db_url.removeprefix(sqlite_prefix) == ":memory:") == in_memory
    return False


postgres_prefix = "postgres://"

def is_postgres(db_url: str):
    return db_url.lower().startswith(postgres_prefix)


def get_dbname(db_url: str):
    return db_url[db_url.rfind('/') + 1:]


def database_exists(engine: Engine):
    """Return true if the database exists. Works for Postgres, MySQL, SQLite.

    :param engine: SQLAlchemy engine or string denoting a database URL.
    """
    # For details, see https://stackoverflow.com/a/3670000

    db_url = str(engine.url)
    if is_sqlite(db_url, in_memory=False):
        # Sqlite file. 'select 1' below might create a db if it does not exist. So:
        file_path = db_url.removeprefix(sqlite_prefix)
        if not os.path.isfile(file_path):
            return False

    try:
        with engine.connect() as conn:  # noqa
            conn.execute(text('SELECT 1'))
            return True
    except (ProgrammingError, OperationalError) as _:
        return False


def resolve_db_path(db_url: str, base_dir_path=None) -> str:
    """
    Resolve SQLite URL to absolute path relative to rel_dir_path.
    Non-SQLite URLs are returned unchanged, as well as SQLite URLs with absolute path.
    If the SQLite URL contains a relative path, it is resolved relative
    to `rel_dir_path` or, if the latter is None, to the current working directory.

    :param db_url: database URL.
    :param base_dir_path: base directory used to resolve relative SQLite paths.
    """
    if not is_sqlite(db_url) or is_sqlite(db_url, in_memory=True):
        return db_url

    # sqlite, not in-memory
    file_path = db_url.removeprefix(sqlite_prefix)
    if isabs(file_path):
        return db_url
    elif base_dir_path is None:
        return sqlite_prefix + abspath(file_path)
    else:
        return sqlite_prefix + abspath(join(base_dir_path, file_path))


def secure_dburl(db_url):
    """Return a printable database name by removing passwords, if any

    :param db_url: database path as string in the format:
        dialect+driver://username:password@host:port/database
        For info see:
        http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
    """
    return re.sub(r"://(.*?):(.*)@", r"://\1:***@", db_url)


def s2s_db_version(engine) -> Literal[4, 5]:
    from stream2segment.io.db.models import MiniSeed

    inspector = inspect(engine)
    if inspector.has_table(MiniSeed.__tablename__) or not inspector.get_table_names():
        return 5
    return 4
