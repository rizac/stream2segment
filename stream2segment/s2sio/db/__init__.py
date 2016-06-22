import os
from StringIO import StringIO
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.exc import OperationalError
from sqlalchemy import BLOB
import pandas as pd
import pandas.io.sql as pdsql
import numpy as np

from sqlalchemy.sql.expression import select, join
from sqlalchemy.orm.attributes import InstrumentedAttribute

from stream2segment.classification import class_labels_df
from obspy import read


class SessionHandler(object):
    """
        Class handling sqlalchemy sessions. Initialize this object with an sqlalchemy engine
        or database uri (string) and then access the two methods:
        - self.session() which returns a new sqlAlchemy session, or
        - self.session_scope() to commit changes to the database (or rolling them back in case of
        exceptions) inside `with` statement:
        ```with self.session_scope() as session:
            ... do something with the session ...```
    """
    def __init__(self, sql_alchemy_engine_or_dburl):
        if isinstance(sql_alchemy_engine_or_dburl, Engine):
            self.engine = sql_alchemy_engine_or_dburl
            self.db_uri = self.engine.engine.url
        else:
            self.db_uri = sql_alchemy_engine_or_dburl
            self.engine = create_engine(sql_alchemy_engine_or_dburl)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations.
        Usage:
            s = SessionScope(engine_or_uri)
            with s.session_scope():  # or s.session_scope() as session:
                ... do your stuff ...
        """
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
        """Returns a new sqlalchemy session, or the sql alchemy session currently used in a with
        statement"""
        return Session(self.engine)


class PandasDbHandler(SessionHandler):
    """
    An object extending `SessionHandler` initialized with a database uri suited for managing
    "standard" pandas DataFrames IO-operations with the underlying db.
    This class is essentially a wrapper around several pandas sql methods, with the additional
    "hack" of allowing specifying a primary key when writing (self.to_sql). This is necessary
    because sqlAlchemy auto-mapping (see below) detects only tables with primary keys

    *SqlAlchemy Auto-map*
    This module uses SQLAlchemy engine (inherited from SessionHandler) which makes it possible to
    use any DB (e.g. postgresql, mysql, sqlite) supported by that library. Moreover, by means of
    sqlalchemy auto-map, tables in the database are stored and mapped to sql-alchemy table objects
    without the need of sqlalchemy Declarative's implementation and mantainance
    (http://docs.sqlalchemy.org/en/latest/orm/extensions/declarative/index.html)

    The sql alchemy tables can be accessed via `self.table(table_name)`, their columns (As
    sqlAlchemy InstrumentedAttribute objects) via self.column(tablename, colname).
    (If you want to force auto-mapping, call self.automap() first)

    Pitfalls:
    - pandas to_sql (version 0.18) does not give the possibility to:
    - write primary keys (solved, see above. Maybe future versions will be featured with that)
    - Foreign keys and complex db stuff
"""

    def __init__(self, db_uri):
        """
            :param: db_uri: the database uri, e.g. "sqlite:///" + filename
        """
        SessionHandler.__init__(self, db_uri)

        # engine, initialize once (already done in superclass)
        # self.engine = create_engine(self.db_uri)
        self.tables = {}

        # create tables from db (if any):
        self.automap()

    def automap(self):
        """
        Automatically maps tables on the database with tables (sql alchemy object) stored here. The
        latter can be accessed as items of this object, like a read-only dict:
        self[table_name], len(self), self.keys()
            Initializes the database connection retrieving automatically the tables stored there
            (those with primary keys) and storing in a class attribute tables (dict of table
            names mapped to relative sqlalchemy table)
        """
        # Code copied from here:
        # http://docs.sqlalchemy.org/en/latest/orm/extensions/automap.html#basic-use

        Base = automap_base()

        # reflect the tables
        # Note: ONLY TABLES WITH PRIMARY KEYS will be mapped!
        Base.prepare(self.engine, reflect=True)

        # mapped classes are now created with names by default
        # matching that of the table name.
        # User = Base.classes.events
        # Address = Base.classes.segments
        # store them in an attribute (dict of table_name [string]: sqlalchemy table)
        # It might be empty if database does not exist

        self.tables.update(Base.classes._data)  # pylint:disable=protected-access
        # line above overwrites existing keys

    def table(self, table_name):
        """Returns the sqlalchemy table object mapped to the given table_name"""
        return self.tables[table_name]

    def column(self, table_name, column_name):
        """Returns the sqlalchemy column object mapped to the given table_name"""
        try:
            table = self.table(table_name)
            try:
                return getattr(table, column_name)
            except AttributeError:
                raise ValueError("No such column '%s'" % column_name)
        except KeyError:
            raise ValueError("No such table: '%s'" % table_name)

    def drop_table(self, table_name):
        """Drops the table identified by the given table name. If the table does not exist, exits
        silently"""
        try:
            self.table(table_name).drop(self.engine)
            self.automap()
        except KeyError:
            pass

    def attcount(self, table_name):
        """Returns the number of attributes (columns in the underlying table) of the table
        identified by table_name"""
        tbl = self.table(table_name)
        return len([a for a in tbl.__dict__
                    if isinstance(getattr(tbl, a), InstrumentedAttribute)])

    def read_sql_query(self, selectable, parse_dates=None):
        """Selects the given tables in table_names (iterable) with the given where clause
        and returns the corresponding DataFrame. Use this method rather than self.read_sql for
        performance reasons (when possible). Note: if table_names has more than one element and
        some column names are shared across those tables, the resulting DataFrame columns will be
        not unique. NOTE also that since foreign keys are not supported in the current (0.18)
        version of pandas, joins are not supported
        :param table_names: an iterable of table names
        :param where: an sqlalchemy where clause
        :Example:
        xxx = PandasDbHandler(db_uri)  # or any of its subclasses...
        t_name = "some_existing_table_name"
        where = xxx.column(table, "Id").in_([55, 56])
        df = self.read_sql_query(xxx.select(t_name).where(where))
        """
        return pd.read_sql_query(selectable, self.engine, parse_dates=parse_dates)

    def to_sql(self, dframe, table_name, pkey_name, index=False, index_label=None,
               if_exists='append', dtype=None):
        """
            Calls dframe.to_sql with given arguments and self.engine. A table will be created if
            it does not exist and if_exist='append' (the default). From pandas.DataFrame.to_sql:
            :param dframe: the pandas DataFrame whose table must be created
            :param: dframe: pandas DataFrame
            :param: table_name: the name of the table mapped to dframe. Its existence will be
            checked and if not found, a table with the given name reflecting the data frame types
            will be created
            :type: table_name: string
            :param: pkey_name: a name of one of the DataFrame columns to be used as table primary
            key. FIXME: what if we want to specify the index? write "index" here? to be checked
            :type: pkey_name: string
            :param: if_exists: {'fail', 'replace', 'append'}, default 'append'
                - fail: If table exists, raises ValueError (note: pandas doc states "do nothing".
                    FALSE! have a look at source code in v.0.18)
                - skip: (self-explanatory, exit silently. Added functionality not present in pandas)
                - replace: If table exists, drop it, recreate it, and insert data.
                - append: If table exists, insert data. Create if does not exist.
            If append, then the table will be created if it does not exist
            :type: if_exists: string
            :param: index : boolean, default False. Write DataFrame index as a column.
            :type: index: boolean
            :type index_label: Column label for index column(s). If None is given (default) and
            index is True, then the index names are used. A sequence should be given if the
            DataFrame uses MultiIndex.
            :type index_label: string or sequence, default None
            :param: dtype: dict of column name to SQL type, default None. Optional specifying the
            datatype for columns. The SQL type should be a SQLAlchemy type, or a string for sqlite3
            fallback connection. Example: dtype={'data': sqlalchemy.BLOB}
            :type: dtype: dict or None
            :return: the sqlalchemy table or a KeyError is raised if such a table could not be
            created
        """
        # we need to handle ourself the if_exist case cause we need to create a schema with
        # the predefined pkeys. pandas.DataFrame.to_sql does not give us this possibility but
        # pandas sql get_schema (in _init_table below) does
        # Here is when it gets a little tricky (to_sql is dframe.to_sql):
        #
        #            table exists              table doesn't
        #            ========================= =========================
        #
        # 'fail'     to_sql(..., 'fail',...)   create table
        #                                      to_sql(..., 'append',...)
        #
        # 'replace'  drop table                < same as above>
        #            create table
        #            to_sql(..., 'append',...)
        #
        # 'append'   <same as right>           <same as above>
        #
        # NOTe: skip is easy, just check if exists and skip (see below)
        #

        table_exists = table_name in self.tables
        if if_exists == 'skip':
            if table_exists:
                return
            if_exists = 'append'  # useless, just provide a pandas valid key ('skip' is not)
        if if_exists == 'replace' and table_exists:
            try:
                self.drop_table(table_name)
                self.automap()  # update tables
            except OperationalError as _:  # table does not exist? whatever error? pass
                # (we nmight not have such a table...)
                pass
            table_exists = table_name in self.tables
            if table_exists:
                raise ValueError("Unable to drop table '%s'" % table_name)

        if not table_exists:
            self._init_table(dframe, table_name, pkey_name, dtype)  # calls self.automap()
            # Now, passing 'fail' to pandas method below would... fail, obviously, cause we just
            # created the table). Rename to 'append', cause now it's that in the user intentions
            if if_exists == 'fail':
                if_exists = 'append'

        if dframe is not None and not dframe.empty:
            # Rename 'replace' to 'append', as the former case wa handled above:
            if_exists = 'append' if if_exists == 'replace' else if_exists
            dframe.to_sql(table_name, self.engine, index=index, if_exists=if_exists,
                          index_label=index_label, dtype=dtype)

    def _init_table(self, dframe, table_name, pkey_name, dtype=None):
        """
            Re-updates the internal tables attribute by creating the given table. This method does
            NOT check if the table already exist on the database. Use ``table_name in self.tables``
            to check for that and in case ``self.drop_table(table_name)``
            :param dframe: the pandas DataFrame whose table must be created. Basically, column
            information and types are transferred to their Sql equivalent
            :param: dframe: pandas DataFrame
            :param: table_name: the name of the table mapped to dframe. Its existence will be
            checked and if not found, a table with the given name reflecting the data frame types
            will be created
            :type: table_name: string
            :param: pkey_name: a name of one of the DataFrame columns to be used as table primary
            key. FIXME: what if we want to specify the index? write "index" here? to be checked
            :type: pkey_name: string
            :param: dtype: dict of column name to SQL type, default None. Optional specifying the
            datatype for columns. The SQL type should be a SQLAlchemy type, or a string for sqlite3
            fallback connection. Example: dtype={'data': sqlalchemy.BLOB}
            :type: dtype: dict or None
        """
        # problem: pandas does not have a direct way to assign a primary key
        # so the hack (waiting for this feature to be implemented, surely) is taken from:
        # http://stackoverflow.com/questions/30867390/python-pandas-to-sql-how-to-create-a-table-with-a-primary-key
        schema = pdsql.get_schema(frame=dframe,
                                  name=table_name,
                                  con=self.engine,  # Using SQLAlchemy makes it possible to use
                                  # any DB supported by that library
                                  keys=[pkey_name],
                                  dtype=dtype)

        # execute the schema (CREATE TABLE etcetera)
        self.engine.execute(schema)
        self.automap()  # updates the table justb added as sqlalchemy table here

    def read_sql(self, table_name, coerce_float=True, index_col=None, parse_dates=None,
                 columns=None, chunksize=None):
        """
            Calls pandas.read_sql_table with self.engine as sql alchemy engine.
            From pandas documentation:
            :param table_name: Name of SQL table in database
            :type table_name: string
            :param index_col: Column(s) to set as index(MultiIndex)
            :type index_col: string or list of strings, optional, default: None
            :param  coerce_float: Attempt to convert values to non-string, non-numeric objects (like
            decimal.Decimal) to floating point. Can result in loss of Precision.
            :type coerce_float: boolean, default True
            :param  parse_dates:
                - List of column names to parse as dates
                - Dict of {column_name: format string} where format string is strftime compatible in
                case of parsing string times or is one of (D, s, ns, ms, us) in case of parsing
                integer timestamps
                - Dict of {column_name: arg dict}, where the arg dict corresponds to the keyword
                arguments of pandas.to_datetime() Especially useful with databases without native
                Datetime support, such as SQLite
            :type parse_dates: list or dict, default: None
            :param columns: List of column names to select from sql table
            :type columns: list, default: None
            :param chunksize: If specified, return an iterator where chunksize is the number of rows
            to include in each chunk.
            :type chunksize: int, default None
            :return a pandas DataFrame (empty in case table not found)
        """
#         if table_name not in self.tables:
#             # FIXME: better way to return a generator of 1 element?
#             return (pd.DataFrame() for _ in xrange(1)) if chunksize is not None else pd.DataFrame()
        return pd.read_sql_table(table_name, self.engine, None, index_col, coerce_float,
                                 parse_dates, columns, chunksize)


class DbHandler(PandasDbHandler):
    """
    An object extending `PandasDbHandler` used for IO operations for this program. As the tables
    are known in advance, they can be accessed via self.table(name) where name can be:

    - self.T_SEG ('segments'): represents the "core" table where each segment is stored (with
    relative station and channel metadata)
    - selg.T_EVT ("events"): represents the table of all seismic events downloaded
    - self.T_CLS ("classes"): represents the classes (for machine learning preprocessing) are stored
    - self.T_RUNS ("runs"): represents the information about each run populating the db is stored

    This class supports also an attribute table_settings (dict) in which we store all necessary
    information about tables (implementation details: any new table should be added to this dict)
    such as primary keys, data types etc. The user usually calls the two methods:
    - self.read
    - self.write
    - self.purge
    which take care of customizing necessary stuff (for instance, prior to writing the table
    T_SEG will be assigned with a primary key column whose name is set in self.table_settings and
    with values set as an hash calculated on network station location channel and time ranges)
    There are also two other handy methods:
    - self.select (which is faster than self.read if one wants to read not the whole table as
    DataFrame, providing also where clause if needed) and
    - self.mseed which returns the obspy stream from an object which can be a Series, a dict, or a
    bytestring
"""

    # initialize here the default table names:
    T_RUN = "runs"
    T_EVT = "events"
    T_SEG = "segments"
    T_CLS = "classes"

    def __init__(self, db_uri):
        """
            :param: db_uri: the database uri, e.g. "sqlite:///" + filename
        """
        PandasDbHandler.__init__(self, db_uri)

        # Not everything can be fully automated. Here specific table settings used in read, purge
        # and write. IMPORTANT NOTE: when adding a new table, add also AT LEAST its primary key here
        seg, evt, run, cls = DbHandler.T_SEG, DbHandler.T_EVT, DbHandler.T_RUN, DbHandler.T_CLS
        self.table_settings = {evt: {'pkey': '#EventID', 'parse_dates': ['Time']},
                               seg: {'pkey': 'Id', 'dtype': {'Data': BLOB},
                                     'parse_dates': ['DataStartTime', 'DataEndTime', 'RunId',
                                                     'StartTime', 'EndTime', 'ArrivalTime']},
                               run: {'pkey': 'Id', 'parse_dates': ['Id']},
                               cls: {'pkey': 'Id'}
                               }

    def select(self, table_names, where=None):
        """Selects the given tables in table_names (iterable) with the given where clause
        and returns the corresponding DataFrame by calling self.read_sql_query.
        Note: if table_names has more than one element and
        some column names are shared across those tables, the resulting DataFrame columns will be
        not unique
        :param table_names: an iterable of table names
        :param where: an sqlalchemy where clause
        :Example:
        xxx = DbHandler(db_uri)
        t_name = xxx.T_SEG
        where = xxx.column(t_name, "Id").in_([55, 56])
        df = xxx.select(t_name, where)
        """
        tables = []
        for tbl in table_names:
            tables.append(self.tables[tbl])

        selectable = select(tables).where(where) if where is not None else select(tables)
        parse_dates = []
        for table_name in table_names:
            parse_dates.extend(self.table_settings[table_name].get("parse_dates", []))

        return self.read_sql_query(selectable, parse_dates=None if not parse_dates else parse_dates)

    def purge(self, dframe, table_name, pkey_name=None):
        """
            Purges the given DataFrame of data already written on the database, based on the table
            primary key
            :param: dframe: the DataFrame
            :type: dframe: pandas DataFrame
            :param: table_name: the name of the table mapped to the given DataFrame
            :type: table_name: string
            :pkey_name: the private key whereby to check if data is already on the database
            :type: pkey_name: string. Must be a column of the given DataFrame. NOTE **If table_name
            is NOT one of the default tables registered in self.table_settings, it *must not be
            None and must be a column of dframe*
            FIXME: not implemented the case where the index is the primary key
            :return: a new DataFrame with the data not stored to the datbase according to pkey_name
            :rtype: pandas DataFrame
        """
        if dframe is None or dframe.empty or table_name not in self.tables:
            return dframe

        dframe = self.prepare_df(table_name, dframe)
        if table_name in self.table_settings:  # one of the default tables:
            if pkey_name is None:  # use default pkey:
                pkey_name = self.table_settings[table_name]['pkey']

        session = self.session()
        column = self.column(table_name, pkey_name)
        ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
        dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]
        session.close()

        return dframe

    def write(self, dframe, table_name, purge_first=False, if_exists='append'):
        """
        Calls self.to_sql(dframe, table_name). NOTE: **table_name must be one of the default tables
        registered on self.table_settings**
        """
        if dframe is None or dframe.empty:
            return
        pkey = self.table_settings[table_name]['pkey']
        if purge_first:
            dframe = self.purge(dframe, table_name, pkey)  # calls self.prepare_df
        else:
            dframe = self.prepare_df(table_name, dframe)
        dtype = self.table_settings[table_name].get('dtype', None)
        self.to_sql(dframe, table_name, pkey, if_exists=if_exists, dtype=dtype)

    def prepare_df(self, table_name, dframe):
        """
            Prepares a default frame for purging or writing. Basically it creates an Id for
            all null Id's if table_name == self.tables.segments, and modifies the data frame.
            Returns the input dataframe at the end (potentially unmodified)
        """
        if table_name == self.T_SEG:
            pkey = self.table_settings[table_name]['pkey']
            recalc = pkey not in dframe.columns
            if recalc:
                dframe.insert(0, pkey, None)
            else:
                recalc = pd.isnull(dframe[pkey]).any()

            if recalc:
                def myfunc(row):
                    row[pkey] = hash((row['#EventID'], row['#Network'], row['Station'],
                                      row['Location'], row['Channel'],
                                      row['DataStartTime'].isoformat(),
                                      row['DataEndTime'].isoformat()))
                    return row
                dframe = dframe.apply(myfunc, axis=1)

        return dframe

    def read(self, table_name, chunksize=None, columns=None, filter_func=None):
        """
        Calls self.read_sql(table_name). NOTE: **table_name must be one of the default tables
        registered on self.table_settings**
        :param table_name: Name of SQL table in database
        :type table_name: string
        :param chunksize: If specified, return an iterator where chunksize is the number of rows
        to include in each chunk.
        :type chunksize: int, default None
        """
        # if chunksize is None and filter func is not None, we might do filter on a big dataframe
        # which is what we do NOT want. Thus read per chunks
        filter_here = chunksize is None and filter_func is not None
        if filter_here:  # we might have overflows on filters, thus do iteration here:
            chunksize = 30

        ret = self.read_sql(table_name, coerce_float=True,
                            index_col=None,
                            parse_dates=self.table_settings[table_name].get('parse_dates', None),
                            columns=columns,
                            chunksize=chunksize)

        if filter_here:
            _tmp = None
            for r in ret:
                r = filter_func(r)
                _tmp = r if _tmp is None else _tmp.append(r, ignore_index=True)
            return _tmp

        if filter_func is None:
            return ret

        return (filter_func(r) for r in ret) if chunksize is not None else filter_func(ret)
    # implement mutable sequence: dbhandler.get(table_name), for table_name in dbhandler: ...

    @staticmethod
    def mseed(obj):
        """Returns an obspy stream object from the rawdata (bytes data)
        :param obj: a pandas Series or any object with a 'Data' attribute (bytestrings), or
        a python dict or any object with a 'Data' key (bytestrings) or a bytestring itself. The
        method will attempt to read bytestring data in the order specified above. Note that
        the 'Data' attribute is not assured to be used in future versions, therefore it is better
        to call this method than e.g., self.mseed(series.Data)
        :return: an obspy stream from the resulting obj
        """
        try:
            return read(StringIO(obj.Data))
        except AttributeError:
            try:
                return read(StringIO(obj['Data']))
            except (TypeError, KeyError):
                return read(StringIO(obj))


class ListReader(DbHandler):
    """
        A sublass of DbHandler which behaves like a list. Items of the list are the pandas Series
        of the "segments" table (self.table(self.T_SEG)) accessed in the order they are read from
        the table (insertion order?).
    """
    def __init__(self, db_uri, filter_func=None, sort_columns=None, sort_ascending=None):
        """
            Initializes a new ListReader via a given db_uri
            :param db_uri: the database uri, e.g. sqlite:///path_to_my_sqlite_file
            :param filter_func: a filter function taking as argument the DataFrame of *segments*
            read and returning a filtered DataFrame. NOTE: filter_func is mainly implemented to
            filter out (remove) DataFrame rows. However, if filter_func is not None, it should
            not filter out the Dataframe columns denoting the associated Table's primary key or any
            of the columns specified in sort_columns, if any. The table primary key name is
            accessible via `self.table_settings[self.T_SEG]['pkey']`
            :param sort_columns: same as pandas DataFrame sort_value's by arg.
            Example: 'A', ['A', 'B'], ...
            :param sort_ascending: same aas pandas DataFrame sort_values's 'ascending' argument.
            Example: True, [True, False], ...
        """
        DbHandler.__init__(self, db_uri)
        # check if database exists: FIXME: why HERE???
        try:
            connection = self.engine.connect()
            connection.execute("SELECT * from segments;")
            connection.close()
        except OperationalError as oerr:
            raise ValueError(str(oerr) + "\nDoes the database exist?")

        iterator = self.read(self.T_SEG, chunksize=30, filter_func=filter_func)
        id_colname = self.table_settings[self.T_SEG]['pkey']
        files = None  # do NOT instantiate a new DataFrame, otherwise append below coerces to
        # the type of the files DataFrame (object) and we want to preserve the db type (so first
        # iteration files is the first chunk read)
        columns = [id_colname] if sort_columns is None else list(sort_columns)
        if id_colname not in columns:
            columns.insert(0, id_colname)

        for data in iterator:
            # remove Data column (avoid too much data in memory, get it with get_segment)
            # data.drop('Data', axis=1, inplace=True)
            # only use ids:
            data = pd.DataFrame({k: data[k] for k in columns})
            if files is None:
                files = data
            else:
                files = files.append(data)

        if files is not None and not files.empty and sort_columns:
            files.sort_values(by=sort_columns, ascending=sort_ascending, inplace=True)

        if files is None:
            files = pd.DataFrame(columns=[id_colname])
        else:
            files = files[[id_colname]]
            files.reset_index(drop=True, inplace=True)
        # files.info()
        self.mseed_ids = files

    def __len__(self):
        """returns the number of segments read from the segments table"""
        return len(self.mseed_ids)

    def __getitem__(self, key):
        return self.get(key)[self.T_SEG]

    def get(self, segment_id, table_name, column_names_list=None):

        where = (self.column(self.T_SEG, "Id") == segment_id)

        if self.T_EVT == table_name:
            where &= (self.column(self.T_EVT, "#EventID") == self.column(self.T_SEG, "#EventID"))

        if self.T_RUN == table_name:
            where &= (self.column(self.T_RUN, "Id") == self.column(self.T_SEG, "RunId"))

        if self.T_CLS == table_name:
            where &= (self.column(self.T_CLS, "Id") == self.column(self.T_SEG, "ClassId"))

        pddf = self.select([table_name], where)

        return pddf if column_names_list is None else \
            pddf[column_names_list]

    def get_stream(self, segment_id, include_same_channel=False):
        columns = ['#Network', 'Station', 'Location', 'DataStartTime', 'DataEndTime',
                   'Channel', 'Data']
        # sser is the pandas Series representing the row of the segments table
        # corresponding to segment_id
        seg_series = self.get(segment_id, self.T_SEG, columns).iloc[0]
        strm = self.mseed(seg_series)
        if include_same_channel:
            def filter_func(df):
                return df[(df['#Network'] == seg_series['#Network']) &
                          (df['Station'] == seg_series['Station']) &
                          (df['Location'] == seg_series['Location']) &
                          (df['DataStartTime'] == seg_series['DataStartTime']) &
                          (df['DataEndTime'] == seg_series['DataEndTime']) &
                          (df['Channel'].str[:2] == seg_series['Channel'][:2]) &
                          (df['Channel'] != seg_series['Channel'])]

            # FIXME: read is NOT EFFICIENT, BETTER SELECT!!
            other_components = self.read(self.T_SEG, filter_func=filter_func)
            for i, seg_sr in other_components.iterrows():
                stre = self.mseed(seg_sr)
                strm.traces.append(stre.traces[0])

        return strm

    def get_id(self, index):
        """Returns the database ID of the index-th item (=db table row)"""
        id_colname = self.table_settings[self.T_SEG]['pkey']
        return self.mseed_ids.iloc[index][id_colname]


#     def get_(self, segment_id, table_names=None, as_annotated_class=True):
#         """gets the row of the index-th segment from the relative table
#         :param: table_name either self.T_SEG (default when missing) or self.T_EVT or self.T_RUN
#         :return a pandas Series if as_series is True (the default), raising a ValueError if
#         the DataFrame resulting from the db query has not 1 element, or a pandas DataFrame otherwise
#         """
#         if table_names is None:
#             table_names = [self.T_SEG]
#         else:
#             # pandas unique preserves the order and is faster than numpy.unique:
#             table_names = pd.unique(table_names).tolist()
# 
#         where = (self.column(self.T_SEG, "Id") == segment_id)
# 
#         if self.T_EVT in table_names:
#             where &= (self.column(self.T_EVT, "#EventID") == self.column(self.T_SEG, "#EventID"))
# 
#         if self.T_RUN in table_names:
#             where &= (self.column(self.T_RUN, "Id") == self.column(self.T_SEG, "RunId"))
# 
#         if self.T_CLS in table_names:
#             if as_annotated_class:
#                 where &= (self.column(self.T_CLS, "Id") ==
#                           self.column(self.T_SEG, "AnnotatedClassId"))
#             else:
#                 where &= (self.column(self.T_CLS, "Id") == self.column(self.T_SEG, "ClassId"))
# 
#         pddf = self.select(table_names, where)
# 
#         if len(pddf) != 1:
#             raise ValueError("Expected 1 row, found %d" % len(pddf))
# 
# 
#         # create the multiindex:
#         tnms = []
#         for tnam in table_names:
#             tnms.extend([tnam]*self.attcount(tnam))
# 
#         index = pd.MultiIndex.from_tuples(list(zip(tnms, pddf.columns)), names=['table', 'column'])
#         return pd.Series(pddf.iloc[0].values, index=index)


class ClassAnnotator(ListReader):
    """
        Class managing the downloaded data.
        An object of this class is initialized with a database URI: r= Reader(db_uri) and stores
        internally each item (=table row) as a list of ids mapped to the relative table rows.
        Each item (table row) holds some "data" (collection of one or more mseed files)
        and associated "metadata" (classId, AnnotatedClassId, Station, Network etcetera).
        This object is an iterable and supports the len function. For each entry, data
        and metadata can be accessed via get_segment, get_metadata, class id get/set via get_class,
        set_class
        :Example:
        .. code:
            r= Reader(db_uri)
            for i in xrange(len(r)):
                r.get(i)  # returns an obspy stream object on which you can call several methods,
                          # e.g. r.get(i).plot()
                r.get_segment(i, raw=True)  # returns raw bytes (string in python2)
                r.get_segment(i)  # returns the obspy stream object
                r.set_class(i, j) # sets the class of the i-th instance
                # j must be one of the values of the 'Id' columns of
                r.set_class(i, j, as_annotated_class=False) # sets the class of the i-th instance
                # but under the column 'class id' not 'annotated class id'
                # j must be one of the values of the 'Id' columns of
                r.get_classes()  # get dataframe of all classes, with columns 'Id' 'Label',
                                     # 'Description', 'Count'
    """
    # the db column name of the annotated class_id:
    annotated_class_id_colname = 'AnnotatedClassId'
    # the db column name of the (classified) class_id:
    class_id_colname = 'ClassId'

    def __init__(self, *args, **kwargs):
        """
            Initializes a new DataManager via a given db_uri
            :param db_uri: the database uri, e.g. sqlite:///path_to_my_sqlite_file
            :param filter_func: a filter function taking as argument the DataFrame of segments
            read and returning a filtered DataFrame
        """
        ListReader.__init__(self, *args, **kwargs)
        self._classes_dataframe = self.read(self.T_CLS)
        self._classes_dataframe.insert(len(self._classes_dataframe.columns), 'Count', 0)
        self.update_classes()

    def get_classes_df(self):
        """Returns the pandas DataFrame representing the classes. The DataFrame is read once
        in the constructor and updated here with a column 'Count' which counts the instances per
        class. Column names might vary across versions but in principle their names are 'Id',
        'Label' and 'Description' (plus the aforementioned 'Count')"""
        return self._classes_dataframe

    def get_class(self, index, as_annotated_class=True):
        """Returns the class id (integer) of the index-th item (=db table row).
        :param as_annotated_class: if True (the default), returns the value of the column
            of the annotated class id (representing the manually annotated class), otherwise the
            column specifying the class id (representing the class id as the result of some
            algorithm, e.g. statistical classifier)
        """
        row = self.get(index, as_annotated_class=as_annotated_class)
        att_name = self.annotated_class_id_colname if as_annotated_class else self.class_id_colname
        return row[self.T_SEG][att_name]

    def set_class(self, index, class_id, as_annotated_class=True):
        """
            Sets the class of the index-th item (=db table row). **IMPORTANT: call
            self._update_classes() after exiting if using this method within a with statement:
            `with self.session_scope():`
                ...
            reader.update_classes()
            :param index: the mseed index
            :param class_id: one of the classes id. To get them, call self.get_classes()['Id']
            (returns a pandas Series object)
            :param as_annotated_class: if True (the default), sets the value of the column
            of the annotated class id (representing the manually annotated class), otherwise the
            column specifying the class id (representing the class id as the result of some
            algorithm, e.g. statistical classifier)
        """
        # store the old class id and the new one:
        # NOTE: we absolutely need a with statement to keep the session open
        with self.session_scope() as session:
            id_colname = self.table_settings[self.T_SEG]['pkey']
            row = session.query(self.tables[self.T_SEG])
            row = row.filter(self.column(self.T_SEG, id_colname) == self.get_id(index)).first()
            # row = self.get_row(index)
            att_name = self.annotated_class_id_colname if as_annotated_class else \
                self.class_id_colname
            setattr(row, att_name, class_id)
        self.update_classes()

    # get the list [(class_id, class_label, count), ...]
    def update_classes(self):
        session = self.session()
        classes_dataframe = self.get_classes_df()
        table = self.table(self.T_SEG)
        id_colname = self.table_settings[self.T_CLS]['pkey']

        def countfunc(row):
            row['Count'] = session.query(table).filter(table.AnnotatedClassId == row[id_colname]).count()
            return row

        self._classes_dataframe = classes_dataframe.apply(countfunc, axis=1)
        session.close()
