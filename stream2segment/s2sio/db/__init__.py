"""
    Database classes utilities for IO operations
"""

import os
from StringIO import StringIO
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.exc import OperationalError
from sqlalchemy import BLOB
from sqlalchemy import distinct
from sqlalchemy.sql.expression import select, join
from sqlalchemy.orm.attributes import InstrumentedAttribute
import pandas as pd
import pandas.io.sql as pdsql
import numpy as np

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
        self._open_session = None
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
            self._open_session = None
            session.close()

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

    The sql alchemy tables can be accessed via `self.table(table_name)`, and their columns (As
    sqlAlchemy InstrumentedAttribute objects) as sql alchemy attributes
    t = self.table(table_name)
    # access the Id column:
    t.Id
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

    def get_name_and_table(self, table):
        """Returns the tuple table_name (string), sqlalchemy table object
        relative to the argument, which can be either a table name or an sqlalchemy table
        Raises value error if no table is found in the automap method of this class"""
        try:
            return table, self.tables[table]
        except KeyError:
            for key, value in self.tables.iteritems():
                if table == value:
                    return key, value

        raise ValueError(("Error in `get_name_and_table` method: argument '%s' (%s) is neither "
                          "a valid table name nor an existing table object") % (str(table),
                                                                                str(type(table))))

#     def drop_table(self, table_name):
#         """Drops the table identified by the given table name. If the table does not exist, exits
#         silently"""
#         try:
#             self.table(table_name).drop(self.engine)
#             self.automap()
#         except KeyError:
#             pass

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
        dbh = PandasDbHandler(db_uri)  # or any of its subclasses...
        table_name = "some_existing_table_name"
        where = dbh.column(table_name, "Id").in_([55, 56])
        df = self.read_sql_query(dbh.select(t_name).where(where))
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
        return pd.read_sql_table(table_name, self.engine, None, index_col, coerce_float,
                                 parse_dates, columns, chunksize)


class DbHandler(PandasDbHandler):
    """
    An object extending `PandasDbHandler` used for IO operations for this program. As the tables
    are known in advance, they can be accessed via :

    - self.T_SEG ('segments'): represents the "core" table where each segment is stored (with
    relative station and channel metadata)
    - selg.T_EVT ("events"): represents the table of all seismic events downloaded
    - self.T_CLS ("classes"): represents the classes (for machine learning preprocessing) are stored
    - self.T_RUNS ("runs"): represents the information about each run populating the db is stored

    Each of the above mentioned classes has its relative table name, e.g. self.T_SEG_NAME,
    self.T_EVT_NAME, ... etcetera. Most of the methods accept either table names or table objects

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

    STATION_TBL_COLUMNS = ["#Network", "Station", "Latitude", "Longitude", "Elevation", "SiteName",
                           "StartTime", "EndTime"]
    CHANNEL_TBL_COLUMNS = ["Location", "Channel", "Depth", "Azimuth", "Dip", "SensorDescription",
                           "Scale", "ScaleFreq", "ScaleUnits", "SampleRate"]

    # initialize here the default table names:
    # ANY NEW TABLE MUST BE ADDED HERE (step 1 of 3):
    T_RUN_NAME = "runs"
    T_EVT_NAME = "events"
    T_STA_NAME = "stations"
    T_CHA_NAME = "channels"
    T_SEG_NAME = "segments"
    T_CLS_NAME = "classes"
    T_PRO_NAME = "processing"

    def __init__(self, db_uri):
        """
            :param: db_uri: the database uri, e.g. "sqlite:///" + filename
        """
        PandasDbHandler.__init__(self, db_uri)

        # follow these steps if you add a new table:
        # 1: add a table object to this class (we could use a loop BUT: we wouldn't see these
        # variables in editors autocompletiion and we would probably see warnings.Quite annoying...

        # NOTE: use the `property` class (same as the `property` decorator) so that the value is
        # returned only when called explicitly (we might have no db at the given db_uri)
        DbHandler.T_RUN = property(lambda self: self.table(self.T_RUN_NAME))
        DbHandler.T_EVT = property(lambda self: self.table(self.T_EVT_NAME))
        DbHandler.T_STA = property(lambda self: self.table(self.T_STA_NAME))
        DbHandler.T_CHA = property(lambda self: self.table(self.T_CHA_NAME))
        DbHandler.T_SEG = property(lambda self: self.table(self.T_SEG_NAME))
        DbHandler.T_CLS = property(lambda self: self.table(self.T_CLS_NAME))
        DbHandler.T_PRO = property(lambda self: self.table(self.T_PRO_NAME))

        # Not everything can be fully automated. Here specific table settings used in read, purge
        # and write. IMPORTANT NOTE: when adding a new table, add also AT LEAST its primary key here
        # (step 2 of 3)

        # (use lambdas to lazily initialize tables. We might not have tables nor a db at this stage)
        self.table_settings = {
            self.T_EVT_NAME: {
                              'pkey': '#EventID',
                              'parse_dates': ['Time'],
                              't_seg_bin_rel': lambda: (getattr(self.T_EVT, "#EventID") ==
                                                        getattr(self.T_SEG, "#EventID")),
                              },
            self.T_STA_NAME: {
                              'pkey': 'Id',
                              'parse_dates': ["StartTime", "EndTime"],
                              't_seg_bin_rel': lambda: (self.T_SEG.ChannelId == self.T_CHA.Id &
                                                        self.T_STA.Id == self.T_CHA.StationId)
                              },
            self.T_CHA_NAME: {
                              'pkey': 'Id',
                              't_seg_bin_rel': lambda: (self.T_SEG.ChannelId == self.T_CHA.Id),
                              },
            self.T_SEG_NAME: {
                             'pkey': 'Id',
                             'dtype': {'Data': BLOB},
                             'parse_dates': ['RunId', 'StartTime', 'EndTime', 'ArrivalTime']
                             },
            self.T_RUN_NAME: {
                              'pkey': 'Id',
                              'parse_dates': ['Id'],
                              't_seg_bin_rel': lambda: (self.T_SEG.RunlId == self.T_RUN.Id)
                              },
            self.T_CLS_NAME: {
                              'pkey': 'Id',
                              't_seg_bin_rel': lambda: (self.T_SEG.ClasslId == self.T_CLS.Id)
                              },
            self.T_PRO_NAME: {
                              'pkey': 'Id',
                              't_seg_bin_rel': lambda: (self.T_SEG.ProcessinglId == self.T_PRO.Id)
                              },
            }

    def join_t_seg_with(self, tables):
        """
            Returns a where clause by "joining" the segments table with the given tables according
            to their "foreign keys". NOTE that this is not a real join, as we cannot implement
            foreign keys in pandas (this is the drawback. The advantage is that we don't have to
            implement an ORM by taking care of all columns data types). So basically the returned
            clause is a where clause of AND (sqlalchemy '&') such as, e.g.:
            tables = [self.T_RUN, self.T_EVT], then
            join_t_seg_with(tables) produces:
            self.T_SEG.RunId == self.T_RUN.Id & \
                getattr(self.T_SEG, "#EventID") == getattr(self.T_EVT, "#EventID")

            (note the getattr in the second & cause we have "invalid" python attr.names)
            :param tables: an iterable of the sqlalchemy tables or table names (string). If "all",
            all tables for which a relation with self.T_SEG has been declared will be used
        """
        tbl_dict = self.tables if tables == "all" else self._todict(tables)

        where_clause = None

        for tname, _ in tbl_dict.iteritems():
            # get the key t_seg_bin_rel and call it (it's a callable), or by default return None
            where = self.table_settings.get(tname, {}).get('t_seg_bin_rel', lambda: None)()

            if where is None:
                continue

            if where_clause is None:
                where_clause = where
            else:
                where_clause &= where

        return where_clause

    def _todict(self, tables):
        """
            Returns a subdict of self.tables according to the arguments
            :param tables: an iterable of either table names or table objects
            :return: a dict of string -> sql alchemy table (subset of self.tables)
        """
        ret = {}
        for tbl in tables:
            tname, tbl = self.get_name_and_table(tbl)
            ret[tname] = tbl
        return ret

    def select(self, tables, where=None):
        """Selects the given tables (iterable) with the given where clause
        and returns the corresponding DataFrame by calling self.read_sql_query.
        Note: if tables has more than one element and
        some column names are shared across those tables, the resulting DataFrame columns will be
        not unique
        :param tables: an iterable of table names, or sqlalchemy tables
        :param where: an sqlalchemy where clause
        :Example:

        dbh = DbHandler(db_uri)
        df = dbh.select([dbh.T_SEG], dbh.T_SEG.Id.in_([55, 56]))

        # if the query has to be done from self.T_SEG on another table's value (e.g., Magnitude),
        # use an sqlalchemy '&' with the relation between the two tables. The method
        # self.join_t_seg_with([T_EVT])
        # equals the sqlalchemy expression:
        # getattr(T_SEG, "#EventID") == getattr(T_SEG, "#EventID")

        df = dbh.select([dbh.T_SEG], T_EVT.Magnitude.in_([3.1, 3.2]) & \
            self.join_t_seg_with([T_EVT]))

        # You can also return a joined DataFrame, BUT NOTE THAT THE COLUMN ORDER
        # of the DATAFRAME SEEMS NOT TO REFLECT THE ORDER OF THE TABLES, THUS COLUMNS
        # WITH THE SAME NAME (e.g. 'Longitude') ARE EITHER MERGED OR DUPLICATED (FIXME: CHECK!)

        df = dbh.select([dbh.T_SEG, dbh.T_EVT], T_EVT.Magnitude.in_([3.1, 3.2]) & \
            getattr(T_SEG, "#EventID") == getattr(T_SEG, "#EventID"))
        """
        tablez = self._todict(tables)

        selectable = select(tablez.values()).where(where) if where is not None else \
            select(tablez.values())

        # get the dates object so thay are parsed to datetime objects. These are stored in the
        # constructor. FIXME: not handled the case of columns sharing the same name on different
        # tables
        parse_dates = []
        for table_name in tablez.keys():
            parse_dates.extend(self.table_settings[table_name].get("parse_dates", []))

        return self.read_sql_query(selectable, parse_dates=None if not parse_dates else parse_dates)

    def purge(self, dframe, table, pkey_name=None):
        """
            Purges the given DataFrame of data already written on the database, based on the table
            primary key
            :param: dframe: the DataFrame
            :type: dframe: pandas DataFrame
            :param: table: the name of the table mapped to the given DataFrame, or the sql alchemy
            table object (one of the values of self.tables dict)
            :type: table_name: string
            :pkey_name: the private key whereby to check if data is already on the database
            :type: pkey_name: string. Must be a column of the given DataFrame. NOTE **If table_name
            is NOT one of the default tables registered in self.table_settings, it *must not be
            None and must be a column of dframe*
            FIXME: not implemented the case where the index is the primary key
            :return: a new DataFrame with the data not stored to the datbase according to pkey_name
            :rtype: pandas DataFrame
        """
        table_name, table = self.get_name_and_table(table)
        if dframe is None or dframe.empty or table_name not in self.tables:
            return dframe

        dframe = self.prepare_df(table_name, dframe)
        if table_name in self.table_settings:  # one of the default tables:
            if pkey_name is None:  # use default pkey:
                pkey_name = self.table_settings[table_name]['pkey']

        session = self.session()
        column = getattr(table_name, pkey_name)
        ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
        dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]
        session.close()

        return dframe

    def write(self, dframe, table, purge_first=False, if_exists='append'):
        """
        Calls self.to_sql(dframe, table_name). NOTE: **table_name must be one of the default tables
        registered on self.table_settings**
        :param table: the name of the table mapped to the given DataFrame, or the sql alchemy
            table object (one of the values of self.tables dict)
        """
        table_name = self.get_name_and_table(table)[0]
        if dframe is None or dframe.empty:
            return
        pkey = self.table_settings[table_name]['pkey']
        if purge_first:
            dframe = self.purge(dframe, table_name, pkey)  # calls self.prepare_df
        else:
            dframe = self.prepare_df(table_name, dframe)
        dtype = self.table_settings[table_name].get('dtype', None)
        self.to_sql(dframe, table_name, pkey, if_exists=if_exists, dtype=dtype)

    def prepare_df(self, table, dframe):
        """
            Prepares a default frame for purging or writing. Basically it creates an Id for
            all null Id's if table_name == self.tables.segments, and modifies the data frame.
            Returns the input dataframe at the end (potentially unmodified)
             :param table: the name of the table mapped to the given DataFrame, or the sql alchemy
            table object (one of the values of self.tables dict)
        """
        table_name = self.get_name_and_table(table)[0]
        if table_name == self.T_SEG_NAME:
            dframe, pkey, recalc = self._checkpkey(table_name, dframe)

            if recalc:
                dframe = self.prepare_df(self.T_CHA_NAME, dframe)
                dframe.rename(columns={self.table_settings[self.T_CHA_NAME]['pkey']:
                                       'ChannelId'}, inplace=True)

                def myfunc(row):
                    row[pkey] = hash((row['#EventID'], row['#Network'], row['Station'],
                                      row['Location'], row['Channel'],
                                      row['DataStartTime'].isoformat(),
                                      row['DataEndTime'].isoformat()))
                    return row
                dframe = dframe.apply(myfunc, axis=1)
                dframe = dframe.drop(self.STATION_TBL_COLUMNS + self.CHANNEL_TBL_COLUMNS, axis=1)

        elif table_name == self.T_STA_NAME:
            dframe, pkey, recalc = self._checkpkey(table_name, dframe)

            if recalc:
                dframe.insert(0, pkey, dframe['#Network'] + "." + dframe['Station'])
            dframe = dframe([pkey] + self.STATION_TBL_COLUMNS)

        elif table_name == self.T_CHA_NAME:
            dframe, pkey, recalc = self._checkpkey(table_name, dframe)

            if recalc:
                dframe = self.prepare_df(self.T_STA_NAME, dframe)
                dframe.rename(columns={self.table_settings[self.T_STA_NAME]['pkey']:
                                       'StationId'}, inplace=True)
                dframe.insert(0, pkey, dframe['StationId'] + "." + dframe['Location'] + "." +
                              dframe['Channel'])
                dframe = dframe([pkey] + self.CHANNEL_TBL_COLUMNS)

        return dframe

    def _checkpkey(self, table, dframe):
        table_name = self.get_name_and_table(table)[0]
        pkey = self.table_settings[table_name]['pkey']
        recalc = pkey not in dframe.columns
        if recalc:
            dframe.insert(0, pkey, None)
            return dframe, pkey, True
        else:
            recalc = pd.isnull(dframe[pkey]).any()
            return dframe, pkey, recalc

    def read(self, table, chunksize=None, columns=None, filter_func=None):
        """
        Calls self.read_sql(table_name). NOTE: **table_name must be one of the default tables
        registered on self.table_settings**
        :param table: the name of the table mapped to the given DataFrame, or the sql alchemy
            table object (one of the values of self.tables dict)
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

        table_name = self.get_name_and_table(table)[0]
        ret = self.read_sql(table_name, coerce_float=True,
                            index_col=None,
                            parse_dates=self.table_settings[table_name].get('parse_dates', None),
                            columns=columns,
                            chunksize=chunksize)

        if filter_here:
            _tmp = None
            for r__ in ret:
                r__ = filter_func(r__)
                _tmp = r__ if _tmp is None else _tmp.append(r__, ignore_index=True)
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
        of the "segments" table (self.T_SEG, or self.tables(self.T_SEG_NAME)) accessed in the order
        they are read from
        the table (insertion order?).
    """
    def __init__(self, db_uri, filter_func=None, sort_columns=None, sort_ascending=None):
        """
            Initializes a new ListReader via a given db_uri
            :param db_uri: the database uri, e.g. sqlite:///path_to_my_sqlite_file
            :param filter_func: a filter function taking as argument the DataFrame of *segments*
            and returning a filtered DataFrame. NOTE: filter_func is mainly implemented to
            filter out (remove) DataFrame rows of the segments table (no filter on other tables
            column are currently supported, e.g. magnitude of T_EVT). It should
            not filter out the Dataframe columns denoting the segments Table's primary key or any
            of the columns specified in sort_columns, if any. The table primary key name is
            accessible via `self.table_settings[self.T_SEG_NAME]['pkey']`. Most methods of this
            class use that segment id as argument to uniquely get/set segments properies on the 
            relative table row
            :Example:

            class_ids = (1,2,-1)
            def filter_func(dframe):
                return dframe[dframe['ClassId'].isin(class_ids)]

            l = Listreader(my_db_uri, filter_func)

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

        iterator = self.read(self.T_SEG_NAME, chunksize=30, filter_func=filter_func)
        id_colname = self.table_settings[self.T_SEG_NAME]['pkey']
        files = None  # do NOT instantiate a new DataFrame, otherwise append below coerces to
        # the type of the files DataFrame (object) and we want to preserve the db type (so first
        # iteration files is the first chunk read)
        columns = [id_colname] if sort_columns is None else list(sort_columns)
        if id_colname not in columns:
            columns.insert(0, id_colname)

        for data in iterator:
            # use only columns of interest (Id and other specified in sort_columns)
            data = pd.DataFrame({k: data[k] for k in columns})
            if files is None:
                files = data
            else:
                files = files.append(data)

        files.reset_index(drop=True, inplace=True)
        # files.info()
        self.mseed_ids = files
        self.sort_columns = sort_columns
        self.sort_ascending = sort_ascending
        self.sort()
        self._mseed_ids = self.mseed_ids.copy()

    def sort(self, by=None, ascending=None):
        """Sorts internally the ids according to by and ascending.
         :param by: same as pandas DataFrame sort_value's by arg.
            Example: 'A', ['A', 'B'], ...
            If None, self.sort_columns is used. If the latter is None, no sort will take place
            and the method silently exists
         :param ascending: same as pandas DataFrame sort_values's 'ascending' argument.
            Example: True, [True, False], ...
        """
        if by is None:
            by = self.sort_columns
        if ascending is None:
            ascending = self.sort_ascending
        if self.mseed_ids is not None and not self.mseed_ids.empty and by:
            self.mseed_ids.sort_values(by=by, ascending=ascending, inplace=True)

    def __len__(self):
        """returns the number of segments read from the segments table"""
        return len(self.mseed_ids)

    def get(self, segment_id, table, column_names_list=None):
        """
            Returns a pandas DataFrame representing the table row of the given segment id,
            for the table specified in table_name.
            filtered with only the given column names in column_names_list (if the latter is not
            None). The dataframe has in principle zero or one rows
            :param segment_id: the segment_id
            :param table: the name of the table mapped to the given DataFrame, or the sql alchemy
            table object (one of the values of self.tables dict)
            :param column_names_list: the columns of the table whose value has to be returned.
            if None, returns all columns from the given table
            :return: a pandas DataFrame reflecting the row(s) of the query
        """
        table_name = self.get_name_and_table(table)[0]
        where = self.join_t_seg_with([table_name])
        additional_where = (self.T_SEG.Id == segment_id)
        if where is None:  # in case table name == self.T_SEG_NAME
            where = additional_where
        else:
            where &= additional_where  # in any other case

        pddf = self.select([table_name], where)
        return pddf if column_names_list is None else \
            pddf[column_names_list]

    def filter(self, where):
        """FIXME: add doc!
        :param table_names: an iterable of table names
        :param where: an sqlalchemy where clause. Can be None (no where clause)
        :Example:
        dbh = DbHandler(db_uri)
        where1 = ~dbh.table(dbh.T_SEG).Id.in_([55, 56])
        where2 = dbh.table(dbh.T_EVT).Magnitude == 51.5
        dbh.filter(where1 & where2)
        """
        where_clause = None
        if where is not None:
            tables = []
            # detect the tables in the where clause
            for tbl in where._from_objects:  # pylint:disable=protected-access
                # tbl seems to be a different type than the tables stored in self.tables.values()
                # so use its name:
                tables.append(str(tbl.name))
            # join those tables with the foreign keys specified in the constructor:
            where_clause = self.join_t_seg_with(tables)  # joins with all tables, might be None
            if where_clause is None:  # we only have T_SEG in tables
                where_clause = where
            else:
                where_clause &= where

        # select the columns used for sort ordering. This also avoid to load all table data
        # so that most likely mseed binary data is skipped avoiding overhead
        cols = list(self.sort_columns)
        id_colname = self.table_settings[self.T_SEG_NAME]['pkey']
        if id_colname not in cols:
            cols.insert(0, id_colname)
        t_cols = [getattr(self.T_SEG, c) for c in cols]
        selectable = select(t_cols) if where_clause is None else select(t_cols).where(where_clause)

        parse_dates = [c for c in self.table_settings[self.T_SEG_NAME]['parse_dates']
                       if c in cols]
        tmp = self.read_sql_query(selectable, parse_dates= None if not parse_dates else parse_dates)
        self.mseed_ids = tmp.drop_duplicates(subset=[id_colname], inplace=True)
        self.sort()

        return self

    def reset_filter(self):
        """Resets a filter previously set with self.filter"""
        self.mseed_ids = self._mseed_ids.copy()

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
            for _, seg_sr in other_components.iterrows():
                stre = self.mseed(seg_sr)
                strm.traces.append(stre.traces[0])

        return strm

    def get_id(self, index):
        """Returns the database ID of the index-th item (=db table row)"""
        id_colname = self.table_settings[self.T_SEG_NAME]['pkey']
        return self.mseed_ids.iloc[index][id_colname]
