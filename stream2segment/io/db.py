from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas.io.sql as pdsql
from sqlalchemy.exc import OperationalError
import pandas as pd
import numpy as np
from sqlalchemy import BLOB
import os

"""
Module holding db io operations
"""


class DbHandler(object):
    """
    An object of this class is initialized with a database uri and manages IO-operations between db
    and data (stored as pandas DataFrame). This module uses SQLAlchemy which makes it possible to
    use any DB (e.g. postgresql, mysql, sqlite) supported by that library. Moreover, by means of
    sqlalchemy auto-map, tables in the database are stored and mapped to sql-alchemy table objects
    without the need of sqlalchemy Declarative's implementation and mantainance
    (http://docs.sqlalchemy.org/en/latest/orm/extensions/declarative/index.html).
    The sql alchemy tables can be accessed via this object which acts as a kind of a read-only dict:
        - self[table_name]
        - table_name in self
        - len(self)
        - for table_name in self: table = self[table_name]

    (If you want to force auto-mapping, call self.automap())

    In general, the user needs only to call read and write method on "default" tables:
    - self.read(table_name,..)
    - self.write(dataFrame, table_name)
    where table_name is one of the following:
    - self.tables.events
    - self.tables.segments
    - self.tables.logs

    Those methods get default settings stored for those default tables, and then pass them to two
    more "low-level" functions:
    - self.read_sql(df, table_name,...) and
    - self.to_sql(df, table_name...) [*]

    The two latter functions can be called for custom operations, but the user has to provide more
    parameters (e.g. a mandatory primary key for the table when writing)

    All write operations take care also to create the table in the database schema,
    if it does not exist. NOTE: **usually self.purge is called in the
    DataFrame prior to writing to get a DataFrame without those elements already
    present on the db table (by comparing with the table primary key which must be given and must be
    a DataFrame column)**.

    Pitfalls:
    - pandas to_sql (version 18) does not give the possibility to write primary keys. That's too bad
    cause later sqlalchemy auto-map won't read the table cause it NEEDS a primary key. The solution
    here is a hack copied from the internet which might be probably solved in future pandas versions
    - Foreign keys and complex stuff cannot be declared
    - We got rid of mantaining a complex model.py file with all our tables reflected to the db but
    still some DataTypes (e.g. BLOB) must be passed explicitly and thus some sort of customization
    is needed. That is why the "write_df", "purg_df" and "read_df" exist: for those tables we store
    settings internally so taht the user does not need to pass them all the time.
"""
    def __init__(self, db_uri):
        """
            :param: db_uri: the database uri, e.g. "sqlite:///" + filename
        """
        self.db_uri = db_uri
        # initialize self.tables, a dict

        class MyDict(dict):  # allow self.tables to have attributes (see below)
            pass
        self.tables = MyDict()

        # self.tables will be populated inside initdb when needed (i.e, lazily) with our
        # sqlAlchemy table objects (using sqlAlchemy auto-mapping): this allows us to be free of
        # implementing our Base class and ORM in python, with the pain of keeping it updated with db
        # migrations.
        # However, a complete automation is not entirely possible. Pandas sql IO operations have to
        # be hacked to account for e.g., primary keys and other parameters.
        # These parameters are saved here below and they will be used when calling read_df, purge_df
        # and write_df, which are shorthand function for read, purge and write
        self.tables._metadata = {"events": {'pkey': '#EventID',
                                            'parse_dates': ['DataStartTime', 'DataEndTime',
                                                            'StartTime', 'EndTime', 'ArrivalTime']
                                            },
                                 "segments": {'pkey': 'Id',
                                              'dtype': {'Data': BLOB},
                                              'parse_dates': ['Time']
                                              },
                                 "logs": {'pkey': 'Time',
                                          'parse_dates': ['Time']
                                          },
                                 "classes": {'pkey': 'Id'}
                                 }
        # Now, we want to use read_df, purge_df, write_df with simplicity, for instance:
        # dbhandler.purge_df(dbhandler.tables.segments), dbhandler.purge_df(dbhandler.tables.events)...
        # To do that, we define custom attributes on self.tables (we can, as we override dict)
        for table_name in self.tables._metadata:
            setattr(self.tables, table_name, table_name)

        # engine, initialize once
        self.engine = create_engine(self.db_uri)
        # create tables from db (if any):
        self.automap()

        if "classes" not in self.tables:
            from stream2segment.classification import class_labels_df
            # write the class labels:
            # well, it should exist, but for practical reasons let's be sure: if_exist is specified
            self.write(class_labels_df, "classes", if_exists='fail')  # fail means: do nothing

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

        self.tables.update(Base.classes._data)  # overwriting existing keys

    def session(self):
        return Session(self.engine)

    def column(self, table_name, column_name):
        if table_name in self.tables:
            table = self.tables[table_name]
            try:
                return getattr(table, column_name)
            except AttributeError:
                raise ValueError("No such column '%s'" % column_name)
        raise ValueError("no such table: '%s'" % table_name)

    def purge(self, dframe, table_name, pkey_name=None):
        """
            Purges the given DataFrame of data already written on the database.
            :param: dframe: the DataFrame
            :type: dframe: pandas DataFrame
            :param: table_name: the name of the table mapped to the given DataFrame
            :type: table_name: string
            :pkey_name: the private key whereby to check if data is already on the database
            :type: pkey_name: string. Must be a column of the given DataFrame. NOTE **If table_name
            is one of the default tables registered on this object (as of April 2015,
            self.tables.segments, self.tables.events, self.tables.logs) and pkey_name is None or
            missing, it will be retrieved from internal settings. Otherwise it must be spceified
            explicitly**. FIXME: not implemented the case where the index is the primary key
            :return: a new DataFrame with the data not stored to the datbase according to pkey_name
            :rtype: pandas DataFrame
        """
        tables = self.tables
        if table_name in tables:
            if table_name in self.tables._metadata:  # one of the default tables:
                dframe = self._prepare_df(table_name, dframe)
                if pkey_name is None:  # use default one:
                    pkey_name = self.get_metadata(table_name)['pkey']

            session = Session(self.engine)
            column = getattr(tables[table_name], pkey_name)
            ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
            dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]
            session.close()

        return dframe

    def to_sql(self, dframe, table_name, pkey_name, index=False, index_label=None,
               if_exists='append', dtype=None):
        """
            Calls dframe.to_sql with given argument and self.engine. Contrarily to dframe.to_sql,
            there is no need to specify the engine and the schema. A table will be created if
            it does not exist and if_exist='append' (the default)
            From pandas.DataFrame.to_sql:
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
        #
        #

        table_exists = table_name in self.tables

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
            self._init_table(dframe, table_name, pkey_name, dtype)
            # Now, passing 'fail' to pandas method below would... fail, obviously, cause we just
            # created the table). Rename to 'append', cause now it's that in the user intentions
            if if_exists == 'fail':
                if_exists = 'append'

        if dframe is not None and not dframe.empty:
            # Rename 'replace' to 'append', as the former case wa handled above:
            if_exists = 'append' if if_exists == 'replace' else if_exists
            dframe.to_sql(table_name, self.engine, index=index, if_exists=if_exists,
                          index_label=index_label, dtype=dtype)

    def drop_table(self, table_name):
        if table_name in self.tables:
            self.tables[table_name].drop(self.engine)
            self.automap()

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
            :param drop_and_create_if_exist: self-explanatory. Defaults to False
            :param replace_all: boolean
            :return: the sqlalchemy table or a KeyError is raised if such a table could not be
            created
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

    def write(self, dframe, table_name, if_exists='append'):
        """
        Calls self.to_sql(dframe, table_name). NOTE: **table_name must be one of the default tables
        registered on this class (as of April 2015, self.tables.segments, self.tables.events,
        self.tables.logs)**
        """
        dframe = self._prepare_df(table_name, dframe)
        pkey = self.get_metadata(table_name)['pkey']
        dtype = self.get_metadata(table_name).get('dtype', None)
        self.to_sql(dframe, table_name, pkey, if_exists=if_exists, dtype=dtype)

    def _prepare_df(self, table_name, dframe):
        """
            Prepares a default frame for purging or writing. Basically it creates an Id for
            all null Id's if table_name == self.tables.segments, and modifies the data frame.
            Returns the input dataframe at the end (potentially unmodified)
        """
        if table_name == self.tables.segments:
            pkey = self.get_metadata(table_name)['pkey']
            recalc = pkey not in dframe.columns
            if recalc:
                dframe.insert(0, pkey, None)
            else:
                recalc = pd.isnull(dframe[pkey]).any()

            if recalc:
                def myfunc(row):
                    if pd.isnull(row[pkey]):
                        row[pkey] = self.get_wav_id(row['#EventID'], row['#Network'],
                                                    row['Station'], row['Location'], row['Channel'],
                                                    row['DataStartTime'], row['DataEndTime'])
                    return row
                dframe = dframe.apply(myfunc, axis=1)

        return dframe

    def get_metadata(self, table_name):
        return self.tables._metadata.get(table_name, {})

    @staticmethod
    def get_wav_id(event_id, network, station, location, channel, start_time, end_time):
        """
            Returns a unique id from the given arguments. The hash of the tuple built from the
            arguments will be returned. No argument can be None
            :param: event_id: the event_id
            :type: event_id: string
            :param: network: the station network
            :type: network: string
            :param: station: the given station
            :type: station: string
            :param: location: the given location
            :type: location: string
            :param: channel: the given channel
            :type: channel: string
            :param: start_time: the wav start time
            :type: start_time: datetime object, or a string representing a datetime
                (e.g. datetime.isoformat())
            :param: end_time: the wav end time, or a string representing a datetime
                (e.g. datetime.isoformat())
            :type: end_time: datetime object
            :return: a unique integer denoting the given wav.
            Two wavs with the same argument have the same id
            :rtype: integer
        """
        start_time = start_time.isoformat()
        end_time = end_time.isoformat()
        val = (event_id, network, station, channel, location, start_time, end_time)
        if None in val:
            raise ValueError("No None value in get_wav_id")
        return hash(val)

    def read_sql(self, table_name, coerce_float=True, index_col=None, parse_dates=None,
                 columns=None, chunksize=None):
        """
            Calls pandas.read_sql_table. From their documentation:
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
        tables = self.tables
        if table_name not in tables:
            return pd.DataFrame()
        return pd.read_sql_table(table_name, self.engine, None, index_col, coerce_float,
                                 parse_dates, columns, chunksize)

    def read(self, table_name, chunksize=None):
        """
        Calls self.read_sql(table_name). NOTE: **table_name must be one of the default tables
        registered on this class (as of April 2015, self.tables.segments, self.tables.events,
        self.tables.logs)**
        :param table_name: Name of SQL table in database
        :type table_name: string
        :param chunksize: If specified, return an iterator where chunksize is the number of rows
        to include in each chunk.
        :type chunksize: int, default None
        """
        return self.read_sql(table_name, coerce_float=True,
                             index_col=None,
                             parse_dates=self.get_metadata(table_name).get('parse_dates', None),
                             columns=None,
                             chunksize=chunksize)  # FIXME!! parse dates!
    # implement mutable sequence: dbhandler.get(table_name), for table_name in dbhandler: ...

    def __len__(self):
        return self.tables.__len__()

    def __getitem__(self, table_name):
        return self.tables.__getitem__(table_name)

    def __iter__(self):
        return self.tables.__iter__()

    def __contains__(self, table_name):
        return self.tables.__contains__(table_name)

    def keys(self):
        return self.tables.keys()

    def values(self):
        return self.tables.values()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    dbio = DbHandler('sqlite:///./mydb.db')  # 'sqlite:///./mydb.db'
    obj = dbio.read_df(dbio.tables.segments)
    for o in obj:
        print "chunksize"
