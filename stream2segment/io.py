from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas.io.sql as pdsql
from sqlalchemy.exc import OperationalError
import pandas as pd
from sqlalchemy import BLOB
import os

"""
 This module holds the DbHandler class, which manages the synchronization between db and data saving
 only the necessary stuff, and creating tables if they do not exist. This class uses sql alchemy's
 automap_base which does not require any ORM declaration in a separate python module (nor its
 maintainance), nor migration scripts: it simply looks at the database stored at the given location
 and returns the tables in form of slq alchemy objects (if any): just specify the path and
 everything is fine.
 Drawbacks: foreign keys and complex stuff cannot be declared
"""


class DbHandler(object):
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
                                 "data": {'pkey': 'Id',
                                          'dtype': {'Data': BLOB},
                                          'parse_dates': ['Time']
                                          },
                                 "logs": {'pkey': 'Time',
                                          'parse_dates': ['Time']
                                          }
                                 }
        # Now, we want to use read_df, purge_df, write_df with simplicity, for instance:
        # dbhandler.purge_df(dbhandler.tables.data), dbhandler.purge_df(dbhandler.tables.events)...
        # To do that, we define custom attributes on self.tables (we can, as we override dict)
        for table_name in self.tables._metadata:
            setattr(self.tables, table_name, table_name)

    def init_db(self):
        """
            Initializes the database connection retrieving automatically the tables stored there
            (those with primary keys) and storing in a class attribute tables (dict of table
            names mapped to relative sqlalchemy table)
        """
        # Code copied from here:
        # http://docs.sqlalchemy.org/en/latest/orm/extensions/automap.html#basic-use

        # engine, initialize once
        if not hasattr(self, 'engine'):
            self.engine = create_engine(self.db_uri)

        Base = automap_base()

        # reflect the tables
        # Note: ONLY TABLES WITH PRIMARY KEYS will be mapped!
        Base.prepare(self.engine, reflect=True)

        # mapped classes are now created with names by default
        # matching that of the table name.
        # User = Base.classes.events
        # Address = Base.classes.data
        # store them in an attribute (dict of table_name [string]: sqlalchemy table)
        # It might be empty if database does not exist

        self.tables.update(Base.classes._data)  # overwriting existing keys

    def purge(self, dframe, table_name, pkey_name):
        """
            Purges the given DataFrame of data already written on the database.
            :param: dframe: the DataFrame
            :type: dframe: pandas DataFrame
            :param: table_name: the name of the table mapped to the given DataFrame
            :type: table_name: string
            :pkey_name: the private key whereby to check if data is already on the database
            :type: pkey_name: string. Must be a column of the given DataFrame. FIXME: not
            implemented the case where the index is the primary key
            :return: a new DataFrame with the data not stored to the datbase according to pkey_name
            :rtype: pandas DataFrame
        """
        self.init_db()
        tables = self.tables
        if table_name in tables:
            session = Session(self.engine)
            column = getattr(tables[table_name], pkey_name)
            ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
            dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]
            session.close()

        return dframe

    def write(self, dframe, table_name, pkey_name, index=False, if_exists='append', dtype=None):
        """
            Writes the given pandas data frame to the table with given table_name. If such a table
            does not exist, and the argument if_exist='append' (the default), then
            the table will be first created
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
                - fail: If table exists, do nothing.
                - replace: If table exists, drop it, recreate it, and insert data.
                - append: If table exists, insert data. Create if does not exist.
            If append, then the table will be created if it does not exist
            :type: if_exists: string
            :param: index : boolean, default False. Write DataFrame index as a column.
            :type: index: boolean
            :param: dtype: dict of column name to SQL type, default None. Optional specifying the
            datatype for columns. The SQL type should be a SQLAlchemy type, or a string for sqlite3
            fallback connection. Example: dtype={'data': sqlalchemy.BLOB}
            :type: dtype: dict or None
            :return: the sqlalchemy table or a KeyError is raised if such a table could not be
            created
        """
        if dframe is not None and not dframe.empty:
            if if_exists == 'append':
                self._init_table(dframe, table_name, pkey_name, dtype)
            else:
                self.init_db()  # in case it was not called (_init_table above does it)
            dframe.to_sql(table_name, self.engine, index=index, if_exists=if_exists,
                          index_label=pkey_name, dtype=dtype)

    def _init_table(self, dframe, table_name, pkey_name, dtype=None):
        """
            Re-updates the internal tables attribute by creating the given table if a table with a
            given table_name does not exist on the database
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
            :return: the sqlalchemy table or a KeyError is raised if such a table could not be
            created
        """
        self.init_db()
        engine, tables = self.engine, self.tables
        if table_name not in tables:
            # problem: pandas does not have a direct way to assign a primary key
            # so the hack (waiting for this feature to be implemented, surely) is taken from:
            # http://stackoverflow.com/questions/30867390/python-pandas-to-sql-how-to-create-a-table-with-a-primary-key
            schema = pdsql.get_schema(frame=dframe,
                                      name=table_name,
                                      con=self.engine,  # Using SQLAlchemy makes it possible to use
                                      # any DB supported by that library
                                      keys=[pkey_name],
                                      dtype=dtype)
            # the following is for safety, we shouldn't have a table UNLESS we wrote it
            # without the workaround above:
            try:
                engine.execute('DROP TABLE ' + table_name + ';')
            except OperationalError as _:  # table does not exist? whatever error? pass
                pass
            # execute the schema (CREATE TABLE etcetera)
            engine.execute(schema)
            self.init_db()  # updates the tables

        return self.tables[table_name]

    def purge_df(self, table_name, dframe):
        """Writes dframe to database, using internal settings (so that arguments of self.write
        default to internal settings according to table)"""
        dframe = self._prepare_df(table_name, dframe)
        return self.purge(dframe, table_name, self.get_metadata(table_name)['pkey'])

    def write_df(self, table_name, dframe):
        dframe = self._prepare_df(table_name, dframe)
        pkey = self.get_metadata(table_name)['pkey']
        dtype = self.get_metadata(table_name).get('dtype', None)
        self.write(dframe, table_name, pkey, dtype=dtype)

    def _prepare_df(self, table_name, dframe):
        if table_name == self.tables.data:
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
                dframe.apply(myfunc, axis=1)

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

    def read(self, table_name, coerce_float=True, index_col=None, parse_dates=None, columns=None,
             chunksize=None):
        """
            Read SQL database table into a DataFrame. Calls pandas read_sql_table
            :type table_name: string
            :param table_name: Name of SQL table in database

            :type index_col: string or list of strings, optional, default: None
            :param index_col: Column(s) to set as index(MultiIndex)
            :type coerce_float: boolean, default True
            :param  coerce_float: Attempt to convert values to non-string, non-numeric objects (like
            decimal.Decimal) to floating point. Can result in loss of Precision.
            :type parse_dates: list or dict, default: None
            :param  parse_dates:
                - List of column names to parse as dates
                - Dict of {column_name: format string} where format string is strftime compatible in
                case of parsing string times or is one of (D, s, ns, ms, us) in case of parsing
                integer timestamps
                - Dict of {column_name: arg dict}, where the arg dict corresponds to the keyword
                arguments of pandas.to_datetime() Especially useful with databases without native
                Datetime support, such as SQLite
            :type columns: list, default: None
            :param columns: List of column names to select from sql table
            :type chunksize: int, default None
            :param chunksize: If specified, return an iterator where chunksize is the number of rows
            to include in each chunk.
            :return a pandas DataFrame (empty in case table not found)
        """
        self.init_db()
        tables = self.tables
        if table_name not in tables:
            return pd.DataFrame()
        return pd.read_sql_table(table_name, self.engine, None, index_col, coerce_float,
                                 parse_dates, columns, chunksize)

    def read_df(self, table_name, chunksize=None):
        """
         Read SQL database table into a DataFrame. Calls pandas read_sql_table
         :param table: one of the class attributes of this class starting with "tbl_" and
         identifying a specific datbase table (with all necessary settings stored and thus not
         to be remembered as in self.read): E.g., tbl_logs, tbl_data, tbl_events
         """
        return self.read(table_name, coerce_float=True,
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


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    dbio = DbHandler('sqlite:///./mydb.db')  # 'sqlite:///./mydb.db'
    dbio.init_db()
    obj = dbio.read_df(dbio.tables.data)
    for o in obj:
        print "chunksize"
