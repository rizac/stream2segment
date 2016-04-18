from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas.io.sql as pdsql
from sqlalchemy.exc import OperationalError
# from sqlalchemy import BLOB
# from pandas.io.sql import SQLTable

fileout = "sqlite:///./mydb.db"


class DbHandler(object):
    def __init__(self, db_uri=fileout):
        """
            :param: db_uri: the database uri, e.g. "sqlite:///" + filename
        """
        self.db_uri = db_uri

    def init_db(self):
        """
            Initializes the database connection retrieving automatically the tables stored there
            (those with primary keys) and storing in a class attribute tables (dict of table
            names mapped to relative sqlalchemy table)
        """
        # Code copied from here:
        # http://docs.sqlalchemy.org/en/latest/orm/extensions/automap.html#basic-use
        Base = automap_base()

        # engine, initialize once
        if not hasattr(self, 'engine'):
            self. engine = create_engine(self.db_uri)

        # reflect the tables
        # Note: ONLY TABLES WITH PRIMARY KEYS will be mapped!
        Base.prepare(self.engine, reflect=True)

        # mapped classes are now created with names by default
        # matching that of the table name.
        # User = Base.classes.events
        # Address = Base.classes.data
        # store them in an attribute (dict of table_name [string]: sqlalchemy table)
        # It might be empty if database does not exist
        self.tables = Base.classes._data

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
        if not dframe.empty:
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
