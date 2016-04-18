from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pandas.io.sql as pdsql
from sqlalchemy.exc import OperationalError
# import sqlalchemy.types as sql_types
from sqlalchemy import BLOB
from pandas.io.sql import SQLTable

fileout = "sqlite:///./mydb.db"


class DbHandler(object):
    def __init__(self, db_uri=fileout, init_db=False):
        self.db_uri = db_uri
        if init_db:
            self.init_db()

    def init_db(self):

        Base = automap_base()

        # engine, suppose it has two tables 'user' and 'address' set up
        if not hasattr(self, 'engine'):
            # engine, suppose it has two tables 'user' and 'address' set up
            self. engine = create_engine(self.db_uri)

        # reflect the tables
        # Note: ONLY TABLES WITH PRIMARY KEYS will be mapped!
        Base.prepare(self.engine, reflect=True)

        # mapped classes are now created with names by default
        # matching that of the table name.
        # User = Base.classes.user
        # Address = Base.classes.address

        # self.session = Session(self.engine)

        # rudimentary relationships are produced
        # session.add(Address(email_address="foo@bar.com", user=User(name="foo")))
        # session.commit()

        # collection-based relationships are by default named
        # "<classname>_collection"
        # print (u1.address_collection)
        self.tables = Base.classes._data

    def write(self, dframe, table_name, pkey_name, index=False, if_exists='append'):
        if not dframe.empty:
            self.init_db()
            self.init_table(dframe, table_name, pkey_name)
            dframe.to_sql(table_name, self.engine, index=index, if_exists=if_exists, 
                          # index_label = pkey
                          dtype={'Data': BLOB})
    # , index=True, index_label=None, chunksize=None, dtype=None)

    def purge(self, dframe, table_name, pkey_name):
        self.init_db()
        tables = self.tables
        if table_name in tables:
            session = Session(self.engine)
            column = getattr(tables[table_name], pkey_name)
            ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
            dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]
            session.close()

        return dframe

    def init_table(self, dframe, table_name, pkey_name):
        self.init_db()
        engine, tables = self.engine, self.tables
        if table_name not in tables:
            # problem: pandas does not have a direct way to assign a primary key
            # so the hack (waiting for this feature to be implemented, surely) is taken from:
            # http://stackoverflow.com/questions/30867390/python-pandas-to-sql-how-to-create-a-table-with-a-primary-key
            schema = pdsql.get_schema(dframe,
                                      table_name,
                                      con=engine,
                                      keys=[pkey_name],
                                      dtype={'Data': BLOB})
            # uncomment this line, we shouldn't have a table UNLESS we wrote it
            # without the workaround above
            try:
                engine.execute('DROP TABLE ' + table_name + ';')
            except OperationalError as _:
                pass
            engine.execute(schema)
            self.init_db()  # updates the tables

        return self.tables[table_name]

    def purge_old(self, dframe, table_name, pkey_name):
        self.init_db()
        session, engine, tables = self.session, self.engine, self.tables
        if table_name not in tables:
            # problem: pandas does not have a direct way to assign a primary key
            # so the hack (waiting for this feature to be implemented, surely) is taken from:
            # http://stackoverflow.com/questions/30867390/python-pandas-to-sql-how-to-create-a-table-with-a-primary-key
            schema = pdsql.get_schema(dframe,
                                      table_name,
                                      con=engine,
                                      keys=[pkey_name],
                                      dtype={'Data': BLOB})
            # uncomment this line, we shouldn't have a table UNLESS we wrote it
            # without the workaround above
            try:
                engine.execute('DROP TABLE ' + table_name + ';')
            except OperationalError as oerr:
                pass
            engine.execute(schema)
        else:
            column = getattr(tables[table_name], pkey_name)
            ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
            dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]

        return dframe

#     def write_dframe(engine, dframe, table_name):
#         if not dframe.empty:
#             dframe.to_sql(table_name, engine, index=False, if_exists='append')
# 
# 
# 
# def get_connection():
#     engine = create_engine('sqlite:///'+fileout)
#     connection = engine.connect()
#     return connection
# 
# 
# def db_check(database_uri, table_name=None):
#     db = create_engine(database_uri)
#     try:
#         db.connect()
#         db.execute("SELECT 1;")
#         return True
#     except OperationalError:
#         # Switch database component of the uri
#         return False
#
# def write_events(dframe, **kwargs):
#     write_data(dframe, "events", '#EventID', **kwargs)
# 
# 
# def write_wavs(dframe, **kwargs):
#     write_data(dframe, "data", 'id', **kwargs)
# 
# 
# def purge_data(dframe, table_name, pkey_name):
#     session, engine, classes = get_session()
#     if table_name not in classes:
#         # problem: pandas does not have a direct way to assign a primary key
#         # so the hack (waiting for this feature to be implemented, surely) is taken from:
#         # http://stackoverflow.com/questions/30867390/python-pandas-to-sql-how-to-create-a-table-with-a-primary-key
#         schema = pdsql.get_schema(dframe,
#                                   table_name,
#                                   con=engine,
#                                   keys=[pkey_name])
#         # uncomment this line, we shouldn't have a table UNLESS we wrote it
#         # without the workaround above
#         engine.execute('DROP TABLE ' + table_name + ';')
#         engine.execute(schema)
#     else:
#         column = getattr(classes[table_name], pkey_name)
#         ids = session.query(column).filter(column.in_(dframe[pkey_name].values)).all()
#         dframe = dframe[~dframe[pkey_name].isin([i[0] for i in ids])]
# 
#     if not dframe.empty:
#         dframe.to_sql(table_name, engine, index=False, if_exists='append')

#     if connection is None:
#         connection = get_connection()
#     result = connection.execute("select * from events")
#     for row in result:
#         j = 9
