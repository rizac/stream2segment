'''
modules for querying mseed objects from a database
Created on Apr 26, 2016

@author: riccardo
'''
from StringIO import StringIO
import pandas as pd
from obspy import read
from stream2segment.io import db
from sqlalchemy import func, distinct

class Reader(object):
    """
        Class reading and returning all mseed files in a given database. 
        An object of this class is initialized with a database URI: r= Reader(db_uri) and returns
        the mseed read from the given database.
        :Example:
        .. code:
            for i in xrange(len(r)):
                r.get(i)  # returns an obspy stream object on which you can call several methods,
                          # e.g. r.get(i).plot()
                r.get_raw(i)  # returns the raw bytes (string in python2)
    """
    def __init__(self, db_uri, keep_session_open=False):
        dbh = db.DbHandler(db_uri)
        iterator = dbh.read(dbh.tables.data, chunksize=10)
        id_col_name = 'Id'
        files = None  # do NOT instantiate a new DataFrame, otherwise append below coerces to
        # the type of files (object) and we want to preserve the db type (so first iteration
        # files is the first chunk read)
        for data in iterator:
            data = pd.DataFrame({id_col_name: data[id_col_name]})
            if files is None:
                files = data
            else:
                files = files.append(data)

        if files is None:
            files = pd.DataFrame(columns=[id_col_name])
        else:
            files.reset_index(drop=True, inplace=True)
        self.dbh = dbh
        self.files = files
        self.keep_session_open = keep_session_open
        classes_table_name = self.dbh.tables.classes
        self.classes_dataframe = self.dbh.read(classes_table_name)
        self.classes_dataframe.insert(len(self.classes_dataframe.columns), 'Count', 0)
        self._update_classes()

    def close(self):
        """closes the session. Does nothing if session_open was False in the constructor"""
        if self.keep_session_open:
            self.session.close()

    def session(self):
        if self.keep_session_open and not hasattr(self, 'session'):
            self.session = self.dbh.session()
        return self.session if self.keep_session_open else self.dbh.session()

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return (i for i in xrange(len(self)))

    def get_data_table(self):
        return self.dbh.tables[self.dbh.tables.data]

    def get_classes_table(self):
        return self.dbh.tables[self.dbh.tables.classes]

    def get_raw(self, index):
        table = self.get_data_table()
        sess = self.session()
        row = sess.query(table).filter(table.Id == int(self.files.iloc[index]['Id'])).first()
        if not self.keep_session_open:
            sess.close()
        return row.Data

    def get_class(self, index):
        table = self.get_data_table()
        sess = self.session()
        row = sess.query(table).filter(table.Id == int(self.files.iloc[index]['Id'])).first()
        if not self.keep_session_open:
            sess.close()
        return row.AnnotatedClassId

    def get(self, index):
        bytez = self.get_raw(index)
        return read(StringIO(bytez))

    def get_metadata(self, index):
        table = self.get_data_table()
        sess = self.session()
        row = sess.query(table).filter(table.Id == int(self.files.iloc[index]['Id'])).first()
        ret = {}
        for attr_name in row.__dict__:
            if attr_name[0] != "_" and attr_name != "Data":
                ret[attr_name] = getattr(row, attr_name)
        if not self.keep_session_open:
            sess.close()
        return ret

    def set_class(self, index, class_id):
        table = self.get_data_table()
        sess = self.session()
        row = sess.query(table).filter(table.Id == int(self.files.iloc[index]['Id'])).first()
        row.AnnotatedClassId = class_id
        sess.commit()
        sess.flush()  # FIXME: needed???
        if not self.keep_session_open:
            sess.close()
        self._update_classes()

    # get the list [(class_id, class_label, count), ...]
    def _update_classes(self):
        classes_dataframe = self.classes_dataframe
        table = self.get_data_table()
        sess = self.session()

        def countfunc(row):
            row['Count'] = sess.query(table).filter(table.AnnotatedClassId == row['Id']).count()
            return row

        self.classes_dataframe = classes_dataframe.apply(countfunc, axis=1)
        if not self.keep_session_open:
            sess.close()
