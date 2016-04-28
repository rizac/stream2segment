'''
modules for querying mseed objects from a database
Created on Apr 26, 2016

@author: riccardo
'''
from StringIO import StringIO
import pandas as pd
from obspy import read
from stream2segment.io import db


class DataManager(object):
    """
        Class managing all data (mseed files) stored in a given database, and their classes.
        An object of this class is initialized with a database URI: r= Reader(db_uri) and returns
        the mseed read from the given database.
        :Example:
        .. code:
            r= Reader(db_uri)
            for i in xrange(len(r)):
                r.get(i)  # returns an obspy stream object on which you can call several methods,
                          # e.g. r.get(i).plot()
                r.get_data(i, raw=True)  # returns raw bytes (string in python2)
                r.get_data(i)  # returns the obspy stream object
                r.set_class(i, j) # sets the class of the i-th instance
                # j must be one of the values of the 'Id' columns of
                r.set_class(i, j, as_annotated_class=False) # sets the class of the i-th instance
                # but under the column 'class id' not 'annotated class id'
                # j must be one of the values of the 'Id' columns of
                r.get_classes()  # get dataframe of all classes, with columns 'Id' 'Label',
                                     # 'Description', 'Count'
    """
    # the db column name of the Id:
    id_colname = 'Id'
    # the db column name of the annotated class_id:
    annotated_class_id_colname = 'AnnotatedClassId'
    # the db column name of the (classified) class_id:
    class_id_colname = 'ClassId'

    def __init__(self, db_uri):
        self.db_uri = db_uri
        dbh = db.DbHandler(db_uri)
        iterator = dbh.read(dbh.tables.data, chunksize=10)
        id_col_name = self.id_colname
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
        classes_table_name = self.dbh.tables.classes
        self._classes_dataframe = self.dbh.read(classes_table_name)
        self._classes_dataframe.insert(len(self._classes_dataframe.columns), 'Count', 0)
        self._update_classes(self.session(), close_session=True)

    def session(self):
        return self.dbh.session()

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return (i for i in xrange(len(self)))

    def get_data_table(self):
        return self.dbh.tables[self.dbh.tables.data]

    def get_classes(self):
        """Returns the pandas DataFrame representing the classes. The dataframe is read once
        in the constructor and updated here with a column 'Count' which counts the instances per
        class"""
        return self._classes_dataframe

    def get_id(self, index):
        """Returns the database ID of the index-th mseed"""
        return self.files.iloc[index][self.id_colname]

    def get_row(self, index, session=None):
        """
            :param session: either None (the default) in which case a session will be opened and
            closed before returning, or a sql alchemy session object (see self.session()). In
            the latter case the session must be closed manually
            :param columns:
            :return: the database row identified by the index-th mseed (in the order returned when
            reading all entries for the data table in the constructor). For accessing its values
            use getattr(row, column_name). For listing the database columns, use a loop on
            row.__dict__ but remember that sqlalchemy stores internal attributes (those starting
            with underscore for instance)
        """
        sess = self.session() if session is None else session
        table = self.get_data_table()
        row = sess.query(table).filter(table.Id == int(self.get_id(index))).first()
        if session is None:
            sess.close()
        return row

    def get_data(self, index, raw=False, session=None):
        """
            Returns the data of the index-th mseed
            :param index: the index
            :param raw: (defaults True) whether to return a raw sequence of bytes
            (string in python2) or an obspy stream (the default)
        """
        row = self.get_row(index, session)
        bytez = row.Data
        if raw:
            return bytes
        return read(StringIO(bytez))

    def get_metadata(self, index, session=None):
        """
            Returns the metadata (all columns except 'Data' column) of the index-th mseed
            :param index: the mseed index
            :return: a dict with keys reflecting the database table columns
        """
        row = self.get_row(index, session)
        ret = {}
        for attr_name in row.__dict__:
            if attr_name[0] != "_" and attr_name != "Data":
                ret[attr_name] = getattr(row, attr_name)
        return ret

    def get_class(self, index, as_annotated_class=True, session=None):
        """Returns the class id (integer) of the index-th mseed.
        :param as_annotated_class: if True (the default) returns the value of the column
        of the annotated class id (i.e., manually annotated), otherwise the column specifying the
        class id (usually as result from some algorithm, e.g. statistical classifier)
        """
        row = self.get_row(index, session)
        att_name = self.annotated_class_id_colname if as_annotated_class else self.class_id_colname
        return getattr(row, att_name)

    def set_class(self, index, class_id, as_annotated_class=True, session=None):
        """
            Sets the class of the index-th mseed file. Note: the column of the mseed which will be
            modified is 'AnnotatedClassId', not 'ClassId' (the latter is supposed to be used as
            a classifier outcome)
            :param index: the mseed index
            :param class_id: one of the classes id. To get them, call self.get_classes()['Id']
            (returns a pandas Series object)
            :param as_annotated_class: if True (the default), sets the value of the column
            of the annotated class id (i.e., manually annotated), otherwise the column specifying
            the class id (usually as result from some algorithm, e.g. statistical classifier)
        """
        sess = session
        if sess is None:
            sess = self.session()  # need to create one so that get_row below does not close it!
        row = self.get_row(index, sess)
        att_name = self.annotated_class_id_colname if as_annotated_class else self.class_id_colname
        setattr(row, att_name, class_id)
        sess.commit()
        sess.flush()  # FIXME: needed???
        sess.close()
        self._update_classes(sess, close_session=session is None)

    # get the list [(class_id, class_label, count), ...]
    def _update_classes(self, session, close_session):
        classes_dataframe = self.get_classes()
        table = self.get_data_table()

        def countfunc(row):
            row['Count'] = session.query(table).filter(table.AnnotatedClassId == row['Id']).count()
            return row

        self._classes_dataframe = classes_dataframe.apply(countfunc, axis=1)
        if close_session:
            session.close()
