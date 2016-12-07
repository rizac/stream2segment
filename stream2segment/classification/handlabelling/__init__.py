from stream2segment.io.db import DbHandler
from stream2segment.classification import UNKNOWN_CLASS_ID


class ClassAnnotator(object):
    """
        Class managing the annotation of class label on a downloaded database.
        An object of this class is initialized with a database handler
        s2sio.db.DbHandler or an URI (string) and allows setting / getting the labels
        for building training sets suitable for further machine learning processes
    """
    # the db column name of the annotated class_id:
    hl_class_id_colname = 'ClassIdHandLabeled'
    # the db column name of the (classified) class_id:
    class_id_colname = 'ClassId'

    def __init__(self, db_handler_or_db_uri):
        """
            Initializes a new DataManager via a given db_handler (s2sio.db.Dbhandler) or a
            db_uri (string)
            :param db_uri: the DbHandler or the database uri, e.g.
                "sqlite:///path_to_my_sqlite_file"
        """
        if isinstance(db_handler_or_db_uri, DbHandler):
            self.dbhandler = db_handler_or_db_uri
        else:
            self.dbhandler = DbHandler(db_handler_or_db_uri)

        self._classes_dataframe = self.dbhandler.read(self.dbhandler.T_CLS)
        self._classes_dataframe.insert(len(self._classes_dataframe.columns), 'Count', 0)
        self._classes_dataframe_needs_update = True

    def get_classes_df(self):
        """Returns the pandas DataFrame representing the classes. The DataFrame is read once
        in the constructor and updated here with a column 'Count' which counts the instances per
        class. Column names might vary across versions but in principle their names are 'Id',
        'Label' and 'Description' (plus the aforementioned 'Count')"""
        if self._classes_dataframe_needs_update:
            self._classes_dataframe_needs_update = False
            self.update_classes_count()
        return self._classes_dataframe

    def get_class(self, segment_id):
        """Returns the class id (integer) of the index-th item (=db table row).
        :param segment_id. The id (primary key) of a given segment. This method is mainly used
            in conjunction to a s2sio.db.ListReader object (which might have been passed to
            the constructor, as it extends s2sio.db.DbHandler). A ListReader is a DbHandler which
            reads and returns segments by their id in a python-list fashion
        """
        dbh = self.dbhandler
        where = (dbh.column(dbh.T_SEG, "Id") == segment_id)
        pddf = dbh.select([dbh.T_SEG], where)

        return pddf.iloc[0][self.class_id_colname]

    def set_class(self, segment_id, class_id):
        """
            Sets the class of the index-th item (=db table row).
            :param segment_id. The id (primary key) of a given segment. This method is mainly used
            in conjunction to a s2sio.db.ListReader object (which might have been passed to
            the constructor, as it extends s2sio.db.DbHandler). A ListReader is a DbHandler which
            reads and returns segments by their id in a python-list fashion
            :param class_id: one of the classes id. To get them, call self.get_classes()['Id']
            (returns a pandas Series object)
            :return: self.get_classes_df (updated with the new 'Count' column)
        """
        dbh = self.dbhandler
        # store the old class id and the new one:
        # NOTE: we absolutely need a with statement to keep the session open
        with dbh.session_scope() as session:
            id_colname = dbh.table_settings[dbh.T_SEG]['pkey']
            row = session.query(dbh.tables[dbh.T_SEG])
            row = row.filter(dbh.column(dbh.T_SEG, id_colname) == segment_id).first()
            # set new class id:
            setattr(row, self.class_id_colname, class_id)
            # set manual hand labelled to true, unless we set to "unknown" in which
            # case we reset the hand labelling to false (as if we didn't annotated it)
            setattr(row, self.hl_class_id_colname, class_id != UNKNOWN_CLASS_ID)

    def update_classes_count(self):
        """internal method used when calling set_class to update the counts of the given
        classes"""
        dbh = self.dbhandler
        session = dbh.session()
        classes_dataframe = self.get_classes_df()
        table = dbh.table(dbh.T_SEG)
        id_colname = dbh.table_settings[dbh.T_CLS]['pkey']

        def countfunc(row):
            """function setting the 'Count' column for each DataFrame row"""
            row['Count'] = session.query(table).filter(table.ClassId ==
                                                       row[id_colname]).count()
            return row

        self._classes_dataframe = classes_dataframe.apply(countfunc, axis=1)
        session.close()
