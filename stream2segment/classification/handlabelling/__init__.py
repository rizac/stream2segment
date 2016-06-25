from stream2segment.s2sio.db import DbHandler


class ClassAnnotator(object):
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

    def __init__(self, db_handler_or_db_uri):
        """
            Initializes a new DataManager via a given db_uri
            :param db_uri: the database uri, e.g. sqlite:///path_to_my_sqlite_file
            :param filter_func: a filter function taking as argument the DataFrame of segments
            read and returning a filtered DataFrame
        """
        if isinstance(db_handler_or_db_uri, DbHandler):
            self.dbhandler = db_handler_or_db_uri
        else:
            self.dbhandler = DbHandler(db_handler_or_db_uri)

        self._classes_dataframe = self.dbhandler.read(self.dbhandler.T_CLS)
        self._classes_dataframe.insert(len(self._classes_dataframe.columns), 'Count', 0)
        self.update_classes()

    def get_classes_df(self):
        """Returns the pandas DataFrame representing the classes. The DataFrame is read once
        in the constructor and updated here with a column 'Count' which counts the instances per
        class. Column names might vary across versions but in principle their names are 'Id',
        'Label' and 'Description' (plus the aforementioned 'Count')"""
        return self._classes_dataframe

    def get_class(self, segment_id, as_annotated_class=True):
        """Returns the class id (integer) of the index-th item (=db table row).
        :param as_annotated_class: if True (the default), returns the value of the column
            of the annotated class id (representing the manually annotated class), otherwise the
            column specifying the class id (representing the class id as the result of some
            algorithm, e.g. statistical classifier)
        """
        dbh = self.dbhandler
        where = (dbh.column(dbh.T_SEG, "Id") == segment_id)
        pddf = dbh.select([dbh.T_SEG], where)

        return pddf.iloc[0][self.annotated_class_id_colname]

#         row = self.get(index, self.T_SEG)
#         att_name = self.annotated_class_id_colname if as_annotated_class else self.class_id_colname
#         return row.iloc[0][att_name]

    def set_class(self, segment_id, class_id, as_annotated_class=True):
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
        dbh = self.dbhandler
        # store the old class id and the new one:
        # NOTE: we absolutely need a with statement to keep the session open
        with dbh.session_scope() as session:
            id_colname = dbh.table_settings[dbh.T_SEG]['pkey']
            row = session.query(dbh.tables[dbh.T_SEG])
            row = row.filter(dbh.column(dbh.T_SEG, id_colname) == segment_id).first()
            # row = self.get_row(index)
            att_name = self.annotated_class_id_colname if as_annotated_class else \
                self.class_id_colname
            setattr(row, att_name, class_id)
        self.update_classes()

    # get the list [(class_id, class_label, count), ...]
    def update_classes(self):
        dbh = self.dbhandler
        session = dbh.session()
        classes_dataframe = self.get_classes_df()
        table = dbh.table(dbh.T_SEG)
        id_colname = dbh.table_settings[dbh.T_CLS]['pkey']

        def countfunc(row):
            row['Count'] = session.query(table).filter(table.AnnotatedClassId == row[id_colname]).count()
            return row

        self._classes_dataframe = classes_dataframe.apply(countfunc, axis=1)
        session.close()
