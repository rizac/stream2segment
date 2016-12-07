'''

Utilities for converting from pandas DataFrames to sqlalchemy tables objects
(according to models.py)
Basically, most of these functions are copied and pasted from pandas.io.sql.SqlTable
Created on Jul 17, 2016

@author: riccardo
'''
from __future__ import division
from datetime import datetime, date
from pandas.io.sql import _handle_date_column
from pandas.types.api import DatetimeTZDtype
import numpy as np
# pandas zip seems a wrapper around itertools.izip (generator instead than list):
from pandas.compat import (lzip, map, zip, raise_with_traceback,
                           string_types, text_type)
# is this below the same as pd.isnull? For safety we leave it like it is (the line is imported
# from pandas.io.sql and used in one of the copied methods below)
from pandas.core.common import isnull
# but we need also pd.isnull so we import it like this for safety:
from pandas import isnull as pd_isnull
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import and_
from pandas import to_numeric
import pandas as pd
from sqlalchemy.engine import create_engine
from itertools import cycle


def _get_dtype(sqltype):
    """Converts a sql type to a numpy type. Copied from pandas.io.sql"""
    from sqlalchemy.types import (Integer, Float, Boolean, DateTime,
                                  Date, TIMESTAMP)

    if isinstance(sqltype, Float):
        return float
    elif isinstance(sqltype, Integer):
        # TODO: Refine integer size.
        return np.dtype('int64')
    elif isinstance(sqltype, TIMESTAMP):
        # we have a timezone capable type
        if not sqltype.timezone:
            return datetime
        return DatetimeTZDtype
    elif isinstance(sqltype, DateTime):
        # Caution: np.datetime64 is also a subclass of np.number.
        return datetime
    elif isinstance(sqltype, Date):
        return date
    elif isinstance(sqltype, Boolean):
        return bool
    return object


def get_cols(table, primary_key_only=False):
    """table is either the table class or a table instance"""
    cols = table.__table__.columns
    return [c for c in cols if c.primary_key] if primary_key_only else cols


def get_col_names(table):
    """table is either the table class or a table instance"""
    return get_cols(table).keys()


def get_non_nullable_cols(table, dataframe):
    """
        Returns the dataframe column names which have a corresponding table attribute
        (reflecting the db table column) which has been set to non-nullable
    """
    non_nullable_cols = []
    dframe_cols = dataframe.columns
    for col in get_cols(table):
        if not col.nullable and col.key in dframe_cols:
            non_nullable_cols.append(col.key)
    return non_nullable_cols


def harmonize_rows(table, dataframe, inplace=True):
    """Make the DataFrame's rows align with the SQL table column nullable value.
    That is, removes the dataframe rows which are NA (None, NaN or NaT) for those values
    corresponding to `table` columns which were set to not be Null (nullable=False).
    Non nullable table attributes (reflecting db table columns) not present in dataframe columns
    are not acounted for: in other words, the non-nullable condition on the dataframe is set for
    those columns only which have a corresponding name in any of the table attributes.
    Consider calling `harmonize_cols` first to make sure the column values
    align with the table column types
    :param inplace: argument to be passed to pandas dropna
    """
    non_nullable_cols = get_non_nullable_cols(table, dataframe)
    if non_nullable_cols:
        dataframe.dropna(subset=[non_nullable_cols], inplace=inplace)
    return dataframe


def harmonize_columns(table, dataframe, parse_dates=None):
    """Make the DataFrame's column types align with the SQL table
    column types. Returns a new dataframe with "correct" types (according to table)
    Columns which are *not* shared with table columns (assuming dataframe columns are strings)
    are left as they are.
    The returned dataframe row numbers is not modified
    """
    _, dfr = _harmonize_columns(table, dataframe, parse_dates)
    return dfr


def _harmonize_columns(table, dataframe, parse_dates=None):
    """
    Copied and modified from pandas.io.sql:
    Make the DataFrame's column types align with the SQL table
    column types. The original dataframe dtypes and values MIGHT be modified in place!

    Modified because it uses pandas.to_numeric when df.as_type fails. It
    should silently converts all non-numeric non-date values to NaN and NaT respectively,
    the only drawback is that int types are float64 in case. This should not be a problem
    for conversion to db's. And in any case is better than having the original type, when
    it was, e.g., object or something else

    Original pandas doc:

    Need to work around limited NA value support. Floats are always
    fine, ints must always be floats if there are Null values.
    Booleans are hard because converting bool column with None replaces
    all Nones with false. Therefore only convert bool if there are no
    NA values.
    Datetimes should already be converted to np.datetime64 if supported,
    but here we also force conversion if required
    by means of parse_dates (a list of strings denoting the name of the additional columns
    to be parsed as dates. None by default)
    """
    # Note by me: _handle_date_column calls pd.to_datetime which coerces invalid dates to NaT
    # However, astype raises Errors, so we replace the "astype" with to_numeric if the former
    # fails

    # handle non-list entries for parse_dates gracefully
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []

    if not hasattr(parse_dates, '__iter__'):
        parse_dates = [parse_dates]

    column_names = []  # added by me
    for sql_col in get_cols(table):
        col_name = sql_col.name
        try:
            df_col = dataframe[col_name]
            # the type the dataframe column should have
            col_type = _get_dtype(sql_col.type)

            if (col_type is datetime or col_type is date or
                    col_type is DatetimeTZDtype):
                dataframe[col_name] = _handle_date_column(df_col)

            elif col_type is float:
                # floats support NA, can always convert!
                # BUT: if we have non-numeric non-NaN values, this fails, so
                # we fall back to pd.to_numeric
                try:
                    dataframe[col_name] = df_col.astype(col_type, copy=False)
                except ValueError:
                    # failed, use to_numeric coercing to None on errors
                    # this also sets the column dtype to float64
                    dataframe[col_name] = pd.to_numeric(df_col, errors='coerce')
            elif col_type is np.dtype('int64'):
                # integers. Try with normal way. The original code checked NaNs like this:
                # if len(df_col) == df_col.count():
                # but this raises on e.g., non numeric strings, which we want to
                # convert to NaN. So, try the "normal way":
                try:
                    dataframe[col_name] = df_col.astype(col_type, copy=False)
                except ValueError:
                    # failed, use to_numeric coercing to None on errors
                    # this also sets the column dtype to float64
                    dataframe[col_name] = pd.to_numeric(df_col, errors='coerce')
            elif col_type is bool:
                # boolean seems to convert without errors but
                # converts NaN's to True, None to False. So, for preserving None's and NaN:
                bool_col = df_col.astype(col_type, copy=True)
                bool_col[pd.isnull(df_col)] = None
                dataframe[col_name] = bool_col

            # OLD CODE (commented out):
#             elif len(df_col) == df_col.count():
#                 # No NA values, can convert ints and bools
#                 if col_type is np.dtype('int64') or col_type is bool:
#                     dataframe[col_name] = df_col.astype(
#                         col_type, copy=False)

            # Handle date parsing
            if col_name in parse_dates:
                try:
                    fmt = parse_dates[col_name]
                except TypeError:
                    fmt = None
                dataframe[col_name] = _handle_date_column(df_col, format=fmt)

            column_names.append(col_name)  # added by me
        except KeyError:
            pass  # this column not in results

    return column_names, dataframe


def df2dbiter(dataframe, table_class, harmonize_columns_first=True, harmonize_rows=True,
              parse_dates=None):
    """
        Returns a generator of of ORM model instances (reflecting the rows database table mapped by
        table_class) from the given dataframe. The returned generator can be used in loops and each
        element can be e.g. added to the database by means of sqlAlchemy `session.add` method.
        NOTE: Only the dataframe columns whose name match the table_class columns will be used.
        Therefore, it is safe to append to dataframe any column whose name is not in table_class
        columns. Some iterations might return None according to the parameters (see below)
        :param table_class: the CLASS of the table whose rows are instantiated and returned
        :param dataframe: the input dataframe
        :param harmonize_columns_first: if True (default when missing), the dataframe column types
        are FIRST harmonized to reflect the table_class column types, and columns without a match
        in table_class are filtered out. See `harmonize_columns(dataframe)`
        :param harmonize_rows: if True (default when missing), NA values are checked for
        those table columns which have the property nullable set to False. In this case, the
        generator **might return None** (the user should in case handle it, e.g. skipping
        these model instances which are not writable to the db)
        :param parse_dates: a list of strings denoting additional column names whose values should
        be parsed as dates. Ignored if harmonize_columns_first is False
    """
    if harmonize_columns_first:
        colnames, dataframe = _harmonize_columns(table_class, dataframe, parse_dates)
    else:
        table_col_names = get_col_names(table_class)
        colnames = [c for c in dataframe.columns if c in table_col_names]  # FIXME: optimize this?

    new_df = dataframe[colnames]

    if dataframe.empty:
        for _ in len(dataframe):
            yield None
        return

    valid_rows = cycle([True])
    if harmonize_rows:
        non_nullable_cols = get_non_nullable_cols(table_class, new_df)
        if non_nullable_cols:
            df = ~pd_isnull(new_df[non_nullable_cols])
            valid_rows = df.apply(lambda row: row.all(), axis=1).values

    cols, datalist = _insert_data(new_df)
    # Note below: datalist is an array of N column, each of M rows (it would be nicer to return an
    # array of N rows, each of them representing a table row. But we do not want to touch pandas
    # code.
    # See _insert_table below). Thus we zip it:
    for is_ok, row_values in zip(valid_rows, zip(*datalist)):
        if is_ok:
            # we could make a single line statement, but two lines are more readable:
            row_args_dict = dict(zip(cols, row_values))
            yield table_class(**row_args_dict)
        else:
            yield None


def _insert_data(dataframe):
    """Copied to pandas.io.sql.SqlTable: basically converts dataframe to a numpy array of arrays
    for insertion inside a db table.
    :return: a tuple of two elements (cols, data):
    cols denotes the dataframe columns (list of strings),
    data denotes the dataframe data (converted and casted) to be inserted.
    NOTE: Each element of data denotes a COLUMN. Thus len(data) == len(cols). For each column,
    data[i] has M elements denoting the M rows (where M = len(dataframe))
    Thus, to get each table row as array, use:
        zip(*data)
    To get each table row as dict colname:value, use:
        for row in zip(*data):
            row_as_dict = dict(zip(cols, row_values))

    (this code is copied from pandas.io.sql.SqlTable._insert_data, we do not want to mess
    around with that code - more than what we've already done, so we keep it as it is)
    """
    column_names = list(map(text_type, dataframe.columns))
    ncols = len(column_names)
    data_list = [None] * ncols
    blocks = dataframe._data.blocks

    for i in range(len(blocks)):
        b = blocks[i]
        if b.is_datetime:
            # convert to microsecond resolution so this yields
            # datetime.datetime
            d = b.values.astype('M8[us]').astype(object)
        else:
            d = np.array(b.get_values(), dtype=object)

        # replace NaN with None
        if b._can_hold_na:
            mask = isnull(d)
            d[mask] = None

        for col_loc, col in zip(b.mgr_locs, d):
            data_list[col_loc] = col

    return column_names, data_list


def flush(session, on_exc=None):
    """
        Flushes the given section. In case of Exception (IntegrityError), rolls back the session
        and returns False. Otherwise True is returned
        :param on_exc: a callable (function) which will be called with the given exception as first
        argument
        :return: True on success, False otherwise
    """
    try:
        session.flush()
        return True
    except SQLAlchemyError as _:
        session.rollback()
        if hasattr(on_exc, "__call__"):  # on_exc=None returns False
            on_exc(_)
        return False


def commit(session, on_exc=None):
    """
        Commits the given section. In case of Exception (IntegrityError), rolls back the session
        and returns False. Otherwise True is returned
        :param on_exc: a callable (function) which will be called with the given exception as first
        argument.
        :return: True on success, False otherwise
    """
    try:
        session.commit()
        return True
    except SQLAlchemyError as _:
        session.rollback()
        if hasattr(on_exc, "__call__"):  # on_exc=None returns False
            on_exc(_)
        return False


def get_or_add_iter(session, model_instances, model_cols_or_colnames=None, on_add='flush'):
    """
        Iterates on all model_rows trying to add each
        instance to the session if it does not already exist on the database. All instances
        in `model_instances` should belong to the same model (python class). For each
        instance, its existence is checked based on `model_cols_or_colnames`: if a database row
        is found, whose values are the same as the given instance for **all** the columns defined
        in `model_cols_or_colnames`, then the *first* database row instance is returned.

        Yields tuple: (model_instance, is_new_and_was_added)

        Note that if `on_add`="flush" (the default) or `on_add`="commit", model_instance might be
        None (see below)

        :Example:
        ```
        instances = ... # list or iterable of some model instances
        # add all instances if they are not found on db according to their primary key
        # (thus no need to specify the argument `model_cols_or_colnames`)
        for instance, is_new in get_or_add_iter(session, instances, on_add='commit'):
            if instance is None:
                # instance was not found on db but adding it raised an exception
                # (including the case where the item of instances is None)
            elif is_new:
                # instance was not found on the db and was succesfully added
            else:
               # instance was found on the db and the first matching instance is returned

        # assuming the model of each instance is a class named 'MyTable' with a primary key 'id':
        instances = [MyTable(...), ..., MyTable(...)]
        # the calls below produce the same results:
        get_or_add_iter(session, instances):
        get_or_add_iter(session, instances, 'id'):
        get_or_add_iter(session, instances, MyTable.id):
        ```
        :param model_instances: an iterable (list tuple generator ...) of ORM model instances. None
        values are valid and will yield the tuple (None, False)
        All instances *MUST* belong to the same class, i.e., represent rows of the same db table.
        An ORM model is the python class reflecting a database table. An ORM model instance is
        simply a python instance of that class, and thus reflects a rows of the database table
        :param model_cols_or_colnames: (iterable, model attribute, string or None) An iterable of
        columns whereby the existience of a database row matching `model_instance` has to be checked
        first. The parameter can be also given as "scalar" as model attribute or string denoting
        a model attribute name (the comparison will be made on that single attribute in case)
        Usually, provide primary keys or a combination of unique constraints.
        The parameter can be given as column attributes of the given ORM model, or as strings
        denoting the given ORM model columns. If None, the model primary keys are taken as
        columns, thus if a row is found, whose primary key values are equal to the primary key
        values of `model instance`, the instance reflecting that row is returned
        :param on_add: 'commit', 'flush' or None. Default: 'flush'. Tells whether a `session.flush`
        or a `session commit` has to be issued after each `session.add`. In case of failure, a
        `session.rollback` will be issued and the tuple (None, False) is yielded
    """
    binexpfunc = None  # cache dict for expressions
    for row in model_instances:
        if row is None:
            yield None, False
        else:
            if not binexpfunc:
                binexpfunc = _bin_exp_func_from_columns(row.__class__, model_cols_or_colnames)

            yield _get_or_add(session, row, binexpfunc, on_add)


def _get_or_add(session, row, binexpr_for_get, on_add='flush'):
    """
    Returns row, is_new, where row is the pased row as argument (added if not existing) or the
    one on the db matching binexpr_for_get. is_new is a boolean indicating whether row was newly
    added or found on the db (according to binexpr_for_get)
    If flush_on_add is True (flush_on_add=False must be carefully used, epsecially for handling
    rollbacks), then row might be None if session.sluch failed
    :param row: the model instance represetning a table row. Cannot be None
    :param on_add: 'flush', 'commit' or everything else (do nothing)
    """
    model = row.__class__
    row_ = session.query(model).filter(binexpr_for_get(row)).first()
    if row_:
        return row_, False
    else:
        session.add(row)
        if (on_add == 'flush' and not flush(session)) or \
                (on_add == 'commit' and not commit(session)):
            return None, False
        return row, True


def _bin_exp_func_from_columns(model, model_cols_or_colnames):
    """
        Returns an slalchemy binary expression *function* for the given model and the given columns
        the function can be passed in a query object for a given model instance.
        :Example:
        # assuming a MyTable model defined somewhere, with columns "col_name_1", "col_name2":
        func = _bin_exp_func_from_columns(MyTable, ["col_name_1", "col_name2"]):
        # Now assume we have a model instance
        row = MyTable(col_name_1='a', col_name_2=5.5)
        # We can use func in a query (assuming we have a session object):
        session.query(model).filter(func(row)).all()
        # which will query the db table mapped by MyTable for all the rows whose col_name_1 value
        is 'a' **and** whose 'col_name_2' value is 5.5
    """
    if not model_cols_or_colnames:
        model_cols_or_colnames = get_cols(model, primary_key_only=True)

    # is string? In py2, check attr "__iter__". In py3, as it has the attr, go for isinstance:
    if not hasattr(model_cols_or_colnames, "__iter__") or isinstance(model_cols_or_colnames, str):
        model_cols_or_colnames = [model_cols_or_colnames]

    columns = []
    for col in model_cols_or_colnames:
        if not hasattr(col, "key"):  # is a column object
            col = getattr(model, col)  # return the attribute object
        columns.append(col)

    def ret_func(row):
        """ Returns the binary expression function according toe the model_cols_or_colnames for
        a given model instance `row`"""
        return and_(*[col == getattr(row, col.key) for col in columns])

    return ret_func


def init_db(dbpath):
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)
    # create a configured "Session" class
    Session = sessionmaker(bind=engine)
    # create a Session
    session = Session()