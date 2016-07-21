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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import and_

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
        dataframe = dataframe.dropna(subset=[non_nullable_cols], inplace=inplace)
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
    column types.
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
                dataframe[col_name] = df_col.astype(col_type, copy=False)

            elif len(df_col) == df_col.count():
                # No NA values, can convert ints and bools
                if col_type is np.dtype('int64') or col_type is bool:
                    dataframe[col_name] = df_col.astype(
                        col_type, copy=False)

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


def df_to_table_iterrows(table_class, dataframe, harmonize_columns_first=True,
                         harmonize_rows=True, parse_dates=None):
    """
        Returns a generator of of ORM model instances (reflecting the rows database table mapped by
        table_class) from the given dataframe. The returned generator can be used in loops and each
        element can be e.g. added to the database by means of sqlAlchemy `session.add` method.
        :param table_class: the CLASS of the table whose rows are instantiated and returned
        :param dataframe: the input dataframe
        :param harmonize_columns_first: if True (default when missing), the dataframe column types
        are FIRST harmonized to reflect the table_class column types, and columns withotu a match
        in table_class are filtered out. See `harmonize_columns(dataframe)`
        :param harmonize_rows: if True (default when missing), NA values are checked for
        those table columns which have the property nullable set to False. In this case, the
        generator MIGHT RETURN None, denoting rows which are not writable to the db
        :param parse_dates: a list of strings denoting additional column names whose values should
        be parsed as dates. Ignored if harmonize_columns_first is False
    """
    if harmonize_columns_first:
        colnames, dataframe = _harmonize_columns(table_class, dataframe, parse_dates)
    else:
        colnames = [c for c in dataframe.columns if c in get_col_names(table_class)]

    dataframe = dataframe[colnames]

    valid_rows = None
    if harmonize_rows:
        non_nullable_cols = get_non_nullable_cols(table_class, dataframe)
        if non_nullable_cols:
            df = pd_isnull(dataframe[non_nullable_cols])
            valid_rows = df.apply(lambda row: pd_isnull(row).any(), axis=1).values

    if valid_rows is None:
        l = len(dataframe)
        valid_rows = (True for _ in l)  # fake generator, basically skip valid_rows check below

    cols, datalist = _insert_data(dataframe)
    # Note below: datalist is an array of N column, each of M rows (it would be nicer to return an
    # array of N rows, each of them representing a table row. But we do not want to touch pandas
    # code.
    # See _insert_table below). Thus we zip it:
    for has_null, row_values in zip(valid_rows, zip(*datalist)):
        if has_null:
            yield None
        else:
            # don't make (even if we could) a single line statement. Two lines are more readable:
            row_args_dict = dict(zip(cols, row_values))
            yield table_class(**row_args_dict)


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


def flush(session):
    try:
        session.flush()
        return True
    except IntegrityError:
        session.rollback()
        return False


def add_or_get(session, db_row, *args):
    lst = list(args)
    model = db_row.__class__
    if not lst:
        lst = get_cols(db_row, primary_key_only=True)

    binary_expr = None
    for col_name in lst:
        try:  # is column name
            col = getattr(model, col_name)
            bin_exp = col == getattr(db_row, col.key)
        except TypeError:
            if not isinstance(col_name, BinaryExpression):  # col_name is a model attribute
                # (db column)
                bin_exp = col_name == getattr(db_row, col_name.key)
            else:
                bin_exp = col

        binary_expr = bin_exp if binary_expr is None else binary_expr & bin_exp

    if binary_expr is None:
        return None, False

    instance = session.query(model).filter(binary_expr).first()
    if instance:
        return instance, False
    else:
        session.add(db_row)
        return db_row, True


def get_or_add_all(session, model_rows, check_existence_on=None, flush_on_add=True):
    for _ in get_or_add_iter(session, model_rows, check_existence_on, flush_on_add):
        pass


def get_or_add_iter(session, model_rows, check_existence_on=None, flush_on_add=True):
    class2expr = {}  # cahce dict for expression
    for row in model_rows:
        model = row.__class__
        binexpfunc = class2expr.get(model, None)
        if not binexpfunc:
            binexpfunc = _bin_exp_func_from_columns(model, check_existence_on)
            class2expr[model] = binexpfunc

            yield _get_or_add(session, model, row, binexpfunc, flush_on_add)


def get_or_add(session, row, check_existence_on=None, flush_on_add=True):
    model = row.__class__
    return _get_or_add(session, model, row, _bin_exp_func_from_columns(model, check_existence_on),
                       flush_on_add)


def _get_or_add(session, model, row, binexpr_for_get, flush_on_add=True):
    row_ = session.query(model).filter(binexpr_for_get(row)).first()
    if row_:
        return row_, False
    else:
        session.add(row)
        if not flush_on_add or flush(session):
            return row, True
        return None, False


def _bin_exp_func_from_columns(model, columns):

    if not columns:
        columns = get_cols(model, primary_key_only=True)

    for col in columns:
        if not hasattr(col, "key"):  # is a column object
            col = [getattr(model, col)]  # return the attribute object
        columns.append(col)

    binary_expr_func = None
    for col in columns:
        def bin_exp_func(row):
            return col == getattr(row, col.key)

        binary_expr_func = bin_exp_func if binary_expr_func is None else \
            lambda row: (binary_expr_func(row)) & (bin_exp_func(row))

    return binary_expr_func


# def _get_or_add(session, row, check_existence_on=None, flush_on_add=True):
#     """check existence_on NOT callable. Either binary expression OR valid_obj OR iterable of
#     valid_obj's, where a valid_obj is either a String or a model column (class attribute)
#     Returns the tuple instance, is_added, bin_exp_func
#     where instance is the model instance (~= db row), either got or newly added
#     is_added is self explanatory: whether we added the instance or we got it cause already existing
#     bin_exp_func: None if check_existence_on is a sqlalchemy Binary expression or a callable
#         (returning a binary expression). Otherwise, it is the callable built here from the
#         `check_existence_on` argument. It accepts any row of the same row's model AND will apply
#         the same criteria set here.
#     """
#     binexpfunc = None
#     model = row.__class__
#     if hasattr(check_existence_on, "__call__"):
#         binary_expr = check_existence_on(row)
#     elif isinstance(check_existence_on, BinaryExpression):
#         binary_expr = check_existence_on
#     else:
#         binexpfunc = _bin_exp_func_from_columns(row, check_existence_on)
#         # now execute it to get the binary expression:
#         binary_expr = binexpfunc(row)
# 
#     instance = session.query(model).filter(binary_expr).first()
#     if instance:
#         return instance, False, binexpfunc
#     else:
#         session.add(row)
#         if flush_on_add:
#             try:
#                 session.flush()
#             except IntegrityError:
#                 session.rollback()  # rollback only last transaction
#                 return None, False, binexpfunc
#         return row, True, binexpfunc
# 
# 
# def _get_or_exec(session, row, check_existence_bin_exp, exec_func_if_no_exist='add', flush_on_exec=True):
#     """check existence_on NOT callable. Either binary expression OR valid_obj OR iterable of
#     valid_obj's, where a valid_obj is either a String or a model column (class attribute)
#     Returns the tuple instance, is_added, bin_exp_func
#     where instance is the model instance (~= db row), either got or newly added
#     is_added is self explanatory: whether we added the instance or we got it cause already existing
#     bin_exp_func: None if check_existence_on is a sqlalchemy Binary expression or a callable
#         (returning a binary expression). Otherwise, it is the callable built here from the
#         `check_existence_on` argument. It accepts any row of the same row's model AND will apply
#         the same criteria set here.
#     """
#     instance = session.query(row.__class__).filter(check_existence_bin_exp).first()
#     if instance:
#         return instance, False
#     else:
#         if exec_func_if_no_exist == 'add':
#             session.add(row)
#         else:
#             exec_func_if_no_exist(session, row)
#         if flush_on_exec:
#             try:
#                 session.flush()
#             except IntegrityError:
#                 session.rollback()  # rollback only last transaction
#                 return None, False
#         return row, True
