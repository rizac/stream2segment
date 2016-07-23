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
from pandas import to_numeric
import pandas as pd


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
                dataframe[col_name] = df_col.astype(col_type, copy=False)

            elif col_type is np.dtype('int64'):
                # integers. Try with normal way. The original code checked NaNs like this:
                # if len(df_col) == df_col.count():
                # but this raises on e.g., non numeric strings, which we want to
                # convert to NaN. So, try the "normal way":
                try:
                    dataframe[col_name] = df_col.astype(col_type, copy=False)
                except ValueError:
                    # failed, use to_numeric coercing to None on errors
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


def df_to_table_iterrows(table_class, dataframe, harmonize_columns_first=True,
                         harmonize_rows=True, parse_dates=None):
    """
        Returns a generator of of ORM model instances (reflecting the rows database table mapped by
        table_class) from the given dataframe. The returned generator can be used in loops and each
        element can be e.g. added to the database by means of sqlAlchemy `session.add` method.
        NOTE: Only the dataframe columns whose name match the table_class columns will be used.
        Therefore, it is safe to append to dataframe any column whose name is not in table_class
        columns.
        :param table_class: the CLASS of the table whose rows are instantiated and returned
        :param dataframe: the input dataframe
        :param harmonize_columns_first: if True (default when missing), the dataframe column types
        are FIRST harmonized to reflect the table_class column types, and columns without a match
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

    if dataframe.empty:
        return

    valid_rows = None
    if harmonize_rows:
        non_nullable_cols = get_non_nullable_cols(table_class, dataframe)
        if non_nullable_cols:
            df = ~pd_isnull(dataframe[non_nullable_cols])
            valid_rows = df.apply(lambda row: row.all(), axis=1).values

    if valid_rows is None:
        l = len(dataframe)
        # fake generator, basically skip valid_rows check below
        valid_rows = (True for _ in xrange(l))

    cols, datalist = _insert_data(dataframe)
    # Note below: datalist is an array of N column, each of M rows (it would be nicer to return an
    # array of N rows, each of them representing a table row. But we do not want to touch pandas
    # code.
    # See _insert_table below). Thus we zip it:
    for is_ok, row_values in zip(valid_rows, zip(*datalist)):
        if is_ok:
            # don't make (even if we could) a single line statement. Two lines are more readable:
            row_args_dict = dict(zip(cols, row_values))
            yield table_class(**row_args_dict)
#         else:
#             yield None


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
    except IntegrityError as _:
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


def get_or_add_all(session, model_rows, model_cols_or_colnames=None, flush_on_add=True):
    ret = []
    for instance, isnew in get_or_add_iter(session, model_rows, model_cols_or_colnames, flush_on_add):
        if instance is not None:
            ret.append(instance)
    return ret


def get_or_add_iter(session, model_rows, model_cols_or_colnames=None, flush_on_add=True):
    class2expr = {}  # cahce dict for expression
    for row in model_rows:
        model = row.__class__
        binexpfunc = class2expr.get(model, None)
        if not binexpfunc:
            binexpfunc = _bin_exp_func_from_columns(model, model_cols_or_colnames)
            class2expr[model] = binexpfunc

        yield _get_or_add(session, model, row, binexpfunc, flush_on_add)


def get_or_add(session, row, model_cols_or_colnames=None, flush_on_add=True):
    model = row.__class__
    return _get_or_add(session, model, row, _bin_exp_func_from_columns(model, model_cols_or_colnames),
                       flush_on_add)


def _get_or_add(session, model, row, binexpr_for_get, flush_on_add=True):
    """Note: flush_on_add=False is dangerous!"""
    row_ = session.query(model).filter(binexpr_for_get(row)).first()
    if row_:
        return row_, False
    else:
        session.add(row)
        if not flush_on_add or flush(session):
            return row, True
        return None, False


def _bin_exp_func_from_columns(model, model_cols_or_colnames):

    if not model_cols_or_colnames:
        model_cols_or_colnames = get_cols(model, primary_key_only=True)

    columns = []
    for col in model_cols_or_colnames:
        if not hasattr(col, "key"):  # is a column object
            col = getattr(model, col)  # return the attribute object
        columns.append(col)

    binary_expr_funcs = []
    for col in columns:
        def bin_exp_func(row):
            return col == getattr(row, col.key)

        binary_expr_funcs.append(bin_exp_func)

    if len(binary_expr_funcs) == 1:
        return binary_expr_funcs[0]
    else:
        def ret_func(row):
            binexprs = [b(row) for b in binary_expr_funcs]
            return and_(*binexprs)
        return ret_func
