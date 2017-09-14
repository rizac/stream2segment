'''
Efficient utilities for converting from pandas DataFrames to sqlalchemy tables objects
(according to `models.py`)
Some of these functions are copied and pasted from `pandas.io.sql.SqlTable`, some other account for
performance improvements in SqlAlchemy (http://docs.sqlalchemy.org/en/latest/faq/performance.html).
This module is a bridge between these libraries. It might be that in future pandas releases
most of these functionalities will be standard in the library

Refs
----

- Underlying mechanism of SqlAlchemy:
  http://docs.sqlalchemy.org/en/latest/glossary.html#term-descriptor
- Key attribute in SqlAlchemy columns:
  http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column.params.key
- Name attribute in SqlAlchemy columns:
  http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column.params.name
- Mapper SqlAlchemy object (for inspecting a table):
  http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper
  http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper.columns

Created on Jul 17, 2016

@author: riccardo
'''
from __future__ import division
from datetime import datetime, date
# from collections import OrderedDict
from itertools import cycle, izip
import numpy as np
import pandas as pd

from pandas.io.sql import _handle_date_column
# from pandas.types.api import DatetimeTZDtype
# pandas zip seems a wrapper around itertools.izip (generator instead than list):
from pandas.compat import (lzip, map, zip, raise_with_traceback,
                           string_types, text_type)
# is this below the same as pd.isnull? For safety we leave it like it is (the line is imported
# from pandas.io.sql and used in one of the copied methods below)
from pandas.core.common import isnull
# but we need also pd.isnull so we import it like this for safety:
# from pandas import isnull as pd_isnull
from pandas.types.dtypes import DatetimeTZDtype
# sql-alchemy:
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
# from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import and_, or_
from sqlalchemy.orm.attributes import InstrumentedAttribute
# from pandas import to_numeric
# from sqlalchemy.engine import create_engine
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import func, bindparam


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


def colnames(table, pkey=None, fkey=None, nullable=None):
    """
        Returns an iterator returning the attributes names (as string) reflecting database
        columns with the given properties specified as argument.
        :param table: an ORM model class (python class)
        :param pkey: boolean or None. If None, filter on primary keys is off. If True, only primary
        key columns are yielded, if False, only non-primary key columns are yielded
        :param fkey: boolean or None. If None, filter on foreign keys is off. If True, only foreign
        key columns are yielded, if False, only non-foreign key columns are yielded
        :param nullable: boolean or None. If None, filter on nullable columns is off.
        If True, only columns where nullable=True are yielded, if False, only columns where
        nullable=False are yielded
    """
    mapper = inspect(table)
    # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper.mapped_table
    table = mapper.mapped_table
    fkeys_cols = set((fk.parent for fk in table.foreign_keys)) if fkey in (True, False) else set([])
    for att_name, column in mapper.columns.items():
        # the dict-like above is keyed based on the attribute name defined in the mapping,
        # not necessarily the key attribute of the Column itself (column). See
        # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper.columns
        if (pkey is None or pkey == column.primary_key) and \
                (fkey is None or (column in fkeys_cols) == fkey) and \
                (nullable is None or nullable == column.nullable):
            yield att_name


def shared_colnames(table, dataframe, pkey=None, fkey=None, nullable=None):
    """Returns an iterator with the columns shared between the given table (ORM model
    instance or class) and the given pandas DataFrame. The table columns are its attribute names
    (as typed in the code in the class definition), the DataFrame should thus have string columns
    to avoid unexpected results in the comparison
    :param table: the ORM table instance or the ORM table model
    :param dataframe: a dataframe, with string columns
    :param pkey: boolean or None. If None, filter on primary keys is off. If True, only primary
    key columns shared between table and dataframe are yielded, if False, only shared non-primary
    key columns are yielded
    :param fkey: boolean or None. If None, filter on foreign keys is off. If True, only shared
    foreign key columns are yielded, if False, only shared non-foreign key columns are yielded
    :param nullable: boolean or None. If None, filter on nullable columns is off.
    If True, only shared columns where nullable=True are yielded, if False, only shared columns
    where nullable=False are yielded
    """
    dfcols = set(dataframe.columns)  # in operator is faster on sets
    for colname in colnames(table, pkey=pkey, fkey=fkey, nullable=nullable):
        if colname in dfcols:
            yield colname


def harmonize_rows(table, dataframe, inplace=True):
    """Make the DataFrame's rows align with the SQL table column nullable value.
    That is, removes the dataframe rows which are NA (None, NaN or NaT) for those values
    corresponding to `table` columns which were set to not be Null (nullable=False).
    Non nullable table attributes (reflecting db table columns) not present in dataframe columns
    are not accounted for: in other words, the non-nullable condition on the dataframe is set for
    those columns only which have a corresponding name in any of the table attributes.
    Consider calling `harmonize_cols` first to make sure the column values
    align with the table column types
    :param inplace: argument to be passed to pandas `dropna`
    """
    non_nullable_cols = list(shared_colnames(table, dataframe, nullable=False))
    # FIXME: actually dropna accepts also generators, so the conversion to list is useless
    if non_nullable_cols:
        tmp = dataframe.dropna(subset=non_nullable_cols, axis=0, inplace=inplace)
        if not inplace:  # if inplace, tmp is None and dataframe has been modified
            dataframe = tmp
    return dataframe


def harmonize_columns(table, dataframe, parse_dates=None):
    """Make the DataFrame's column types align with the SQL table
    column types. Returns a new dataframe with "correct" types (according to table)
    Columns which are *not* shared with table columns (assuming dataframe columns are strings)
    are left as they are. Columns of the table are assumed to be its attribute names (as typed
    in the code), thus the DataFrame is assumed to have string columns as well.
    The returned dataframe row numbers is not modified
    :param table: an ORM model class
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
    # note below: it seems that the original pandas module uses Column objects
    # however, some properties (such as type) are also available in InstrumentedAttribute
    # (getattr(table, col_name)). So we use the latter
    for col_name in colnames(table):
        sql_col = getattr(table, col_name)
        # changed, in pandas was: 'sql_col.name', but we want to harmonize according to
        # the attribute names
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
                except (TypeError, ValueError):
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

    for i in xrange(len(blocks)):
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


def dfrowiter(dataframe, columns=None):
    """Returns an efficient iterator over `dataframe` rows. The i-th returned values is
    a `dict`s of `dataframe` columns (strings) keyed to the i-th row values. Each value is
    assured to be
    a python type (str, bool, datetime, int and float are currently supported) with pandas
    null values (NaT, NaN) converted to None, if any.
    :param dataframe: the input dataframe
    """
    cols, datalist = _insert_data(dataframe[columns] if columns is not None else dataframe)
    # Note below: datalist is an array of N column, each of M rows (it would be nicer to return an
    # array of N rows, each of them representing a table row. But we do not want to touch pandas
    # code.
    # See _insert_table below). Thus we zip it:
    for row_values in zip(*datalist):
        # we could make a single line statement, but two lines are more readable:
        row_args_dict = dict(zip(cols, row_values))
        yield row_args_dict


def _get_max(session, numeric_column):
    """Returns the maximum value from a given numeric column, usually a primary key with
    auto-increment=True. If it's the case, from `_get_max() + 1` we assure unique identifier for
    adding new objects to the table of `numeric_column`. SqlAlchemy has the ability to set
    autoincrement for us but telling explicitly the id value for an autoiincrement primary key
    speeds up *a lot* the insertion (especially if used in conjunction with slqlachemy core methods
    :param session: an sqlalchemy session
    :param numeric_column: a column of an ORM model (mapping a db table column)
    """
    return session.query(func.max(numeric_column)).scalar() or 0


def dbquery2df(query):
    """Returns a query result as a set of tuples where each tuple value is the db row values
    according to columns
    :param session: sqlalchemy session
    :param columns: a list of ORM instance columns
    :param query_filter: optional filter to be apoplied to the query, defaults to None (no filter)
    """
    colnames = [c['name'] for c in query.column_descriptions]
    return pd.DataFrame(columns=colnames, data=query.all())


def syncdf(dataframe, session, matching_columns, autoincrement_pkey_col, buf_size=10,
           drop_duplicates=True, return_df=True, onerr=None):
    """
    Efficiently synchronizes `dataframe` with the corresponding database table T.
    Returns the tuple `(d, new)` where:

    `return_df`  `d`:                                        `new`:
    ===========  =========================================== ===================================
    `True`       `dataframe` with only rows with a           the number of rows of which
                 corresponding row in T (according to        where inserted on T (`new<=len(d)`)
                 `matching_columns`) and the column
                 `autoincrement_pkey_col` set (the column
                 needs not to be in `dataframe.columns`)
    -----------  ------------------------------------------- ------------------------------------
    `False`      the total number of rows `<=len(dataframe)` the number of rows of which
                 (depending on `drop_duplicates`)            where inserted on T (`new<=d`)
    ===========  =========================================== ====================================

    `return_df=False` is in principle faster as less operations are involved.

    This function works by:
    1. Setting first the value of `autoincrement_pkey_col` for those rows found on T
       (according to `matching_columns`)
    2. Auto-incrementing `autoincrement_pkey_col` values for the remaining rows
       (not found on the db), and finally writing those rows to T

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe` rows and T rows:
    when two rows are found that are equal (according to all `matching_columns` values), then
    the data frame row `autoincrement_pkey_col` value is set = T row value
    :param autoincrement_pkey_col: the ORM column denoting an auto-increment primary key of T.
    Unexpected results if the column does not match those criteria. The column
    needs not to be a column of `dataframe`. The returned `dataframe` will have in any case this
    column set
    :param buf_size: integer, defaults to 10. The buffer size before committing. Increase this
    number for better performances (speed) at the cost of some "false negative" (committing a
    series of operations where one raise an integrity error discards all subsequent operations
    regardless if they would raise as well or not)
    :param drop_duplicates: boolean, True. After having fetched the primary keys and set it to
    the dataframe corresponding column, drop duplicates under `matching_columns`. You should
    always set this argument to True unless you are really sure `dataframe` has no duplicates
    under `matching_columns`, and you really want to save the extra time of dropping again
    (but is that saved time actually remarkable?)

    :return: the tuple `(d, new)` where
    1. `d` is `dataframe` where rows without a corresponding row on T have been filtered out.
    `d` has surely the column `autoincrement_pkey_col` (see Technical notes below for details),
    which on the other hand needs not to to be a column of `dataframe`, and
    2. `new` (`<= len(d)`) is the number of rows of `d` which are "new" (i.e., whose corresponding
    row was not present on T before this function call). `d` index is not reset, so a ref.
    to `dataframe` is always possible

    Technical notes
    ================================================================================================

    1. T is obtained as the `class_` attribute of the first passed
    `Column <http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column>`_,
    therefore `autoincrement_pkey_col` and each element of `matching_columns` must refer to the
    same db table T.
    2. The mapping between an sql-alchemy Column C and a pandas dataframe *string*
    column K is based on the sql-alchemy `key` attribute: `C.key == K`
    3. On the db session side, we do not use ORM functionalities but lower
    level sql-alchemy core methods, which are faster (FIXME: ref needed). This, together with
    the "buffer size" argument, speeds up a lot items insertion on the database (for update
    operations this needs tests).
    The drawbacks of these approaches is that we need to create by ourself the primary keys before
    inserting a row to T, and that if a single item of a buffer raises an SqlAlchemtError, all
    following items are not added to the db, even if they where well formed
    """
    dframe_with_pkeys = fetchsetpkeys(dataframe, session, matching_columns, autoincrement_pkey_col)
    if drop_duplicates:
        subset_cols = [k.key for k in matching_columns]
        dframe_with_pkeys = dframe_with_pkeys.drop_duplicates(subset=subset_cols)
        dframe_with_pkeys.is_copy = False
    return insertdf_napkeys(dframe_with_pkeys, session, autoincrement_pkey_col, buf_size,
                            return_df)


def insertdf_napkeys(dataframe, session, autoincrement_pkey_col, buf_size=10, return_df=True,
                     onerr=None):
    """
    Efficiently inserts rows of `dataframe` in the corresponding database table T, but only
    where `autoincrement_pkey_col` is N/A (handling auto-incrementing of `autoincrement_pkey_col`
    internally for good performances). `autoincrement_pkey_col` needs not to be in
    `dataframe.columns` (the case is treated as if all `autoincrement_pkey_col` values were N/A).
    Returns the tuple `(d, new)` where

    `return_df`  `d`:                                     `new`:
    ===========  ======================================== ===================================
    `True`       `dataframe` with only rows with a        the number of rows of which
                 corresponding row in T (either because   where inserted on T (`new<=len(d)`)
                 `autoincrement_pkey_col` is not NA or
                 because the row has been successfully
                 inserted on T)
    -----------  ---------------------------------------- ------------------------------------
    `False`      the total number of rows:                the number of rows of which
                 `d==len(dataframe)`                      where inserted on T (`new<=d`)
    ===========  ======================================== ====================================

    `return_df=False` is in principle faster as less operations are involved.

    The remainder of the documentation is the same as `syncdf`, so please see there for details
    """
    dtmp = None
    df_pkey_col = autoincrement_pkey_col.key
    dframe_with_pkeys = dataframe
    df_has_pkey = df_pkey_col in dataframe.columns
    if df_has_pkey:
        mask = pd.isnull(dframe_with_pkeys[df_pkey_col])
        dtmp = dframe_with_pkeys[mask]
        dtmp.is_copy = False  # avoid pandas SettingWithCopyWarning, we are trying to modify dtmp
        # modifications will not affect dframe_with_pkeys but we are aware of it
    else:
        dtmp = dframe_with_pkeys

    ret_first_arg, new = (len(dframe_with_pkeys), 0) if not return_df else (dframe_with_pkeys, 0)
    numnans = len(dtmp)
    if numnans:
        max_pkey = _get_max(session, autoincrement_pkey_col) + 1
        new_pkeys = np.arange(max_pkey, max_pkey+numnans, dtype=int)
        dtmp[df_pkey_col] = new_pkeys
        new_df, new = insertdf(dtmp, session, [autoincrement_pkey_col], buf_size,
                               query_first=False, drop_duplicates=False, return_df=return_df,
                               onerr=onerr)
        if return_df:
            if df_has_pkey:
                dframe_with_pkeys.loc[new_df.index, df_pkey_col] = new_df[df_pkey_col]
                dframe_with_pkeys = dframe_with_pkeys.dropna(subset=[df_pkey_col])  # for safety
            else:
                dframe_with_pkeys = new_df
            ret_first_arg = dframe_with_pkeys

    return ret_first_arg, new


def insertdf(dataframe, session, matching_columns, buf_size=10, query_first=True,
             drop_duplicates=True, return_df=True,
             onerr=None):
    """
    Efficiently inserts row of `dataframe` to the corresponding database table T. Rows found on
    T (according to `matching_columns`) are not inserted again. Returns the tuple `(d, new)` where:

    `return_df`  `d`:                                        `new`:
    ===========  =========================================== ===================================
    `True`       `dataframe` with only rows with a           the number of rows of which
                 corresponding row in T (according to        where inserted on T (`new<=len(d)`)
                 `matching_columns`) either because already
                 existing or newly inserted
    -----------  ------------------------------------------- ------------------------------------
    `False`      the total number of rows `<=len(dataframe)` the number of rows of which
                 (depending on `drop_duplicates`)            where inserted on T (`new<=d`)
    ===========  =========================================== ====================================

    `return_df=False` is in principle faster as less operations are involved.

    :param query_first: boolean (defaults to True): queries T for rows already present. If this
    argument is False no skip is done, i.e. for all rows of `dataframe` the function will
    attempt to add them to T. **Set to False to speed up the function as long as you are sure no
    row of `dataframe` violates any T constraint**

    The remainder of the documentation is the same as `syncdf`, so please see there for details
    """
    if dataframe.empty:
        return (0, 0) if not return_df else (dataframe, 0)

    buf_size = max(buf_size, 1)
    buf = {}
    matching_colnames = [c.key for c in matching_columns]
    if drop_duplicates:
        dataframe.drop_duplicates(subset=matching_colnames, inplace=True)
    table_model = matching_columns[0].class_
    # allocate all primary keys for the given model
    existing_keys = set() if not query_first else _dbquery2set(session, matching_columns)
    shared_cnames = list(shared_colnames(table_model, dataframe))
    last = len(dataframe) - 1
    existing = 0
    not_inserted = 0
    indices_discarded = []

    for i, rowdict in enumerate(dfrowiter(dataframe, shared_cnames)):
        if existing_keys:
            rowtup = tuple(rowdict[col] for col in matching_colnames)
            if rowtup not in existing_keys:
                # _tup2idx[rowtup] = i
                buf[i] = rowdict
            else:
                existing += 1
        else:
            buf[i] = rowdict

        if len(buf) == buf_size or (i == last and buf):
            try:
                session.connection().execute(table_model.__table__.insert(), buf.values())
                session.commit()
            except SQLAlchemyError as _:
                session.rollback()
                if onerr is not None:
                    onerr(dataframe, buf.iterkeys(), _)
                not_inserted += len(buf)
                if return_df:
                    indices_discarded.extend(buf.iterkeys())

            buf.clear()

    new = len(dataframe) - not_inserted - existing
    if not return_df:
        first_ret_arg = len(dataframe) - not_inserted
    else:
        first_ret_arg = dataframe
        if not_inserted:
            if not_inserted == len(dataframe):
                first_ret_arg = dataframe.iloc[[]]  # basically, empty dataframe, preserving cols
            else:
                indices_discarded = np.array(indices_discarded, dtype=int)
                indices = np.in1d(np.arange(len(dataframe)), indices_discarded, assume_unique=True,
                                  invert=True)
                first_ret_arg = dataframe.iloc[indices]

    return first_ret_arg, new


def updatedf(dataframe, session, where_col, update_columns, buf_size=10, return_df=True,
             onerr=None):
    """
    Efficiently updates row of `dataframe` to the corresponding database table T.
    Returns `d`, where:

    `return_df`  `d`:
    ===========  ========================================
    `True`       `dataframe` with only rows
                 successfully updated
    -----------  ----------------------------------------
    `False`      the total number of rows successfully
                 updated
    ===========  ========================================

    `return_df=False` is in principle faster as less operations are involved.

    The remainder of the documentation is the same as `syncdf`, so please see there for details
    """
    if dataframe.empty:
        return 0 if not return_df else dataframe

    table_model = where_col.class_
    shared_columns = update_columns + [where_col]
    shared_cnames = [c.key for c in shared_columns]
    # find a col not present for where_col. Otherwise error is raised:
    # bindparam() name where_col.key is reserved for automatic usage in the VALUES or SET clause
    # of this  insert/update statement.   Please use a name other than column name when using
    # bindparam() with insert() or update() (for example, 'b_id').
    where_col_bindname = where_col.key + "_"
    while where_col_bindname in shared_cnames:
        where_col_bindname += "_"
    stmt = table_model.__table__.update().\
        where(where_col == bindparam(where_col_bindname)).\
        values({c.key: bindparam(c.key) for c in update_columns})
    buf = {}
    last = len(dataframe) - 1
    indices_discarded = []
    not_updated = 0

    for i, rowdict in enumerate(dfrowiter(dataframe, shared_cnames)):
        # replace the where column:
        rowdict[where_col_bindname] = rowdict.pop(where_col.key)
        buf[i] = rowdict
        if len(buf) == buf_size or (i == last and buf):
            try:
                session.connection().execute(stmt, buf.values())
                session.commit()
            except SQLAlchemyError as _:
                session.rollback()
                if onerr is not None:
                    onerr(dataframe, buf.iterkeys(), _)
                not_updated += len(buf)
                if return_df:
                    indices_discarded.extend(buf.iterkeys())

            buf.clear()

    if not return_df:
        first_ret_arg = last + 1 - not_updated
    else:
        first_ret_arg = dataframe
        if not_updated:
            if not_updated == len(dataframe):
                first_ret_arg = dataframe.iloc[[]]  # basically, empty dataframe, preserving cols
            else:
                indices_discarded = np.array(indices_discarded, dtype=int)
                indices = np.in1d(np.arange(len(dataframe)), indices_discarded, assume_unique=True,
                                  invert=True)
                first_ret_arg = dataframe.iloc[indices]

    return first_ret_arg


def _existing_insts(tuple_instances, session, columns):
    """returns a sub-set (python `set`) of only `tuple_instances` existing on the db
    :param tuple_instances: a list of instances, each represented by a tuple of values. For each
    tuple, the ith element is the value of the i-th column in `columns`
    :param session: sql-alchemy session
    :param columns: ORM columns. The i-th value in each tuple of `tuple_instances` must be the
    value of the i-th column of `columns`
    :return: a python set of tuples, sub-set of `set(tuple_instances)`
    """
    exprs = [and_(*[col == v for col, v in izip(columns, tup)]) for tup in tuple_instances]
    # 2. Get those elements and add them, if saved to the db
    existing_inst = _dbquery2set(session, columns, or_(*exprs)) if exprs else set()
    return existing_inst & set(tuple_instances)


def _dbquery2set(session, columns, query_filter=None):
    """Returns a query result as a set of tuples where each tuple value is the db row values
    according to columns
    :param session: sqlalchemy session
    :param columns: a list of ORM instance columns
    :param query_filter: optional filter to be apoplied to the query, defaults to None (no filter)
    """
    qry = session.query(*columns) if query_filter is None else \
        session.query(*columns).filter(query_filter)
    return set(tuple(x) for x in qry)


def fetchsetpkeys(dataframe, session, matching_columns, pkey_col):
    """Fetches the primary keys of the table T corresponding to `dataframe` and sets their values
    on `dataframe[pkey_col.key]`. `dataframe` does not need to have that column in the first place
    (it will be added if not present)

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe` rows and T rows:
    when two rows are found that are equal (according to all `matching_columns` values), then
    the primary key value of T row is set on the `dataframe` corresponding row
    :param pkey_col: the ORM column denoting T primary key. It does not need to be a column
    of `dataframe`

    :return: a new data frame with the column `pkey_col` populated with the primary keys
    of T. Values that are n/a, None's or NaN's (see `pandas.DataFrameisnull`) denote rows that do
    not have corresponding T row and might need to be added to T.
    The index of `d` is **not** reset, so that a track
    to the original dataframe is always possible (the user must issue a `d.reset_index` to reset
    the index).

    Technical notes:
    1. T is retrieved by means of the passed
    `Columns <http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column>`_,
    therefore `autoincrement_pkey_col` and each element of `matching_columns` must refer to the
    same db table T.
    2. The mapping between an sql-alchemy Column C and a pandas dataframe *string*
    column K is based on the sql-alchemy `key` attribute: `C.key == K`
    3. On the db session side, we do not use ORM functionalities but lower
    level sql-alchemy core methods, which are faster (FIXME: ref needed). This, together with
    the "buffer size" argument, speeds up a lot items insertion on the database.
    The drawback of the former is that we need to create by ourself the primary keys, the drawback
    of the latter is that if a single item of a buffer raises an SqlAlchemtError, all following
    items are not added to the db, even if they where well formed
    """
    cols = matching_columns + [pkey_col]
    df_new = dbquery2df(session.query(*cols).distinct())
    # colnames = [c.key for c in cols]
    # objs = [x for x in session.query(*cols)]
    # d = pd.DataFrame(columns=colnames, data=objs)
    return mergeupdate(dataframe, df_new, [c.key for c in matching_columns], [pkey_col.key], False)


def mergeupdate(df_old, df_new, matching_columns, set_columns, drop_df_new_duplicates=True):
    """
        Kind-of pandas.DataFrame update: sets
        `df_old[set_columns]` = `df_new[set_columns]`
        for those row where `df_old[matching_columns]` = `df_new[matching_columns]` only.
        `df_new` **should** have unique rows under `matching columns` (see argument
        `drop_df_new_duplicates`)
        :param df_old: the pandas DataFrame whose values should be replaced
        :param df_new: the pandas DataFrame which should set the new values to `df_old`
        :param matching_columns: list of strings: the columns to be checked for matches. They must
        be shared between both data frames
        :param set_columns: list of strings denoting the column to be set from `df_new` to
        `df_old` for those rows matching under `matching_cols`
        :param drop_df_new_duplicates: If True (the default) drops duplicates of `df_new` under
        `matching_columns` before updating `df_old`
    """
    if df_new.empty or df_old.empty:  # for safety (avoid useless calculations)
        return df_old

    if drop_df_new_duplicates:
        df_new = df_new.drop_duplicates(subset=matching_columns)

    # use df_new[matching_columns + set_columns] only for relevant columns
    # (should speed up merging?):
    mergedf = df_old.merge(df_new[matching_columns + set_columns], how='left',
                           on=list(matching_columns), indicator=True)

    # set values of new_df by means of the _merge column created via the arg indicator=True above:
    # _merge is in ('both', 'right_only', 'left_only'). We should never have 'right_only because of
    # the how='left' above. Skip checking for the moment
    for col in set_columns:
        if col not in df_old:
            ser = mergedf[col].values
        else:
            ser = np.where(mergedf['_merge'] == 'both', mergedf[col+"_y"], mergedf[col+"_x"])
        # ser = mergedf[col+"_y"].where(mergedf['_merge'] == 'both', mergedf[col+"_x"])
        df_old[col] = ser

    return df_old
