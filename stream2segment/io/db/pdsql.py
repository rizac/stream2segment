"""
Utilities for interaction between pandas DataFrames and sqlalchemy tables objects
(according to `models.py`). As of 2016, some of these functions are modified from
`pandas.io.sql.SqlTable` in most cases for performance improvements
(http://docs.sqlalchemy.org/en/latest/faq/performance.html)

Refs (URL are split in two when too long):
----

- Underlying mechanism of SqlAlchemy:
  http://docs.sqlalchemy.org/en/latest/glossary.html#term-descriptor
- Key attribute in SqlAlchemy columns:
  http://docs.sqlalchemy.org/en/latest/core/metadata.html
    #sqlalchemy.schema.Column.params.key
- Name attribute in SqlAlchemy columns:
  http://docs.sqlalchemy.org/en/latest/core/metadata.html
    #sqlalchemy.schema.Column.params.name
- Mapper SqlAlchemy object (for inspecting a table):
  http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    #sqlalchemy.orm.mapper.Mapper
  http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    #sqlalchemy.orm.mapper.Mapper.columns

:date: Jul 17, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import division

# make the following(s) behave like python3 counterparts if running from py2.7
# (http://python-future.org/imports.html#explicit-imports):
from builtins import range

from datetime import datetime, date
# Returns a list over dictionary values (and keys)
# with the same list-like behaviour on Py2.7 as on Py3 (scroll at the end of
# imports for other custom implementations of iterkeys and listkeys):
from future.utils import listvalues

import numpy as np
import pandas as pd

from pandas import to_datetime

# is this below the same as pd.isnull? For safety we leave it like it is (the
# line is imported from pandas.io.sql and used in one of the methods below)
from pandas.core.common import isnull

# Sql-alchemy:
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import func, bindparam
from sqlalchemy.types import Integer, Float, Boolean, DateTime, Date  # , TIMESTAMP

# future package implements a iterkeys which checks if the attribute iterkeys
# is defined on a dict. This looks inefficient. Moreover it lacks a listkeys
# function. Let's implement both here:
try:
    dict.iteritems
except AttributeError:
    # Python 3
    text_type = str

    def listkeys(d):
        return list(d.keys())

    def iterkeys(d):
        return d.keys()
else:
    # Python 2:
    text_type = unicode  # noqa
    import itertools
    zip = itertools.izip  # noqa
    map = itertools.imap  # noqa

    def listkeys(d):
        return d.keys()

    def iterkeys(d):
        return d.iterkeys()


def _get_dtype(sqltype):
    """Converts a sql type to a numpy type. Modified from pandas.io.sql

    :param sqltype: one of the following: Integer, Float, Boolean, DateTime,
        Date. Any other sql type will result in `object` being returned. This
        includes TIMESTAMPs as no support for TIMESTAMP as we assume all
        datetime(s) are in UTC thus they do not require a timezone set
    """
    if isinstance(sqltype, Float):
        return float
    if isinstance(sqltype, Integer):
        # TODO: Refine integer size.
        return np.dtype('int64')
    # Drop compatibility with TIMESTAMPS. If needed in the future, here the old code:
    # if isinstance(sqltype, TIMESTAMP):
    #     # we have a timezone capable type
    #     if not sqltype.timezone:
    #         return datetime
    #     return DatetimeTZDtype
    if isinstance(sqltype, DateTime):
        # Caution: np.datetime64 is also a subclass of np.number.
        return datetime
    if isinstance(sqltype, Date):
        return date
    if isinstance(sqltype, Boolean):
        return bool
    return object


def colnames(table, pkey=None, fkey=None, nullable=None):
    """Returns an iterator returning the attributes names (as string) reflecting
    database columns with the given properties specified as argument.

    :param table: an ORM model class (python class)
    :param pkey: boolean or None. If None, filter on primary keys is off.
        If True, only primary key columns are yielded, if False, only
        non-primary key columns are yielded
    :param fkey: boolean or None. If None, filter on foreign keys is off.
        If True, only foreign key columns are yielded, if False, only
        non-foreign key columns are yielded
    :param nullable: boolean or None. If None, filter on nullable columns is off.
        If True, only columns where nullable=True are yielded, if False,
        only columns where nullable=False are yielded
    """
    mapper = inspect(table)
    # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    #   #sqlalchemy.orm.mapper.Mapper.mapped_table
    table = _get_mapped_table(mapper)
    fkeys_cols = set((fk.parent for fk in table.foreign_keys)) \
        if fkey in (True, False) else set([])
    for att_name, column in mapper.columns.items():
        # the dict-like above is keyed based on the attribute name defined in
        # the mapping, not necessarily the key attribute of the Column itself
        # (column). See
        # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
        #   #sqlalchemy.orm.mapper.Mapper.columns
        # Note also: mapper.columns.items() returns a list, if performances are
        # a concern, we should iterate over the underlying mapper.columns._data
        # (but is cumbersome)
        if (pkey is None or pkey == column.primary_key) and \
                (fkey is None or (column in fkeys_cols) == fkey) and \
                (nullable is None or nullable == column.nullable):
            yield att_name


def _get_mapped_table(mapper):
    """Return the mapped table from the given SQLAlchemy mapper
    Note that there is a twin method in Inspector class ('sqlevalexpr' module)
    (FIXME: merge in future releases)
    """
    # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    #   #sqlalchemy.orm.mapper.Mapper.mapped_table
    # Note that from v 1.3+, we need to use .persist_selectable:
    try:
        table = mapper.persist_selectable
    except AttributeError:
        table = mapper.mapped_table
    return table


def shared_colnames(table, dataframe, pkey=None, fkey=None, nullable=None):
    """Returns an iterator with the columns shared between the given table
    (ORM model instance or class) and the given pandas DataFrame. The table
    columns are its attribute names (as typed in the code in the class
    definition), the DataFrame should thus have string columns to avoid
    unexpected results in the comparison

    :param table: the ORM table instance or the ORM table model
    :param dataframe: a dataframe, with string columns
    :param pkey: boolean or None. If None, filter on primary keys is off.
        If True, only primary key columns shared between table and dataframe
        are yielded, if False, only shared non-primary key columns are yielded
    :param fkey: boolean or None. If None, filter on foreign keys is off.
        If True, only shared foreign key columns are yielded, if False, only
        shared non-foreign key columns are yielded
    :param nullable: boolean or None. If None, filter on nullable columns is
        off. If True, only shared columns where nullable=True are yielded, if
        False, only shared columns where nullable=False are yielded
    """
    dfcols = set(dataframe.columns)  # in operator is faster on sets
    for colname in colnames(table, pkey=pkey, fkey=fkey, nullable=nullable):
        if colname in dfcols:
            yield colname


def harmonize_rows(table, dataframe, inplace=True):
    """Make the DataFrame's rows align with the SQL table column nullable value.
    That is, removes the dataframe rows which are NA (None, NaN or NaT) for
    those values corresponding to `table` columns which were set to not be Null
    (nullable=False). Non nullable table attributes (reflecting db table
    columns)  not present in dataframe columns are not accounted for: in other
    words, the non-nullable condition on the dataframe is set for those columns
    only which have a corresponding name in any of the table attributes.
    Consider calling `harmonize_cols` first to make sure the column values
    align with the table column types

    :param inplace: argument to be passed to pandas `dropna`
    """
    non_nullable_cols = list(shared_colnames(table, dataframe, nullable=False))
    # `dropna` below accepts also generators, so list(...) is only used
    # to check if we have elements:
    if non_nullable_cols:
        tmp = dataframe.dropna(subset=non_nullable_cols, axis=0, inplace=inplace)
        if not inplace:  # if inplace, tmp is None and dataframe has been modified
            dataframe = tmp
    return dataframe


def harmonize_columns(table, dataframe, parse_dates=None):
    """Make the DataFrame's column types align with the SQL table
    column types. Returns a new dataframe with "correct" types (according to
    table)
    Columns which are *not* shared with table columns (assuming dataframe
    columns are strings) are left as they are. Columns of the table are assumed
    to be its attribute names (as typed in the code), thus the DataFrame is
    assumed to have string columns as well. The returned dataframe row numbers
    is not modified

    :param table: an ORM model class
    """
    _, dfr = _harmonize_columns(table, dataframe, parse_dates)
    return dfr


def _harmonize_columns(table, dataframe, parse_dates=None):
    """Copied and modified from pandas.io.sql:
    Make the DataFrame's column types align with the SQL table
    column types. The original dataframe dtypes and values MIGHT be modified
    in place!

    Modified because it uses pandas.to_numeric when df.as_type fails. It
    should silently converts all non-numeric non-date values to NaN and NaT
    respectively, the only drawback is that int types are float64 in case,
    which **might** be a problem with some SQL backends which require ints
    and are not casting it. The issue is handled by all method of this modules
    (for details, see `setpkeys` in case)

    :param parse_dates: a dict of `dataframe` column names which need to be
        forcibly casted to datetime(s). The dict values can be None (infer the
        cast format) or a letter in ['D', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']
        denoting the unit of the column values, if they are numeric. If this
        parameter is falsy (e.g. None) it defaults to the empty dict

    Original pandas doc:
    Need to work around limited NA value support. Floats are always
    fine, ints must always be floats if there are Null values.
    Booleans are hard because converting bool column with None replaces
    all Nones with false. Therefore only convert bool if there are no
    NA values.
    Datetimes should already be converted to np.datetime64 if supported,
    but here we also force conversion if required
    by means of parse_dates (a list of strings denoting the name of the
    additional columns to be parsed as dates. None by default)
    """
    if not parse_dates:
        parse_dates = {}

    column_names = []  # added by me
    # note below: it seems that the original pandas module uses Column objects
    # however, some properties (such as type) are also available in
    # InstrumentedAttribute (getattr(table, col_name)). So we use the latter
    for col_name in colnames(table):
        sql_col = getattr(table, col_name)
        # changed, in pandas was: 'sql_col.name', but we want to harmonize
        # according to the attribute names
        try:
            df_col = dataframe[col_name]
            # the type the dataframe column should have
            col_type = _get_dtype(sql_col.type)

            if col_type is datetime or col_type is date:
                # (removed `or col_type is DatetimeTZDtype` from the if above)
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
                # integers. Try with normal way. The original code checked
                # NaNs like this: if len(df_col) == df_col.count():
                # but this raises on e.g., non numeric strings, which we want
                # to convert to NaN. So, try the "normal way":
                try:
                    dataframe[col_name] = df_col.astype(col_type, copy=False)
                except (TypeError, ValueError):
                    # failed, use to_numeric coercing to None on errors
                    # this also sets the column dtype to float64
                    dataframe[col_name] = pd.to_numeric(df_col, errors='coerce')
            elif col_type is bool:
                # boolean seems to convert without errors but
                # converts NaN's to True, None to False. So, for preserving
                # None's and NaN:
                bool_col = df_col.astype(col_type, copy=True)
                # in the presence of Nones, the line below will convert type
                # from bool to float
                # to account for Nones (stored as nans):
                bool_col[pd.isnull(df_col)] = None
                dataframe[col_name] = bool_col

            # OLD CODE (commented out):
#             elif len(df_col) == df_col.count():
#                 # No NA values, can convert ints and bools
#                 if col_type is np.dtype('int64') or col_type is bool:
#                     dataframe[col_name] = df_col.astype(
#                         col_type, copy=False)

            # Handle date parsing. Try to get if it is a dict, in that case
            # use the values as the format argument, otherwise use None:
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


def _handle_date_column(col, format=None):  # noqa
    """Modified from `pandas.io.sql._handle_date_column` for parsing
    datetime(s) which are supposed to be in UTC.

    :param col: pandas Series or numpy array with values to be cated to datetime
    :param format: either None (in that case the column values are supposed to
        be parsable as datetime) or a string in
        ['D', 'd', 'h', 'm', 's', 'ms', 'us', 'ns'] denoting the unit of `col`,
        in which case `col` must be numeric. If `col` is numeric and `format`
        is None, then `format` defaults to 's' (seconds)
    """
    if format is None and (issubclass(col.dtype.type, np.floating) or
                           issubclass(col.dtype.type, np.integer)):
        format = 's'  # @ReservedAssignment
    if format in ['D', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']:
        return to_datetime(col, errors='coerce', unit=format)
    return to_datetime(col, errors='coerce', format=format)


def _insert_data(dataframe):
    """Convert dataframe to a numpy array of arrays for insertion inside a db
    table.

    :return: a tuple of two elements (cols, data):
        cols denotes the dataframe columns (list of strings),
        data denotes the dataframe data (converted and casted) to be inserted.
        NOTE: Each element of data denotes a COLUMN. Thus len(data)==len(cols).
        For each column, data[i] has M elements denoting the M rows
        (where M = len(dataframe)). Thus, to get each table row as array, use:
            zip(*data)
        To get each table row as dict colname:value, use:
            for row in zip(*data):
                row_as_dict = dict(zip(cols, row_values))

    (this code is copied from pandas.io.sql.SqlTable._insert_data)
    """
    column_names = list(map(text_type, dataframe.columns))
    ncols = len(column_names)
    data_list = [None] * ncols
    blocks = dataframe._data.blocks

    for i in range(len(blocks)):
        b = blocks[i]
        if b.is_datetime:
            # in pandas versions before 1.0, b.values is a numpy ndarray of
            # dtype datetime64[ns]
            # After, it might be a DatetimeArray IF ALL items are all not
            # pd NaT (so we might jump in any case to the else below)
            # Looking at the code
            # (https://github.com/pandas-dev/pandas/blob/master/pandas/io/sql.py#L700)
            # we can do like this:
            if b.is_datetimetz:
                # this is equivalent to say: we have a DatetimeArray
                # (or DatetimeIndex?). Why we will never be here in pd < 0.24
                # using the same tests is also a mystery. However, we use
                # the code provided at the link above with a single addition:
                # remove the timezone as we do not work with timezone aware
                # datetimes and also make the objects returned by this "if"
                # branch and the "else" below the same
                # (tz_convert(tz) convert tz-aware Datetime Array/Index from
                # one time zone to another. A `tz` of None will
                # convert to UTC and remove the timezone information)
                d = b.values.tz_convert(None).to_pydatetime()
                # Need to return 2-D data; DatetimeArray is 1D
                d = np.atleast_2d(d)
            else:
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
    """Returns an efficient iterator over `dataframe` rows. The i-th returned
    values is a `dict`s of `dataframe` columns (strings) keyed to the i-th row
    values. Each value is assured to be a Python type (str, bool, datetime, int
    and float are currently supported) with pandas null values (NaT, NaN)
    converted to None, if any.

    :param dataframe: the input dataframe
    """
    cols, datalist = _insert_data(dataframe[columns] if columns is not None
                                  else dataframe)
    # Note below: datalist is an array of N column, each of M rows (it would be
    # nicer to return an array of N rows, each of them representing a table row.
    # But we do not want to touch pandas code. See _insert_table below).
    # Thus we zip it:
    for row_values in zip(*datalist):
        yield dict(zip(cols, row_values))


def _get_max(session, numeric_column):
    """Return the maximum value from a given numeric column, usually a primary
    key with auto-increment=True. If it's the case, from `_get_max() + 1` we
    assure unique identifier for adding new objects to the table of
    `numeric_column`. SqlAlchemy has the ability to set autoincrement for us
    but telling explicitly the id value for an autoincrement primary key
    speeds up *a lot* the insertion (especially if used in conjunction with
    SQLAlchemy core methods

    :param session: an sqlalchemy session
    :param numeric_column: a column of an ORM model (mapping a db table column)
    """
    return session.query(func.max(numeric_column)).scalar() or 0


def dbquery2df(query):
    """Return a query result as a dataframe

    :param query: SqlAlchemy query. IT MUST BE GIVEN WITH ALL COLUMNS OF
        INTEREST SEPARATED, e.g.:
        ```session.query(Table.columna, Table.column_b)```
        and **not**:
        ```session.query(Table)```

        It accepts joins and filters, e.g.:
        ```session.query(Table.columna, Table.column_b).join(...).filter(...)```

        And also expressions as column, e.g.:
        ```session.query(Table.columna, (Table.column_b >0).label('abc'))```
        Where `label` associated to the query will be the dataframe column name
        (when passing normal columns, the data frame column name is inferred
        from the SQLAlchemy column name)
    """
    columns = [c['name'] for c in query.column_descriptions]
    return pd.DataFrame(columns=columns, data=query.all())


def syncdf(dataframe, session, matching_columns, id_col, update=False,
           buf_size=10, keep_duplicates=False, onduplicates_callback=None,
           oninsert_err_callback=None, onupdate_err_callback=None):
    """Synchronize efficiently `dataframe` with the corresponding database
    table T.

    Returns the tuple:

    inserted, not_inserted, updated, not_updated, synced_dataframe

    where:

    * inserted: the number of rows of `dataframe` inserted (new in the table)
    * not_inserted: the number of rows of `dataframe` NOT inserted (sql
      constraint error)
    * updated: 0 if `update` is False, otherwise the number of rows of
      `dataframe` updated
    * not_updated: 0 if `update` is False, otherwise the number of rows of
      `dataframe` NOT updated (sql constraint error)
    * synced_dataframe: The pandas Data frame subset of `dataframe` with rows
      SURELY mapped with an existing row on the table T. This includes rows of
      `dataframe` which already had a mapped row on T, or were successfully
      inserted or updated. `id_col` should be a Numeric Unique SQLAlchemy
      Column (e.g., INTEGER primary key); `dataframe[id_col]` is assured to
      exist and will NOT have NA (nan's, None's), and its dtype will be casted
      to the python type corresponding to the SQL type of its matching column
      on T.
      NOTE: **The order of rows of `synced_dataframe` might not match the order
      of `dataframe`, nor its index (pd.Index), so do not rely on them to match
      a row of `dataframe` with a row of `synced_dataframe`.**
      The length of `synced_dataframe` will be `>= inserted`, or equal to
      `inserted+updated` if `update` is True or non empty list (see below)

    This function first fetches the primary keys
    from the database table into `dataframe[id_col.key]`, matching columns with
    `matching_columns`, then uses a `DbManager` which internally splits
    `dataframe` into rows to insert (`id_col` NA) and rows to update, and
    inserts/update them committing chunks of `buf_size` rows.
    If you need to insert/update a lot of items and/or you do not care about
    the returned Data frame, you can use a `DbManager` which has a lower level
    approach (i.e., more typing) but its faster

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe`
        rows and T rows: when two rows are found that are equal (according to
        all `matching_columns` values), then the data frame row `id_col` value
        is set = T row value
    :param id_col: the ORM column denoting a NUMERIC and UNIQUE Column of T
        (e.g., INTEGER primary key): unexpected results if the column does not
        match those criteria. The column needs not to be a column of
        `dataframe`. The returned `dataframe` will have in any case this column
        set with non-NA values and the proper python type (corresponding to
        the column  SQL type)
    :param update: boolean or list of strings. Whether to update or not:
        - If True, all shared columns between dataframes and table model will
          be updated (except id_col): the shared columns are calculated only
          the first time a dataframe is added to this object.
        - If list of STRINGS, then the columns which matching names are updated
          only (the string name of id_col should not be in the list)
        - If False (or, in general falsy, so empty list or None is the same):
          do not update
    :param buf_size: integer, defaults to 10. The buffer size before committing.
        Increase this number for better performances (speed) at the cost of some
        "false negative" (committing a series of operations where one raise an
        integrity error discards all subsequent operations regardless if they
        would raise as well or not)
    :param keep_duplicates: boolean or string in 'first', 'last': if True,
        duplicates of `dataframe` under `matching_columns` are not checked, if
        False, they are dropped, if 'first' ('last'), only first (last) row of
        each duplicate group are kept, and the rest is dropped (when not True,
        this is the `keep` argument passed to :meth:`DataFrame.drop_duplicates`)
    :param onduplicates_callback: function, or None. A function executed when
        removing duplicates, if `drop_duplicates` is True. It is called with
        two arguments:
        - `dataframe` with duplicated rows only
        - an exception
        Set to None to not execute any callback on duplicates
    :param oninsert_err_callback: function, or None. A function executed on SQL
        insert errors. It is called with two arguments:
        - `dataframe` with non-inserted rows only (its maximum length will be
          `buf_size`)
        - the sqlalchemy exception
        Set to None to not execute any callback on insert errors
    :param onupdate_err_callback: function, or None. A function executed on SQL
        update errors, if `update` is True or non-empty string list. It is
        called with two arguments:
        - `dataframe` with non-updated rows only (its maximum length will be
          `buf_size`)
        - the sqlalchemy exception
        Set to None to not execute any callback on update errors

    Technical notes
    ===========================================================================

    1. T is obtained as the `class_` attribute of the first passed
    `Column <http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column>`_,
    therefore `id_col` and each element of `matching_columns` must refer to the
    same db table T.
    2. The mapping between an sql-alchemy Column C and a pandas dataframe *str*
    column K is based on the sql-alchemy `key` attribute: `C.key == K`
    3. On the db session side, we do not use ORM functionalities but lower
    level sql-alchemy core methods, which are faster (FIXME: ref needed).
    This, together with the "buffer size" argument, speeds up a lot items
    insertion on the database. The drawbacks of these approaches is that the
    method needs to create the primary keys before inserting a row to T, and
    that if a single item of a buffer raises an SqlAlchemtError, all following
    items are not added to the db, even if they where well formed
    """

    if keep_duplicates is not True:
        dupes_mask = dataframe.duplicated(subset=[k.key for k in matching_columns],
                                          keep=keep_duplicates)
        if dupes_mask.any():
            if onduplicates_callback:
                onduplicates_callback(dataframe[dupes_mask],
                                      Exception("Duplicated instances violate "
                                                "db constraint"))
            dataframe = dataframe[~dupes_mask].copy()

    dframe_with_pkeys = syncdfcol(dataframe, session, matching_columns, id_col)
    dbm = DbManager(session, id_col,
                    update, buf_size, return_df=True,
                    oninsert_err_callback=oninsert_err_callback,
                    onupdate_err_callback=onupdate_err_callback)
    dbm.add(dframe_with_pkeys)
    table, inserted, not_inserted, updated, not_updated = dbm.close()
    # dframe_with_pkeys's `id_col` might not be castable to `id_col`
    # SQL type: think about SQL type = INTEGER, and dframe_with_pkeys has
    # Nones: then dframe_with_pkeys[id_col].dtype = float, not int.
    # d.dataframe's `id_col` is surely castable to the SQL type, and
    # *in general* DbManager already casted it. But not always. Thus for safety:
    dataframe = cast_column(dbm.dataframe, id_col)

    return inserted, not_inserted, updated, not_updated, dataframe


class DbManager(object):
    """Class managing the insertion of table rows into db. This class is
    optimized for adding several dataframes in series, but can be used also to
    insert/update a single dataframe in one
    shot:

    ```
        d = DbManager(..., return_df=False)  # make insertion / updates faster
        d.add(first_dataframe)
        ...
        d.add(last_dataframe)
        # get table model and stats:
        table, inserted, not_inserted, updated, not_updated = d.close()
    ```

    or

    ```
        d = DbManager(..., return_df=True)
        d.add(first_dataframe)
        ...
        d.add(last_dataframe)
        d.close()
        # get the dataframe synchronized with the db (primary keys set):
        synced_dataframe = d.dataframe
    ```

    NOTES:

    `id_col` is a sql-alchemy Column which MUST be NUMERIC and UNIQUE (e.g.,
    the typical case of an INTEGER primary key)

    The database SHOULD NOT BE MODIFIED in within each `add` call, as the
    "next value" of `id_col` is retrieved once at the beginning and stored
    internally for performance reasons

    Do not rely on the index of `d.dataframe` to be the same of whatever
    DataFrame passed in `d.add`

    The `add` method takes care to split between rows to update and rows to
    insert, eventually writing to the database when the amount of rows exceeds
    `buf_size`: this should avoid memory issues and be more efficient. The
    dataframes passed to `add` should have always the same columns and types,
    the columns need to be a subset of the underlying table columns. If you
    want to return the dataframe **synchronized** with the db, pass
    return_df=True and, after closing this classs, call the `dataframe`
    property
    """

    def __init__(self, session, id_col, update, buf_size, return_df=False,
                 oninsert_err_callback=None, onupdate_err_callback=None):
        """Initialize a new `DbManager`

        :param id_col: an SQLAlchemy Column. It MUST be of type numeric and
            UNIQUE (e.g. INTEGER primary key). The mapped ORM Table will be
            inferred from this attribute. For each dataframe passed to `add`,
            identified the dataframe column C mapped to id_col (having the same
            name), the rows to insert will be those with Nones under C, and the
            others those to update
        :param return_df: (boolean, False by default) if True, the property
            `self.dataframe` will return the db-synced dataframe. Setting this
            argument to True might increase memory allocation and time speed

        :param update: boolean or list of strings. Whether to update or not:
            - If True, all shared columns between dataframes and table model
              will be updated (except id_col): the shared columns are
              calculated only the first time a dataframe is added to this
              object.
            - If list of STRINGS, then the columns which matching
              names are updated only (the string name of id_col should not be
              in the list)
            - If False (or, in general falsy, so None works): do not update

        """
        # inserted, total_to_insert, updated, total_to_update:
        self.info = [0, 0, 0, 0]
        self.inserts = []
        self.updates = []
        self.buf_size = buf_size
        self.session = session
        self.id_col = id_col
        # True or a list of strings. True: update all shared columns:
        self.colnames2update = update
        self.colnames2insert = None  # will be populated at the 1st insert only
        self.table = id_col.class_
        self.return_df = return_df
        self.dfs = []
        self._toinsert_count = 0
        self._toupdate_count = 0
        self._max = _get_max(session, id_col)
        self.oninsert_err_callback = oninsert_err_callback
        self.onupdate_err_callback = onupdate_err_callback

    def add(self, dframe):
        """Add the given dataframe to be inserted or updated (according to
        dframe[self.id_col.key])

        :param dframe: the dataframe. It MUST have `self.id_col.key` as column,
            either NA or non-NA
        """
        if dframe.empty:
            # if the dataframe is empty do nothing and return
            # But append a copy to self.dfs if self.return_df is True.
            # This way if self.dataframe should return an empty dataframe
            # it will have at least the proper (expected) columns
            if self.return_df and not self.dfs:
                self.dfs.append(dframe.copy())
            return

        bufsize = self.buf_size
        dfinsert, dfupdate = None, None

        mask = pd.isnull(dframe[self.id_col.key])
        if mask.all():
            dfinsert = dframe
        else:  # some elements have non-na id, thus they SHOULD be updated

            if mask.any():
                dfinsert = dframe[mask]

            performupdate = self.colnames2update

            if performupdate or self.return_df:
                dfupdate = dframe if dfinsert is None else \
                    dframe[~mask]  # pylint: disable=invalid-unary-operand-type
                if not performupdate:
                    # in this case, we do not want to update existing rows, but
                    # we want to return the dataframe in the `self.dataframe`
                    # property. Thus, append dfupdate to self.dfs AND set
                    # dfupdate to None (==skip update see below)
                    self.dfs.append(dfupdate)
                    dfupdate = None

        if dfinsert is not None:
            newinserts = len(dfinsert)
            if newinserts:
                self.inserts.append(dfinsert)
                total_inserts_count = self._toinsert_count + newinserts
                if total_inserts_count >= bufsize:
                    self._insert(bufsize)  # reset self._toinsert_count to 0
                else:
                    self._toinsert_count = total_inserts_count

        if dfupdate is not None:
            newupdates = len(dfupdate)
            if newupdates:
                self.updates.append(dfupdate)
                total_updates_count = self._toupdate_count + newupdates
                if total_updates_count >= bufsize:
                    self._update(bufsize)  # reset self._toupdate_count to 0
                else:
                    self._toupdate_count = total_updates_count

    @property
    def dataframe(self):
        """Return the dataframe of all inserted / updated / existing instances
        (rows).
        You should close this object or call flush before calling this method.

        Note that if id_col passed in the constructor is of type int, it might
        be of type float. Use :meth:`cast_column` in case, the returned
        dataframe[id_col.key] is assured NOT to have NaNs Raises if
        self.return_df is False
        """
        if not self.return_df:
            raise ValueError('return_df is False')
        dfs = self.dfs
        return pd.DataFrame() if not dfs else \
            pd.concat(dfs, axis=0, ignore_index=False, copy=False,
                      verify_integrity=False)

    def _insert(self, buf_size=None):
        inserts = self.inserts
        # pd.concat *must* copy the data exceot for trivial cases, e.g. the
        # data frame to be concat is 1: in this case we set the arg. copy=True
        # to avoid SettingsWithCopy warning. Also `ignore index` only if we
        # have to concat more dataframes: we should simply set
        # ignore_index=True but we have legacy code tests failing (false
        # positives) if we don't
        dfr = pd.concat(inserts, axis=0, ignore_index=len(inserts) > 1, copy=True,
                        verify_integrity=False)
        session = self.session
        id_col = self.id_col
        # Set the dfr[id_col.key] with primary key values for all rows
        # (dfr must not have such a column). See also NOTE below
        dfr = syncdfseq(dfr, session, id_col, overwrite=True, pkeycol_maxval=self._max)
        insert_cols = self.colnames2insert
        if insert_cols is None:
            insert_cols = self.colnames2insert = _get_shared_colnames(self.table, dfr)
        total = len(dfr)
        self._max += total
        if buf_size is None:
            buf_size = min(self.buf_size, self._toinsert_count)
        return_df = self.return_df
        # IMPORTANT NOTE on line below: id_col must be of sql INTEGER type.
        # On insert, if dfr[id_col] is float (which might happen as pandas
        # converts integer columns with NaNs/Nones to float), THEN WE HAVE A
        # PROBLEM on postgres because it complains if IDs are not strict
        # integers (so e.g. 6.0 is NOT a valid id). The problem was solved by
        # calling `syncdfseq` above
        new, dfr = insertdf(dfr, session, id_col.class_, insert_cols,
                            buf_size=buf_size,
                            return_df=return_df,
                            onerr=self.oninsert_err_callback)
        if return_df:
            self.dfs.append(dfr)
        info = self.info
        info[0] += new
        info[1] += total
        # cleanup:
        self._toinsert_count = 0
        # this is faster in py3 than del inserts[:] (=`del self.inserts[:]`):
        self.inserts = []

    def _update(self, buf_size=None):
        updates = self.updates
        # pd.concat *must* copy the data except for trivial cases, e.g. the
        # data frame to be concatenated is 1: in this case we set the arg.
        # copy=True to avoid SettingsWithCopy warning. Also `ignore index` only
        # if we have to concat more dataframes: we should simply set
        # ignore_index=True but we have legacy code tests failing (false
        # positives) if we don't
        dfr = pd.concat(updates, axis=0, ignore_index=len(updates) > 1, copy=True,
                        verify_integrity=False)
        if buf_size is None:
            buf_size = min(self.buf_size, self._toupdate_count)
        id_col = self.id_col
        # Set the dfr[id_col.key] with proper type
        # (dfr must have such a column, and no NaN/None). See also NOTE below
        dfr = cast_column(dfr, id_col)
        update_cols = self.colnames2update
        if update_cols is True:
            update_cols = self.colnames2update = \
                _get_shared_colnames(self.table, dfr, id_col)
        total = len(dfr)
        return_df = self.return_df
        # IMPORTANT NOTE on line below: id_col must be of sql INTEGER type.
        # On update, if dfr[id_col] is float (which might happen as pandas
        # handles integer columns with NaNs/Nones by storing the column as
        # float), THEN WE HAVE A PROBLEM on postgres when using the primary key
        # in a where clause: updates work but are simply horribly SLOW (3 to 4
        # times slower). The problem was solved by calling `cast_column` above
        updated, dfr = updatedf(dfr, self.session, id_col, update_cols,
                                buf_size=buf_size, return_df=return_df,
                                onerr=self.onupdate_err_callback)
        if return_df:
            self.dfs.append(dfr)
        info = self.info
        info[2] += updated
        info[3] += total
        # cleanup:
        self._toupdate_count = 0
        # this is faster in py3 than del inserts[:] (=`del self.inserts[:]`):
        self.updates = []

    def flush(self):
        """Flushe remaining stuff to insert/ update, if any"""
        if self.inserts:
            self._insert()
        if self.updates:
            self._update()

    def close(self):
        """Flushe remaining stuff to insert/ update, if any, prints to log
        updates and inserts. Returns the tuple
        `table, inserted, not_inserted, updated, not_updated`
        """
        self.flush()
        new, ntot, upd, utot = self.info
        return self.table, new, ntot - new, upd, utot - upd


def _get_shared_colnames(table_model, dataframe, where_col=None):
    """Return a list of shared column names between table_model and dataframe.
    If where_col is not None, it will be excluded from the returned list
    (where_col is assumed to be a column used in the where clause of an update
    and thus it should not be included in the columns to update)
    """
    shared_colnames_gen = shared_colnames(table_model, dataframe)
    if where_col is not None:
        wherecolname = where_col.key
        shared_colnames_gen = (cname for cname in shared_colnames_gen
                               if cname != wherecolname)
    return list(shared_colnames_gen)


def syncdfseq(dataframe, session, seq_col, overwrite=False, pkeycol_maxval=None):
    """Synchronize `dataframe[seq_col.key]` with the underlying database table T,
    setting values not in T by auto-incrementing the sequence of values (thus
    `seq_col` must be numeric and having unique constraint, e.g. an integer
    primary key).

    If 'overwrite', it overwrites the values of dataframe[seq_col], otherwise
    writes only NA values. This argument is ignored if `seq_col` is not a column
    of `dataframe` (the column will be added in case).
    If `pkeycol_maxval` is not None, sets the `seq_col` values from
    `pkeycol_maxval + 1`: this is faster as it does not query the db but the
    user is repsonsible not to violate constraints, if the dataframe is later
    inserted / updated to the db. If None, `pkeycol_maxval` default to the
    Database Table's maximum.

    The database Table is retrieved as the table mapped by the model of
    `seq_col`. Regardless of whether dataframe has the column or not,
    After this call, `dataframe` will have the column with name `seq_col.key`
    casted to the pandas type corresponding to `seq_col` type.

    :param session: an sql-alchemy session object
    :param seq_col: an SQLAlchemy Column, i.e. an attribute of some ORM class
        representing a db Table. The column must denote a sequence, i.e. must
        be of SQL type **NUMERIC** and unique (e.g. integer primary key),
        otherwise this method should not be used.
    :param dataframe: the dataframe with values to be inserted/updated/deleted
        from the table mapped by `seq_col`
    """
    if pkeycol_maxval is None:
        pkeycol_maxval = _get_max(session, seq_col)
    pkeycol_maxval += 1
    pkeyname = seq_col.key
    if not overwrite and pkeyname in dataframe:
        # Treat here the case where we have to set 0 to len(dataframe)-1 values
        # If we have all NaNs values, treat the case as if we did not
        # have the column (goto case below)
        mask = pd.isnull(dataframe[pkeyname])
        nacount = mask.sum()
        if nacount != len(dataframe):
            if nacount > 0:
                dataframe.loc[mask, pkeyname] = \
                    np.arange(pkeycol_maxval, pkeycol_maxval+nacount,
                              dtype=_get_dtype(seq_col.type))
            # cast values if we modified only SOME row values of
            # dataframe[pkeyname]. E.g., if dataframe[seq_col.key] was of type
            # float because it has NaNs, and seq_col is of type integer, we
            # must cast (E.g., postgres raises or is extremely slow if we pass
            # 6.0 instead of 6 in an insert/update!)
            return cast_column(dataframe, seq_col)

    # if we are here
    # either we want to set all values of dataframe[pkeyname] (overwrite=True),
    # or pkeyname is not a column of dataframe,
    # or all dataframe[pkeyname] are na
    # In ALL these cases we do not need `cast_column`, but simply set the dtype
    # in np.arange:
    new_pkeys = np.arange(pkeycol_maxval, pkeycol_maxval+len(dataframe),
                          dtype=_get_dtype(seq_col.type))
    dataframe[pkeyname] = new_pkeys
    return dataframe


def cast_column(dataframe, sql_column):
    """Cast the dataframe column mapped to `sql_column` to the Python type
    mapped to `sql_column`'s sql type.
    dataframe[sql_column.key] MUST be a valid column in dataframe, and must
    have values castable (e.g., non-NAN's in case of int's - the usual case as
    sql_column is often a primary key).

    :return: dataframe with the column casted
    """
    col_type = _get_dtype(sql_column.type)
    pkeyname = sql_column.key
    if dataframe[pkeyname].dtype != col_type:
        dataframe[pkeyname] = dataframe[pkeyname].astype(col_type, copy=False)
    return dataframe


def insertdf(dataframe, session, table_model, colnames2insert=None,
             buf_size=10, return_df=True,
             onerr=None):
    """Efficiently inserts row of `dataframe` to the Table T mapped by the ORM
    `table_model`. This function performs a sort of "raw" insert with no check,
    thus any kind of constraint defined on T must be satisfied by `dataframe`.
    For instance, if T defines a primary key with some sort of auto sequence
    (INTEGER auto increment), then `dataframe` needs to define such a column,
    with correct values and types (Note: SQLite seems to handle missing primary
    keys, auto-incrementing them, postgres not. Thus it is not safe to omit
    those columns in `dataframe`.  If you want to set automatically primary key
    value / numeric sequence / numeric column with unique constraint, see
    :meth:`syncdfseq`). If you want a more "high-level" method taking care of
    handling insert/updates and synchronization, see :meth:`syncdf`.

    Returns the tuple `new, df` where:

    * new: is the number of new rows inserted
    * df is the pandas DataFrame with same columns as `dataframe` and only
        rows that are succesfully inserted. If return_df=False, this argument
        is None (in case, this function should run faster)

    .. seealso:: `syncdfseq`
    .. seealso::  `syncdf`

    :param dataframe: a pandas dataframe
    :param session: the sql-alchemy session
    :param table_model: an SQLAlchemy ORM class mapping some database table
    :param colnames2insert: a list of columns to be inserted. None will default
        to all `dataframe` columns. This latter case might be more time
        consuming if this method is called several times

    The remainder of the documentation is the same as `syncdf`, so please see
    there for details
    """
    if dataframe.empty:
        return 0, dataframe if return_df else None

    buf_size = max(buf_size, 1)
    buf = {}

    if colnames2insert is None:
        colnames2insert = _get_shared_colnames(table_model, dataframe)

    last = len(dataframe) - 1
    not_inserted = 0
    indices_discarded = []

    for i, rowdict in enumerate(dfrowiter(dataframe, colnames2insert)):
        buf[i] = rowdict
        if len(buf) == buf_size or (i == last and buf):
            try:
                session.connection().execute(table_model.__table__.insert(),
                                             listvalues(buf))
                session.commit()
            except SQLAlchemyError as sa_exc:
                session.rollback()
                not_inserted += len(buf)
                if onerr is not None:
                    onerr(dataframe.iloc[listkeys(buf)], sa_exc)
                if return_df:
                    indices_discarded.extend(iterkeys(buf))

            buf.clear()

    new = len(dataframe) - not_inserted
    ret_df = None
    if return_df:
        ret_df = dataframe
        if not_inserted:
            if not_inserted == len(dataframe):
                ret_df = dataframe.iloc[[]]  # empty dataframe, preserving cols
            else:
                indices_discarded = np.array(indices_discarded, dtype=int)
                indices = np.in1d(np.arange(len(dataframe)), indices_discarded,
                                  assume_unique=True, invert=True)
                ret_df = dataframe.iloc[indices]

    return new, ret_df


def updatedf(dataframe, session, where_col, colnames2update=None, buf_size=10,
             return_df=True, onerr=None):
    """Update efficiently rows of `dataframe` to the corresponding database
    table T (whose ORM will be retrieved by means of `where_col`).

    Returns the tuple:
    ```
    (updated, d)
    ```
    where:

    * updated is the number of rows successfully updated (no sql errors)
    * d is None if return_df = None, otherwise the sub-set of `dataframe` with
      only updated rows. Its length is 'updated'

    :param where_col: a SQLALchemy Column indicating the column whereby the SQL
        where clause is issued (usually, a primary key or a column with unique
        constraints). IMPORTANT: `dataframe[where_col.key]` type should match
        the SQL type. See :meth:`cast_column` in case
    :param colnames2update: a list of columns to be updated. None will default
        to all `dataframe` columns EXCEPT `where_col`. This latter case might
        be more time consuming if this method is called several times

    `return_df=False` is in most cases faster, use it if you do not need a
    database-synchronized version of `dataframe`

    The remainder of the documentation is the same as `syncdf`, so please see
    there for details
    """
    if dataframe.empty:
        return (0, dataframe if return_df else None)

    table_model = where_col.class_
    if colnames2update is None:
        colnames2update = _get_shared_colnames(table_model, dataframe, where_col)

    where_col_name = where_col.key
    shared_cnames = [where_col_name] + colnames2update
    # find a col not present for where_col. Otherwise error is raised:
    # bindparam() name where_col.key is reserved for automatic usage in the
    # VALUES or SET clause of this  insert/update statement.   Please use a
    # name other than column name when using bindparam() with insert() or
    # update() (for example, 'b_id').
    where_col_bindname = where_col_name + "_"
    while where_col_bindname in shared_cnames:  # assure uniqueness
        where_col_bindname += "_"
    stmt = table_model.__table__.update().\
        where(where_col == bindparam(where_col_bindname)).\
        values({c: bindparam(c) for c in colnames2update})
    buf = {}
    last = len(dataframe) - 1
    indices_discarded = []
    not_updated = 0

    for i, rowdict in enumerate(dfrowiter(dataframe, shared_cnames)):
        # replace the where column:
        rowdict[where_col_bindname] = rowdict.pop(where_col_name)
        buf[i] = rowdict
        if len(buf) == buf_size or (i == last and buf):
            try:
                session.connection().execute(stmt, listvalues(buf))
                session.commit()
            except SQLAlchemyError as sa_exc:
                session.rollback()
                not_updated += len(buf)
                if onerr is not None:
                    onerr(dataframe.iloc[listkeys(buf)], sa_exc)
                if return_df:
                    indices_discarded.extend(iterkeys(buf))

            buf.clear()

    updated, ret_df = last + 1 - not_updated, None

    if return_df:
        ret_df = dataframe
        if not_updated:
            if not_updated == len(dataframe):
                ret_df = dataframe.iloc[[]]  # empty dataframe, preserving cols
            else:
                indices_discarded = np.array(indices_discarded, dtype=int)
                indices = np.in1d(np.arange(len(dataframe)), indices_discarded,
                                  assume_unique=True, invert=True)
                ret_df = dataframe.iloc[indices]

    return updated, ret_df


def syncdfcol(dataframe, session, matching_columns, sync_col):
    """Synchronize `dataframe[sync_col.key]` from the underlying database
    Table T. Fetches the values from T, identifies matching rows by means of
    `matching_columns`, and sets the value of `dataframe[sync_col.key]` for the
    matching rows. `dataframe` does not need to have that column in the first
    place (it will be added if not present). Dataframe rows not identified on
    the database will have NaN/Null under `sync_col`

    NOTE: If sync_col is of SQL type INTEGER, the dtype of the returned
        dataframe[sync_col.key]'s dtype might be float to accomodate NaN's, if
        any. Note that postgres is strict and will issue an
        `sqlalchemy.exc.DataError` if inserting/updating a non-nan value (e.g.,
        6.0 instead of 6), and it's also terribly slow in some updates when a
        where clause is made on a float column supposed to be 'int'. The cast
        cannot be done here as the column might have nan's not convertible to
        int. If there are non-NaNs, see :function:`cast_column` for casting.

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe`
        rows and T rows: when two rows are found that are equal (according to
        all `matching_columns` values), then the value of T row's `sync_col`
        is set on the `dataframe` corresponding row
    :param sync_col: the ORM column denoting the column to be synchronized.
        It does not need to be a column of `dataframe`

    :return: a new data frame with the column `sync_col` populated with the
        values of T. Values that are n/a, None's or NaN's (see
        `pandas.DataFrameisnull`) denote rows that do not have corresponding T
        row and might need to be added to T. The index of `d` is **not** reset,
        so that a track to the original dataframe is always possible (the user
        must issue a `d.reset_index` to reset the index).

    Technical notes:
    1. T is retrieved by means of the passed
       `Columns <http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column>`_,
       therefore `autoincrement_pkey_col` and each element of
       `matching_columns` must refer to the same db table T.
    2. The mapping between an sql-alchemy Column C and a pandas dataframe *str*
       column K is based on the sql-alchemy `key` attribute: `C.key == K`
    3. On the db session side, we do not use ORM functionalities but lower
       level sql-alchemy core methods, which are faster (FIXME: ref needed).
       This, together with the "buffer size" argument, speeds up a lot items
       insertion on the database. The drawback of the former is that we need to
       create by ourself the primary keys, the drawback of the latter is that
       if a single item of a buffer raises an `SqlAlchemyError`, all following
       items are not added to the db, even if they where well formed
    """
    cols = matching_columns + [sync_col]
    df_new = dbquery2df(session.query(*cols).distinct())
    return mergeupdate(dataframe, df_new, [c.key for c in matching_columns],
                       [sync_col.key], False)


def mergeupdate(dataframe, other_df, matching_columns, merge_columns,
                drop_other_df_duplicates=True):
    """Merge `other_df` into `dataframe` and returns the latter, by setting
    `dataframe[merge_columns]` = `other_df[merge_columns]` for those row where
    `dataframe[matching_columns]` = `other_df[matching_columns]` only.

    Example:
    `dataframe` and `other_df` have three columns in common: `id` (int),
    `name` (str) and `time` (datetime).

    `dataframe` has also a column `data` (int):
    ```
    >>> dataframe
        id  name       time  data
    0   45     a        NaT     4
    1   45     b 2006-01-01     5
    ```

    `other_df` has also the columns `count` (int) and `value` (float):
    ```
    >>> other_df
         id name       time  count  value
    0  45.0    a 2008-01-01      5    NaN
    1  45.0    c 2006-01-01      5    4.5
    ```

    If we merge the to dataframes using 'id' and 'name' tuples as row
    identifiers, and merging only the columns 'time' and 'value', we get:
    ```
    >>> mergeupdate(dataframe, other_df, ['id', 'name'], ['time', 'value'])
       id name       time  data  value
    0  45    a 2008-01-01     4    NaN
    1  45    b 2006-01-01     5    NaN
    ```

    Note:
    1. The second row of `other_df` is NOT added to `dataframe` as according to
       `matching_columns` it does not exist on `dataframe`
    2. `other_df` **should** have unique rows under `matching columns`
       (see argument drop_other_df_duplicates`)

    :param dataframe: the pandas DataFrame whose values should be replaced
    :param other_df: the pandas DataFrame which should set the new values to
        `dataframe`
    :param matching_columns: list of strings: the columns to be checked for
        matches. They must be shared between both data frames
    :param merge_columns: list of strings denoting the column(s) to be merged
        or set from `other_df` to `dataframe` for those rows matching under
        `matching_cols`. They must be present in `other_df` columns
    :param drop_other_df_duplicates: If True (the default) drops ALL duplicates
        of `other_df` under `matching_columns` before updating `dataframe`. If
        'first', drops duplicates except for the first occurrence. if 'last'
        drops duplicates except for the last occurrence.
    """
    if drop_other_df_duplicates:
        keep = False if drop_other_df_duplicates is True else \
            drop_other_df_duplicates
        other_df = other_df.drop_duplicates(subset=matching_columns, keep=keep)

    otherdf = other_df[matching_columns + merge_columns]  # only  relevant columns
    try:
        # Use dataframe.merge. For any column C in
        # `matching_columns + merge_columns` which is shared between
        # `dataframe` and `other_df`, then `merge_df` will have two columns:
        # C + '_x' (populated with `dataframe` values) and C + '_y'
        # (with `other_df` values)
        mergedf = dataframe.merge(otherdf, how='left',
                                  on=list(matching_columns), indicator=True)
    except ValueError:
        # Apparently, pandas 0.23+ raises if the the dtypes of a column does
        # not match across the two dataframes (in previous pandas versions, the
        # dtypes where upcasted if needed, e.g.: dataframe[C] = datetime,
        # other_df[C] = object, mergedf[C] = object). We handle here the only
        # "false positive" of this new behaviour. i.e. when one of the two
        # columns has all Nones, we try to cast it to the type of the other
        # column. Eventually, we call again `merge`: it raises again? then fine
        retry = False
        # if there is a mismatch, it is surely for a column in BOTH dataframes:
        for col in set(dataframe.columns) & set(otherdf.columns):
            if dataframe[col].dtype == otherdf[col].dtype:
                continue
            # the casting below might raise (e.g., ints do not accept nones)
            # which is fine
            if pd.isnull(otherdf[col]).all():
                retry = True
                otherdf[col] = otherdf[col].astype(dataframe[col].dtype)
            elif pd.isnull(dataframe[col]).all():
                retry = True
                dataframe[col] = dataframe[col].astype(otherdf[col].dtype)
        if retry:
            mergedf = dataframe.merge(otherdf, how='left', on=list(matching_columns),
                                      indicator=True)
        else:
            raise  # raise original ValueError

    # Now set the `merge_df` columns back into `dataframe`. The idea is that
    # for all shared columns, then perform **row-wise** the following: if value
    # was specified in both `dataframe` and `other_df`, then take `other_df`
    # value. Otherwise `dataframe` value. Remember that the indicator=True
    # argument above has created also a '_merge' column in `merge_df`: the
    # column values are catagorical and can be 'both', 'left_only' (value only
    # in `dataframe`). We should never have 'right_only because of the
    # how='left' above (skip this check for the moment)
    for col in merge_columns:
        if col not in dataframe:
            # trivial case: `dataframe` did not have a column, add it
            ser = mergedf[col].values
        else:
            # if value was in both, take `other_df` value (mergedf[col+"_y"]),
            # otherwise take `dataframe` value (mergedf[col+"_x"])
            ser = np.where(mergedf['_merge'] == 'both', mergedf[col+"_y"],
                           mergedf[col+"_x"])
        dataframe[col] = ser

    return dataframe
