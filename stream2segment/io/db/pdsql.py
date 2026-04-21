"""
Utilities for interaction between pandas DataFrames, optimized for our workflow
"""
from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np
import pandas as pd

from sqlalchemy import select, UpdateBase, Column, Select, Insert, Update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.expression import func, bindparam
from sqlalchemy.types import Integer, Float, Boolean, DateTime

from stream2segment.io.db.inspection import colnames

pandas_version = float('.'.join(pd.__version__.split('.')[:2]))


def get_dtype(sql_type):
    """Converts a SQL type to a numpy type. Given a SQLAlchemy column (ORM model
    attribute) the SQL type is given by its `.type` attribute

    sqlType   | returned type
    ----------+--------------
    Integer   | np.int64
    Float     | np.float64
    DateTime  | np.datetime64
    Boolean   | np.bool_
    Any other | np.object_

    :param sql_type: one of the following: Integer, Float, Boolean, DateTime.
    """
    if isinstance(sql_type, Float):
        return np.float64
    if isinstance(sql_type, Integer):
        return np.int64
    if isinstance(sql_type, DateTime):
        # Caution: np.datetime64 is also a subclass of np.number.
        return np.datetime64
    if isinstance(sql_type, Boolean):
        return np.bool_  # same as np.bool
    return np.object_  # same as np.object


def apply_table_dtypes(
    table: type[DeclarativeBase], dataframe, drop_non_nullable=True
):
    """
    Applies the ORM table data types on the given dataframe, converting data as necessary.
    Columns whose names mismatch between table and dataframe are not modified.
    For ints and bools with NaN / None, the array dtype is converted to `object`.

    :param table: an ORM model class
    :param dataframe: the Data frame
    :param drop_non_nullable: if True (the deault), drop rows that have NaN / NULL
        values for columns that are non-nullable
    """
    for col_name in colnames(table):
        sql_col = getattr(table, col_name)
        col_type = get_dtype(sql_col.type)
        try:
            df_col = dataframe[col_name]
            # the type the dataframe column should have
            if col_type == np.datetime64:
                if issubclass(df_col.dtype.type, np.number):
                    format_ = 's' if issubclass(df_col.dtype.type, np.number)
                elif pandas_version >= 2:
                    format_ = 'ISO8601'
                else:
                    format_ = None
                # let's assume utc and then remove tz info so comparison to "standard"
                # Python date-times (i.e., with no tzinfo) is possible
                dataframe[col_name] = pd.to_datetime(
                    df_col, errors='coerce', format=format_, utc=True
                ).dt.tz_localize(None)
            elif col_type == np.float64:
                try:
                    dataframe[col_name] = df_col.astype(col_type, copy=False)
                except (TypeError, ValueError):
                    dataframe[col_name] = pd.to_numeric(df_col, errors='coerce')
            elif col_type  == np.bool_:
                # bool does not raise, but converts None to False, everything "truthy"
                # to True. So first store nulls, if any:
                invalid = pd.isna(df_col)
                dataframe[col_name] = df_col.astype(col_type, copy=False)
                if invalid.any():
                    # convert to object otherwise the next operation upcasts the column
                    # datat type to float:
                    dataframe[col_name] = dataframe[col_name].astype(object)
                    # Reset back `None`s:
                    dataframe.loc[invalid, col_name] = None
            elif col_type == np.int64:
                # int is stricter than bool, and does not coerce invaldi values, So:
                try:
                    dataframe[col_name] = df_col.astype(col_type, copy=False)
                except (TypeError, ValueError):
                    # Force coercion to numeric (set NaN):
                    dataframe[col_name] = pd.to_numeric(df_col, errors='coerce')
                    # now keep track of the NaNs indices:
                    invalid = pd.isna(dataframe[col_name])
                    # Temporarily set as 0 the NaNs, so casting later is feasible:
                    dataframe.loc[invalid, col_name] = 0
                    # Cast to our type and then to object to avoid upcasting to float
                    # when we set None later
                    dataframe[col_name] = (
                        dataframe[col_name].astype(col_type).astype(object)
                    )
                    # Reset back `None`s in place:
                    dataframe.loc[invalid, col_name] = None
        except KeyError:
            pass  # this column not in results

    if drop_non_nullable:
        non_nullable_cols = dataframe.columns.intersection(
            set(c.name for c in table.__table__.c if not c.nullable)  # noqa
        )
    else:
        non_nullable_cols = []  # simply skip if below

    if non_nullable_cols:
        pre_len = len(dataframe)
        dataframe = dataframe.dropna(subset=non_nullable_cols, axis=0, inplace=False)
        discarded = pre_len - len(dataframe)
        if discarded:
            # Cast bools and ints as they might have been object:
            for col in non_nullable_cols:
                dtype = get_dtype(getattr(table, col).type)
                if dtype in (np.int64, np.bool_):
                    dataframe[col] = dataframe[col].astype(dtype, copy=False)
            dataframe.attrs['discarded'] = discarded

    return dataframe


def df2db(
    dfr,
    table_model,
    engine,
    id_col: str,
    uc_cols: list[str],
    update_cols: Optional[list[str]]=None,
    chunksize=1000
):
    """
    Write the given dataframe to the relative table. Return the tuple

    (dfr, failed_insert, failed_update)

    where:
        dfr is the passed dataframe with 'id_col' set (it must not be present
            before calling this method) minus failed_i (see below)
        failed_insert: the subset of the passed dataframe that could not
            be inserted for some db error
        failed_update: the subset of the passed dataframe that could not be
            updated for some db error. NOTE: contrarily to `failed_insert`,
            these rows are still included in the returned dataframe

    :param dfr: a pandas dataframe
    :param engine: a sql-alchemy engine
    :param uc_cols: a list of ORM columns for comparing `dataframe`
        rows and T rows: when two rows are found that are equal (according to
        all `matching_columns` values), then the data frame row `id_col` value
        is set = T row value
    :param id_col: the ORM column denoting a NUMERIC and UNIQUE Column of T
        (e.g., INTEGER primary key): unexpected results if the column does not
        match those criteria. The column needs not to be a column of
        `dataframe`. The returned `dataframe` will have in any case this column
        set with non-NA values and the proper python type (corresponding to
        the column  SQL type)
    :param update: optional list of strings. I provided, then the columns which
        matching names are updated only (the string name of id_col should not be
        in the list)
    :param chunksize: integer, defaults to 10. The buffer size before committing.
        Increase this number for better performances (speed) at the cost of some
        "false negative" (committing a series of operations where one raise an
        integrity error discards all subsequent operations regardless if they
        would raise as well or not)
    """
    dfr_with_pkeys = sync_pkey(
        dfr,
        table_model,
        engine,
        id_col,
        uc_cols,
        chunksize=chunksize
    )
    failed_i = pd.DataFrame(columns=dfr_with_pkeys.columns, data=[])
    failed_u = pd.DataFrame(columns=dfr_with_pkeys.columns, data=[])

    id_max = dfr_with_pkeys.attrs[f'{id_col}_max']
    to_insert: pd.Series = dfr_with_pkeys[id_col] > id_max

    if to_insert.any():
        stmt = [create_insert_statement(table_model)]
        start = 0
        ids = set()
        dfr_tmp = dfr_with_pkeys.loc[to_insert]
        while start < len(dfr_tmp):
            rows = list(iter_rows(dfr_tmp[start: start+chunksize]))
            ids.update(r[id_col] for r in execute_sql(engine, stmt, rows))  # set update, not sql!
            start += chunksize
        if len(ids) < len(dfr_tmp):
            mask = (~to_insert) | dfr_with_pkeys[id_col].isin(ids)
            failed_i = dfr_with_pkeys.loc[~mask]
            dfr_with_pkeys = dfr_with_pkeys.loc[mask]

    if update_cols:
        to_update = ~to_insert
        if to_update.any():
            stmt = [create_update_statement(table_model, id_col, update_cols)]
            start = 0
            ids = set()
            dfr_tmp = dfr_with_pkeys.loc[to_update]
            while start < len(dfr_tmp):
                rows = list(iter_rows(dfr_tmp[start: start + chunksize]))
                ids.update(r[id_col] for r in execute_sql(engine, stmt, rows))  # set update, not sql!
                start += chunksize
            if len(ids) < len(dfr_tmp):
                mask = to_update & (~dfr_with_pkeys[id_col].isin(ids))
                failed_u = dfr_with_pkeys.loc[mask]

    if not pd.api.types.is_integer_dtype(dfr_with_pkeys[id_col]):  # for safety
        dfr_with_pkeys[id_col] = dfr_with_pkeys[id_col].astype(int)

    return dfr_with_pkeys, failed_i, failed_u


def sync_pkey(
    dfr,
    table_model,
    engine,
    pkey_col:str,
    uc_cols: list[str],
    select_where=None,
    assign_new_ids=True,
    chunksize=50000
):
    """
    Synchronize the primary key `id_col` using the related `table_model` and
    `uc_cols` to match existing db rows. Return `dfr` with an `id_col` set
    (replacing entirely the existing dataframe `id_col`, if any) of type int.
    The returned dataframe will also have an attribute `dfr.attrs[id_col+"_max"]`
    denoting the current maximum of `id_col` on the db: consequently, dataframe
    rows whose `id_col` value is greater than that maximum **ARE NOT YET INSERTED
    ON THE DATABASE**: the id was assigned to be safely used in insert operation.


    :param pkey_col: string denoting the ID (primary key) table column. It
        MUST be a SQL numeric column (preferably, an auto increment primary key
        of type int)
    :param and uc_cols: list of strings denoting the unique constraint columns,
        i.e. the columns that must be unique for each row and can then be used to
        match equal rows. They should be relatively few and of type integer for better
        performance
    """
    columns = table_model.__table__.c  # columns collection
    col_names = [pkey_col] + list(uc_cols)
    stmt = select(*(columns[c] for c in col_names))
    if select_where is not None:
        stmt = stmt.where(select_where)
    # set column nullable int type (for now):
    dfr[pkey_col] = pd.Series(pd.NA, index = dfr.index, dtype="Int64")

    if chunksize <= 0:  # fetch at once (db table small to medium size)
        chunksize = get_row_count(engine, table_model)

    if chunksize > 0:  # fetch in chunks (db table huge):
        # create an id_col + suffix where we put fetched db values:
        suffix = '_'
        while pkey_col + suffix in dfr.columns:
            suffix += '_'
        for db_df in db2dfs(stmt, engine, chunksize=chunksize):
            if db_df.empty:
                continue
            dfr = dfr.merge(db_df, how='left', on=uc_cols, suffixes=('', suffix))
            # For each row, if dfr[pkey_col + suffix] is not null (row exists on db),
            # set it on dfr[pkey_col]. Otherwise, keep dfr[pkey_col]:
            dfr[pkey_col] = dfr[pkey_col + suffix].combine_first(dfr[pkey_col])
            # drop new ids (already merged):
            dfr = dfr.drop(columns=[pkey_col + suffix])

    pkey_max = get_max(engine, columns[pkey_col])
    if assign_new_ids:
        nans = pd.isna(dfr[pkey_col])
        nan_count = nans.sum()
        if nan_count > 0:
            dfr.loc[nans, pkey_col] = range(pkey_max + 1, pkey_max + nan_count + 1, 1)
        dfr[pkey_col] = dfr[pkey_col].astype(int)  # from Int64 back to natural int (faster)

    dfr.attrs[f'{pkey_col}_max'] = pkey_max

    return dfr


def get_max(engine, numeric_column: Column):
    """
    Return the table maximum value from a given numeric column, usually a primary
    key with auto-increment=True. Values from `get_max() + 1` can be safely inserted on
    the table we
    assure unique identifier for adding new objects to the table of
    `numeric_column`.

    :param engine:
    :param numeric_column: a sqlalchemy.schema.Column object
    """
    # return session.query(func.max(numeric_column)).scalar() or 0
    with engine.connect() as conn:
        return conn.execute(select(func.max(numeric_column))).scalar() or 0


def get_row_count(engine, table_model):
    """
    Return the table number of rows.

    :param engine:
    :param table_model: ORM model
    """
    # return session.query(func.max(numeric_column)).scalar() or 0
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table_model)).scalar() or 0


def db2dfs(query: Select, engine, chunksize=20000) -> Iterable[pd.DataFrame]:
    columns = [c['name'] for c in query.column_descriptions]
    with engine.connect() as conn:
        result = conn.execute(query)
        while True:
            rows = result.fetchmany(chunksize)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=columns)


def create_insert_statement(table_model) -> Insert:
    return table_model.__table__.insert()


def create_update_statement(
    table_model, where_col: str, update_cols: list[str]
) -> Update:
    table = table_model.__table__
    columns = table.c
    return (
        table.update()
        .where(columns[where_col] == bindparam(where_col))
        .values({col: bindparam(col) for col in update_cols})
    )


def execute_sql(
    engine: Engine, statement: Sequence[UpdateBase], data: Sequence[dict]
) -> Iterable[dict]:
    with engine.begin() as conn:  # noqa
        yield from _execute_sql(conn, statement, data)


def _execute_sql(
    conn, statement: Sequence[UpdateBase], data: Sequence[dict]
) -> Iterable[dict]:
    # use a stack for recursion. start_index will be set on failure
    stack = [data]

    while stack:
        chunk = stack.pop()
        if len(chunk) == 0:
            continue

        try:
            with conn.begin_nested():  # SAVEPOINT
                for stmt in statement:
                    conn.execute(stmt, chunk)
            yield from chunk
        except IntegrityError as e:
            # rollback of this chunk happens automatically
            if len(chunk) > 1:
                mid = len(chunk) // 2
                stack.append(chunk[mid:])
                stack.append(chunk[:mid])


def iter_rows(dataframe: pd.DataFrame, columns=None) -> Iterable[dict]:
    """
    Yield dataframe rows as `dict`s for insertion into a database. The output is
    `dataframe.to_dict(orient='records')` but with dict values converted to Python
    objects, including all pandas NA (Nat, NaN, None) converted to `None`. Supported
    data types are int, float, datetime, str / object and bool. Data type matching
    between `dataframe` and the underlying database table are not checked for.
    """

    if columns is not None:
        dataframe = dataframe[columns]

    data_list = []
    columns = []
    for col, series in dataframe.items():
        columns.append(str(col))
        # if series.dtype.kind == "M":  # FIXME REMOVE AFTER TESTING THAT DATETIMES ARE OK IN SQL
        #     d = series.dt.to_pydatetime()
        # else:
        #    d = series.values.astype(object, copy=False)
        d = series.values.astype(object, copy=False)

        # assert isinstance(d, np.ndarray), type(d)

        mask = pd.isna(d)
        if mask.any():
            d[mask] = None

        data_list.append(d)

    # columns = list(map(str, dataframe.columns))
    for row_values in zip(*data_list):
        yield dict(zip(columns, row_values))
