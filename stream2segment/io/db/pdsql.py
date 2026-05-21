"""
Utilities for interaction between pandas DataFrames, optimized for our workflow
"""
# :date: Jul 17, 2016
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

# leave all imports below because they might be used elsewhere:
from sqlalchemy import select, update, insert, UpdateBase, Column, Select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.expression import func
from sqlalchemy.types import Integer, Float, Boolean, DateTime

pandas_version = float('.'.join(pd.__version__.split('.')[:2]))


def apply_table_dtypes(
    table: type[DeclarativeBase], dataframe, drop_non_nullable=True
):
    """
    Applies the ORM table data types on the given dataframe, converting data as
    necessary. Columns whose names mismatch between table and dataframe are not
    modified. For ints and bools with NaN / None, the array dtype is converted to
    `object`.

    :param table: an ORM model class
    :param dataframe: the Data frame
    :param drop_non_nullable: if True (the default), drop rows that have NaN / NULL
        values for columns that are non-nullable
    """
    table_cols = [c for c in table.__table__.c if c.name in dataframe.columns]  # noqa
    keep = None
    non_nullable_cols = set()
    if drop_non_nullable:
        non_nullable_cols = set(c.name for c in table_cols if not c.nullable)  # noqa
        if non_nullable_cols:
            keep = pd.Series(True, index=dataframe.index)

    for col in table_cols:  # noqa
        col_name = col.name
        sql_type = getattr(table, col_name).type
        df_col = dataframe[col_name]
        # the type the dataframe column should have
        if (
            isinstance(sql_type, DateTime) and not
            pd.api.types.is_datetime64_dtype(df_col)
        ):
            if issubclass(df_col.dtype.type, np.number):
                format_ = 's'
            elif pandas_version >= 2:
                format_ = 'ISO8601'
            else:
                format_ = None
            # let's assume utc and then remove tz info so comparison to "standard"
            # Python date-times (i.e., with no tzinfo) is possible
            dataframe[col_name] = pd.to_datetime(
                df_col, errors='coerce', format=format_, utc=True
            ).dt.tz_localize(None)
        elif isinstance(sql_type, Float) and not pd.api.types.is_float_dtype(df_col):
            try:
                dataframe[col_name] = df_col.astype(float, copy=False)
            except (TypeError, ValueError):
                dataframe[col_name] = pd.to_numeric(df_col, errors='coerce')
        elif isinstance(sql_type, Boolean) and not pd.api.types.is_bool_dtype(df_col):
            # bool does not raise, but converts None to False, everything "truthy"
            # to True. So first store nulls, if any:
            invalid = pd.isna(df_col)
            dataframe[col_name] = df_col.astype(bool, copy=False)
            if invalid.any():
                # convert to object otherwise the next operation upcasts the column
                # datat type to float:
                dataframe[col_name] = dataframe[col_name].astype(
                    pd.CategoricalDtype([True, False])
                )
                # Reset back `None`s:
                dataframe.loc[invalid, col_name] = None
        elif (
            isinstance(sql_type, Integer) and
            not pd.api.types.is_integer_dtype(df_col)
        ):
            # int is stricter than bool, and does not coerce invalid values, So:
            try:
                dataframe[col_name] = df_col.astype(int, copy=False)
            except (TypeError, ValueError):
                # support for NaNs:
                dataframe[col_name] = df_col.astype("Int64", copy=False)

        if col_name in non_nullable_cols:
            keep &= dataframe[col_name].notna()

    if keep is not None and (~keep.all()):
        pre_len = len(dataframe)
        dataframe = dataframe[keep]
        discarded = pre_len - len(dataframe)
        if discarded:
            # Cast bools and ints as they might have been object:
            dataframe.attrs['discarded'] = discarded

    return dataframe


def get_col_max(engine, numeric_column: Column):
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


def sync_pkey(
    dfr: pd.DataFrame,
    engine: Engine,
    table_model: type[DeclarativeBase],
    unique_cols: list[str],
    chunksize=5000
):
    """
    Synchronize the primary key `id_col` using the related `table_model` and
    `unique_cols` to match fetched db rows with dataframe rows.
    Return `dfr` with an `id_col` set (replacing entirely the existing dataframe
    `id_col`, if any) of type int or Int64 (supporting NaN).

    :param unique_cols: list of strings denoting the unique constraint columns,
        i.e. the columns that must be unique for each row and can then be used to
        match equal rows
    """
    columns = table_model.__table__.c  # columns collection
    pkey_col = [col.name for col in table_model.__table__.primary_key.columns][0]
    col_names = [pkey_col] + list(unique_cols)
    stmt = select(*(columns[c] for c in col_names))
    # set column nullable int type (for now):
    dfr[pkey_col] = pd.Series(pd.NA, index = dfr.index, dtype="Int64")

    if chunksize > 0:  # fetch in chunks (db table huge):
        # create an id_col + suffix where we put fetched db values:
        suffix = '_'
        while pkey_col + suffix in dfr.columns:
            suffix += '_'
        for db_df in fetch_df(engine, stmt, chunksize=chunksize):
            if db_df.empty:
                continue
            dfr = dfr.merge(db_df, how='left', on=unique_cols, suffixes=('', suffix))
            # For each row, if dfr[pkey_col + suffix] is not null (row exists on db),
            # set it on dfr[pkey_col]. Otherwise, keep dfr[pkey_col]:
            # dfr[pkey_col] = dfr[pkey_col + suffix].combine_first(dfr[pkey_col])
            dfr[pkey_col] = dfr[pkey_col].fillna(dfr[pkey_col + suffix])
            # drop new ids (already merged):
            dfr = dfr.drop(columns=[pkey_col + suffix])

    return dfr


def set_pkeys(dfr: pd.DataFrame, engine: Engine, table_model: type[DeclarativeBase]):
    """
    Set the primary key column of the given table model on the passed dataframe,
    replacing or creating a new column named as the table primary key and composed of
    sequential, non-null integers, starting with the current max on the DB table + 1
    """
    id_col = [col.name for col in table_model.__table__.primary_key.columns][0]
    # empty case: just add id_col for compatibility with pd ops (e.g. concat, merge):
    if dfr.empty:
        if id_col not in dfr.columns:
            dfr[id_col] = pd.Series(dtype=int)
    else:
        id_max = get_col_max(engine, table_model.__table__.c[id_col]) + 1
        dfr[id_col] = np.arange(id_max, id_max + len(dfr), dtype=int)
    return dfr


def insert_df(
    dfr: pd.DataFrame,
    engine: Engine,
    table_model: type[DeclarativeBase],
    chunksize=5000,
) -> pd.DataFrame:
    """
    Insert dfr to the given table model. The primary key column of the table must be
    an auto-increment (sequential) integer (int or Int64), and the relative column must
    be supplied on the passed DataFrame (see `set_pkeys`). Null primary keys are allowed
    but risky in that the relative dataframe row will most likely be inserted
    with an auto incremented ID which will be unknown without additional DB queries
    """
    if dfr.empty:
        return dfr

    stmt = [insert(table_model)]
    start = 0
    ids_failed = set()

    id_col = [col.name for col in table_model.__table__.primary_key.columns][0]
    while start < len(dfr):
        rows = list(iter_rows(dfr[start: start + chunksize]))
        ids_inserted = set(r[id_col] for r in executemany(engine, stmt, rows))
        if len(ids_inserted) < len(rows):
            ids_failed.update(set(r[id_col] for r in rows) - ids_inserted)
        start += chunksize

    if len(ids_failed):
        failed_mask = dfr[id_col].isin(ids_failed)
        dfr = dfr[~failed_mask]

    return dfr


def fetch_df(engine, select_stmt: Select, chunksize=5000) -> Iterable[pd.DataFrame]:
    """
    Fetch a dataframe from a select statement, yielding chunks of the table as
    Iterable of dataframes. For small data, you can do:
    dataframe = pd.concat(list(fetch_df(engine, select_stmt)))
    """
    # columns = [c['name'] for c in query.column_descriptions]
    select_stmt = select_stmt.execution_options(stream_results=True)

    with engine.connect() as conn:
        result = conn.execute(select_stmt)  # .yield_per(chunksize)

        while rows := result.fetchmany(chunksize):
            yield pd.DataFrame(rows, columns=result.keys())


def executemany(
    engine: Engine, statement: Sequence[UpdateBase], data: Sequence[dict]
) -> Iterable[dict]:
    """
    Execute a SQL `executemany` by running the given statement(s) on each dict
    passed in the given data
    """
    with engine.begin() as conn:  # noqa

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

        mask = pd.isna(series)
        is_na = mask.any()

        if pd.api.types.is_datetime64_dtype(series):
            d = series.dt.to_pydatetime()
        else:
           d = series.values.astype(object, copy=is_na)

        if is_na:
            d[mask] = None

        data_list.append(d)

    # columns = list(map(str, dataframe.columns))
    for row_values in zip(*data_list):
        yield dict(zip(columns, row_values))
