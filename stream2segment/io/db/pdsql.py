"""
Utilities for interaction between pandas DataFrames, optimized for our workflow
"""
from collections.abc import Iterable, Sequence
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from sqlalchemy import select, UpdateBase, Column, Select, Insert, Update, ColumnElement
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, InstrumentedAttribute
from sqlalchemy.sql.expression import func, bindparam
from sqlalchemy.types import Integer, Float, Boolean, DateTime

pandas_version = float('.'.join(pd.__version__.split('.')[:2]))


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
            pd.api.types.is_datetime64_any_dtype(df_col)
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
        elif isinstance(sql_type, Integer) and not pd.api.types.is_integer_dtype(df_col):
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
    pkey_col: str,
    uc_cols: list[str],
    select_where=None,
    chunksize=5000
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
        for db_df in select_df(engine, stmt, chunksize=chunksize):
            if db_df.empty:
                continue
            dfr = dfr.merge(db_df, how='left', on=uc_cols, suffixes=('', suffix))
            # For each row, if dfr[pkey_col + suffix] is not null (row exists on db),
            # set it on dfr[pkey_col]. Otherwise, keep dfr[pkey_col]:
            dfr[pkey_col] = dfr[pkey_col + suffix].combine_first(dfr[pkey_col])
            # drop new ids (already merged):
            dfr = dfr.drop(columns=[pkey_col + suffix])

    return dfr


def insert_df(
    dfr: pd.DataFrame,
    engine: Engine,
    table_model: type[DeclarativeBase],
    chunksize=5000,
    on_missing_pkey_col: Literal["auto-increment", "raise"] = 'auto-increment'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Insert dfr to the given table model. The primary key column of the table must be
    an auto-increment (sequential) integer. If not present in the dataframe, the id
    values will be inserted according to the current max on the DB, unless
    on_missing_pkey is 'raise' (then the column must already be present).

    See `sync_pkey` for more information
    """
    stmt = [create_insert_statement(table_model)]
    start = 0
    index_inserted = set()
    failed = pd.DataFrame(columns=dfr.columns, data=[])

    id_col = [col.name for col in table_model.__table__.primary_key.columns][0]
    if id_col not in dfr.columns:
        if on_missing_pkey_col != 'auto-increment':
            raise ValueError(f'Missing {id_col}')
        id_max = get_col_max(engine, table_model.__table__.c[id_col]) + 1
        dfr[id_col] = np.arange(id_max, id_max + len(dfr), dtype=int)

    while start < len(dfr):
        rows = list(iter_rows(dfr[start: start + chunksize]))
        index_inserted.update(  # Python Set update, not SQL!
            r[id_col] for r in execute_sql(engine, stmt, rows)
        )
        start += chunksize

    if len(index_inserted) < len(dfr):
        _mask = dfr.index.isin(index_inserted)
        failed = dfr[~_mask]
        dfr = dfr[_mask]

    return dfr, failed


def select_df(engine, query: Select, chunksize=5000) -> Iterable[pd.DataFrame]:
    # columns = [c['name'] for c in query.column_descriptions]
    query = query.execution_options(stream_results=True)

    with engine.connect() as conn:
        result = conn.execute(query).yield_per(chunksize)

        while True:
            rows = result.fetchmany(chunksize)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=result.keys())


def create_insert_statement(table_model) -> Insert:
    return table_model.__table__.insert()


ColumnLike: TypeAlias = str | InstrumentedAttribute | Column

def create_update_statement(
    table_model,
    update_cols: ColumnLike | list[ColumnLike],
    where_clause: ColumnElement[bool]
) -> Update:

    if not isinstance(update_cols, (list, tuple)):
        update_cols = [update_cols]

    return (
        table_model.__table__.update()
        .where(where_clause)
        #.where(columns[where_col] == bindparam(where_col))
        .values({col: bindparam(getattr(col, "key", col)) for col in update_cols})
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
        if pd.api.types.is_datetime64_dtype(series):
            d = series.dt.to_pydatetime()
        else:
           d = series.values.astype(object, copy=False)

        mask = pd.isna(d)
        if mask.any():
            d[mask] = None

        data_list.append(d)

    # columns = list(map(str, dataframe.columns))
    for row_values in zip(*data_list):
        yield dict(zip(columns, row_values))
