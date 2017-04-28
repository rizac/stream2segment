'''

Utilities for converting from pandas DataFrames to sqlalchemy tables objects
(according to models.py)
Some of these functions are copied and pasted from pandas.io.sql.SqlTable
A particular function, `colitems`, deals with SqlAlchemy mapping, returning the columns defined in
a model class.
This deserves a little description of the underlying mechanism of SqlAlchemy (skip
this if you are not a developer).

In SQLAlchemy, descriptors are used heavily in order to provide attribute behavior on mapped
classes. When a class is mapped as such:
```
class MyClass(Base):
    __tablename__ = 'foo'

    id = Column(Integer, primary_key=True)
    data = Column(String)
```
The `MyClass` class will be mapped when its definition is complete, at which point the id and
data attributes, starting out as **Column** objects, will be replaced by the instrumentation
system with instances of **InstrumentedAttribute**, which are descriptors that provide the
`__get__()`, `__set__()` and `__delete__()` methods. The InstrumentedAttribute will generate a
SQL expression when used at the class level:
```
>>> print(MyClass.data == 5)
data = :data_1
```
(http://docs.sqlalchemy.org/en/latest/glossary.html#term-descriptor)
Each InstrumentedAttribute has a **key** attribute which is the attribute name (as
typed in python code), and can be used dynamically via python `setattr` and `getattr` function.
This has **not to be confused** with the `key` attribute of the relative Column, which
is "an optional string identifier which will identify this Column object on the Table."
(http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column.params.key).

Note that the Column has also a name attribute which is the name of this column as represented in
the database. This argument may be the first positional argument, or specified via keyword ('name').
See http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column.params.name

Typically, one wants to work with **InstrumentedAttribute**s and their key (attribute names), not
with **Column**s names and keys. But it is important the distinction.
In a previous version of `colitems`, we used the `__table__.columns` dict-like object, but it is
keyed according to the keys of the Column, which is by default the class attribute name, which means
that the attribute names are lost if one provides custom keys for the Column objects.
The `inspect` function called with a class or an instance produces a `Mapper` object
(http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper)
which apparently has also a `columns` dict like object of Column's object
**keyed based on the attribute name defined in the mapping, not necessarily the key attribute of
the Column itself**
(http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper.columns),
This is used in the method `colitems` which returns the **attribute names** mapped to the
relative **Column**. All in all:
```
for att_name, column_obj in colitems(MyClass):
    instrumented_attribute_obj = MyClass.getattr(att_name)
    # create a SQL expression:
    instrumented_attribute_obj == 5
    # this actually also works and produces the same SQL expression,
    # but I didn't found references about:
    column_obj == 5
```

Created on Jul 17, 2016

@author: riccardo
'''
from __future__ import division
from datetime import datetime, date
from pandas.io.sql import _handle_date_column
# from pandas.types.api import DatetimeTZDtype
import numpy as np
# pandas zip seems a wrapper around itertools.izip (generator instead than list):
from pandas.compat import (lzip, map, zip, raise_with_traceback,
                           string_types, text_type)
# is this below the same as pd.isnull? For safety we leave it like it is (the line is imported
# from pandas.io.sql and used in one of the copied methods below)
from pandas.core.common import isnull
# but we need also pd.isnull so we import it like this for safety:
from pandas import isnull as pd_isnull
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import and_, or_
from sqlalchemy.orm.attributes import InstrumentedAttribute
from pandas import to_numeric
import pandas as pd
from sqlalchemy.engine import create_engine
from itertools import cycle, izip
from sqlalchemy.inspection import inspect
from pandas.types.dtypes import DatetimeTZDtype
from sqlalchemy.sql.expression import func
from collections import OrderedDict


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


# def df2dbiter(dataframe, table_class, harmonize_cols_first=True, harmonize_rows_first=True,
#               parse_dates=None):
#     """
#         Returns a generator of of ORM model instances (reflecting the rows database table mapped by
#         table_class) from the given dataframe. The returned generator can be used in loops and each
#         element can be e.g. added to the database by means of sqlAlchemy `session.add` method.
#         NOTE: Only the dataframe columns whose name match the table_class columns will be used.
#         Therefore, it is safe to append to dataframe any column whose name is not in table_class
#         columns. Some iterations might return None according to the parameters (see below)
#         :param table_class: the CLASS of the table whose rows are instantiated and returned
#         :param dataframe: the input dataframe
#         :param harmonize_cols_first: if True (default when missing), the dataframe column types
#         are harmonized to reflect the table_class column types, and columns without a match
#         in table_class are filtered out. See `harmonize_columns(dataframe)`. If
#         `harmonize_cols_first` and `harmonize_rows_first` are both True, the harmonization is
#         executed in that order (first columns, then rows).
#         :param harmonize_rows_first: if True (default when missing), NA values are checked for
#         those table columns which have the property nullable set to False. In this case, the
#         generator **might return None** (the user should in case handle it, e.g. skipping
#         these model instances which are not writable to the db). If
#         `harmonize_cols_first` and `harmonize_rows_first` are both True, the harmonization is
#         executed in that order (first columns, then rows).
#         :param parse_dates: a list of strings denoting additional column names whose values should
#         be parsed as dates. Ignored if harmonize_cols_first is False
#     """
#     if harmonize_cols_first:
#         colnames, dataframe = _harmonize_columns(table_class, dataframe, parse_dates)
#     else:
#         colnames = list(shared_colnames(table_class, dataframe))
# 
#     new_df = dataframe[colnames]
# 
#     if dataframe.empty:
#         for _ in len(dataframe):
#             yield None
#         return
# 
#     valid_rows = cycle([True])
#     if harmonize_rows_first:
#         non_nullable_cols = list(shared_colnames(table_class, new_df, nullable=False))
#         if non_nullable_cols:
#             valid_rows = new_df[non_nullable_cols].notnull().all(axis=1).values
# 
#     cols, datalist = _insert_data(new_df)
#     # Note below: datalist is an array of N column, each of M rows (it would be nicer to return an
#     # array of N rows, each of them representing a table row. But we do not want to touch pandas
#     # code.
#     # See _insert_table below). Thus we zip it:
#     for is_ok, row_values in zip(valid_rows, zip(*datalist)):
#         if is_ok:
#             # we could make a single line statement, but two lines are more readable:
#             row_args_dict = dict(zip(cols, row_values))
#             yield table_class(**row_args_dict)
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


def withdata(model_column):
    """
    Returns a filter argument for returning instances with values of
    `model_column` NOT *empty* nor *null*. `model_column` type must be STRING or BLOB
    :param model_column: A valid column name, e.g. an attribute Column defined in some
    sqlalchemy orm model class (e.g., 'User.data'). **The type of the column must be STRING or BLOB**,
    otherwise result is undefined. For instance, numeric column with zero as value
    are *not* empty (as the sql length function applied to numeric returns the number of
    bytes)
    :example:
    ```
    # given a table User, return empty or none via "~"
    session.query(User.id).filter(~withdata(User.data)).all()

    # return "valid" columns:
    session.query(User.id).filter(withdata(User.data)).all()
    ```
    """
    return (model_column.isnot(None)) & (func.length(model_column) > 0)


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


# def get_or_add(session, model_instances, columns=None, on_add='flush'):
#     return [x for x in get_or_add_iter(session, model_instances, columns, on_add)]
# 
# 
# def get_or_add_iter(session, model_instances, columns=None, on_add='flush'):
#     """
#         Iterates on all model_rows trying to add each
#         instance to the session if it does not already exist on the database. All instances
#         in `model_instances` should belong to the same model (python class). For each
#         instance, its existence is checked based on `columns`: if a database row
#         is found, whose values are the same as the given instance for **all** the columns defined
#         in `columns`, then the *first* database row instance is returned.
# 
#         Yields tuple: (model_instance, is_new_and_was_added)
# 
#         Note that if `on_add`="flush" (the default) or `on_add`="commit", model_instance might be
#         None (see below)
# 
#         :Example:
#         ```
#         # assuming the model of each instance is a class named 'MyTable' with a primary key 'id':
#         instances = [MyTable(...), ..., MyTable(...)]
#         # add all instances if they are not found on db according to 'id'
#         # (thus no need to specify the argument `columns`)
#         for instance, is_new in get_or_add_iter(session, instances, on_add='commit'):
#             if instance is None:
#                 # instance was not found on db but adding it raised an exception
#                 # (including the case where instances was already None)
#             elif is_new:
#                 # instance was not found on the db and was succesfully added
#             else:
#                # instance was found on the db and the first matching instance is returned
# 
#         # Note that the calls below produce the same results:
#         get_or_add_iter(session, instances):
#         get_or_add_iter(session, instances, 'id'):
#         get_or_add_iter(session, instances, MyTable.id):
#         ```
#         :param model_instances: an iterable (list tuple generator ...) of ORM model instances. None
#         values are valid and will yield the tuple (None, False)
#         All instances *MUST* belong to the same class, i.e., represent rows of the same db table.
#         An ORM model is the python class reflecting a database table. An ORM model instance is
#         simply a python instance of that class, and thus reflects a rows of the database table
#         :param columns: iterable of strings or class attributes (objects of type
#         InstrumentedAttribute), a single Instrumented Attribute, string, or None: the
#         column(s) to check if a model instance has a corresponding row in the database table
#         (in that case the instance reflecting that row is returned and nothing is added). A database
#         column matches the current model instance if **all** values of `columns`
#         are the same. If not iterable, the argument is converted to `[columns]`.
#         If None, the model primary keys are taken as columns.
#         **In most cases, also `Column` objects can be passed, but this method will fail for
#         Columns which override their `key` attribute, as Column's keys will not reflect
#         the class attribute names anymore**. For info see:
#         http://docs.sqlalchemy.org/en/latest/glossary.html#term-descriptor
#         http://docs.sqlalchemy.org/en/latest/core/metadata.html#sqlalchemy.schema.Column.params.key
#         :param on_add: 'commit', 'flush' or None. Default: 'flush'. Tells whether a `session.flush`
#         or a `session commit` has to be issued after each `session.add`. In case of failure, a
#         `session.rollback` will be issued and the tuple (None, False) is yielded
#     """
#     binexpfunc = None  # cache dict for expressions
#     for row in model_instances:
#         if row is None:
#             yield None, False
#         else:
#             if not binexpfunc:
#                 binexpfunc = _bin_exp_func_from_columns(row.__class__, columns)
# 
#             yield _get_or_add(session, row, binexpfunc, on_add)
# 
# 
# def _get_or_add(session, row, binexpr_for_get, on_add='flush'):
#     """
#     Returns row, is_new, where row is the pased row as argument (added if not existing) or the
#     one on the db matching binexpr_for_get. is_new is a boolean indicating whether row was newly
#     added or found on the db (according to binexpr_for_get)
#     If flush_on_add is True (flush_on_add=False must be carefully used, epsecially for handling
#     rollbacks), then row might be None if session.sluch failed
#     :param row: the model instance represetning a table row. Cannot be None
#     :param on_add: 'flush', 'commit' or everything else (do nothing)
#     """
#     model = row.__class__
#     row_ = session.query(model).filter(binexpr_for_get(row)).first()
#     if row_:
#         return row_, False
#     else:
#         session.add(row)
#         if (on_add == 'flush' and not flush(session)) or \
#                 (on_add == 'commit' and not commit(session)):
#             return None, False
#         return row, True
# 
# 
# def _bin_exp_func_from_columns(model, model_cols_or_colnames):
#     """
#         Returns an slalchemy binary expression *function* for the given model and the given columns
#         the function can be passed in a query object for a given model instance.
#         :Example:
#         # assuming a MyTable model defined somewhere, with columns "col_name_1", "col_name2":
#         func = _bin_exp_func_from_columns(MyTable, ["col_name_1", "col_name2"]):
#         # Now assume we have a model instance
#         row = MyTable(col_name_1='a', col_name_2=5.5)
#         # We can use func in a query (assuming we have a session object):
#         session.query(model).filter(func(row)).all()
#         # which will query the db table mapped by MyTable for all the rows whose col_name_1 value
#         is 'a' **and** whose 'col_name_2' value is 5.5
#     """
#     if not model_cols_or_colnames:
#         # Note: the value of colitems are Column object. For those objects, SQL expressions
#         # such as column=5 are valid. But A Column key might differ from
#         # the attribute names, so use the keys of colitems, which are the attribute names.
#         # This does not prevent us from failing
#         # if model_cols_or_colnames is not empty and contains Column objects which have a
#         # particular key set and different from the attribute name.
#         # But we will raise an error in case
#         model_cols_or_colnames = colnames(model, pkey=True)
# 
#     # is string? In py2, check attr "__iter__". In py3, as it has the attr, go for isinstance:
#     if not hasattr(model_cols_or_colnames, "__iter__") or isinstance(model_cols_or_colnames, str):
#         model_cols_or_colnames = [model_cols_or_colnames]
# 
#     columns = []
#     for col in model_cols_or_colnames:
#         if not hasattr(col, "key"):  # is NOT a Column object, nor an Instrumented attribute
#             # (http://docs.sqlalchemy.org/en/latest/glossary.html#term-descriptor)
#             col = getattr(model, col)  # return the attribute object
#         columns.append(col)
# 
#     def ret_func(row):
#         """ Returns the binary expression function according toe the model_cols_or_colnames for
#         a given model instance `row`"""
#         return and_(*[col == getattr(row, col.key) for col in columns])
# 
#     return ret_func


def dfrowiter(dataframe, columns=None):
    """
        Returns an efficient iterator over `dataframe` rows. The i-th returned values is
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


# def add(dataframe, session, table_model, pkey_col, buf_size=10):
#     ret = []
#     buf = []
#     oks = [2] * buf_size
#     err = [0] * buf_size
#     pkeyname = pkey_col.key
#     # allocate all primary keys for the given model
#     existing_pkeys = set(x[0] for x in session.query(pkey_col))
#     for rowdict in dfrowiter(dataframe):
#         if rowdict[pkeyname] in existing_pkeys:
#             ret.append(1)
#         else:
#             buf.append(rowdict)
#             if len(buf) == buf_size:
#                 try:
#                     session.bind.execute(table_model.__table__.insert(), buf)
#                     ret.extend(oks)
#                 except SQLAlchemyError:
#                     ret.extend(err)
#                 buf = []
#     if buf:
#         try:
#             session.bind.execute(table_model.__table__.insert(), buf)
#             ret.extend(oks[:len(buf)])
#         except SQLAlchemyError:
#             ret.extend(err[:len(buf)])
# 
#     return ret


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
    colnames = []
    for c in query.column_descriptions:
        ctype = type(c['expr'])
        if ctype != InstrumentedAttribute:
            raise ValueError("the query needs to be built with columns only, %s found" %
                             str(ctype))
        colnames.append(c['name'])
    return pd.DataFrame(columns=colnames, data=query.all())


def sync(dataframe, session, matching_columns, autoincrement_pkey_col, add_buf_size=10,
         drop_duplicates=True):
    """
    Efficiently synchronizes the values of `autoincrement_pkey_col` on `dataframe` with the
    corresponding database table T, either by fetching the primary key, or by autoincrementing it
    (after adding the row to T).
    Returns the tuple (d, discarded, new) where `d` is `dataframe`
    with successfully retrieved/added rows only, `discarded` is the number of rows discarded from
    `dataframe` (not in `d`) and `new` is the number of rows of `d` which where newly added and
    not present on T before this function call.

    This function works:
    1. First, by setting the value of `autoincrement_pkey_col` for those rows found on T
       (according to `matching_columns`)
    2. Second, by auto-incrementing `autoincrement_pkey_col` values for the remaining rows
       (not found on the db), and finally writing those rows to T

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe` rows and T rows:
    when two rows are found that are equal (according to all `matching_columns` values), then
    the data frame row `autoincrement_pkey_col` value is set as the corresponding T row value
    :param autoincrement_pkey_col: the ORM column denoting an auto-increment primary key of T.
    Unexpected results if the column does not match tose criteria. The column
    needs not to be a column of `dataframe`. The returned `dataframe` will have in any case this
    column set
    :param add_buf_size: integer, defaults to 10. When adding new items to T, another factor that
    speeds up a lot insertion is to reduce the commits by committing
    buffers if items (basically, lists) instead of committing each time. Increase this argument to
    speed up insertion, at the cost of loosing more good instances in case of errors (when a
    single item in a buffer raises, all subsequent items are discarded, regardless if they
    would have raised or not), decrease it if speeds does not matter: in the lowest case (1)
    you will be sure to write all good instances without false negative
    :param drop_duplicates: boolean, True. After having fetched the primary keys and set it to
    the dataframe corresponding column, drop duplicates
    under `matching_columns`. You should always set this argument to True unless you are really sure
    `dataframe` is with no duplicates under `matching_columns`, and you really want to save
    the extra time of dropping again (but is that saved time actually remarkable?)

    :return: the tuple (d, discarded, new) where `d` is `dataframe`
    with successfully retrieved/added rows only, `discarded` is the number of rows discarded from
    `dataframe` (i.e., not in `d`. It *should* hold `discarded = len(dataframe) - len(d)`)
    and `new` is the number of rows of `d` which where newly added and
    not present on T before this function call (`new <= len(d)`).
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
#     discarded = 0
#     dframe_with_pkeys = fetchsetpkeys(dataframe, session, matching_columns, autoincrement_pkey_col)
#     mask = pd.isnull(dframe_with_pkeys[autoincrement_pkey_col.key])
#     dtmp = dframe_with_pkeys[mask]
#     dtmp.is_copy = False  # avoid next lines issuing a CopyWarning
#     if drop_newinst_duplicates:
#         oldlen = len(dtmp)
#         dtmp = dtmp.drop_duplicates(subset=[k.key for k in matching_columns])
#         discarded += oldlen - len(dtmp)
#     new_df, _, new = add2db_onpkey(dtmp, session, autoincrement_pkey_col, add_buf_size)
#     discarded += _
#     return new_df, discarded, new
    discarded = 0
    dframe_with_pkeys = fetchsetpkeys(dataframe, session, matching_columns, autoincrement_pkey_col)
    if drop_duplicates:
        oldlen = len(dframe_with_pkeys)
        subset_cols = [k.key for k in matching_columns]
        dframe_with_pkeys = dframe_with_pkeys.drop_duplicates(subset=subset_cols)
        discarded += oldlen - len(dframe_with_pkeys)
        dframe_with_pkeys.is_copy = False
    new_df, _, new = syncnullpkeys(dframe_with_pkeys, session, autoincrement_pkey_col, add_buf_size)
    discarded += _
    return new_df, discarded, new


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


def add2db(dataframe, session, matching_columns, add_buf_size=10, query_first=True,
           drop_duplicates=True):
    """
    Efficiently adds the rows of `dataframe` to the corresponding database table T, skipping
    db insertion for those `dataframe` rows found on T (according to `matching_columns`).
    Returns the tuple (d, discarded, new) where `d` is `dataframe`
    with successfully added rows only, `discarded` is the number of rows discarded from
    `dataframe` (not in `d`) and `new` is the number of rows of `d` which where newly added and
    not present on T before this function call. Discarded is basically the number of rows which
    violate some constraint (e.g., missing primary key, unique constraint etcetera)

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe` rows and T rows:
    when two rows are found that are equal (according to all `matching_columns` values), then
    the data frame row is not added to T and will be simply returned as a row of `d`
    :param add_buf_size: integer, defaults to 10. When adding new items to T, another factor that
    speeds up a lot insertion is to reduce the commits by committing
    buffers if items (basically, lists) instead of committing each time. Increase this argument to
    speed up insertion, at the cost of loosing more good instances in case of errors (when a
    single item in a buffer raises, all subsequent items are discarded, regardless if they
    would have raised or not), decrease it if speeds does not matter: in the lowest case (1)
    you will be sure to write all good instances without false negative
    :param query_first: boolean (defaults to True): queries T for rows already present. If this
    argument is False no skip is done, i.e. for all rows of `dataframe` the function will
    attempt to add them to T. **Set to False to speed up the function but only if you are
    sure no row of `dataframe` violates any T constraint**
    :param drop_duplicates: boolean, True. Before adding new items, drop duplicates under
    `matching_columns`. You should always set this argument to True unless you are really sure
    `dataframe` is with no duplicates under `matching_columns`, and you really want to save
    the extra time of dropping again (but is that saved time actually remarkable?)

    :return: the tuple (d, discarded, new) where `d` is `dataframe`
    with successfully added rows only, `discarded` is the number of rows discarded from
    `dataframe` (i.e., not in `d`. It *should* hold `discarded = len(dataframe) - len(d)`)
    and `new` is the number of rows of `d` which where newly added and
    not present on T before this function call (`new <= len(d)`).
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
    buf_size = max(add_buf_size, 1)
    buf_inst = OrderedDict()  # preserve insertion order
    buf_indices = OrderedDict()  # see above
    new = 0
    discarded = 0
    matching_colnames = [c.key for c in matching_columns]
    if drop_duplicates:
        oldlen = len(dataframe)
        dataframe.drop_duplicates(subset=matching_colnames, inplace=True)
        discarded = oldlen - len(dataframe)
    table_model = matching_columns[0].class_
    # allocate all primary keys for the given model
    existing_keys = set() if not query_first else _dbquery2set(session, matching_columns)

    shared_cnames = list(shared_colnames(table_model, dataframe))

    conn = session.connection()

    # Note: we might remove dupes on the dataframe, but then we would return a different
    # dataframe
    # as we want to use this method for stations and channels, and we need to add stations first
    # this might remove some channels.

    last = len(dataframe) - 1
    indices = []  # indices to keep (already existing or successfully written)
    for i, rowdict in enumerate(dfrowiter(dataframe, shared_cnames)):
        rowtup = tuple(rowdict[k] for k in matching_colnames)
        if rowtup not in existing_keys:
            buf_inst[rowtup] = rowdict
            buf_indices[rowtup] = i
        else:
            indices.append(i)
        if len(buf_inst) == buf_size or (i == last and buf_inst):
            update = False
            try:
                conn.execute(table_model.__table__.insert(), buf_inst.values())
                update = True
            except IntegrityError as _:
                # if we had an integrity error, it might be that buf had dupes
                # in this case, some rows might be added! So to be sure we issue a
                # query to know which elements have been added
                # 1. Create a query specific to the buf elements using pkey_cols:
                exprs = []
                for key in buf_inst.keys():
                    exprs.append(and_(*[pkeycol == v
                                        for pkeycol, v in izip(matching_columns, key)]))
                # 2. Get those elements and add them, if saved to the db
                if exprs:
                    added_set = _dbquery2set(session, matching_columns, or_(*exprs))
                    if added_set:
                        update = True
                        buf_indices = {k: buf_indices[k] for k in added_set if k in buf_indices}
                        buf_inst = {k: buf_inst[k] for k in added_set if k in buf_inst}
            except SQLAlchemyError as _:
                pass

            if update:
                new += len(buf_inst)
                indices.extend(buf_indices.values())

            buf_inst.clear()
            buf_indices.clear()

    oldlen = len(dataframe)
    indices.sort()  # indices might not be sorted, we want to return the same orders
    # (for matching rows)
    df = dataframe.iloc[indices]
    df.is_copy = False  # avoid settings with copy warning
    discarded += oldlen - len(df)
    return df, discarded, new


def syncnullpkeys(dataframe, session, autoincrement_pkey_col, add_buf_size=10):
    """
    Efficiently adds the rows of `dataframe` to the corresponding database table T, skipping
    db insertion for those `dataframe` rows where the values of `autoincrement_pkey_col` are not
    n/a, (e.g., Null or NaN's,...) and inserting only n/a rows, after setting the primary
    key according to the T maximum (mimicking db autoincrementing feature).
    `autoincrement_int_pkey_col.key` needs not to be a column of `dataframe`.

    Returns the tuple (d, discarded, new) where `d` is `dataframe`
    with successfully added rows only, `discarded` is the number of rows discarded from
    `dataframe` (not in `d`) and `new` is the number of rows of `d` which where newly added and
    not present on T before this function call.

    :param dataframe: a pandas dataframe
    :param session: an sql-alchemy session
    :param matching_columns: a list of ORM columns for comparing `dataframe` rows and T rows:
    when two rows are found that are equal (according to all `matching_columns` values), then
    the data frame row is not added to T and will be simply returned as a row of `d`
    :param autoincrement_pkey_col: the ORM column denoting an auto-increment primary key of T.
    Unexpected results if the column does not match tose criteria. The column
    needs not to be a column of `dataframe`. The returned `dataframe` will have in any case this
    column set
    :param add_buf_size: integer, defaults to 10. When adding new items to T, another factor that
    speeds up a lot insertion is to reduce the commits by committing
    buffers if items (basically, lists) instead of committing each time. Increase this argument to
    speed up insertion, at the cost of loosing more good instances in case of errors (when a
    single item in a buffer raises, all subsequent items are discarded, regardless if they
    would have raised or not), decrease it if speeds does not matter: in the lowest case (1)
    you will be sure to write all good instances without false negative
    :param drop_duplicates: boolean, True. Before adding new items, drop duplicates under
    `matching_columns`. You should always set this argument to True unless you are really sure
    `dataframe` is with no duplicates under `matching_columns`, and you really want to save
    the extra time of dropping again (but is that saved time actually remarkable?)

    :return: the tuple (d, discarded, new) where `d` is `dataframe`
    with successfully added rows only, `discarded` is the number of rows discarded from
    `dataframe` (i.e., not in `d`. It *should* hold `discarded = len(dataframe) - len(d)`)
    and `new` is the number of rows of `d` which where newly added and
    not present on T before this function call (`new <= len(d)`).
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
    discarded = new = 0
    dtmp = None
    df_pkey_col = autoincrement_pkey_col.key
    dframe_with_pkeys = dataframe
    if df_pkey_col not in dataframe.columns:
        dframe_with_pkeys[df_pkey_col] = None
        dtmp = pd.DataFrame(dframe_with_pkeys, copy=False)
    else:
        mask = pd.isnull(dframe_with_pkeys[df_pkey_col])
        dtmp = dframe_with_pkeys[mask]

    numnans = len(dtmp)
    if numnans:
        dtmp.is_copy = False  # avoid pandas SettingWithCopyWarning, we are trying to modify dtmp
        # modifications will not affect dframe_with_pkeys but we are aware of it
        max_pkey = _get_max(session, autoincrement_pkey_col) + 1
        new_pkeys = np.arange(max_pkey, max_pkey+numnans, dtype=int)
        dtmp[df_pkey_col] = new_pkeys
        new_df, discarded, new = add2db(dtmp, session, [autoincrement_pkey_col], add_buf_size,
                                        query_first=False, drop_duplicates=False)
        dframe_with_pkeys.loc[new_df.index, df_pkey_col] = new_df[df_pkey_col]
    return dframe_with_pkeys.dropna(subset=[df_pkey_col]), discarded, new


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
    return dfupdate(dataframe, df_new,
                    [c.key for c in matching_columns], [pkey_col.key], False)


def dfupdate(df_old, df_new, matching_columns, set_columns, drop_df_new_duplicates=True):
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


# def fdsn2sql(fdsn_param):
#     """returns a sql "like" expression from a given fdsn channel constraint parameters
#     (network,    station,    location    and channel) converting wildcards, if any"""
#     return fdsn_param.replace('*', "%").replace("?", "_")

# class Adder(object):
# 
#     def __init__(self, session, pkey_cols, add_buf_size=10):
#         """
#             Creates a new Adder, which will add new dataframes to the db table identified by
#             `pkey_cols`, skipping already saved (according to `pkey_cols`). See class-method `add`
#             :param session: the sql alchemy session
#             :param pkey_cols: sql alchemy instrumented attributes (As list). They *must* belong to
#             the same table ORM. Usually, it's a single primary key, but any set of keys is
#             possible. In principle, the keys should have a unique constraint set.
#             If not, the check for existing db entries might be inconsistent
#             :param add_buf_size: integer, defaults to 10. The buffer size which, when full (or the
#             iteration is ended) will issue a commit for those instances to be added. Setting lower
#             numbers assures all "well formed" instances are added, but is slower, setting to a
#             higher number speeds up a lot the db insertion but if the N-th instance violates some
#             db constraint, then buffer instances from N+1-th on will not be added
#         """
#         self.existing_keys = None
#         self.pkey_cols = pkey_cols
#         self.shared_colnames = None
#         self.session = session
#         self.add_buf_size = add_buf_size
#         self._conn = None
#         self._discarded = 0
#         self._new = 0
# 
#     @property
#     def discarded(self):
#         return self._discarded
# 
#     @property
#     def new(self):
#         return self._new
# 
#     def add(self, dataframe, drop_duplicates='inplace'):
#         """
#             Adds each row of `dataframe` to the db. Rows already present will not be added again
#             Rows are compared using `pkey_cols` (specified in the constructor), thus rows are equal
#             if they equal in values for those columns.
#             Note that This method uses
#             sqlalchemy-core methods speed up the insertion into the db. Thus,
#             the session will *not* be updated with the new inserted instances, but the returned
#             dataframe will
#             :param dataframe: the dataframe to add. Obviously, it must hold *at least* the column
#             names matching `self.pkey_cols` names. The data frame can hold columns
#             not present on the db table. They will not added to the db nor they will be removed
#             from the returned data frame
#             :param drop_duplicates: string or boolan: (True, False, 'inplace'). Default: 'inplace'
#             The dataframe *MUST NOT HAVE DUPLICATES* (equal rows under `pkey_cols` names). If the
#             user is *sure* it hasn't, or it called explicitly:
#             ```
#             dataframe.drop_duplicates(subset=pkeynames, ...)
#             ```
#             specify False to skip the check and speed up the procedure.
#             Otherwise, specify True to drop duplicates on the dataframe
#             preserving the original dataframe, or 'inplace' to modify inplace the given dataframe
#             The duplicates drop happens before the db insertion
#             :return: a new dataframe with only rows added or already present in the db. Check
#             `self.new` and `self.discarded` to know how many new items where added to the db and
#             how many `dataframe` rows were discarded
#         """
#         buf_size = max(self.add_buf_size, 1)
#         buf_inst = OrderedDict()
#         buf_indices = OrderedDict()
#         new = 0
#         pkeynames = [c.key for c in self.pkey_cols]
#         if drop_duplicates is not False:
#             oldlen = len(dataframe)
#             if drop_duplicates == 'inplace':
#                 dataframe.drop_duplicates(subset=pkeynames, inplace=True)
#             else:
#                 dataframe = dataframe.drop_duplicates(subset=pkeynames)
#             self._discarded += oldlen - len(dataframe)
#         table_model = self.pkey_cols[0].class_
#         # allocate all primary keys for the given model
#         existing_keys = self.existing_keys if self.existing_keys is not None else \
#             set(tuple(x) for x in self.session.query(*self.pkey_cols))
# 
#         if self.shared_colnames is None:
#             self.shared_colnames = list(shared_colnames(table_model, dataframe))
# 
#         if self._conn is None:
#             self._conn = self.session.connection()
#         conn = self._conn
# 
#         # Note: we might remove dupes on the dataframe, but then we would return a different
#         # dataframe
#         # as we want to use this method for stations and channels, and we need to add stations first
#         # this might remove some channels.
# 
#         last = len(dataframe) - 1
#         indices = []  # indices to keep (already existing or successfully written)
#         for i, rowdict in enumerate(dfrowiter(dataframe, self.shared_colnames)):
#             rowtup = tuple(rowdict[k] for k in pkeynames)
#             if rowtup not in existing_keys:
#                 buf_inst[rowtup] = rowdict
#                 buf_indices[rowtup] = i
#             else:
#                 indices.append(i)
#             if len(buf_inst) == buf_size or (i == last and buf_inst):
#                 update = False
#                 try:
#                     conn.execute(table_model.__table__.insert(), buf_inst.values())
#                     update = True
#                 except IntegrityError as _:
#                     # if we had an integrity error, it might be that buf had dupes
#                     # in this case, some rows might be added! So to be sure we issue a
#                     # query to know which elements have been added
#                     # 1. Create a query specific to the buf elements using pkey_cols:
#                     exprs = []
#                     for key in buf_inst.keys():
#                         exprs.append(and_(*[pkeycol == v
#                                             for pkeycol, v in izip(self.pkey_cols, key)]))
#                     # 2. Get those elements and add them, if saved to the db
#                     if exprs:
#                         added_set = set(tuple(x) for x in self.session.query(*self.pkey_cols).
#                                         filter(or_(*exprs)))
#                         if added_set:
#                             update = True
#                             buf_indices = {k: buf_indices[k] for k in added_set if k in buf_indices}
#                             buf_inst = {k: buf_inst[k] for k in added_set if k in buf_inst}
#                 except SQLAlchemyError as _:
#                     pass
# 
#                 if update:
#                     existing_keys |= set(buf_inst.keys())
#                     new += len(buf_inst)
#                     indices.extend(buf_indices.values())
# 
#                 buf_inst.clear()
#                 buf_indices.clear()
# 
#         # update existing keys:
#         self.existing_keys = existing_keys
#         oldlen = len(dataframe)
#         indices.sort()  # indices might not be sorted, we want to return the same orders
#         # (for matching rows)
#         df = dataframe.iloc[indices]
#         df.is_copy = False  # avoid settings with copy warning
#         self._discarded += oldlen - len(df)
#         self._new += new
#         return df
