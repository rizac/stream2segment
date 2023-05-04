"""
Module implementing the functionalities that allow issuing sql select
statements from config files, command line or via GUI input controls
via string expression on database tables columns

:date: Mar 6, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from datetime import datetime
import shlex
import warnings

import numpy as np
from sqlalchemy import asc, and_, desc, inspect


def exprquery(sa_query, conditions, orderby=None):
    """Enhance the given SQLAlchemy query `sa_query` with conditions
    and ordering given in form of **string** expression, returning a new
    SQLAlchemy query.

    This method first infers the reference Model from **the first entity
    passed** in `sa_query` (thus **pay attention to the arguments order when
    you build sa_query**). These for example all consider the class 'Parent'
    as reference model:
    ```
        session.query(Parent)
        session.query(Parent, Child)
        session.query(count(distinct(Parent)))

    ```
    Then columns (and relationships, if any) are extracted from `conditions`,
    which is a `dict[str, str]` where keys are any queryable attribute of the
    **reference model** (columns, hybrid properties, relationships), and the
    mapped values are string expressions that will be converted to their SQL
    counterparts. E.g.:
    ```
    {
        'id' : '<6',
        'child.age': '>8'
    }
    ```
    The order by condition is then applied, if present.

    The returned query is a valid sql-alchemy query and can be further
    manipulated **in most cases**: a case when it's not possible is when
    issuing a `group_by` in `postgres`
    (for info, see https://stackoverflow.com/a/18061451).
    In these case a normal SqlAlchemy query must be issued

    Example:
    ```
    # pseudo code:

    Parent:
        id = Column(primary_key=True,...)
        child_id: foreign_key(Child.id)
        age = Column(Integer, ...)
        birth = Column(DateTime, ...)
        children = relationship(Child,...)

    Child:
        id = Column(primary_key=True,...)
        age = Column(Integer, ...)
        birth = Column(DateTime, ...)
        parent = relationship(Parent,...)

    sess = ...  # sql-alchemy session
    sa_query = sess.query  # sql-alchemy session's query object
    ```

    Then:

    Return all parents who have children:
    `exprquery(sa_query(Parent), {'children', 'any'})`

    Return all parents id's who have children:
    `exprquery(sa_query(Parent.id), {'children', 'any'})`

    Return all parents who have adult children:
    `exprquery(sa_query(Parent), {'children.age', '>=18'})`

    Return all parents born before 1980 who have adult children:
    ```
    exprquery(sa_query(Parent), {'birth': '<1980-01-01',
                                 'children.age', '>=18'})
    ```

    Return all parents who have adult children, sorted (ascending)
    by parent's age (2 solutions):
    `exprquery(sa_query(Parent), {'children.age', '>=18'}, ['age'])` or
    `exprquery(sa_query(Parent), {'children.age', '>=18'}, [('age', 'asc')])`

    Return all parents who have adult children, sorted (ascending) by
    parent's age and then descending by parent's id:
    ```
    exprquery(sa_query(Parent), {'children.age', '>=18'},
              [('age', 'asc'), ('id', 'desc')])
    ```

    Finally, note that, called `date1980 = datetime(1980, 1, 1)`, these three
    are equivalent and valid:
    ```
    exprquery(sa_query(Parent).filter(Parent.birth < date1980),
              {'children.age', '>=18'})
    ```
    ```
    exprquery(sa_query(Parent), {'children.age', '>=18'})\
        .filter(Parent.birth < date1980)
    ```
    exprquery(sa_query(Parent),
              {'birth': '<1980-01-01', 'children.age', '>=18'})
    ```

    :param sa_query: any sql-alchemy query object
    :param conditions: a dict of string columns mapped to **string**
        expression, e.g. "column2": "[1, 45]" or "column1": "true" (note:
        string, not the boolean True). A string column is an expression
        denoting an attribute of the reference model class and can include
        relationships.
        Example: if the reference model tablename is 'mymodel', then a string
        column 'name' will refer to 'mymodel.name', 'name.id' denotes on the
        other hand a relationship 'name' on 'mymodel' and will refer to the
        'id' attribute of the table mapped by 'mymodel.name'. The values of
        the dict on the other hand are string expressions in the form
        recognized by `binexpr`. E.g. '>=5', '["4", "5"]' ...
        For each condition mapped to a falsy value (e.g., None or empty
        string), the condition is discarded. See note [*] below for auto-added
        joins  from columns.
    :param orderby: a list of string columns (same format
        as `conditions` keys), or a list of tuples where the first element is
        a string column, and the second is either "asc" (ascending) or "desc"
        (descending). In the first case, the order is "asc" by default. See
        note [*] below for auto-added joins from orderby columns.

    :return: a new SQLalchemy query including the given conditions and ordering

    [*] Note on auto-added joins: if any given `condition` or `orderby` key
        refers to relationships defined on the reference model class, then
        necessary joins are appended to `sa_query`, *unless already present*
        (this should also avoid the warning 'SAWarning: Pathed join target',
        currently in `sqlalchemy.orm.query.py:2105`).
    """
    # get the table model from the query's FIRST column description
    model = sa_query.column_descriptions[0]['entity']
    parsed_conditions = []
    joins = set()  # relationships have an hash, this assures no duplicates
    # set already joined tables. We use the private method _join_entities
    # although it's not documented anywhere (we inspected via eclipse debug to
    # find the method):
    already_joined_models = set(_.class_ for _ in sa_query._join_entities)
    # if its'an InstrumentedAttribute, use the class
    relations = inspect(model).relationships

    if conditions:
        for attname, expression in conditions.items():
            if not expression:  # discard empty strings, None's, ...
                # note that expressions MUST be strings
                continue
            relationship, column = _get_rel_and_column(model, attname, relations)
            if relationship is not None and \
                    get_rel_refmodel(relationship) not in already_joined_models:
                joins.add(relationship)
            condition = binexpr(column, expression)
            parsed_conditions.append(condition)

    directions = {"asc": asc, "desc": desc}
    orders = []
    if orderby:
        for order in orderby:
            try:
                column_str, direction = order
            except ValueError:
                column_str, direction = order, "asc"
            directionfunc = directions[direction]
            relationship, column = _get_rel_and_column(model, column_str, relations)
            if relationship is not None and \
                    get_rel_refmodel(relationship) not in already_joined_models:
                joins.add(relationship)
            # FIXME: we might also write column.asc() or column.desc()
            orders.append(directionfunc(column))

    if joins:
        sa_query = sa_query.join(*joins)
    if parsed_conditions:
        sa_query = sa_query.filter(and_(*parsed_conditions))
    if orders:
        sa_query = sa_query.order_by(*orders)
    return sa_query


def _get_rel_and_column(model, colname, relations=None):
    if relations is None:
        relations = inspect(model).relationships  # ['station'].remote_side
    cols = colname.split(".")
    obj = model
    rel = None
    for col in cols:
        tmp = getattr(obj, col)
        if col in relations and obj is model:
            rel = tmp
            obj = get_rel_refmodel(relations[col])
        else:
            obj = tmp
    return rel, obj


def get_rel_refmodel(relationship):
    """Return the relationship's reference table model

    :param relationship: the InstrumentedAttribute relative to a relationship.
        Example. Given a model `model`, and a relationship e.g.
        `r_name=inspect(model).relationships.keys()[i],
        then `relationship=getattr(model, r_name)`
    """
    return relationship.mapper.class_


def binexpr(column, expr):
    """Return an :class:`sqlalchemy.sql.expression.BinaryExpression` to be
    used as `query.filter` argument from the given column and the given
    expression. Supports the operators given in
    :func:`stream2segment.io.db.sqlevalexpr.split` and the types given in
    `parsevals`: (`int`s, `float`s, `datetime`s, `bool`s and `str`s)

    :param column: an sqlkalchemy model column
    :param expr: a string expression (see `split`)

    Example:
    ```
    # given a model with column `column1`
    binexpr(model.column1, '>=5')
    ```
    """
    try:
        operator, values = split(expr)
        values = parsevals_sql(column, values)
        if operator == '=':
            return column == values[0] if len(values) == 1 else column.in_(values)
        if operator == "!=":
            return column != values[0] if len(values) == 1 else ~column.in_(values)
        if operator == ">":
            return and_(*[column > val for val in values])
        if operator == "<":
            return and_(*[column < val for val in values])
        if operator == ">=":
            return and_(*[column >= val for val in values])
        if operator == "<=":
            return and_(*[column <= val for val in values])
        else:
            cond = column.between(values[0], values[1])
            if operator == 'open':
                cond = cond & (column != values[0]) & (column != values[1])
            elif operator == 'leftopen':
                cond = cond & (column != values[0])
            elif operator == 'rightopen':
                cond = cond & (column != values[1])
            elif operator != 'closed':
                raise ValueError("Invalid operator %s" % operator)
            return cond
    except (AssertionError, ValueError, IndexError, AttributeError, TypeError):
        raise ValueError("Invalid expression for column '%s': %s" % (column, expr))


def split(expr):
    """Split the expression into its operator(s) and its value.

    :param: expression: a string which is first stripped (i.e., leading and
        trailing spaces are omitted) and then either:
        1. starts with (zero or more spaces and):
            "<", "=", "==", "!=", ">", "<=", ">="
        2. starts with "[", "(", "]" **and** ends with "]" , "[", ")", where
           "[", "]" denote the closed interval (endpoints included) and the
           other symbols an open interval (endpoints excluded)

    :return: the operator (one of the symbol above) and the remaining string.
        Note that the operator is normalized to "=" in case 1 if either "=" or
        "==", and in case 2 is "open", "leftopen", "rightopen", "closed"
    """
    expr = expr.strip()
    if expr[:2] in ("<=", ">=", "==", "!="):
        return '=' if expr[:2] == '==' else expr[:2], expr[2:].strip()
    if expr[0] in ("<", ">", "="):
        return expr[0], expr[1:].strip()
    if expr[0] in ("(", "]", "["):
        assert expr[-1] in (")", "[", "]")
        newexpr = expr[1:-1].replace(",", " ")
        assert len(shlex.split(newexpr)) == 2
        if expr[0] == '[':
            val = "closed" if expr[-1] == ']' else "rightopen"
        else:
            val = "leftopen" if expr[-1] == ']' else "open"
        return val, newexpr
    return "=", expr


def parsevals_sql(column, expr_value):
    """Parse `expr_value` according to the model column type. Supports `int`s,
    `float`s, `datetime`s, `bool`s and `str`s.

    :param expr_value: a value given as command line argument(s). Thus, quoted
        strings will be recognized removing the quotation symbols. The list of
        values will then be casted to the python type of the given column. Note
        that the values are intended to be in SQL syntax, thus NULL or null for
        Python None's. Datetime's must be input in ISO format (with or without
        spaces)

    Example. Given a model with int column 'column1':
    `parsevals(model.column1, '4 null 5 6') = [4, None, 5, 6]`
    """
    try:
        return parsevals(get_pytype(get_sqltype(column)), expr_value)
    except ValueError as verr:
        raise ValueError("column %s: %s" % (str(column), str(verr)))


def parsevals(pythontype, expr_value):
    """Parse `expr_value` according to the given python type. Supports `int`s,
    `float`s, `datetime`s, `bool`s and `str`s.

    :param expr_value: if bool, int, float, None or datetime, or iterable of
        those values, a value given as command line argument(s). Thus, quoted
        strings will be recognized removing the quotation symbols. The list of
        values will then be casted to the python type of the given column.
        Note that the values are intended to be in SQL syntax, thus NULL or
        null for python None's. Datetime's must be input in ISO format
        (with or without spaces)

    Example. Given a model with int column 'column1':
    `parsevals(int, '4 null 5 6') = [4, None, 5, 6]`
    """
    _NONES = ("null", "NULL")
    vals = shlex.split(expr_value)
    if pythontype == float:
        return [None if x in _NONES else float(x) for x in vals]
    elif pythontype == int:
        return [None if x in _NONES else int(x) for x in vals]
    elif pythontype == bool:
        # bool requires a user defined function for parsing javascript/python
        # strings (see below)
        return [None if x in _NONES else _bool(x) for x in vals]
    elif pythontype == datetime:
        # numpy complains if we have timezone aware strings. This is also
        # the case when we insert programmatically some error (say, a comma)
        # at the end: it is interpreted as timezone. No big deal except
        # we want to suppress the warning. Note that in future numpy releases
        # this will raise, which is even better so that one must pass
        # utc datetime strings with no timezone info:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # array casting below works with None's:
            return np.array(vals, dtype="datetime64[us]").tolist()
    elif pythontype == str:
        return [None if x in _NONES else str(x) for x in vals]

    raise ValueError('Unsupported python type %s' % pythontype)


def _bool(val):
    """Parse javascript booleans true false and returns a python boolean"""
    if val in ('false', 'False', 'FALSE'):
        return False
    elif val in ('true', 'True', 'TRUE'):
        return True
    return bool(val)


def get_sqltype(obj):
    """Return the sql type associated with `obj`.

    :param obj: an object with an 'expression' method, e.g.
        sqlalchemy.sql.schema.Column or
        :class:`sqlalchemy.orm.attributes.QueryableAttribute` (for instance
        :class:`sqlalchemy.orm.attributes.InstrumentedAttribute`, i.e. the
        model's attributes mapping db columns)

    :return: An object defined in :class:`sqlalchemy.sql.sqltypes`, e.g.
        `Integer` the method  :function:`get_pytype` of the returned object
        defines the relative python type
    """
    try:
        return obj.expression.type
    except NotImplementedError:
        return None


def get_pytype(sqltype):
    """Returns the python type associated to the given sqltype.
    :param sqltype: an object as returned by `get_sqltype`
    :return: a python type class asscoaiated to `sqltype`, or None
    """
    try:
        return sqltype.python_type
    except NotImplementedError:
        return None
