'''
Module implementing the functionalities that allow querying an sql database
via string expression on table columns

Created on Mar 6, 2017

@author: riccardo
'''
from builtins import str
import shlex
import numpy as np
from datetime import datetime
from sqlalchemy import asc, and_, desc, inspect
from sqlalchemy.exc import InvalidRequestError


def exprquery(sa_query, conditions, orderby=None, distinct=None):
    """
    Enhance the given sql-alchemy query `sa_query` with conditions
    and ordering given in form of (string) expression, returning a new sql alchemy query.
    Joins are automatically added inside this
    method. That is, if any given `condition` key refers to relationships defined on the model class
    (retrieved from the first ORM model found in `sa_querysa_query.column_descriptions`), then
    necessary joins are appended to `sa_query`. If `sa_query` already contains joins, the join
    is not added again, and sql-alchemy issues a warning 'SAWarning: Pathed join target'
    (currently in `sqlalchemy.orm.query.py:2105`).
    The returned query is a valid sql-alchemy query and can be further manipulated
    **in most cases**: a case when it's not possible is when issuing a `group_by` in `postgres`
    (for info, see
    http://stackoverflow.com/questions/18061285/postgresql-must-appear-in-the-group-by-clause-or-be-used-in-an-aggregate-functi).
    In these cases a normal SqlAlchemy query must be issued

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

    # some queries, given a session object
    sess = ...  # sql-alchemy session

    #return all parents who have children:
    exprquery(sess.query(Parent), {'children', 'any'})

    #return all parents id's who have children:
    exprquery(sess.query(Parent.id), {'children', 'any'})

    #return all parents who have adult children:
    exprquery(sess.query(Parent), {'children.age', '>=18'})

    #return all parents born before 1980 who have children not minor:
    date1980 = datetime(1980,1,1))
    # all these query are equivalent:
    exprquery(sess.query(Parent).filter(Parent.birth < date1980), {'children.age', '>=18'})
    exprquery(sess.query(Parent), {'children.age', '>=18'}).filter(Parent.birth < date1980)
    exprquery(sess.query(Parent), {'birth': '1980-01-01', 'children.age', '>=18'})

    #return all parents who have non-minor children, with age sorted ascending:
    exprquery(sess.query(Parent), {'children.age', '>=18'}, ['age'])
    # same as above (providing 'asc' which if missing is the default):
    exprquery(sess.query(Parent), {'children.age', '>=18'}, [('age', 'asc')])
    # You can also provide more than one order (which in this case is quote trivial/redundant):
    exprquery(sess.query(Parent), {'children.age', '>=18'}, [('age', 'desc'), ('birth', 'asc')])
    ```

    :param query: any sql-alchemy query object
    :param conditions: a dict of string columns mapped to strings expression, e.g.
    "column2": "[1, 45]".
    A string column is an expression denoting an attribute of the underlying model (retrieved
    as the first ORM model found in `sa_querysa_query.column_descriptions`) and can include
    relationships. Example: if the model tablename is 'mymodel', then a string column 'name'
    will refer to 'mymodel.name', 'name.id' denotes on the other hand a relationship 'name'
    on 'mymodel' and will refer to the 'id' attribute of the table mapped by 'mymodel.name'.
    The values of the dict on the other hand are string expressions in the form recognized
    by `get_condition`. E.g. '>=5', '["4", "5"]' ...
    For each condition mapped to a falsy value (e.g., None or empty string), the condition is
    discarded
    :param orderby: a list of string columns (same format
    as `conditions` keys), or a list of tuples where the first element is
    a string column, and the second is either "asc" (ascending) or "desc" (descending). In the
    first case, the order is "asc" by default
    """
    # get the table model from the query
    model = sa_query.column_descriptions[0]['entity']
    parsed_conditions = []
    joins = set()  # relationships have an hash, this assures no duplicates

    # if its'an InstrumentedAttribute, use the class
    relations = inspect(model).relationships

    if conditions:
        for attname, expression in conditions.items():
            if not expression:  # discard falsy expressions (empty strings, None's)
                continue
            relationship, column = _get_rel_and_column(model, attname, relations)
            if expression.strip() in ('any', 'none'):  # any or none:
                condition = relationship.any() if expression.strip() == 'any' \
                    else ~relationship.any()
            else:
                if relationship is not None:
                    joins.add(relationship)
                condition = get_condition(column, expression)
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
            if relationship is not None:
                joins.add(relationship)
            orders.append(directionfunc(column))

    if joins:
        sa_query = sa_query.join(*joins)
    if parsed_conditions:
        sa_query = sa_query.filter(and_(*parsed_conditions))
    if orders:
        sa_query = sa_query.order_by(*orders)
    if distinct is True:
        sa_query = sa_query.distinct()
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
            obj = relations[col].mapper.class_
        else:
            obj = tmp
    return rel, obj


def get_condition(column, expr):
    """Returns an sql alchemy binary expression to be used as `query.filter` argument
    from the given column and the given expression. Supports the operators given in `split` and the
    types given in `parsevals` ()
    :param column: an sqlkalchemy model column
    :param expr: a string expression (see `split`)

    :example:
    ```
    # given a model with column `column1`
    get_condition(model.column1, '>=5')
    ```
    """
    operator, values = split(expr)
    values = parsevals_sql(column, values)
    if operator == '=':
        return column == values[0] if len(values) == 1 else column.in_(values)
    elif operator == "!=":
        return column != values[0] if len(values) == 1 else ~column.in_(values)
    elif operator == ">":
        return and_(*[column > val for val in values])
    elif operator == "<":
        return and_(*[column < val for val in values])
    elif operator == ">=":
        return and_(*[column >= val for val in values])
    elif operator == "<=":
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
    raise ValueError("Invalid expression %s" % expr)


def split(expr):
    """
        Splits the expression into its operator(s) and its value.

        :param: expression: a string which is first stripped (i.e., leading and trailing spaces
        are omitted) and then either:
        1. starts with (zero or more spaces and):
            "<", "=", "==", "!=", ">", "<=", ">="
        2. starts with "[", "(", "]" **and** ends with "]" , "[", ")", where "[", "]" denote the
        closed interval (endpoints included) and the other symbols an open interval (endpoints
        excluded)

        :return: the operator (one of the symbol above) and the remaining string. Note that the
        operator is normalized to "=" in case 1 if either "=" or "==", and in case 2 is "open",
        "leftopen", "rightopen", "closed"
    """
    expr = expr.strip()
    if expr[:2] in ("<=", ">=", "==", "!="):
        return '=' if expr[:2] == '==' else expr[:2], expr[2:].strip()
    elif expr[0] in ("<", ">", "="):
        return expr[0], expr[1:].strip()
    elif expr[0] in ("(", "]", "["):
        assert expr[-1] in (")", "[", "]")
        newexpr = expr[1:-1].replace(",", " ")
        assert len(shlex.split(newexpr)) == 2
        if expr[0] == '[':
            val = "closed" if expr[-1] == ']' else "rightopen"
        else:
            val = "leftopen" if expr[-1] == ']' else "open"
        return val, newexpr
    else:
        return "=", expr


def parsevals_sql(column, expr_value):
    """
        parses `expr_value` according to the model column type. Supports `int`s, `float`s,
        `datetime`s, `bool`s and `str`s.
        :param expr_value: a value given as command line argument(s). Thus, quoted strings will
        be recognized removing the quotation symbols.
        The list of values will then be casted to the python type of the given column.
        Note that the values are intended to be
        in SQL syntax, thus NULL or null for python None's. Datetime's must be input in ISO format
        (with or without spaces)

        :Example:
        ```
        # given a model with int column 'column1'
        parsevals(model.column1, '4 null 5 6') = [4, None, 5, 6]
        ```
    """
    try:
        return parsevals(column.type.python_type, expr_value)
    except ValueError as verr:
        raise ValueError("column %s: %s" % (str(column), str(verr)))


def parsevals(pythontype, expr_value):
    """
        parses `expr_value` according to the given python type. Supports `int`s, `float`s,
        `datetime`s, `bool`s and `str`s.
        :param expr_value: if bool, int, float, None or datetime, or iterable of those values,
        a value given as command line argument(s). Thus, quoted strings will
        be recognized removing the quotation symbols.
        The list of values will then be casted to the python type of the given column.
        Note that the values are intended to be
        in SQL syntax, thus NULL or null for python None's. Datetime's must be input in ISO format
        (with or without spaces)

        :Example:
        ```
        # given a model with int column 'column1'
        parsevals(int, '4 null 5 6') = [4, None, 5, 6]
        ```
    """
    _NONES = ("null", "NULL")
    vals = shlex.split(expr_value)
    if pythontype == float:
        return [None if x in _NONES else float(x) for x in vals]
    elif pythontype == int:
        return [None if x in _NONES else int(x) for x in vals]
    elif pythontype == bool:
        # bool requires a user defined function for parsing javascript/python strings (see below)
        return [None if x in _NONES else _bool(x) for x in vals]
    elif pythontype == datetime:
        return np.array(vals, dtype="datetime64[us]").tolist()  # works with None's
    elif pythontype == str:
        return [None if x in _NONES else str(x) for x in vals]

    raise ValueError('Unsupported python type %s' % pythontype)


def _bool(val):
    '''parses javascript booleans true false and returns a python boolean'''
    if val in ('false', 'False', 'FALSE'):
        return False
    elif val in ('true', 'True', 'TRUE'):
        return True
    return bool(val)
