'''
Module for extracting queries / queries result from input given from strings
Easing the effort for users to do queries in a simplified manner

Created on Mar 6, 2017

@author: riccardo
'''
import shlex
import numpy as np
from datetime import datetime
from sqlalchemy import asc, and_, desc, inspect
from sqlalchemy.exc import InvalidRequestError

_NONES = ("null", "NULL")


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
        :param expr_value: a value given as command line argument(s). Thus, quoted strings will
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
    vals = shlex.split(expr_value)
    if pythontype == float:
        return [None if x in _NONES else float(x) for x in vals]
    elif pythontype == int:
        return [None if x in _NONES else int(x) for x in vals]
    elif pythontype == bool:
        return [None if x in _NONES else bool(x) for x in vals]
    elif pythontype == datetime:
        return np.array(vals, dtype="datetime64[us]").tolist()  # works with None's
    elif pythontype == str:
        return [None if x in _NONES else str(x) for x in vals]

    raise ValueError('Unsupported python type %s' % pythontype)


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


def query(sa_query, model, conditions, orderby=None):
    """
    Builds a query o sql-alchemy query object according to
    `conditions` and `orderby`, which are dicts / lists of string evaluable expressions

    As an example

    The returned query can be further manipulated **in most cases**, e.g.:
    ```
        query(session, mytable.id, ....).all()
        query(session, mytable.id, ....).distinct()
    ```
    However, exceptions might be raised, for instance a
    "must appear in the GROUP BY clause or be used in an aggregate function" error
    if using Postgres as underlying db:
    ```
        query(session, mytable.id, ....).group_by(someothertable.attr)
    ```
    (for info, see http://stackoverflow.com/questions/18061285/postgresql-must-appear-in-the-group-by-clause-or-be-used-in-an-aggregate-functi)

    When a workaround is not feasible (for instance, replacing `group_by` with `distinct`), this
    function has limited performances

    :param session: an sql-alchemy session
    :param query_arg: the argument to the query: can be a model instance ('mymodel')
    or one of its columns ('mymodel.id')
    :param conditions: a dict of string columns mapped to strings expression, e.g.
    "column2": "[1, 45]".
    A string column is an expression denoting an attribute of the underlying model (retrieved
    from `query_arg`) and can include relationships. Example: if query arg is 'mymodel' or
    'mymodel.id', then a string column 'name' will refer to 'mymodel.name', 'name.id' denotes
    on the other hand a relationship 'name' on 'mymodel' and will refer to the 'id' attribute of the
    table mapped by 'mymodel.name'.
    The values of the dict on the other hand are string expressions in the form recognized
    by `get_condition`. E.g. '>=5', '["4", "5"]' ...
    If this argument is None or evaluates to False, no filter is applied
    :param orderby: a list of string columns (same format
    as `conditions` keys), or a list of tuples where the first element is
    a string column, and the second is either "asc" (scending, the default) or "desc" (descending)
    """

    # contrarily to query_args, this function returns the arguments for
    # an sqlalchemy join(...).filter(...).order_by(...).
    # Thus, providing query_arg ans session, it builds:
    parsed_conditions = []
    joins = set()  # relationships have an hash, this assures no duplicates

    # if its'an InstrumentedAttribute, use the class
    relations = inspect(model).relationships

    if conditions:
        for attname, expression in conditions.iteritems():
            if not expression:
                continue
            relationship, column = get_rel_and_column(model, attname, relations)
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
            relationship, column = get_rel_and_column(model, column_str, relations)
            if relationship is not None:
                joins.add(relationship)
            orders.append(directionfunc(column))

    if joins:
        sa_query = sa_query.join(*joins)
    if parsed_conditions:
        sa_query = sa_query.filter(and_(*parsed_conditions))
    if orders:
        sa_query = sa_query.order_by(*orders)
    return sa_query


def queryold(session, query_arg, conditions, orderby=None, *sqlalchemy_query_expressions):
    """
    Builds a query o sql-alchemy query object according to
    `conditions` and `orderby`, which are dicts / lists of string evaluable expressions

    As an example
    
    The returned query can be further manipulated **in most cases**, e.g.:
    ```
        query(session, mytable.id, ....).all()
        query(session, mytable.id, ....).distinct()
    ```
    However, exceptions might be raised, for instance a
    "must appear in the GROUP BY clause or be used in an aggregate function" error
    if using Postgres as underlying db:
    ```
        query(session, mytable.id, ....).group_by(someothertable.attr)
    ```
    (for info, see http://stackoverflow.com/questions/18061285/postgresql-must-appear-in-the-group-by-clause-or-be-used-in-an-aggregate-functi)

    When a workaround is not feasible (for instance, replacing `group_by` with `distinct`), this
    function has limited performances

    :param session: an sql-alchemy session
    :param query_arg: the argument to the query: can be a model instance ('mymodel')
    or one of its attribute ('mymodel.id')
    :param conditions: a dict of string columns mapped to strings expression, e.g.
    "column2": "[1, 45]".
    A string column is an expression denoting an attribute of the underlying model (retrieved
    from `query_arg`) and can include relationships. Example: if query arg is 'mymodel' or
    'mymodel.id', then a string column 'name' will refer to 'mymodel.name', 'name.id' denotes
    on the other hand a relationship 'name' on 'mymodel' and will refer to the 'id' attribute of the
    table mapped by 'mymodel.name'.
    The values of the dict on the other hand are string expressions in the form recognized
    by `get_condition`. E.g. '>=5', '["4", "5"]' ...
    If this argument is None or evaluates to False, no filter is applied
    :param orderby: a list of string columns (same format
    as `conditions` keys), or a list of tuples where the first element is
    a string column, and the second is either "asc" (scending, the default) or "desc" (descending)
    """

    # contrarily to query_args, this function returns the arguments for
    # an sqlalchemy join(...).filter(...).order_by(...).
    # Thus, providing query_arg ans session, it builds:
    parsed_conditions = list(sqlalchemy_query_expressions) or []
    joins = set()  # relationships have an hash, this assures no duplicates

    # if its'an InstrumentedAttribute, use the class
    model = query_arg.class_ if hasattr(query_arg, "class_") else query_arg
    relations = inspect(model).relationships

    if conditions:
        for attname, expression in conditions.iteritems():
            if not expression:
                continue
            relationship, column = get_rel_and_column(model, attname, relations)
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
            relationship, column = get_rel_and_column(model, column_str, relations)
            if relationship is not None:
                joins.add(relationship)
            orders.append(directionfunc(column))

    query = session.query(query_arg)
    if joins:
        query = query.join(*joins)
    if parsed_conditions:
        query = query.filter(and_(*parsed_conditions))
    if orders:
        query = query.order_by(*orders)
    return query


def query_args(model, conditions):
    """Returns the query arguments to be passed to a sqlalchemy query
       The function checks for relationship established on the `model` and creates a
       list of conditions (concatenated with sqlalchemy `and_`) to be passed to the query.
       Returns None if conditions is None (or it evaluates to False)
       **IMPORTANT**
       This method can avoid joins on the query by means of relationships passed in
       `conditions` key (by using 'has' or 'any'). However, this means that exist clause will
       be issued (sometimes decreasing performances - ref needed) but more importantly the query
       built with the returned arguments cannot exploit its full potentiality (todo: example needed)

       :param conditions: a dict of string columns mapped to strings expression, e.g.
        "column2": "[1, 45]".
        A string column is an expression denoting an attribute of the underlying model (retrieved
        from `query_arg`) and can include relationships. Example: if query arg is 'mymodel' or
        'mymodel.id', then a string column 'name' will refer to 'mymodel.name', 'name.id' denotes
        on the other hand a relationship 'name' on 'mymodel' and will refer to the 'id' attribute
        of the table mapped by 'mymodel.name'.
        The values of the dict on the other hand are string expressions in the form recognized
        by `get_condition`. E.g. '>=5', '["4", "5"]' ...
        If this argument is None or evaluates to False, no filter is applied
        :Example:
       ```
       query_args = query_args(model, {'att1': "[1,4]", 'att3': 'null'})
       session.query(model).query(query_args).all()
       ```
    """
    parsed_conditions = []
    # if its'an InstrumentedAttribute, use the class
    relations = inspect(model).relationships

    if conditions:
        for attname, condition_expr in conditions.iteritems():
            if not condition_expr:
                continue
            relationship, column = get_rel_and_column(model, attname, relations)
            condition = get_condition(column, condition_expr)
            if relationship is not None:
                try:
                    condition = relationship.any(condition)
                except InvalidRequestError:
                    condition = relationship.has(condition)
            parsed_conditions.append(condition)

    if not parsed_conditions:
        return None
    return and_(*parsed_conditions)


def get_columns(instance, keys):
    ret = {}
    for field in keys:
        try:
            attval = get_column(instance, field)
        except AttributeError:
            continue
        ret[field] = attval
    return ret


def get_column(instance, colname):
    cols = colname.split(".")
    obj = instance
    for i, col in enumerate(cols):
        try:
            obj = getattr(obj, col)
        except AttributeError:  # maybe a relationship with uselist=True?
            # but only if last attribute (otherwise an error is thrown)
            try:
                if i == len(cols)-1:
                    return [getattr(o, col) for o in obj]
            except:
                pass
            raise ValueError("Invalid column '%s' in '%s'" % (col, colname))
    return obj


def get_rel_and_column(model, colname, relations=None):
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
