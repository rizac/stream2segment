'''
Created on Mar 6, 2017

@author: riccardo
'''
import shlex
import numpy as np
from datetime import datetime
from sqlalchemy import asc, and_, desc, inspect
import pytest
from stream2segment.io.db.models import Segment, Class

_NONES = ("null", "NULL", "None")


def split(expr):
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


def parsevals(column, expr_value):
    pythontype = column.type.python_type
    if pythontype is None:
        return None
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
    else:
        raise ValueError('unsupported python type: %s' % pythontype)


def get_condition(column, expr):
    operator, values = split(expr)
    values = parsevals(column, values)
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


# def get_queryfilter(model, **atts_and_conditionexprs):
#     columns = get_columns(model, [k for k, v in atts_and_conditionexprs.iteritems() if v])
#     conditions = []
#     for attname, condition_expr in atts_and_conditionexprs.iteritems():
#         if not condition_expr or attname not in columns:
#             continue
#         column = columns[attname]
#         if column is Segment.classes:
#             # pre-split: if None, set the expression accordingly:
#             operator, values = split(condition_expr)
#             _any = Segment.classes.any  # @UndefinedVariable
#             if values in _NONES and operator in ("=", "!="):
#                 condition = ~_any() if operator == '=' else _any()
#             else:
#                 # fall back to the "normal" case, but using Class.id as column, not Segment.classes
#                 condition = _any(get_condition(Class.id, condition_expr))  # @UndefinedVariable
#         else:
#             condition = get_condition(column, condition_expr)
#         conditions.append(condition)
#     return None if not conditions else and_(*conditions)


def query(session, query_arg, atts_and_conditionexprs, orderby_list=None):
    # columns = get_columns(model, [k for k, v in atts_and_conditionexprs.iteritems() if v])
    conditions = []
    joins = []
    # if its'an InstrumentedAttribute, use the class
    model = query_arg.class_ if hasattr(query_arg, "class_") else query_arg
    relations = inspect(model).relationships

    if atts_and_conditionexprs:
        for attname, condition_expr in atts_and_conditionexprs.iteritems():
            if not condition_expr:
                continue
            relationship, column = get_rel_and_column(model, attname, relations)
            if column is Segment.classes:
                # pre-split: if None, set the expression accordingly:
                operator, values = split(condition_expr)
                _any = Segment.classes.any  # @UndefinedVariable
                if values in _NONES and operator in ("=", "!="):
                    condition = ~_any() if operator == '=' else _any()
                else:
                    # fall back to the "normal" case, but using Class.id as column, not Segment.classes
                    condition = _any(get_condition(Class.id, condition_expr))  # @UndefinedVariable
            else:
                if relationship is not None:
                    joins.append(relationship)
                try:
                    condition = get_condition(column, condition_expr)
                except AttributeError:
                    operator, values = split(condition_expr)
                    _any = relationship.any  # @UndefinedVariable
                    if values in _NONES and operator in ("=", "!="):
                        condition = ~_any() if operator == '=' else _any()
                    else:
                        obj = column
                        column = inspect(obj).primary_key[0].key
                        # fall back to the "normal" case, but using Class.id as column, not Segment.classes
                        condition = _any(get_condition(getattr(obj, column), condition_expr))  # @UndefinedVariable
    
            conditions.append(condition)

    query = session.query(query_arg)
    if joins:
        query = query.join(*joins)
    if conditions:
        query = query.filter(and_(*conditions))

    if orderby_list:
        query = query.order_by(*get_order_by_args(model, orderby_list))
    return query


def get_order_by_args(model, order_list):
    """ordeR_list: a list of 2 element lists/tuple, where the first element is a
    column of `model` in string format (e.g. "id", or "event.id" for relations defined in there)
    and the second is either "asc" or "desc"""
    if order_list:
        assert all(c[1] in ("asc", "desc") for c in order_list)
        return [asc(get_column(model, c[0])) if c[1] == 'asc' else
                desc(get_column(model, c[0])) for c in order_list]
    return None


def get_columns(model_or_instance, keys):
    ret = {}
    for field in keys:
        try:
            attval = get_column(model_or_instance, field)
        except AttributeError:
            continue
            # attval = [getattr(x, attname) for x in obj]
        ret[field] = attval
    return ret


def get_column(instance, colname):
    cols = colname.split(".")
    obj = instance
    for col in cols:
        obj = getattr(obj, col)
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

#     idx = colname.find(".")
#     obj = model
#     attname = colname
#     rel = None
#     while True:
#         idx = attname.find(".")
#         if idx < 0:
#             break
#         obj = getattr(obj, colname[:idx])
#         attname = colname[idx+1:]
#         if colname[:idx] in relations:
#             if rel is not None:
#                 raise ValueError("Cannot get deeper than one relationship in: %s.%s" %
#                                  model.__tablename__, colname)
#             rel = obj
#             obj = relations[colname[:idx]].mapper.class_
# 
#     return rel, getattr(obj, attname)


if __name__ == '__main__':
    # thius should be made available in a test
    from stream2segment.io.db.models import Segment, Event, Station, Channel

    c = Segment.arrival_time
    cond = get_condition(c, "=2016-01-01T00:03:04")
    assert str(cond) == "segments.arrival_time = :arrival_time_1"

    cond = get_condition(c, "!=2016-01-01T00:03:04")
    assert str(cond) == "segments.arrival_time != :arrival_time_1"

    cond = get_condition(c, ">=2016-01-01T00:03:04")
    assert str(cond) == "segments.arrival_time >= :arrival_time_1"
    
    cond = get_condition(c, "<=2016-01-01T00:03:04")
    assert str(cond) == "segments.arrival_time <= :arrival_time_1"
    
    cond = get_condition(c, ">2016-01-01T00:03:04")
    assert str(cond) == "segments.arrival_time > :arrival_time_1"
    
    cond = get_condition(c, "<2016-01-01T00:03:04")
    assert str(cond) == "segments.arrival_time < :arrival_time_1"
    
    with pytest.raises(ValueError):
        cond = get_condition(c, "2016-01-01T00:03:04, 2017-01-01")
    
    cond = get_condition(c, "2016-01-01T00:03:04 2017-01-01")
    assert str(cond) == "segments.arrival_time IN (:arrival_time_1, :arrival_time_2)"
    
    cond = get_condition(c, "[2016-01-01T00:03:04 2017-01-01]")
    assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2"
    
    cond = get_condition(c, "(2016-01-01T00:03:04 2017-01-01]")
    assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3"
    
    cond = get_condition(c, "[2016-01-01T00:03:04 2017-01-01)")
    assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3"
    
    cond = get_condition(c, "(2016-01-01T00:03:04 2017-01-01)")
    assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3 AND segments.arrival_time != :arrival_time_4"
    
    