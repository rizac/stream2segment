"""
Module implementing the functionalities that allow issuing sql select
statements from config files, command line or via GUI input controls
via string expression on database tables columns
"""
# Mar 6, 2017
from datetime import datetime
import shlex
import warnings

import numpy as np
from sqlalchemy import and_, ColumnElement

from stream2segment.io.db.models import Event, Channel, Segment

base_attrs = {
    "event_time": Event.time,
    "event_latitude": Event.latitude,
    "event_longitude": Event.longitude,
    "event_depth_km": Event.depth_km,
    # author = Column(String)
    # catalog = Column(String, nullable=False)
    # contributor = Column(String)
    # contributor_id = Column(String)
    "event_mag_type": Event.mag_type,
    "event_magnitude_type": Event.mag_type,
    "event_magnitude": Event.magnitude,
    "network_code": Channel.network_code,
    "station_code": Channel.station_code,
    "latitude": Channel.latitude,
    "longitude": Channel.longitude,
    "elevation": Channel.elevation,
    # start_time = Channel.start_time,
    "location_code": Channel.location_code,
    "band_code": Channel.band_code,
    "instrument_code": Channel.instrument_code,
    "orientation_code": Channel.orientation_code,
    "channel_code": Channel.channel_code,
    "depth": Channel.depth,
    "azimuth": Channel.azimuth,
    "dip": Channel.dip,
    "event_distance_km": Segment.event_distance_km,
    "noise_window_s": Segment.noise_window_s,
    "signal_window_s": Segment.signal_window_s,
    "gap_score_percent": Segment.gap_score_percent
    # sample_rate = Column(Float, nullable=False)
}

def build_where_clause(conditions: dict) -> ColumnElement[bool]:
    """"""
    parsed_conditions = None
    for attname, expression in conditions.items():
        if not expression:  # discard empty strings, None's, ...
            # note that expressions MUST be strings
            continue
        if attname not in base_attrs:
            raise AttributeError('Column "{attname}" not found in DB')

        condition = binexpr(base_attrs[attname], expression)
        if parsed_conditions is None:
            parsed_conditions = condition
        else:
            parsed_conditions &= condition

    return parsed_conditions


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
