"""
Module implementing the functionalities that allow issuing sql select
statements from config files, command line or via GUI input controls
via string expression on database tables columns
"""
# Mar 6, 2017
from dataclasses import dataclass, fields
from datetime import datetime
import shlex
import warnings

import numpy as np
from sqlalchemy import and_, ColumnElement, select, func, tuple_, Engine, Select
from sqlalchemy.ext.hybrid import hybrid_property

from stream2segment.io.db import s2s_db_version
from stream2segment.io.db import models
from stream2segment.io.db.legacy import models as legacy_models  # v <= 4


def is_legacy_db(db: Engine):
    return s2s_db_version(db) < 5


@dataclass(frozen = True, slots = True, kw_only=True)
class CommonFields:
    id: ColumnElement
    webservice_id: ColumnElement
    # Event-related stuff:
    event_id: ColumnElement
    event_time: ColumnElement  # sqlalchemy.sql.elements.SQLColumnExpression
    event_latitude: ColumnElement
    event_longitude: ColumnElement
    event_depth_km: ColumnElement
    event_magnitude_type: ColumnElement
    event_magnitude: ColumnElement
    event_webservice_id: ColumnElement
    # channel-related stuff:
    station_id: ColumnElement
    channel_id: ColumnElement
    network_code: ColumnElement
    station_code: ColumnElement
    location_code: ColumnElement
    band_code: ColumnElement | hybrid_property
    instrument_code: ColumnElement | hybrid_property
    orientation_code: ColumnElement | hybrid_property
    latitude: ColumnElement
    longitude: ColumnElement
    elevation: ColumnElement
    depth: ColumnElement
    # azimuth: ColumnElement
    # dip: ColumnElement
    noise_window_s: ColumnElement | hybrid_property
    signal_window_s: ColumnElement | hybrid_property


@dataclass(frozen = True, slots = True, kw_only=True)
class SelectFields(CommonFields):
    data: ColumnElement

def get_select_fields(db: Engine) -> SelectFields:
    """"""
    if is_legacy_db(db):
        lm = legacy_models

        return SelectFields(
            id=lm.Segment.id,
            event_id=lm.Segment.event_id,
            event_time=lm.Event.time,
            event_latitude=lm.Event.latitude,
            event_longitude=lm.Event.longitude,
            event_depth_km=lm.Event.depth_km,
            event_magnitude=lm.Event.magnitude,
            event_magnitude_type=lm.Event.mag_type,
            event_webservice_id=lm.Event.webservice_id,
            station_id=lm.Channel.station_id,
            channel_id=lm.Segment.channel_id,
            webservice_id=lm.Station.datacenter_id,
            network_code=lm.Station.network,
            station_code=lm.Station.station,
            location_code=lm.Channel.location,
            band_code=lm.Channel.band_code,
            instrument_code=lm.Channel.instrument_code,
            orientation_code=lm.Channel.orientation_code,
            latitude=lm.Station.latitude,
            longitude=lm.Station.longitude,
            elevation=lm.Station.elevation,
            depth=lm.Channel.depth,
            # azimuth=lm.Channel.azimuth,
            # dip=lm.Channel.dip,
            data=lm.Segment.data,
            noise_window_s=lm.Segment.noise_window_s,
            signal_window_s = lm.Segment.signal_window_s
        )

    else:
        m = models

        return SelectFields(
            id=m.Segment.id,
            event_id=m.Segment.event_id,
            event_time=m.Event.time,
            event_latitude=m.Event.latitude,
            event_longitude=m.Event.longitude,
            event_depth_km=m.Event.depth_km,
            event_magnitude_type=m.Event.mag_type,
            event_magnitude=m.Event.magnitude,
            event_webservice_id=m.Event.webservice_id,
            station_id=m.Channel.stationxml_id,
            channel_id=m.Segment.channel_id,
            webservice_id=m.Channel.data_webservice_id,
            network_code=m.Channel.network_code,
            station_code=m.Channel.station_code,
            location_code=m.Channel.location_code,
            band_code=m.Channel.band_code,
            instrument_code=m.Channel.instrument_code,
            orientation_code=m.Channel.orientation_code,
            latitude=m.Channel.latitude,
            longitude=m.Channel.longitude,
            elevation=m.Channel.elevation,
            depth=m.Channel.depth,
            # azimuth=m.Channel.azimuth,
            # dip=m.Channel.dip,
            data=m.MiniSeed.data,
            noise_window_s=m.Segment.noise_window_s,
            signal_window_s=m.Segment.signal_window_s
        )


@dataclass(frozen = True, slots = True, kw_only=True)
class WhereFields(CommonFields):
    event_distance_km: ColumnElement | hybrid_property
    event_distance_deg: ColumnElement | hybrid_property
    noise_window_s: ColumnElement | hybrid_property
    signal_window_s: ColumnElement | hybrid_property
    gap_score_percent: ColumnElement | hybrid_property
    orientation_code: ColumnElement | hybrid_property
    channel_code: ColumnElement | hybrid_property


def get_where_fields(db: Engine) -> WhereFields:
    """"""
    if is_legacy_db(db):
        lm = legacy_models

        return WhereFields(
            id=lm.Segment.id,
            station_id=lm.Channel.station_id,
            event_id=lm.Segment.event_id,
            event_time=lm.Event.time,
            event_latitude=lm.Event.latitude,
            event_longitude=lm.Event.longitude,
            event_depth_km=lm.Event.depth_km,
            event_magnitude=lm.Event.magnitude,
            event_magnitude_type=lm.Event.mag_type,
            event_webservice_id=lm.Event.webservice_id,
            channel_id=lm.Segment.channel_id,
            webservice_id=lm.Station.datacenter_id,
            network_code=lm.Station.network,
            station_code=lm.Station.station,
            location_code=lm.Channel.location,
            band_code=lm.Channel.band_code,
            instrument_code=lm.Channel.instrument_code,
            latitude=lm.Station.latitude,
            longitude=lm.Station.longitude,
            elevation=lm.Station.elevation,
            orientation_code=lm.Channel.orientation_code,
            channel_code=lm.Channel.channel,
            depth=lm.Channel.depth,
            # azimuth=lm.Channel.azimuth,
            # dip=lm.Channel.dip,
            event_distance_deg=lm.Segment.event_distance_deg,
            event_distance_km=lm.Segment.event_distance_km,
            noise_window_s=lm.Segment.noise_window_s,
            signal_window_s=lm.Segment.signal_window_s,
            gap_score_percent=lm.Segment.gap_score_percent,
        )

    else:
        m = models

        return WhereFields(
            id=m.Segment.id,
            station_id=m.Channel.stationxml_id,
            event_id=m.Segment.event_id,
            event_time=m.Event.time,
            event_latitude=m.Event.latitude,
            event_longitude=m.Event.longitude,
            event_depth_km=m.Event.depth_km,
            event_magnitude_type=m.Event.mag_type,
            event_magnitude=m.Event.magnitude,
            event_webservice_id=m.Event.webservice_id,
            channel_id=m.Segment.channel_id,
            webservice_id=m.Channel.data_webservice_id,
            network_code=m.Channel.network_code,
            station_code=m.Channel.station_code,
            location_code=m.Channel.location_code,
            band_code=m.Channel.band_code,
            instrument_code=m.Channel.instrument_code,
            latitude=m.Channel.latitude,
            longitude=m.Channel.longitude,
            elevation=m.Channel.elevation,
            orientation_code=m.Channel.orientation_code,
            channel_code=m.Channel.channel_code,
            depth=m.Channel.depth,
            # azimuth=m.Channel.azimuth,
            # dip=m.Channel.dip,
            event_distance_deg=m.Segment.event_distance_deg,
            event_distance_km=m.Segment.event_distance_km,
            noise_window_s=m.Segment.noise_window_s,
            signal_window_s=m.Segment.signal_window_s,
            gap_score_percent=m.Segment.gap_score_percent,
        )


def build_select(
    db: Engine,
    segments_selection: dict | None = None,
    group_components: bool = False,
) -> Select:
    """
    Build the segments-selection SELECT statement, transparently supporting
    both the current and the legacy (v<=4) database schema.

    :param db: an Engine. See sqlalchemy `create_engine(db_url)`
    :param segments_selection: optional dict of selection filters, forwarded
        to `build_where_clause` exactly as before.
    :param group_components: if True, also selects per-component channel
        codes (location/band/instrument). NOT supported on legacy databases
        (this option did not exist in any legacy s2s version) - raises
        NotImplementedError if `db` is a legacy database.
    """
    legacy = s2s_db_version(db) < 5

    if legacy:
        if group_components:
            raise NotImplementedError(
                "`group_components` is not supported on "
                "legacy (v<=4) databases: Either omit `group_components` and handle it "
                "in your code, or re-download the data with this program"
            )

    sel_attrs = get_select_fields(db)
    select_cols = [
        getattr(sel_attrs, f.name).label(f.name) for f in fields(sel_attrs)
    ]
    if legacy:
        stmt = _build_select_legacy(select_cols)
    else:
        stmt = _build_select_new(select_cols)

    if segments_selection:
        stmt = stmt.where(build_where_clause(db, segments_selection))

    return stmt


def _build_select_new(select_cols) -> Select:
    """
    build select statement with provided where_conditions
    """
    return (
        select(*select_cols)
        .select_from(models.Segment)  # probably unnecessary, set main table for clarity
        .join(models.Channel, models.Channel.id == models.Segment.channel_id)
        .join(models.Event, models.Event.id == models.Segment.event_id)
        .join(models.MiniSeed, models.MiniSeed.id == models.Segment.id)
    )


def _build_select_legacy(select_cols) -> Select:
    """
    Legacy-schema equivalent of `_build_select_new`, producing the SAME
    labeled output columns. Mapping notes:

    - MiniSeed.data (new, separate table, INNER JOIN -> only segments with
      a data row are returned) maps to legacy Segment.data (nullable, same
      table). To preserve the same RESULT SEMANTICS (not just same column),
      this query filters on `Segment.has_valid_data` instead of returning
      raw possibly-null/empty data - this is the closest legacy equivalent
      of "a MiniSeed row exists for this segment".
    - Channel.data_webservice_id (new) maps to legacy
      StationXML.datacenter_id (reached via Channel.station_id). See the
      WebService/DataCenter docstring in legacy_models.py: this is a
      SEPARATE id space from Event.webservice_id below, do not assume they
      are comparable.
    - Channel.stationxml_id (new) maps to legacy Channel.station_id - see
      the StationXML docstring in legacy_models.py.
    - Channel.network_code/station_code/latitude/longitude/elevation (new,
      live on Channel) map to legacy StationXML.network_code/station_code/
      latitude/longitude/elevation (live on the Station/StationXML row,
      reached via Channel.station_id).
    - Event.webservice_id (new) maps to legacy Event.webservice_id
      directly - but again, see the id-space-overlap caveat above: this is
      NOT guaranteed comparable to Channel.data_webservice_id / DataCenter.id.
    """
    l_models = legacy_models

    return (
        select(*select_cols)
        .select_from(l_models.Segment)
        .join(l_models.Channel, l_models.Channel.id == l_models.Segment.channel_id)
        .join(l_models.Station, l_models.Station.id == l_models.Channel.station_id)
        .join(l_models.Event, l_models.Event.id == l_models.Segment.event_id)
        .where(l_models.Segment.has_valid_data)
    )


def get_segments_count(db: Engine, segments_selection: dict) -> int:
    stmt = build_select(db, segments_selection)
    stmt_count = select(func.count()).select_from(stmt.subquery())
    with db.connect() as conn:
        return conn.execute(stmt_count).scalar_one()


def build_where_clause(db: Engine, conditions: dict) -> ColumnElement[bool]:
    """"""
    where_fields = get_where_fields(db)
    f_names = {f.name for f in fields(WhereFields)}
    parsed_conditions = None
    for f_name, expression in conditions.items():
        if not expression:  # discard empty strings, None's, ...
            # note that expressions MUST be strings
            continue
        if f_name not in f_names:
            raise AttributeError(f'Column "{f_name}" not found or not implemented')

        condition = binexpr(getattr(where_fields, f_name), expression)
        if parsed_conditions is None:
            parsed_conditions = condition
        else:
            parsed_conditions &= condition

    return parsed_conditions


def get_orderby_columns(
    db, group_components
) -> dict[str, ColumnElement | hybrid_property]:

    seg_attrs = get_where_fields(db)

    if group_components:
        return {
            c: getattr(seg_attrs, c) for c in [
                'webservice_id',
                'network_code',
                'station_code',
                'location_code',
                'instrument_code',
                'band_code',
                'event_id',
                'id'
            ]
        }
    elif is_legacy_db(db):
        return {
            c: getattr(seg_attrs, c) for c in [
                'station_id',
                'id'
            ]
        }
    else:
        return {
            c: getattr(seg_attrs, c) for c in [
                'webservice_id',
                'network_code',
                'station_code',
                'id'
            ]
        }


def binexpr(column, expr):
    """Return an :class:`sqlalchemy.sql.expression.BinaryExpression` to be
    used as `query.filter` argument from the given column and the given
    expression. Supports the operators given in
    :func:`stream2segment.io.db.sqlevalexpr.split` and the types given in
    `parsevals`: (`int`s, `float`s, `datetime`s, `bool`s and `str`s)

    :param column: a sqlalchemy model column
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
        raise ValueError(f"Invalid expression for column '{column}': {expr}")


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
