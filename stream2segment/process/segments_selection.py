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

from stream2segment.io.db import s2s_db_version
from stream2segment.io.db import models
from stream2segment.io.db.legacy import models as legacy_models  # v <= 4


@dataclass(frozen = True, slots = True, kw_only=True)
class SelectableAttributes:
    event_time: ColumnElement  # sqlalchemy.sql.elements.SQLColumnExpression
    event_latitude: ColumnElement
    event_longitude: ColumnElement
    event_depth_km: ColumnElement
    event_id: ColumnElement
    # author = ColumnElement
    # catalog = ColumnElement
    # contributor = ColumnElement
    # contributor_id = ColumnElement
    event_mag_type: ColumnElement
    event_magnitude_type: ColumnElement
    event_magnitude: ColumnElement
    channel_id: ColumnElement
    network_code: ColumnElement
    station_code: ColumnElement
    latitude: ColumnElement
    longitude: ColumnElement
    elevation: ColumnElement
    # start_time = ColumnElement
    location_code: ColumnElement
    band_code: ColumnElement
    instrument_code: ColumnElement
    orientation_code: ColumnElement
    channel_code: ColumnElement
    depth: ColumnElement
    azimuth: ColumnElement
    dip: ColumnElement
    event_distance_km: ColumnElement
    noise_window_s: ColumnElement
    signal_window_s: ColumnElement
    gap_score_percent: ColumnElement
    # sample_rate = Column(Float, nullable=False)


def get_selectable_attributes(db: Engine) -> SelectableAttributes:
    """"""
    if s2s_db_version(db) < 5:
        Segment = legacy_models.Segment
        Channel = models.Channel
        Event = legacy_models.Event
        Station = legacy_models.Station

        return SelectableAttributes(**{
            'id': Segment.id,
            "event_id": Segment.event_id.label('event_id'),
            "event_time": Event.time.label('event_time'),
            "event_latitude": Event.latitude.label('event_latitude'),
            "event_longitude": Event.longitude.label('event_longitude'),
            "event_depth_km": Event.depth_km.label('event_depth_km'),
            "event_mag_type": Event.mag_type.label('event_magnitude_type'),
            # "event_magnitude": Event.magnitude.label('event_magnitude'),
            "event_webservice_id": Event.webservice_id.label('event_webservice_id'),
            "channel_id": Segment.channel_id,
            'data_webservice_id': Station.datacenter_id.label('data_webservice_id'),
            "network_code": Station.network.label('network_code'),
            "station_code": Station.station.label('station_code'),
            "latitude": Station.latitude,
            "longitude": Station.longitude,
            "elevation": Station.elevation,
            "location_code": Channel.location.label('location_code'),
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
        })
    else:
        Segment = models.Segment
        Channel = models.Channel
        Event = models.Event

        return SelectableAttributes(**{
            'id': Segment.id,
            "event_id": Segment.event_id.label('event_id'),
            "event_time": Event.time.label('event_time'),
            "event_latitude": Event.latitude.label('event_latitude'),
            "event_longitude": Event.longitude.label('event_longitude'),
            "event_depth_km": Event.depth_km.label('event_depth_km'),
            "event_mag_type": Event.mag_type.label('event_magnitude_type'),
            # "event_magnitude_type": Event.mag_type.label('event_magnitude_type'),
            "event_magnitude": Event.magnitude.label('event_magnitude'),
            "event_webservice_id": Event.webservice_id.label('event_webservice_id'),
            "channel_id": Segment.channel_id,
            'data_webservice_id': Channel.data_webservice_id,
            "network_code": Channel.network_code,
            "station_code": Channel.station_code,
            "latitude": Channel.latitude,
            "longitude": Channel.longitude,
            "elevation": Channel.elevation,
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
        })


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
        stmt = _build_select_legacy(db, segments_selection)
    else:
        stmt = _build_select_new(db, segments_selection, group_components)

    return stmt


def _build_select_new(
    db: Engine, segments_selection: dict | None = None, group_components:bool = False
) -> Select:
    """
    build select statement with provided where_conditions
    """
    Segment = models.Segment
    Channel = models.Channel
    MiniSeed = models.MiniSeed
    Event = models.Event

    sel_attrs = get_selectable_attributes(db)
    select_cols = [
        getattr(sel_attrs, f.name) for f in fields(sel_attrs)
    ]

    if group_components:
        select_cols += [
            Channel.location_code,
            Channel.band_code,
            Channel.instrument_code
        ]

    stmt = (
        select(*select_cols)
        .select_from(Segment)  # probably unnecessary, set main table for clarity
        .join(Channel, Channel.id == Segment.channel_id)
        .join(Event, Event.id == Segment.event_id)
        .join(MiniSeed, MiniSeed.id == Segment.id)
    )

    if segments_selection:
        stmt = stmt.where(build_where_clause(db, segments_selection))

    return stmt


def _build_select_legacy(db: Engine, segments_selection) -> Select:
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
    Segment = legacy_models.Segment
    Channel = legacy_models.Channel
    Event = legacy_models.Event
    StationXML = legacy_models.StationXML

    sel_attrs = get_selectable_attributes(db)
    select_cols = [
        getattr(sel_attrs, f.name) for f in fields(sel_attrs)
    ]

    stmt = (
        select(*select_cols)
        .select_from(Segment)
        .join(Channel, Channel.id == Segment.channel_id)
        .join(StationXML, StationXML.id == Channel.station_id)
        .join(Event, Event.id == Segment.event_id)
        .where(Segment.has_valid_data)
    )

    if segments_selection:
        stmt = stmt.where(build_where_clause(db, segments_selection))

    return stmt


def get_segments_count(db: Engine, segments_selection: dict) -> int:
    stmt = build_select(db, segments_selection)
    stmt_count = select(func.count()).select_from(stmt.subquery())
    with db.connect() as conn:
        return conn.execute(stmt_count).scalar_one()


def build_where_clause(db: Engine, conditions: dict) -> ColumnElement[bool]:
    """"""
    base_attrs = get_selectable_attributes(db)
    f_names = {f.name for f in fields(SelectableAttributes)}
    parsed_conditions = None
    for attname, expression in conditions.items():
        if not expression:  # discard empty strings, None's, ...
            # note that expressions MUST be strings
            continue
        if attname not in f_names:
            raise AttributeError(f'Column "{attname}" not found or not implemented')

        condition = binexpr(getattr(base_attrs, attname), expression)
        if parsed_conditions is None:
            parsed_conditions = condition
        else:
            parsed_conditions &= condition

    return parsed_conditions


def get_orderby_columns(db, group_components):
    if s2s_db_version(db) > 4:
        Channel = models.Channel
        Segment = models.Segment

        if group_components:
            return (
                Channel.data_webservice_id,
                Channel.network_code,
                Channel.station_code,
                Channel.location_code,
                Channel.instrument_code,
                Channel.band_code,
                Segment.event_id,
                Segment.id
            )
        else:
            return (
                Channel.network_code,
                Channel.station_code,
                Segment.event_id,
                Segment.id
            )

    return (
        legacy_models.Station.network,
        legacy_models.Station.station,
        legacy_models.Segment.event_id,
        legacy_models.Segment.id
    )
    return orderby_columns


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
