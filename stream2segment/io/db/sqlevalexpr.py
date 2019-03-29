'''
Module implementing the functionalities that allow issuing sql select statements from
config files, command line or via GUI input controls
via string expression on database tables columns

:date: Mar 6, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

from datetime import datetime
import shlex

# iterating over dictionary keys with the same set-like behaviour on Py2.7 as on Py3
from future.utils import viewitems

import numpy as np
from sqlalchemy import asc, and_, desc, inspect
from sqlalchemy.orm.attributes import QueryableAttribute
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.orm.collections import InstrumentedList, InstrumentedSet,\
    InstrumentedDict
from sqlalchemy.orm import mapper

# from sqlalchemy.exc import InvalidRequestError


def exprquery(sa_query, conditions, orderby=None, distinct=None):
    """
    Enhance the given sql-alchemy query `sa_query` with conditions
    and ordering given in form of (string) expression, returning a new sql alchemy query.
    Columns (and relationships, if any) are extracted from the string keys of `conditions`
    by detecting the reference model class from `sa_query` first column
    (`sa_query.column_descriptions[0]`): thus **pay attention to the argument order of sa_query**.
    Consqeuently, joins are automatically added inside this method, if needed (if a join is
    already present, in `sa_query` and should be required by any of `conditions`, it won't be
    added twice)
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

    sess = ...  # sql-alchemy session
    sa_query = sess.query  # sql-alchemy session's query object

    # return all parents who have children:
    exprquery(sa_query(Parent), {'children', 'any'})

    # return all parents id's who have children:
    exprquery(sa_query(Parent.id), {'children', 'any'})

    # return all parents who have adult children:
    exprquery(sa_query(Parent), {'children.age', '>=18'})

    # return all parents born before 1980 who have adult children:
    exprquery(sa_query(Parent), {'birth': '<1980-01-01', 'children.age', '>=18'})

    # return all parents who have adult children, sorted (ascending) by parent's age (2 solutions):
    exprquery(sa_query(Parent), {'children.age', '>=18'}, ['age'])  # or
    exprquery(sa_query(Parent), {'children.age', '>=18'}, [('age', 'asc')])

    # return all parents who have adult children, sorted (ascending) by parent's age and then
    # descending by parent's id:
    exprquery(sa_query(Parent), {'children.age', '>=18'}, [('age', 'asc'), ('id', 'desc')])

    # Finally, note that these three are equivalent and valid:
    date1980 = datetime(1980, 1, 1)
    exprquery(sa_query(Parent).filter(Parent.birth < date1980), {'children.age', '>=18'})
    exprquery(sa_query(Parent), {'children.age', '>=18'}).filter(Parent.birth < date1980)
    exprquery(sa_query(Parent), {'birth': '<1980-01-01', 'children.age', '>=18'})
    ```

    :param sa_query: any sql-alchemy query object
    :param conditions: a dict of string columns mapped to **strings** expression, e.g.
    "column2": "[1, 45]" or "column1": "true" (note: string, not the boolean True)
    A string column is an expression denoting an attribute of the reference model class
    and can include relationships.
    Example: if the reference model tablename is 'mymodel', then a string column 'name'
    will refer to 'mymodel.name', 'name.id' denotes on the other hand a relationship 'name'
    on 'mymodel' and will refer to the 'id' attribute of the table mapped by 'mymodel.name'.
    The values of the dict on the other hand are string expressions in the form recognized
    by `binexpr`. E.g. '>=5', '["4", "5"]' ...
    For each condition mapped to a falsy value (e.g., None or empty string), the condition is
    discarded. See note [*] below for auto-added joins from columns.
    :param orderby: a list of string columns (same format
    as `conditions` keys), or a list of tuples where the first element is
    a string column, and the second is either "asc" (ascending) or "desc" (descending). In the
    first case, the order is "asc" by default. See note [*] below for auto-added joins from
    orderby columns.

    :return: a new sel-alchemy query including the given conditions and ordering

    [*] Note on auto-added joins: if any given `condition` or `orderby` key refers to
    relationships defined on the reference model class, then necessary joins are appended to
    `sa_query`, *unless already present* (this should also avoid the
    warning 'SAWarning: Pathed join target', currently in `sqlalchemy.orm.query.py:2105`).
    """
    # get the table model from the query's FIRST column description
    model = sa_query.column_descriptions[0]['entity']
    parsed_conditions = []
    joins = set()  # relationships have an hash, this assures no duplicates
    # set already joined tables. We use the private method _join_entities although it's not
    # documented anywhere (we inspected via eclipse debug to find the method):
    already_joined_models = set(_.class_ for _ in sa_query._join_entities)
    # if its'an InstrumentedAttribute, use the class
    relations = inspect(model).relationships

    if conditions:
        for attname, expression in viewitems(conditions):
            if not expression:  # discard falsy expressions (empty strings, None's)
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
            obj = get_rel_refmodel(relations[col])
        else:
            obj = tmp
    return rel, obj


def get_rel_refmodel(relationship):
    '''returns the relationship's reference table model
    :param relationship: the InstrumentedAttribute retlative to a relationship. Example. Given
        a model `model`, and a relationship e.g. `r_name=inspect(model).relationships.keys()[i],
        then `relationship=getattr(model, r_name)`'''
    return relationship.mapper.class_


def binexpr(column, expr):
    """Returns an :class:`sqlalchemy.sql.expression.BinaryExpression` to be used as `query.filter`
    argument from the given column and the given expression. Supports the operators given in
    :function:`stream2segment.io.db.sqlevalexpr.split` and the types given in `parsevals`:
    (`int`s, `float`s, `datetime`s, `bool`s and `str`s)
    :param column: an sqlkalchemy model column
    :param expr: a string expression (see `split`)

    :example:
    ```
    # given a model with column `column1`
    binexpr(model.column1, '>=5')
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
        return parsevals(get_pytype(get_sqltype(column)), expr_value)
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


def get_sqltype(obj):
    '''Returns the sql type associated with `obj`.

    :param obj: an object with an 'expression' method, e.g. sqlalchemy.sql.schema.Column or
        :class:`sqlalchemy.orm.attributes.QueryableAttribute` (for instance
         :class:`sqlalchemy.orm.attributes.InstrumentedAttribute`, i.e. the model's attributes
        mapping db columns)

    :return: An object defined in :class:`sqlalchemy.sql.sqltypes`, e.g. `Integer`
        the method  :function:`get_pytype` of the returned object defines the relative python type
    '''
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


class Inspector(object):
    '''Class for inspecting a ORM model (Python class) or an instance (Python object)
    reflecting a database row'''

    PKEY = 1
    FKEY = 2
    COL = 4
    QATT = 8
    REL = 16

    def __init__(self, model_or_instance):
        '''Initializes the current object with a model or instance argument'''
        self.mapper = self._model = None
        self.pknames, self.fknames, self.colnames, self.relnames, self.qanames = \
            set(), set(), set(), set(), set()
        self.attrs = {}

        if not model_or_instance:
            return

        try:
            mapper = inspect(model_or_instance)
        except NoInspectionAvailable:
            return
    
        if mapper is model_or_instance:  # this happens when the object has anything to inspect, eg.
            # an AppenderQuery object resulting from some relationship configured in some particular way
            return
    
        # if model_or_instance is an instance, reload Mapper on the model:
        _real_model = model_or_instance
        self.instance = None
        if mapper.mapper.class_ == model_or_instance.__class__:
            self.instance = model_or_instance
            _real_model = model_or_instance.__class__
            mapper = inspect(_real_model)

        self.model = _real_model
        self.mapper = mapper
        # To be clear, when writing
        # class Table:
        #     id = Column(...)
        # 
        # you might expect that Table.id is a Column object. It is indeeed an
        # InstrumentedAttribute (subclass of QueryableAttribute). To get the column
        # we should do ` mapper(Table).columns` which returns a dict-like object of names
        # mapped to Column object.
        # Also, a ORM model might have relationships or simple hybrid properties
        # All these things seem to be instance of QueryableAttribute. So:
        qanames = self.qanames = set(_ for _ in dir(_real_model) if _[:2] != '__' and
                                     isinstance(getattr(_real_model, _), QueryableAttribute))
        self.attrs = {_: getattr(_real_model, _) for _ in qanames}
        # we will remove keys from qanames leaving only queryable attributes not
        # belonging to any other class (that's why we associated it to self.qanames)

        # now, get Foreign Keys Columns
        fk_columns = set(f.parent for f in mapper.mapped_table.foreign_keys)
        # and the pkeys Columns:
        pk_columns = set(mapper.mapped_table.primary_key.columns)

        pknames, fknames, colnames = self.pknames, self.fknames, self.colnames
        try:
            for pkeycol in pk_columns:
                qanames.remove(pkeycol.key)
                pknames.add(pkeycol.key)
            for fkeycol in fk_columns:
                qanames.remove(fkeycol.key)
                fknames.add(fkeycol.key)
            for col in mapper.columns:
                if col not in fk_columns and col not in pk_columns:
                    qanames.remove(col.key)
                    colnames.add(col.key)
        except KeyError as kerr:
            raise ValueError('Attribute to "%s" not defined in the ORM class'
                             % str(kerr))

        relnames = self.relnames
        for rel in self.mapper.relationships:
            qanames.remove(rel.key)
            relnames.add(rel.key)

    def _exclude_set(self, exclude):
        if not isinstance(exclude, set):
            exclude = set(exclude or [])
        if exclude:
            for key, val in self.attrs.items():
                if val in exclude:
                    exclude.remove(val)
                    exclude.add(key)
        return exclude

    def attnames(self, flags=None, sort=True, deep=False, exclude=None):
        '''Yields a set of strings identifying the attributes of the model
        or instance passed in the constructor

        :param flags: one or more of :class:`Inspector.PKEY`, :class:`Inspector.FKEY`,
            :class:`Inspector.QATT`, :class:`Inspector.REL`: concatenate those values
            with the "|" operator for more options, where:

            - :class:`Inspector.PKEY`: return the attribute names denoting SQL primary
                keys
            - :class:`Inspector.FKEY`: return the attribute names denoting SQL foreign
                keys
            - :class:`Inspector.COL`: return the attribute names denoting any SQL column
                on the database (excluding primary keys and foreign keys).
            - :class:`Inspector.QATT`: return the attribute names denoting any queryable
                attribute. i.e.,custom queryable attributes NOT mapped to any database column
                (e.g. hybrid properties)
            - :class:`Inspector.REL`: return the attribute names denoting any relationship
                defined at the Python level on the given model
        :param sort: boolean (default True), yield the strings sorted. Sorting is done
            within each category defined in `flags` (thus first
            primary keys sorted alphabetically, then foreign keys sorted alphabetically, and
            so on)
        :param deep: boolean (default False): whether to return the attributes of all mapped
            relationships. Ignored if :class:`Inspector.REL` is not in `flags`. If True, the
            for each attribute defining a relationship will not be yielded, but all the
            relation's model attributes instead
            (using the same `flags` passed here, except that further relationships will not be
            expanded further). The attributes will be yielded with the relationship model name
            plus a 'dot' pluts the model attribute name
            to avoid potential duplicates. E.g.: 'parent.id', 'parent.name', and so on
        :param exclude: a list of strings or model attributes to be excluded, i.e. not
            yielded. If `deep=True`, the list can include related models attributes
            (with simple strings there is no way to identify the nested attribute to be
            excluded)
        '''
        ret = []
        exclude = self._exclude_set(exclude)
        if flags is None:
            flags = self.PKEY | self.FKEY | self.REL | self.QATT | self.COL

        if flags & self.PKEY:
            ret.extend(self._matchingnames(self.pknames, sort, exclude))

        if flags & self.FKEY:
            ret.extend(self._matchingnames(self.fknames, sort, exclude))

        if flags & self.COL:
            ret.extend(self._matchingnames(self.colnames, sort, exclude))

        if flags & self.QATT:
            ret.extend(self._matchingnames(self.qanames, sort, exclude))

        if flags & self.REL:
            for _ in sorted(self.relnames) if sort else self.relnames:
                if _ not in exclude:
                    if deep:
                        inspector = Inspector(self.relatedmodel(_))
                        for __ in inspector.attnames(flags - self.REL, sort, False, exclude):
                            ret.append('%s.%s' % (_, __))
                    else:
                        ret.append(_)

        return ret

    @staticmethod
    def _matchingnames(names, sort, exclude):
        for name in sorted(names) if sort else names:
            if name not in exclude:
                yield name

    def relatedmodel(self, relname):
        '''Returns the model class related to the given relation name, as returned
        from `self.attnames`'''
        return get_rel_refmodel(self.mapper.relationships[relname])

    def atttype(self, attname):
        '''Returns the Python type corresponding to the SQL type of the
        given attribut name, as returned from `self.attnames`. Returns None
        if no type can be found'''
        if attname in self.relnames:
            return object
        model = self.model
        relnames, aname = self._splitatt(attname)
        for relname in relnames:
            model = Inspector(model).relatedmodel(relname)
        try:
            return get_pytype(get_sqltype(getattr(model, aname)))
        except Exception as _:
            return None

    def attval(self, attname):
        '''Returns the value corresponding to the given attribut name,
        as returned from `self.attnames`. The value is an `InstrumentedAttribute`
        if this object was initialized with a model class, or the value
        of the given attribute (int, float etcetera) if this object was initialized
        with an instance object'''
        relnames, aname = self._splitatt(attname)
        if self.instance is not None:
            obj = self.instance
            for relname in relnames:
                obj = getattr(obj, relname)
        else:
            obj = self.model
            for relname in relnames:
                obj = self.relatedmodel(relname)
        try:
            return getattr(obj, aname)
        except AttributeError:
            if isinstance(obj, (InstrumentedList, InstrumentedSet)):
                ret = [getattr(subobj, aname) for subobj in obj]
                return ret if isinstance(obj, InstrumentedList) else set(ret)
            if isinstance(obj, InstrumentedDict):
                return obj[aname]
            raise

    def _splitatt(self, attname):
        atts = attname.split('.')
        return (atts[:-1], atts[-1]) if len(atts) > 1 else ([], attname)
