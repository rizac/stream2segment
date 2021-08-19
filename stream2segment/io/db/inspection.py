"""
ORM inspection tools
"""
from sqlalchemy import inspect
# from sqlalchemy.exc import NoInspectionAvailable
# from sqlalchemy.ext.declarative import DeclarativeMeta
# from sqlalchemy.orm import object_mapper
from sqlalchemy.orm.attributes import QueryableAttribute


def colnames(model_or_instance, pkey=None, fkey=None, nullable=None):
    """Yield the attributes names (as string) describing db table columns, matching all
    the criteria (logical "and") given as argument.

    :param model_or_instance: an ORM model (Python class representing a db table), i.e.
        a :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`, or an instance of
        such a class
    :param pkey: boolean or None. If True, only names of primary key columns are yielded.
        If False, only names of non-primary key columns are yielded.
        If None, the filter is off (yield all)
    :param fkey: If True, only names of foreign key columns are yielded.
        If False, only names of non-foreign key columns are yielded.
        If None, the filter is off (yield all)
    :param nullable: boolean or None. If True, only names of columns where nullable=True
        are yielded. If False, only names of columns where nullable=False are yielded.
        If None, the filter is off (yield all)
    """

    # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    table = get_table(model_or_instance)
    fkeys_cols = set(fk_colnames(table)) if fkey in (True, False) else set()
    pkey_cols = set(pk_colnames(table)) if pkey in (True, False) else set()

    for att_name, column in _columns(get_mapper(model_or_instance)).items():
        # the dict-like above is keyed based on the attribute name defined in
        # the mapping, not necessarily the key attribute of the Column itself
        # (column). See
        # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
        #   #sqlalchemy.orm.mapper.Mapper.columns
        # Note also: mapper.columns.items() returns a list, if performances are
        # a concern, we should iterate over the underlying mapper.columns._data
        # (but is cumbersome)
        if (pkey is None or (att_name in pkey_cols) == pkey) and \
                (fkey is None or (att_name in fkeys_cols) == fkey) and \
                (nullable is None or nullable == column.nullable):
            yield att_name


def _columns(mapper):
    """The dict of columns from a given mapper"""
    return mapper.columns


def get_mapper(model_or_instance):
    """Return the mapper ot the given model or instance

    :param model_or_instance: an ORM model (Python class representing a db table), i.e.
        a :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`, or an instance of
        such a class

    :return: A :class:`sqlalchemy.orm.Mapper` object
    """
    # this is the same as calling `class_mapper` and object_mapper. For info see:
    # https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    mapper = inspect(model_or_instance)
    if not is_model(model_or_instance):
        return mapper.mapper
    return mapper


def is_model(model_or_instance):
    """Perform a shallow check and return whether the given argument is a model, False
    otherwise.

    :param model_or_instance: an ORM model (Python class representing a db table), i.e.
        a :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`, or an instance of
        such a class
    """
    return isinstance(model_or_instance, type)  # sqlalchemy does this in class_object


def get_table(model_or_instance):
    """Return the :class:`sqlalchemy.sql.schema.Table` instance
    associated to the given ORM model or instance

    :param model_or_instance: an ORM model (Python class representing a db table), i.e.
        a :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`, or an instance of
        such a class
    """
    return model_or_instance.__table__


def fk_colnames(table):
    """Yield the column names denoting Foreign keys.

    :param table: a :class:`sqlalchemy.sql.schema.Table` instance. See also `get_table`
    """
    # mapped_table.foreign_keys: set of Foreign Keys (fk). Each fk has a `parent` attr
    # (the foreign key Column on `mapped_table`) and a `column` attr
    # (the referred column on some another table). Return a dict of `parant` name mapped
    # to `column`:
    for f in table.foreign_keys:
        yield f.parent.key
    # return {f.parent.key: f.column for f in mapped_table.foreign_keys}


def pk_colnames(table):
    """Yields the column names denoting Primary keys

    :param table: a :class:`sqlalchemy.sql.schema.Table` instance. See also `get_table`
    """
    yield from table.primary_key.columns.keys()
    # cols = mapped_table.primary_key.columns
    # return {cname: cols[cname] for cname in cols.keys()}


def attnames(model_or_instance, pkey=None, fkey=None, col=None, rel=None, qatt=None):
    """Yield all attribute names of the given ORM model (or instance) that can be
    inspected and match the criteria given as argument. Inspected attributes are
    those returned by the SQLAlchemy :func:`inspect(model_class).all_orm_descriptors`,
    e.g. :class:`QueryableAttribute`s (which includes table columns), relationships,
    hybrid properties, and so on (technically, all yielded attributes are subclasses of
    :class:`sqlalchemy.orm.InspectionAttr`)

    Note that some attributes might match different arguments: an attribute denoting
    a primary key (`pkey=True`) is also an attribute denoting a database table column
    (`col=True`), thus  `pkey=True, col=False` is inconsistent and yields no item.

    Regardless of the filer given as arguments, the yielded model attributes are
    only those inheriting from :class:`.InspectionAttr`, e.g.
    :class:`.QueryableAttribute` (e.g., table columns, hybrid properties with associated
    expression), as well as extension types such as
    :class:`.hybrid_property`, :class:`.hybrid_method` and :class:`.AssociationProxy`.
    Normal Python methods, attributes and properties defined on the class are not
    included (for that, a normal `dir(model_or_instance)` is the way to go)

    :param model_or_instance: an ORM model (Python class representing a db table), i.e.
        a :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`, or an instance of
        such a class
    :param pkey: boolean or None. If True, only names of attributes denoting primary key
        columns are yielded. If False, only names of non-primary key columns are yielded.
        If None, the filter is off (yield all)
    :param fkey: boolean or None. If True, only names of attributes denoting foreign key
        columns are yielded. If False, only names of non-foreign key columns are yielded.
        If None, the filter is off (yield all)
    :param col: boolean or None. If True, only names of attributes denoting table columns
        are yielded (i.e., columns defined on the database). If False, only names not
        associated to table columns are yielded. If None, the filter is off (yield all)
    :param qatt: bool or None. If True, only names of attributes which are
        :class:`sqlalchemy.orm.attributes.QueryableAttribute` are yielded. If False, only
        non queryable attributes are yielded. If None, the filter is off (yield all).
        Queryable attributes are those that can be used for a db query (e.g. SELECT
        statement) such as a table column, or an hybrid property/method with associated
        expression, whereas a non queryable attribute is any other
        :class:`sqlalchemy.orm.InspectionAttr` (e.g., hybrid property with no associated
        expression)
    """

    table = get_table(model_or_instance)
    pkey_cols = set(pk_colnames(table)) if pkey in (True, False) else None
    fkey_cols = set(fk_colnames(table)) if fkey in (True, False) else None

    mapper = get_mapper(model_or_instance)

    all_cols = set(_columns(mapper).keys()) if col in (True, False) else None
    rel_cols = set(_relationships(mapper).keys()) if rel in (True, False) else None

    # get model if we passed an instance:
    model = model_or_instance
    if not is_model(model_or_instance):
        model = model_or_instance.__class__

    for attname in mapper.all_orm_descriptors.keys():
        if pkey_cols is not None and (attname in pkey_cols) != pkey:
            continue

        if fkey_cols is not None and (attname in fkey_cols) != fkey:
            continue

        if all_cols is not None and (attname in all_cols) != col:
            continue

        if rel_cols is not None and (attname in rel_cols) != rel:
            continue

        if qatt is not None:
            try:
                is_qatt = isinstance(getattr(model, attname), QueryableAttribute)
            except Exception:  # noqa
                # this might happen if the attribute is defined at the instance
                # level (not class) and refers to relationships not setup on the
                # instance. Simply skip it, it is not a queryable attribute:
                is_qatt = False
            if is_qatt != qatt:
                continue

        yield attname


def rel_colnames(mapper):
    """Yield the column names denoting a relationship to some other model

    :param mapper: a mapper from a given ORM model or instance. See also `get_mapper`
    """
    yield from mapper.relationships.keys()
    # cols = mapped_table.primary_key.columns
    # return {cname: cols[cname] for cname in cols.keys()}


def get_related_models(model_or_instance):
    """Return a dict of relationship implemented on the model (or instance) as
    a dict[str, model_class], where each key is the relationship name (attribute of
    `model_or_instance`) and `model_class` is the associated Model class
    (:class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`). For info see:
    https://docs.sqlalchemy.org/en/latest/orm/basic_relationships.html
    """
    mapper = get_mapper(model_or_instance)
    relationships = _relationships(mapper)
    return {name: val.mapper.class_ for name, val in relationships.items()}


def _relationships(mapper):
    """The dict of relationships from a given mapper"""
    return mapper.relationships