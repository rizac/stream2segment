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
    mapper = get_mapper(model_or_instance)
    # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    #   #sqlalchemy.orm.mapper.Mapper.mapped_table
    table = _get_mapped_table(mapper)
    fkeys_cols = set(get_fk_columns(table)) if fkey in (True, False) else set([])
    pkey_cols = set(get_pk_columns(table)) if pkey in (True, False) else set([])
    for att_name, column in mapper.columns.items():
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


def _get_mapped_table(mapper):
    """Return the mapped table from the given SQLAlchemy mapper

    :param mapper: A :class:`sqlalchemy.orm.Mapper` object, see e.g. `get_mapper`
    """
    # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html
    #   #sqlalchemy.orm.mapper.Mapper.mapped_table
    # Note that from v 1.3+, we need to use .persist_selectable:
    try:
        table = mapper.persist_selectable
    except AttributeError:
        table = mapper.mapped_table
    return table


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
    """Return the table instance from tjhe given model or instance

    :param model_or_instance: an ORM model (Python class representing a db table), i.e.
        a :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`, or an instance of
        such a class
    """
    return model_or_instance.__table__


def get_fk_columns(mapped_table):
    """Yields the column objects referred to in the Foreign keys of `mapped_table`
    """
    # mapped_table.foreign_keys: set of Foreign Keys (fk). Each fk has a `parent` attr
    # (the foreign key Column on `mapped_table`) and a `column` attr
    # (the referred column on some another table). Return a dict of `parant` name mapped
    # to `column`:
    return {f.parent.key: f.column for f in mapped_table.foreign_keys}
    # for f in mapped_table.foreign_keys:
    #     yield f.parent


def get_pk_columns(mapped_table):
    """Return a dict of column objects denoting the primary key(s) of `mapped_table`
    """
    cols = mapped_table.primary_key.columns
    return {cname: cols[cname] for cname in cols.keys()}


def attnames(model_or_instance, pkey=None, fkey=None, col=None, qatt=None, rel=None):
    """Yield all attribute names of the given ORM model or instance, matching all the
    criteria (logical and) given as argument

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
        are yielded. If False, only names not associated to table columns are yielded.
        If None, the filter is off (yield all)
    :param qatt: bool or None. If True, only names of attributes which are
        :class:`sqlalchemy.orm.attributes.QueryableAttribute` are yielded. If False, only
        non queryable attributes are yielded. If None, the filter is off (yield all).
        Queryable attributes are those denoting e.g. a table column, or an hybrid
        property, or everything that can be used for a db query (e.g. SELECT statement),
        whereas a non queryable attribute is any other normal Python attribute (e.g.
        normal property, or method) defined on the class
    :param rel: bool or None.If True, only names of attributes denoting a SQLAlchemy
        relationship are yielded. If False, only non-relationship attributes are yielded.
        If None, the filter is off (yield all). For info see:
        https://docs.sqlalchemy.org/en/latest/orm/basic_relationships.html
    """

    if not model_or_instance:
        return {}

    # build filter sets:
    pkeys = None if pkey is None else set(colnames(model_or_instance, pkey=True))
    fkeys = None if fkey is None else set(colnames(model_or_instance, fkey=True))
    cols = None if col is None else set(colnames(model_or_instance))
    rels = None if rel is None else set(get_related_models(model_or_instance))

    # get model if we passed an instance:
    model = model_or_instance
    if not is_model(model_or_instance):
        model = model_or_instance.__class__

    # yield matching queryable attributes:
    for attname in dir(model_or_instance):
        if attname[:2] == '__':
            continue
        # if qatt is not None and \
        #         isinstance(getattr(model, attname), QueryableAttribute) != qatt:
        #     continue
        if qatt is not None:
            try:
                if isinstance(getattr(model, attname), QueryableAttribute) != qatt:
                    continue
            except Exception:  # noqa
                # this might happen if the attribute is defined at the instance
                # level (not class) and refers to relationships not setup on the
                # instance. Simply skip it, it is not a queryable attribute:
                continue
        if pkey is not None and (attname in pkeys) != pkey:
            continue
        if fkey is not None and (attname in fkeys) != fkey:
            continue
        if col is not None and (attname in cols) != col:
            continue
        if rel is not None and (attname in rels) != rel:
            continue
        yield attname


def get_related_models(model_or_instance):
    """Return a dict of relationship implemented on the model (or instance) as
    a dict[str, model_class], where each key is the relationship name (attribute of
    `model_or_instance`) and `model_class` is the associated Model class
    (:class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`). For info see:
    https://docs.sqlalchemy.org/en/latest/orm/basic_relationships.html
    """
    mapper = get_mapper(model_or_instance)
    relationships = mapper.relationships
    return {_: relationships[_].mapper.class_ for _ in relationships.keys()}
