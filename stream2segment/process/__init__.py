from stream2segment.io.db import close_session
from stream2segment.process.db.models import Segment
from stream2segment.process.db.sqlevalexpr import exprquery
from stream2segment.process.main import process, s2smap, SkipSegment
from stream2segment.process.db import get_session


def get_segments(dburl, conditions, orderby=None):
    """Return a query object (iterable of `Segment`s) from teh given conditions
    Example of conditions (dict):
    ```
    {
        'id' : '<6',
        'has_data': 'true'
    }
    ```
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
        joins  from columns
    :param orderby: a list of string columns (same format
        as `conditions` keys), or a list of tuples where the first element is
        a string column, and the second is either "asc" (ascending) or "desc"
        (descending). In the first case, the order is "asc" by default. See
        note [*] below for auto-added joins from orderby columns
    """
    sess = dburl
    close_sess = False
    try:
        if isinstance(sess, str):
            sess = get_session(dburl)
            close_sess = True
        yield from exprquery(sess.query(Segment), conditions, orderby)
    finally:
        if close_sess:
            close_session(sess, dispose_engine=True)