'''
Created on Mar 27, 2017

@author: riccardo
'''
from stream2segment.io.db.models import Segment
from sqlalchemy.sql.expression import and_
from stream2segment.utils.sqlevalexpr import query
from stream2segment.io.db.pd_sql_utils import withdata


def segment_query(sa_query, conditions, withdataonly=None, distinct=True, orderby=None):
    """Returns sqlevalexpr.query but optimized for models.Segment query

    withdataonly might be in the future moved to a hybrid attribute. For the moment means if we want
    segments wit data or not (set to None to ignore it)
    conditions can have the field classes.id which can be 'none' or 'any' and will be translated
    to a relative '~any() or any() (note: none is not null: the latter will select classes id which
    are null in a db meaning)
    sql alchemy expression
    """
    # first parse separately the classes.id 'none' or 'any'

    additional_atts = []
    if conditions:
        val = conditions.get('classes.id', None)
        if val:
            if val.strip() == 'none':
                conditions.pop('classes.id')
                additional_atts.append(~Segment.classes.any())  # @UndefinedVariable
            elif val.strip() == 'any':
                conditions.pop('classes.id')
                additional_atts.append(Segment.classes.any())  # @UndefinedVariable

        val = conditions.get('classes.id', None)

    if withdataonly in (True, False):
        additional_atts.append(withdata(Segment.data))

    if additional_atts:
        sa_query = sa_query.filter(and_(*additional_atts))
    # we might want to use group_by to remove possibly duplicates at the end of the query,
    # especially if we query classes if which is a many to many relationship (For info see:
    # http://stackoverflow.com/questions/23786401/why-do-multiple-table-joins-produce-duplicate-rows)
    # Problem: If any join
    # will be built inside `query` function (e.g. by conditions or orderby not None), then
    # POSTGRES wants those columns also in the group_by clause, as it cannot guess what to do with
    # dupes (for info see
    # http://stackoverflow.com/questions/18061285/postgresql-must-appear-in-the-group-by-clause-or-be-used-in-an-aggregate-functi)
    # Turns out, we only need to issue a `distinct` at sqlalchemy query level, at the end, and it
    # will add necessary columns for us in the select. This means the resulting query might be
    # MORE than the Segment.id, but who cares as long as we get only the first item (see last line)
    ret = query(sa_query, Segment, conditions, orderby)
    return ret.distinct() if distinct else ret
