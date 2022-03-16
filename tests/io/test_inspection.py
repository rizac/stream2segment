"""
Created on Jul 15, 2016

@author: riccardo
"""
from sqlalchemy.ext.hybrid import hybrid_property

import stream2segment.io.db.inspection as insp
from stream2segment.io.db.inspection import get_related_models
from stream2segment.process.db.models import Segment


def test_attnames():
    # Attach a property to the Segment attribute that is not Queriable, so that
    # `attnames(qatt=False)` returns something (currently, it yields nothing).
    # A hacky but efficient way is to attach a hybrid
    # property that raises:
    def raising_func(self):
        # this raises because we will access this proeprty on the class, which
        # does not have segment.station defined yet:
        return self.station.network
    Segment._non_queriable_att = hybrid_property(raising_func)
    # Now, attnames below will try to check:
    # isinstance(Segment._non_queriable_att, QueriableAttribute) -> raises
    # => attnames will determine that

    try:
        # queryable attributes keyed by their argument name:
        qatts = {'pkey': ['id'],
                 'fkey': ['event_id'],
                 'col': ['data', 'event_id', 'id'],
                 'rel': ['station'],
                 'qatt': ['id', 'event_id', 'data', 'station', 'has_data']}

        qatts = {'pkey', 'fkey', 'col', 'rel', 'qatt'}

        segment = Segment()
        assert sorted(insp.attnames(Segment)) == sorted(insp.attnames(segment))

        def attnames(**args):
            return list(insp.attnames(Segment, **args))

        anames = attnames()
        # assert we do NOT have stream and inventory:
        assert len(set(['stream', 'inventory', 'url']) & set(anames)) == 0
        # # assert we also have other expected attributes:
        # for k in qatts:
        #     assert len(set(qatts[k]) & set(anames)) == len(qatts[k])

        _ = attnames(pkey=True, fkey=True)
        assert not _

        _ = attnames(qatt=False)
        assert sorted(_) == \
               sorted(attnames(**{_: False for _ in qatts}))
        assert '_non_queriable_att' in set(_)

        attnames(qatt=False, pkey=True) == attnames(qatt=False, fkey=True) == \
            attnames(qatt=False, col=True) == attnames(qatt=False, col=True, rel=False) == []

        assert sorted(attnames(pkey=True)) == sorted(attnames(pkey=True, col=True))
        assert sorted(attnames(fkey=True)) == sorted(attnames(fkey=True, col=True))
        assert sorted(attnames(pkey=True)) == sorted(attnames(pkey=True, qatt=True))
        assert sorted(attnames(pkey=True)) == sorted(attnames(pkey=True, qatt=True))

        _ = set(attnames(pkey=True))
        assert _ & set(attnames(col=True)) == _
        assert _ & set(attnames(qatt=True)) == _

        _ = set(attnames(fkey=True))
        assert _ & set(attnames(col=True)) == _
        assert _ & set(attnames(qatt=True)) == _

        _ = set(attnames(rel=True))
        assert _ & set(attnames(qatt=True)) == _

        assert sorted(attnames(qatt=True, rel=True)) == sorted(attnames(rel=True))

        assert not set(attnames(qatt=True, rel=False)) - set(attnames(qatt=True))
        assert set(attnames(qatt=True)) - set(attnames(qatt=True, rel=False))

        relnames = set(get_related_models(Segment).keys())
        assert sorted(attnames(rel=True)) == sorted(relnames)
        assert not (relnames - {'download', 'station', 'classes', 'channel',
                                'event', 'datacenter'})

    finally:
        if hasattr(Segment, '_non_queriable_att'):
            del Segment._non_queriable_att
            assert not hasattr(Segment, '_non_queriable_att')

    # all_attnames = set(insp.attnames(Segment))
    # for pkey, fkey, col, rel, qatt in product([[False, True, None]] * 5):
    #     attnames = set(insp.attnames(Segment, pkey, fkey, col, rel, qatt))
    #
    #     if qatt is False:
    #         fkey = pkey = col = rel = False
    #
    #     if col is False:
    #         fkey = pkey = False
    #
    #     if pkey is False:
    #
    #
    #     expected_attnames = set(qatts[comb[0]]).intersection(
    #         *[qatts[c] for c in comb[1:]])
    #     if not expected_attnames:
    #         assert not attnames
    #     else:
    #         try:
    #             assert len(expected_attnames & set(attnames)) == len(expected_attnames)
    #         except AssertionError:
    #             asd = 9



    # combine all possible arguments:
    # count = 0
    # for k in range(1, len(qatts)+1):
    #     for comb in combinations(qatts, k):
    #         attnames = list(insp.attnames(Segment, **{_: True for _ in comb}))
    #
    #         for
    #
    #         for seg in [Segment, segment]:
    #             count += 1
    #             attnames = list(insp.attnames(Segment, **{_: True for _ in comb}))
    #             expected_attnames = set(qatts[comb[0]]).intersection(*[qatts[c] for c in comb[1:]])
    #             if not expected_attnames:
    #                 assert not attnames
    #             else:
    #                 try:
    #                     assert len(expected_attnames & set(attnames)) == len(expected_attnames)
    #                 except AssertionError:
    #                     asd = 9
    # assert not (set(get_related_models(Segment).keys()) -
    #            {'download', 'station', 'classes', 'channel', 'event', 'datacenter'})