'''
Created on Jul 15, 2016

@author: riccardo
'''
# from builtins import str
# from datetime import datetime
from itertools import combinations

# import pytest
import stream2segment.io.db.inspection as insp
from stream2segment.process.db.models import Segment


def test_attnames():
    # attach a fake method to Segment where the type is unknown:
    defval = 'a'
    # Segment._fake_method = \
    #     hybrid_property(lambda self: defval,
    #                     expr=lambda cls: func.substr(cls.download_code, 1, 1))


    # queryable attributes keyed by their argument name:
    qatts = {'pkey': ['id'], 'fkey': ['event_id'],
             'col': ['data', 'event_id', 'id'],
             'rel': ['station'],
             'qatt': ['id', 'event_id', 'data', 'station', 'has_data']}

    # test normal methods/properties are returned only when all arguments are False
    attnames = list(insp.attnames(Segment, **{_: False for _ in qatts}))
    # assert we have stream and inventory:
    assert len(set(['stream', 'inventory']) & set(attnames)) == 2
    # assert we do not have other expected attributes:
    assert len(set([_ for e in qatts for _ in e]) & set(attnames)) == 0

    attnames = list(insp.attnames(Segment))
    # assert we have stream and inventory:
    assert len(set(['stream', 'inventory']) & set(attnames)) == 2
    # assert we also have other expected attributes:
    for k in qatts:
        assert len(set(qatts[k]) & set(attnames)) == len(qatts[k])

    segment = Segment()
    # combine all possible arguments:
    count = 0
    for k in range(1, len(qatts)+1):
        for comb in combinations(qatts, k):
            for seg in [Segment, segment]:
                count += 1
                attnames = list(insp.attnames(Segment, **{_: True for _ in comb}))
                expected_attnames = set(qatts[comb[0]]).intersection(*[qatts[c] for c in comb[1:]])
                if not expected_attnames:
                    assert not attnames
                else:
                    assert len(expected_attnames & set(attnames)) == len(expected_attnames)

