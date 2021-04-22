'''
Created on Jul 15, 2016

@author: riccardo
'''
from builtins import str
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.sql.expression import desc
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.functions import func
from sqlalchemy.orm.attributes import QueryableAttribute

from stream2segment.process.db.models import (Download, Event, WebService, Station,
                                              Channel, Class, ClassLabelling, Segment,
                                              DataCenter)


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)

        sess = db.session
        run = Download()
        sess.add(run)
        sess.commit()

        dcen = DataCenter(station_url="x/station/abc")  # invalid fdsn name
        with pytest.raises(IntegrityError):
            sess.add(dcen)
            sess.commit()
        sess.rollback()

        dcen = DataCenter(station_url="x/station/fdsnws/station/1/")  # another invalid fdsn name
        with pytest.raises(IntegrityError):
            sess.add(dcen)
            sess.commit()
        sess.rollback()

        # https://service.iris.edu/fdsnws/station/1/

        dcen = DataCenter(station_url="domain/fdsnws/station/1/")  # this is save (fdsn)
        sess.add(dcen)
        sess.commit()

        # this is safe (both provided): FIXME!! should we pass here??
        dcen = DataCenter(station_url="x/station/abc", dataselect_url="x/station/abc")
        sess.add(dcen)
        sess.commit()

        ws = WebService(url='abc')
        sess.add(ws)
        sess.commit()

        event1 = Event(id=1, event_id='a', webservice_id=ws.id, time=datetime.utcnow(), magnitude=5,
                       latitude=66, longitude=67, depth_km=6)
        event2 = Event(id=2, event_id='b', webservice_id=ws.id, time=datetime.utcnow(), magnitude=5,
                       latitude=66, longitude=67, depth_km=6)
        sess.add_all([event1, event2])
        sess.commit()

        sta1 = Station(id=1, network='n1', station='s1', datacenter_id = dcen.id,
                       latitude=66, longitude=67, start_time=datetime.utcnow())
        sta2 = Station(id=2, network='n2', station='s1', datacenter_id = dcen.id,
                       latitude=66, longitude=67, start_time=datetime.utcnow())
        sess.add_all([sta1, sta2])
        sess.commit()

        cha1 = Channel(id=1, location='l1', channel='c1', station_id=sta1.id, sample_rate=6)
        cha2 = Channel(id=2, location='l2', channel='c2', station_id=sta1.id, sample_rate=6)
        cha3 = Channel(id=3, location='l3', channel='c3', station_id=sta1.id, sample_rate=6)
        cha4 = Channel(id=4, location='l4', channel='c4', station_id=sta2.id, sample_rate=6)
        sess.add_all([cha1, cha2, cha3, cha4])
        sess.commit()

        # segment 1, with two class labels 'a' and 'b'
        seg1 = Segment(event_id=event1.id, channel_id=cha3.id, datacenter_id=dcen.id,
                       event_distance_deg=5, download_id=run.id,
                       arrival_time=datetime.utcnow(), request_start=datetime.utcnow(),
                       request_end=datetime.utcnow())
        sess.add(seg1)
        sess.commit()

        cls1 = Class(label='a')
        cls2 = Class(label='b')

        sess.add_all([cls1, cls2])
        sess.commit()

        clb1 = ClassLabelling(segment_id=seg1.id, class_id=cls1.id)
        clb2 = ClassLabelling(segment_id=seg1.id, class_id=cls2.id)

        sess.add_all([clb1, clb2])
        sess.commit()

        # segment 2, with one class label 'a'
        seg2 = Segment(event_id=event1.id, channel_id=cha2.id, datacenter_id=dcen.id,
                       event_distance_deg=6.6, download_id=run.id,
                       arrival_time=datetime.utcnow(), request_start=datetime.utcnow(),
                       request_end=datetime.utcnow())

        sess.add(seg2)
        sess.commit()

        clb1 = ClassLabelling(segment_id=seg2.id, class_id=cls1.id)

        sess.add_all([clb1])
        sess.commit()

        # segment 3, no class label 'a' (and with data attr, useful later)
        seg3 = Segment(event_id=event1.id, channel_id=cha1.id, datacenter_id=dcen.id,
                       event_distance_deg=7, download_id=run.id, data=b'data',
                       arrival_time=datetime.utcnow(), request_start=datetime.utcnow(),
                       request_end=datetime.utcnow())
        sess.add(seg3)
        sess.commit()

    def test_inspect(self, db):
        # attach a fake method to Segment where the type is unknown:
        defval = 'a'
        Segment._fake_method = \
            hybrid_property(lambda self: defval,
                            expr=lambda cls: func.substr(cls.download_code, 1, 1))

        import stream2segment.io.db.inspection as insp

        attnames = list(insp.attrs(pkey=True))
        assert attnames == ['id']
        attnames = list(insp.attrs(pkey=None))
        assert attnames == ['id']
        attnames = list(insp.attrs(pkey=False))
        assert attnames == ['id']

        assert 'event_id' in attnames and 'id' not in attnames \
            and 'classes' not in attnames and '_fake_method' not in attnames
        attnames2 = list(insp.attnames(Inspector.FKEY, sort=False))
        # sort=False MIGHT return the same attributes order as sorted=True
        # thus perform a check only if they differ:
        if attnames != attnames:
            assert sorted(attnames) == sorted(attnames2)
        attnames = list(insp.attnames(Inspector.QATT))
        assert '_fake_method' in attnames and not 'id' in attnames and \
            not 'event_id' in attnames
        attnames = list(insp.attnames(Inspector.REL, sort=True))
        assert 'classes' in attnames and 'id' not in attnames \
            and 'event_id' not in attnames and '_fake_method' not in attnames
        attnames2 = list(insp.attnames(Inspector.REL, sort=False))
        # sort=False MIGHT return the same attributes order as sorted=True
        # thus perform a check only if they differ:
        if attnames != attnames:
            assert sorted(attnames) == sorted(attnames2)
        
        attnames = insp.attnames(deep=True)
        for attname in attnames:
            attval = insp.attval(attname)
            assert isinstance(attval, QueryableAttribute)
            if attname == '_fake_method':
                assert insp.atttype(attname) is None
            else:
                assert insp.atttype(attname) is not None

        seg = db.session.query(Segment).first()
        attnames = insp.attnames()
        for attname in attnames:
            val = insp.attval(attname)
            if attname == '_fake_method':
                assert val == defval
            if attname.startswith('classes.'):
                assert isinstance(val, list)
            else:
                assert not isinstance(val, (dict, list, set))
