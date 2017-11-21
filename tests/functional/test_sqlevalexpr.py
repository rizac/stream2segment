#@PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''
from builtins import str
import pytest, os
import unittest
import numpy as np
import os
from stream2segment.io.db.models import Base  # This is your declarative base class
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, InvalidRequestError, ProgrammingError
from stream2segment.io.db.pd_sql_utils import _harmonize_columns, harmonize_columns,\
    harmonize_rows, colnames, dbquery2df
from stream2segment.io.utils import dumps_inv, loads_inv
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from datetime import datetime, timedelta
from sqlalchemy.orm.session import object_session
from stream2segment.io.db.sqlevalexpr import exprquery, binexpr
from stream2segment.io.db.models import ClassLabelling, Class, Segment, Station, Channel,\
    Event, DataCenter, Download, WebService
from sqlalchemy.sql.expression import desc
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.orm.query import aliased
class Test(unittest.TestCase):
        

    
    def setUp(self):
        url = os.getenv("DB_URL", "sqlite:///:memory:")
        self.engine = create_engine(url, echo=False)
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        
        # session = sessionmaker(bind=self.engine)()
    
        # create a configured "Session" class
        Session = sessionmaker(bind=self.engine)
        # create a Session
        self.session = Session()

    def tearDown(self):
        try:
            self.session.flush()
            self.session.commit()
        except SQLAlchemyError as _:
            pass
            # self.session.rollback()
        self.session.close()
        self.session.close()
        Base.metadata.drop_all(self.engine)

    @property
    def is_sqlite(self):
        return str(self.engine.url).startswith("sqlite:///")
    
    @property
    def is_postgres(self):
        return str(self.engine.url).startswith("postgresql://")

    def test_query_joins(self):
        sess = self.session
        run = Download()
        sess.add(run)
        sess.commit()
        
        dcen = DataCenter(station_url="x/station/abc")  # invalid fdsn name
        with pytest.raises(IntegrityError):
            sess.add(dcen)
            sess.commit()
        sess.rollback()
        
        # https://service.iris.edu/fdsnws/station/1/
        
        dcen = DataCenter(station_url="x/station/fdsnws/station/1/")  # this is save (fdsn)
        sess.add(dcen)
        sess.commit()
        
        dcen = DataCenter(station_url="x/station/abc", dataselect_url="x/station/abc")  # this is safe (both provided):
        sess.add(dcen)
        sess.commit()
        
        
        ws = WebService(url='abc')
        sess.add(ws)
        sess.commit()
        
        event1 = Event(id=1, eventid='a', webservice_id=ws.id, time=datetime.utcnow(), magnitude=5,
                              latitude=66, longitude=67, depth_km=6)
        event2 = Event(id=2, eventid='b', webservice_id=ws.id, time=datetime.utcnow(), magnitude=5,
                              latitude=66, longitude=67, depth_km=6)
        
        sess.add_all([event1, event2])
        sess.commit()

        
        sta1 = Station(id=1, network='n1', station='s1', datacenter_id = dcen.id,
                              latitude=66, longitude=67, start_time= datetime.utcnow())
        sta2 = Station(id=2, network='n2', station='s1', datacenter_id = dcen.id,
                              latitude=66, longitude=67, start_time= datetime.utcnow())
        
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
                              arrival_time = datetime.utcnow(), request_start = datetime.utcnow(),
                              request_end=datetime.utcnow())
        
        sess.add(seg1)
        sess.commit()
        
        h = 9
        
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
                              arrival_time = datetime.utcnow(), request_start = datetime.utcnow(),
                              request_end=datetime.utcnow())
        
        sess.add(seg2)
        sess.commit()
        
        clb1 = ClassLabelling(segment_id=seg2.id, class_id=cls1.id)
        
        sess.add_all([clb1])
        sess.commit()
        
        # segment 3, no class label 'a' (and with data attr, useful later)
        seg3 = Segment(event_id=event1.id, channel_id=cha1.id, datacenter_id=dcen.id,
                              event_distance_deg=7, download_id=run.id, data=b'data',
                              arrival_time = datetime.utcnow(), request_start = datetime.utcnow(),
                              request_end=datetime.utcnow())
        sess.add(seg3)
        sess.commit()
        
        # ok so let's see how relationships join for us:
        # this below is wrong, it does not return ANY join cause none is specified in models
        with pytest.raises(Exception):
            sess.query(Channel).join(Event)
        
        # this below works, but since we didn't join is simply returning two * three elements
        # why? (in any case because we have TWO stations with station column = 's1', and three
        # segments all in all):
        res = sess.query(Segment.id).filter(Station.station=='s1').all()
        assert len(res) == 6  # BECAUSE WE HAVE TWO STATIONS with station column == 's1'
        
        # this on the other hand works, and recognizes the join for us:
        res1 = sess.query(Segment.id).join(Segment.station).filter(Station.station=='s1').all()
        assert len(res1) == 3

        # this is the same as above, but uses exist instead on join
        # the documentation says it's slower, but there is a debate in the internet and in any case
        # it will be easier to implement when providing user input in "simplified" sql
        res3 = sess.query(Segment.id).filter(Segment.station.has(Station.station=='s1')).all()
        assert res1 == res3
        
        # Note that the same as above for list relationships (one to many or many to many)
        # needs to use 'any' instead of 'has':
        res4 = sess.query(Segment.id).filter(Segment.classes.any(Class.label=='a')).all()
        assert len(res4) == 2
        
        # ============================
        
        # current segments are these:
        # id  channel_id  event_distance_deg  class_id
        # 1   3           5.0                 1
        # 1   3           5.0                 2
        # 2   2           6.6                 1
        
        # now we try to test the order_by with relationships:
        # this fails:
        with pytest.raises(AttributeError):
            sess.query(Segment.id).order_by(Segment.station.id).all()
        sdgf = 9
        
        # this works:
        k1 = sess.query(Segment.id).join(Segment.station).order_by(Station.id).all()
        k2 = sess.query(Segment.id).join(Segment.station, Segment.channel).order_by(Station.id, Channel.id).all()
        
        # curiously, k1 is like k2 (which is  [(3,), (2,), (1,)]). This is not a weird behaviour, simply the order might have been
        # returned differently cause all segments have the same station thus [(3,), (1,), (2,)] would be also fine
        
        # we order first by event distance degree. Each segment created has an increasing event_distance_degree
        k3 = sess.query(Segment.id).join(Segment.channel).order_by(Segment.event_distance_deg, Channel.id).all()
        res = 456
        
        # So, ordering is by default ascending
        assert k3 == [(1,), (2,), (3,)]
        k4 = sess.query(Segment.id).join(Segment.channel).order_by(desc(Segment.event_distance_deg), Channel.id).all()
        assert k4 == [(3,), (2,), (1,)]
        
        # we order now by event channel id first. Each segment created has an decreasing channel id
        k5 = sess.query(Segment.id).join(Segment.channel).order_by(Channel.id,Segment.event_distance_deg).all()        
        assert k5 == [(3,), (2,), (1,)]
        
        
        res0 = sess.query(Segment.id).join(Segment.channel).order_by(Channel.id,Segment.event_distance_deg).all()        
        # now we test the query function. Set channel.id !=null in order to take all channels
        # The two queries below should be the same
        res1 = exprquery(sess.query(Segment.id), {'channel.id': '!=null'},
                     ['channel.id', 'event_distance_deg']).all()
        res2 = exprquery(sess.query(Segment.id), {'channel.id': '!=null'},
                     [('channel.id', 'asc'), ('event_distance_deg', 'asc')]).all()
        assert res0 == res1
        assert res0 == res2
        gh = 9
        
        
        # test the case where we supply a model instead of a column as first arg
        res1 = exprquery(sess.query(Segment), {'event_distance_deg': '==5'}).all()
        # now a double column
        res2 = exprquery(sess.query(Segment.id, Station.id).join(Segment.station),
                         {'event_distance_deg': '[5,5]'}).all()
        assert res1[0].id == res2[0][0]
        # now the same as above in inversed order (should raise, as Station is considered
        # the model class, but has no event_distance_deg attribute
        with pytest.raises(AttributeError):
            res3 = exprquery(sess.query(Station.id, Segment.id).join(Segment.station),
                             {'event_distance_deg': '[5,5]'}).all()
            sess.rollback()  # for safety

        # what happens if we supply more than one join? we issue a warning, but the query is ok
        # (so the following does not raise)
        # the warning is at sqlalchemy.orm.query.py:2105:
        # 'SAWarning: Pathed join target Segment.channel has already been joined to; skipping
        # "been joined to; skipping" % prop)'
        res1 = exprquery(sess.query(Segment).join(Segment.channel).\
                         filter(Channel.id>0), {'channel.id': '!=null'}).all()
        
        
        # test any and none:
        # Note: contrarily to classes.id specified as interval, which issues a join
        # in exprquery, 'any' and 'none' issue a less-performant 'exist' at sql level, whcih
        # DOES NOT PRODUCE DUPLICATES. Check few line below for the duplicate case
        res1 = exprquery(sess.query(Segment.id), {'classes.id': 'any'},
                             ['channel.id', 'event_distance_deg']).all()
        res2 = exprquery(sess.query(Segment.id), {'classes.id': 'none'},
                     ['channel.id', 'event_distance_deg']).all()
        res3 = exprquery(sess.query(Segment.id), {'classes': 'any'},
                     ['channel.id', 'event_distance_deg']).all()
        res4 = exprquery(sess.query(Segment.id), {'classes': 'none'},
                     ['channel.id', 'event_distance_deg']).all()
        assert res1 == res3 and res2 == res4
        
        # classes is a many to many relationship on Segment,
        # what if we provide a many-to-one (column)? it does not work. From the docs:
        #     :meth:`~.RelationshipProperty.Comparator.any` is only
        #     valid for collections, i.e. a :func:`.relationship`
        #     that has ``uselist=True``.  For scalar references,
        #     use :meth:`~.RelationshipProperty.Comparator.has`.
        with pytest.raises(InvalidRequestError):
            res1 = exprquery(sess.query(Segment.id), {'station.id': 'any'}).all()
            sess.rollback()  # for safety
        # what if we provide a normal attribute? it does not work either cause station is 'scalar':
        with pytest.raises(AttributeError):
            res2 = exprquery(sess.query(Segment.id), {'id': 'any'}).all()
            sess.rollback()  # for safety
        # what if we provide a one-to-many? it works
        res3 = exprquery(sess.query(Station.id), {'segments': 'any'}).all()
                             
#         reminder
        # current segments are these:
        # id  channel_id  event_distance_deg  class_id
        # 1   3           5.0                 1
        # 1   3           5.0                 2
        # 2   2           6.6                 1
        
        
        # test many to many with specific values, not 'any' and 'none' as above
        res1 = exprquery(sess.query(Segment.id), {'classes.id': '[0 1]'},
                     ['channel.id', 'event_distance_deg']).all()
        res1 = exprquery(sess.query(Segment.id), {'classes.id': '[0 1]'},
                     ['channel.id', 'event_distance_deg']).all()
        #classes have ids 1 and 2
        # segment 1 has classes 1 and 2
        # segment 2 has class 2.
        res1 = exprquery(sess.query(Segment.id), {'classes.id': '[0 1]'},
                     ['channel.id', 'event_distance_deg']).all()
        seg_ids = [c[0] for c in res1]
        assert sorted(seg_ids) == [1, 2]  # regardless of order, we are interested in segments
        
        # BUT notice this: now segment 1 is returned TWICE
        # http://stackoverflow.com/questions/23786401/why-do-multiple-table-joins-produce-duplicate-rows
        res1 = exprquery(sess.query(Segment.id), {'classes.id': '[1 2]'},
                     ['channel.id', 'event_distance_deg']).all()
        seg_ids = [c[0] for c in res1]
        assert sorted(seg_ids) == [1,1,2]  # regardless of order, we are interested in segments
        # solution? distinct!. You could use also group_by BUT NOTE:
        # group_by HAS PROBLEMS in postgres, as the grpup by column must be specified also in the
        # group_by argument!
        if self.is_postgres:
            with pytest.raises(ProgrammingError):
                res1 = exprquery(sess.query(Segment.id), {'classes.id': '[1 2]'},
                             ['channel.id', 'event_distance_deg']).group_by(Segment.id).all()
            self.session.rollback()
        else:
            res1 = exprquery(sess.query(Segment.id), {'classes.id': '[1 2]'},
                         ['channel.id', 'event_distance_deg']).group_by(Segment.id).all()
            assert sorted([c[0] for c in res1]) == [1, 2]  # regardless of order, we are interested in segments
        
        #distinct is ok as it's without args, sqlalchemy does the job for us of getting the columns:
        res1 = exprquery(sess.query(Segment.id), {'classes.id': '[1 2]'},
                     ['channel.id', 'event_distance_deg']).distinct().all()
        assert sorted([c[0] for c in res1]) == [1, 2]  # regardless of order, we are interested in segments
        
        
        # again, as above, test with postgres fails:
        if self.is_postgres:
            with pytest.raises(ProgrammingError):
                res1 = exprquery(sess.query(Segment.id).filter(~Segment.has_data),
                             {'classes.id': '[1 2]'}, ['channel.id', 'event_distance_deg'],
                     ).group_by(Segment.id).all()
            self.session.rollback()
        else:
            res1 = exprquery(sess.query(Segment.id).filter(~Segment.has_data),
                         {'classes.id': '[1 2]'}, ['channel.id', 'event_distance_deg'],
                     ).group_by(Segment.id).all()
            assert sorted([c[0] for c in res1]) == [1, 2]  # regardless of order, we are interested in segments
        
        # test hybrid attrs:
        res3 = sess.query(Segment.id).filter(Segment.has_data).all()
        res2 = sess.query(Segment.id).filter(Segment.has_data == True).all()
        res1 = exprquery(sess.query(Segment.id), {'has_data': 'true'}).all()
        assert res1 == res2 and res1 == res3
        # now test the opposite query and assert result sets have no intersection
        res1 = exprquery(sess.query(Segment.id), {'has_data': 'false'}).all()
        assert not(set((_[0] for _ in res1)) & set((_[0] for _ in res2)))
        
    def test_eval_expr(self):
        from stream2segment.io.db.models import Segment, Event, Station, Channel

        c = Segment.arrival_time
        cond = binexpr(c, "=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time = :arrival_time_1"
    
        cond = binexpr(c, "!=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time != :arrival_time_1"
    
        cond = binexpr(c, ">=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time >= :arrival_time_1"
        
        cond = binexpr(c, "<=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time <= :arrival_time_1"
        
        cond = binexpr(c, ">2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time > :arrival_time_1"
        
        cond = binexpr(c, "<2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time < :arrival_time_1"
        
        with pytest.raises(ValueError):
            cond = binexpr(c, "2016-01-01T00:03:04, 2017-01-01")
        
        cond = binexpr(c, "2016-01-01T00:03:04 2017-01-01")
        assert str(cond) == "segments.arrival_time IN (:arrival_time_1, :arrival_time_2)"
        
        cond = binexpr(c, "[2016-01-01T00:03:04 2017-01-01]")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2"
        
        cond = binexpr(c, "(2016-01-01T00:03:04 2017-01-01]")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3"
        
        cond = binexpr(c, "[2016-01-01T00:03:04 2017-01-01)")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3"
        
        cond = binexpr(c, "(2016-01-01T00:03:04 2017-01-01)")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3 AND segments.arrival_time != :arrival_time_4"
        
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()