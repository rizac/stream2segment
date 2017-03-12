#@PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''
import pytest, os
import unittest
import numpy as np
import os
from stream2segment.io.db import models
from stream2segment.io.db.models import Base  # This is your declarative base class
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, InvalidRequestError
from stream2segment.io.db.pd_sql_utils import _harmonize_columns, harmonize_columns,\
    get_or_add_iter, harmonize_rows, colnames
from stream2segment.io.utils import dumps_inv, loads_inv
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from datetime import datetime, timedelta
from sqlalchemy.orm.session import object_session
from stream2segment.utils.sqlevalexpr import  get_column, query, query_args, get_condition
from stream2segment.io.db.models import ClassLabelling, Class
from sqlalchemy.sql.expression import desc

class Test(unittest.TestCase):
        

    
    def setUp(self):
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(engine)
        from sqlalchemy.orm import sessionmaker
        session = sessionmaker(bind=engine)()
    
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
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
        # self.DB.drop_all()


    def test_query_joins(self):
        sess = self.session
        run = models.Run()
        sess.add(run)
        sess.commit()
        
        dcen = models.DataCenter(station_query_url="x/station/abc")
        sess.add(dcen)
        sess.commit()
        
        event1 = models.Event(id='a', time=datetime.utcnow(), magnitude=5,
                              latitude=66, longitude=67, depth_km=6)
        event2 = models.Event(id='b', time=datetime.utcnow(), magnitude=5,
                              latitude=66, longitude=67, depth_km=6)
        
        sess.add_all([event1, event2])
        sess.commit()

        
        sta1 = models.Station(network='n1', station='s1', datacenter_id = dcen.id,
                              latitude=66, longitude=67, )
        sta2 = models.Station(network='n2', station='s1', datacenter_id = dcen.id,
                              latitude=66, longitude=67, )
        
        sess.add_all([sta1, sta2])
        sess.commit()
        
        cha1 = models.Channel(location='l1', channel='c1', station_id=sta1.id, sample_rate=6)
        cha2 = models.Channel(location='l2', channel='c2', station_id=sta1.id, sample_rate=6)
        cha3 = models.Channel(location='l3', channel='c3', station_id=sta1.id, sample_rate=6)
        cha4 = models.Channel(location='l4', channel='c4', station_id=sta2.id, sample_rate=6)
        
        sess.add_all([cha1, cha2, cha3, cha4])
        sess.commit()
        
        # segment 1, with two class labels 'a' and 'b'
        seg1 = models.Segment(event_id=event1.id, channel_id=cha3.id, datacenter_id=dcen.id,
                              event_distance_deg=5, run_id=run.id, 
                              arrival_time = datetime.utcnow(), start_time = datetime.utcnow(),
                              end_time=datetime.utcnow())
        
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
        seg2 = models.Segment(event_id=event1.id, channel_id=cha2.id, datacenter_id=dcen.id,
                              event_distance_deg=6.6, run_id=run.id, 
                              arrival_time = datetime.utcnow(), start_time = datetime.utcnow(),
                              end_time=datetime.utcnow())
        
        sess.add(seg2)
        sess.commit()
        
        clb1 = ClassLabelling(segment_id=seg2.id, class_id=cls1.id)
        
        sess.add_all([clb1])
        sess.commit()
        
        # segment 3, no class label 'a'
        seg3 = models.Segment(event_id=event1.id, channel_id=cha1.id, datacenter_id=dcen.id,
                              event_distance_deg=7, run_id=run.id, 
                              arrival_time = datetime.utcnow(), start_time = datetime.utcnow(),
                              end_time=datetime.utcnow())
        sess.add(seg3)
        sess.commit()
        
        # ok so let's see how relationships join for us:
        # this below is wrong, it does not return ANY join cause none is specified in models
        with pytest.raises(Exception):
            sess.query(models.Channel).join(models.Event)
        
        # this below works, but since we didn't join is simply returning two * three elements
        # why? (in any case because we have TWO stations with station column = 's1', and three
        # segments all in all):
        res = sess.query(models.Segment.id).filter(models.Station.station=='s1').all()
        assert len(res) == 6  # BECAUSE WE HAVE TWO STATIONS with station == 's1'
        
        # this on the other hand works, and recognizes the join for us:
        res1 = sess.query(models.Segment.id).join(models.Segment.station).filter(models.Station.station=='s1').all()
        assert len(res1) == 3

        # this is the same as above, but uses exist instead on join
        # the documentation says it's slower, but there is a debate in the internet and in any case
        # it will be easier to implement when providing user input in "simplified" sql
        res3 = sess.query(models.Segment.id).filter(models.Segment.station.has(models.Station.station=='s1')).all()
        assert res1 == res3
        
        # Note that the same as above for list relationships (one to many or many to many)
        # needs to use 'any' instead of 'has':
        res4 = sess.query(models.Segment.id).filter(models.Segment.classes.any(models.Class.label=='a')).all()
        assert len(res4) == 2
        
        # ============================
        
        
        # redo the same queries with our implemented method, used to parse simple output from user:
        
        res3b = sess.query(models.Segment.id).filter(query_args(models.Segment, {'station.station': 's1'})).all()
        res4b = sess.query(models.Segment.id).filter(query_args(models.Segment, {'classes.label': "a"})).all()
        
        assert res3b == res3 and res4b == res4
        
        # test that passing null as argument returns segments which do NOT have any class set:
        res5 = sess.query(models.Segment.id).filter(query_args(models.Segment, {'classes.id': "null"})).all()
        # test that passing not null as argument returns segments which do NOT have any class set:
        res6 = sess.query(models.Segment.id).filter(query_args(models.Segment, {'classes.id': "!=null"})).all()
        
        assert len(res5) == 0 and len(res6) == 2
        
        
        # now we try to test the order_by with relationships:
        # this fails:
        with pytest.raises(AttributeError):
            sess.query(models.Segment.id).order_by(models.Segment.station.id).all()
        sdgf = 9
        
        # this works:
        k1 = sess.query(models.Segment.id).join(models.Segment.station).order_by(models.Station.id).all()
        k2 = sess.query(models.Segment.id).join(models.Segment.station, models.Segment.channel).order_by(models.Station.id, models.Channel.id).all()
        
        # curiously, k1 is like k2 (which is  [(3,), (2,), (1,)]). This is not a weird behaviour, simply the order might have been
        # returned differently cause all segments have the same station thus [(3,), (1,), (2,)] would be also fine
        
        # we order first by event distance degree. Each segment created has an increasing event_distance_degree
        k3 = sess.query(models.Segment.id).join(models.Segment.channel).order_by(models.Segment.event_distance_deg, models.Channel.id).all()
        res = 456
        
        # So, ordering is by default ascending
        assert k3 == [(1,), (2,), (3,)]
        k4 = sess.query(models.Segment.id).join(models.Segment.channel).order_by(desc(models.Segment.event_distance_deg), models.Channel.id).all()
        assert k4 == [(3,), (2,), (1,)]
        
        # we order now by event channel id first. Each segment created has an decreasing channel id
        k5 = sess.query(models.Segment.id).join(models.Segment.channel).order_by(models.Channel.id,models.Segment.event_distance_deg).all()        
        assert k5 == [(3,), (2,), (1,)]
        
        
        res0 = sess.query(models.Segment.id).join(models.Segment.channel).order_by(models.Channel.id,models.Segment.event_distance_deg).all()        
        # now we test the query function. Set channel.id !=null in order to take all channels
        # The two queries below should be the same
        res1 = query(sess, models.Segment.id, {'channel.id': '!=null'}, ['channel.id', 'event_distance_deg']).all()
        res2 = query(sess, models.Segment.id, {'channel.id': '!=null'}, [('channel.id', 'asc'), ('event_distance_deg', 'asc')]).all()
        assert res0 == res1
        assert res0 == res2
        gh = 9
        
        #classes have ids 1 and 2
        # segment 1 has classes 1 and 2
        # segment 2 has class 2.
        res1 = query(sess, models.Segment.id, {'classes.id': '[0 1]'}, ['channel.id', 'event_distance_deg']).all()
        assert sorted([c[0] for c in res1]) == [1, 2]  # regardless of order, we are interested in segments
        f = 9
        
        # BUT notice this: now segment 1 is returned TWICE
        # http://stackoverflow.com/questions/23786401/why-do-multiple-table-joins-produce-duplicate-rows
        res1 = query(sess, models.Segment.id, {'classes.id': '[1 2]'}, ['channel.id', 'event_distance_deg']).all()
        assert sorted([c[0] for c in res1]) == [1,1,2]  # regardless of order, we are interested in segments
        # solution? group_by:
        res1 = query(sess, models.Segment.id, {'classes.id': '[1 2]'}, ['channel.id', 'event_distance_deg']).group_by(models.Segment.id).all()
        assert sorted([c[0] for c in res1]) == [1, 2]  # regardless of order, we are interested in segments
        
        
        assert sorted(get_column(seg1, "classes.id")) == [1,2]
        
    def test_eval_expr(self):
        from stream2segment.io.db.models import Segment, Event, Station, Channel

        c = Segment.arrival_time
        cond = get_condition(c, "=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time = :arrival_time_1"
    
        cond = get_condition(c, "!=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time != :arrival_time_1"
    
        cond = get_condition(c, ">=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time >= :arrival_time_1"
        
        cond = get_condition(c, "<=2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time <= :arrival_time_1"
        
        cond = get_condition(c, ">2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time > :arrival_time_1"
        
        cond = get_condition(c, "<2016-01-01T00:03:04")
        assert str(cond) == "segments.arrival_time < :arrival_time_1"
        
        with pytest.raises(ValueError):
            cond = get_condition(c, "2016-01-01T00:03:04, 2017-01-01")
        
        cond = get_condition(c, "2016-01-01T00:03:04 2017-01-01")
        assert str(cond) == "segments.arrival_time IN (:arrival_time_1, :arrival_time_2)"
        
        cond = get_condition(c, "[2016-01-01T00:03:04 2017-01-01]")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2"
        
        cond = get_condition(c, "(2016-01-01T00:03:04 2017-01-01]")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3"
        
        cond = get_condition(c, "[2016-01-01T00:03:04 2017-01-01)")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3"
        
        cond = get_condition(c, "(2016-01-01T00:03:04 2017-01-01)")
        assert str(cond) == "segments.arrival_time BETWEEN :arrival_time_1 AND :arrival_time_2 AND segments.arrival_time != :arrival_time_3 AND segments.arrival_time != :arrival_time_4"
        
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()