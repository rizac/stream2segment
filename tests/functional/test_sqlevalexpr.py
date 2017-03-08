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
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from stream2segment.io.db.pd_sql_utils import _harmonize_columns, harmonize_columns,\
    get_or_add_iter, harmonize_rows, colnames
from stream2segment.io.utils import dumps_inv, loads_inv
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from datetime import datetime, timedelta
from sqlalchemy.orm.session import object_session
from stream2segment.utils.sqlevalexpr import  get_column, query
from stream2segment.io.db.models import ClassLabelling, Class

class Test(unittest.TestCase):
        

    
    def setUp(self):
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///:memory:', echo=True)
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
        cha3 = models.Channel(location='l3', channel='c3', station_id=sta2.id, sample_rate=6)
        cha4 = models.Channel(location='l4', channel='c4', station_id=sta2.id, sample_rate=6)
        
        sess.add_all([cha1, cha2, cha3, cha4])
        sess.commit()
        
        seg1 = models.Segment(event_id=event1.id, channel_id=cha1.id, datacenter_id=dcen.id,
                              event_distance_deg=5, run_id=run.id, 
                              arrival_time = datetime.utcnow(), start_time = datetime.utcnow(),
                              end_time=datetime.utcnow())
        
        sess.add(seg1)
        sess.commit()
        
        h = 9
        
        cls1 = Class(label='a')
        cls2 = Class(label='a')
        
        sess.add_all([cls1, cls2])
        sess.commit()
        
        clb1 = ClassLabelling(segment_id=seg1.id, class_id=cls1.id)
        clb2 = ClassLabelling(segment_id=seg1.id, class_id=cls2.id)
        
        sess.add_all([clb1, clb2])
        sess.commit()
        
        # ok so let's see how relationships join for us:
        # that's wrong, it does not return ANY join cause none is specified
        with pytest.raises(Exception):
            sess.query(models.Channel).join(models.Event)
        
        # that works, but since we didn't join is simply retunring two elements
        res = sess.query(models.Segment.id).filter(models.Station.station=='s1').all()
        assert len(res) == 2  # BECAUSE WE HAVE TWO STATIONS with station == 's1'
        
        # this on the other hand works, and recognizes the join for us:
        res1 = sess.query(models.Segment.id).join(models.Segment.station).filter(models.Station.station=='s1').all()
        assert len(res1) == 1  # BECAUSE WE HAVE TWO STATIONS with station == 's1'

        res2 = query(sess, models.Segment.id, {'station.station': 's1'}).all()
        res3 = query(sess, models.Segment, {'station.station': 's1'}).all()
        
        assert get_column(seg1, "id") == 1
        
        assert res1 == res2
        assert res3[0].id == res1[0][0]
        
        res2 = query(sess, models.Segment.id, {'station.station': 's1'})
        res2b = sess.query(models.Segment.id).filter(models.Segment.station.has(models.Station.station >= 's1'))
        res3 = sess.query(models.Segment.id).filter(models.Segment.classes.has(models.Class.id == 1))
        assert not len(res4)
        assert not len(res4)
        
        
        res4 = query(sess, models.Segment.id, {'classes': 'null'}).all()
        assert not len(res4)
        
        res5 = query(sess, models.Segment.id, {'classes': '>=0'}).all()
        assert len(res5) == 1 and res5 == res1
        
        res6 = query(sess, models.Segment.id, {'station.station': 's1', 'classes': '>=0'}).all()
        assert len(res6) == 1 and res6 == res1


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()