'''
Created on Jul 15, 2016

@author: riccardo
'''
import pytest
import unittest
import datetime
import numpy as np
import os
from stream2segment.s2sio.db import models
from stream2segment.s2sio.db.models import Base  # This is your declarative base class
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sqlalchemy.exc import IntegrityError
from stream2segment.s2sio.db.pd_sql_utils import _harmonize_columns, harmonize_columns, add_or_get


class Test(unittest.TestCase):
    
    engine = None

    @classmethod
    def setUpClass(cls):
        file = os.path.dirname(__file__)
        filedata = os.path.join(file,"..","data")
        url = os.path.join(filedata, "_test.sqlite")
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        cls.engine = create_engine('sqlite:///'+url)
        Base.metadata.drop_all(cls.engine)
        Base.metadata.create_all(cls.engine)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(cls.engine)
        
    def setUp(self):
        # create a configured "Session" class
        Session = sessionmaker(bind=self.engine)
        # create a Session
        self.session = Session()

    def tearDown(self):
        try:
            self.session.flush()
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
        self.session.close()
        # self.DB.drop_all()

#     def testParseCol(self):
#         e = models.Event()
#         leng = len(e.get_col_names())
#         df = pd.DataFrame(columns = [chr(i) for i in xrange(leng)],
#                           data=[])
#         df = e.parse_df(df)
#         assert all(df.columns[i] == e.get_col_names()[i] for i in xrange(leng))
#         
#         df = pd.DataFrame(columns = [chr(i) for i in xrange(leng+1)],
#                           data=[])
#         with pytest.raises(ValueError):
#             df = e.parse_df(df)
        
    def testSqlAlchemy(self):
        #run_cols = models.Run.__table__.columns.keys()
        #run_cols.remove('id')  # remove id (auto set in the model)
        #d = pd.DataFrame(columns=run_cols, data=[[None for _ in run_cols]])
        #records = d.to_dict('records') # records(index=False)
        # record = records[0]

        # pass a run_id without id and see if it's updated as utcnow:
        run_row = models.Run()
        assert run_row.id is None

        run_row = models.Run(id=None)
        assert run_row.id is None

        # test that methods of the base class work:
        assert len(run_row.get_cols()) > 0
        assert len(run_row.get_col_names()) > 0

        # test id is auto added:
        self.session.add_all([run_row])
        # self.session.flush()
        self.session.commit()
        assert run_row.id is not None

        # now pass a utcdatetime and see if we keep that value:
        utcnow = datetime.datetime.utcnow()
        run_row = models.Run(run_time=utcnow)
        assert run_row.run_time == utcnow
        self.session.add_all([run_row])
        # self.session.flush()
        self.session.commit()
        assert run_row.run_time == utcnow

        # test column names:
#         colz = run_row.get_col_names()
#         colz.remove('id')  # the primary key
#         d = pd.DataFrame(columns=colz, data=[[None for _ in colz]])
#         run_row._check_columns(d)

        # test types. string ints are parsed automatically? YES
        val = '6'
        e = models.Class(id=val)
        assert e.id != int(val)
        self.session.add(e)
        self.session.commit()
        assert e.id == int(val)

        # test types. string floats are parsed automatically as int? YES if INT
        # so this is NO:
        val = '5.2'
        e = models.Class(id=val)
        assert e.id != float(val)
        self.session.add(e)

        with pytest.raises(IntegrityError):
            self.session.commit()
        # necessary after failure? FIXME: check!
        self.session.rollback()
            
        # this is YES:
        val = '5.0'
        e = models.Class(id=val)
        assert e.id != int(float(val))
        self.session.add(e)
        self.session.commit()
        assert e.id == int(float(val))


        # test types. String floats are parsed automatically? YES
        val = '6.7'
        e = models.Event(id='abc', time=datetime.datetime.utcnow(),
                         latitude=val, longitude=78, magnitude=56, depth_km=45)
        assert e.latitude != float(val)
        self.session.add(e)
        self.session.commit()
        assert e.latitude == float(val)

        dc = models.DataCenter(station_query_url='abc')
        self.session.add(dc)
        self.session.flush()

        # test stations auto id (concat):
        # first test non-specified non-null fields (should reaise an IntegrityError)
        e = models.Station(network='abc', station='f', datacenter_id=dc.id)
        assert e.id is None
        self.session.add(e)
        # we do not have specified all non-null fields:
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # now test auto id
        e = models.Station(network='abc', station='f', latitude='89.5', longitude='56',
                           datacenter_id=dc.id)
        assert e.id is None
        self.session.add(e)
        # we do not have specified all non-null fields:
        self.session.commit()
        assert e.id == "abc.f"

        # test unique constraints by changing only network
        sta = models.Station(network='a', station='f', latitude='89.5', longitude='56',
                             datacenter_id=dc.id)
        self.session.add(sta)
        # we do not have specified all non-null fields:
        self.session.commit()
        assert sta.id == "a.f"

        # now re-add it. Unique constraint failed
        sta = models.Station(network='a', station='f', latitude='189.5', longitude='156',
                             datacenter_id=dc.id)
        self.session.add(sta)
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # test stations channels relationship:
        sta = models.Station(network='ax', station='f', latitude='89.5', longitude='56',
                             datacenter_id=dc.id)
        # write channels WITHOUT foreign key
        cha1 = models.Channel(location='l', channel='HHZ', sample_rate=56)
        cha2 = models.Channel(location='l', channel='HHN', sample_rate=12)
        # add channels to the stations.channels relationships
        sta.channels.append(cha1)
        sta.channels.append(cha2)
        # Now when adding and commiting we should see channels foreign keys updated according to
        # sta id:
        self.session.add(sta)
        self.session.add(cha1)
        self.session.add(cha2)
        self.session.commit()
        # foreign keys are auto updated!!! TEST IT:
        assert cha1.station_id == sta.id
        assert cha2.station_id == sta.id

        # now test the same with a station read from the database. We don't acutally need
        # a commit, flush is sufficient
        sta = self.session.query(models.Station).filter(models.Station.id == 'a.f').first()
        cha1 = models.Channel(location='l2', channel='HHZ', sample_rate=56)
        cha2 = models.Channel(location='l', channel='HHW', sample_rate=56)
        sta.channels.append(cha1)
        sta.channels.append(cha2)
        assert cha1.station_id != sta.id
        assert cha2.station_id != sta.id
        self.session.flush()
        # foreign keys are auto updated!!! TEST IT:
        assert cha1.station_id == sta.id
        assert cha2.station_id == sta.id
        
        k = self.session.query(models.Event).all()
        assert len(k) == 1

    def test_add_and_flush(self):
        # bad event entry (no id):
        e = models.Event(time = datetime.datetime.utcnow(), latitude = 6, longitude=8, depth_km=6,
                         magnitude=56)
        events = len(self.session.query(models.Event).all())
        
        self.session.add(e)
        with pytest.raises(IntegrityError):
            self.session.flush()
        self.session.rollback()  # necessary for the query below (all()). Try to comment and see
        assert len(self.session.query(models.Event).all()) == events
        
        # single flush after two add: rollback rolls back BOTH
        e2 = models.Event(id = 'Column(String, primary_key=True', 
                         time = datetime.datetime.utcnow(), latitude = 6, longitude=8, depth_km=6,
                         magnitude=56)
        self.session.add(e)
        self.session.add(e2)
        with pytest.raises(IntegrityError):
            self.session.flush()

        self.session.rollback()  # necessary for the query below (all()). Try to comment and see        
        assert len(self.session.query(models.Event).all()) == events

        # CORRECT WAY TO DO:
        # two flushes after two adds: rollback rolls back only those that failed:
        self.session.add(e)
        with pytest.raises(IntegrityError):
            self.session.flush()
        self.session.rollback()  # necessary for the query below (all()). Try to comment and see        
        self.session.add(e2)
        
        assert len(self.session.query(models.Event).all()) == events+1
            
    def test_pd_to_sql(self):
        dc = models.DataCenter(station_query_url='awergedfbvdbfnhfsnsbstndggf ')
        self.session.add(dc)
        self.session.commit()
        
        id = 'abcdefghilmnopq'
        utcnow = datetime.datetime.utcnow()
        e = models.Station(id="a.b", network='a', station='b', latitude=56, longitude=78,
                           datacenter_id=dc.id)
        self.session.add(e)
        self.session.commit()
#         
        df = pd.DataFrame(columns=models.Station.get_col_names(), data=[[None for _ in models.Station.get_col_names()]])
        df.loc[0, 'id'] = id + '.j'
        df.loc[0, 'network'] = id
        df.loc[0, 'station'] = 'j'
        df.loc[0, 'latitude'] = 43
        df.loc[0, 'longitude'] = 56.7
        df.loc[0, 'datacenter_id'] = dc.id
        
        df.to_sql(e.__table__.name, self.engine, if_exists='append', index=False)
         
        # same id as above, but check that data exist (i.e., error)
        df.loc[0, 'id'] = id
        with pytest.raises(IntegrityError):
            df.to_sql(e.__table__.name, self.engine, if_exists='append', index=False)

    def test_add_or_get(self):
        id = 'some_id_not_used_elsewhere'
        utcnow = datetime.datetime.utcnow()
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7, magnitude=56)
        e_, new_added = add_or_get(self.session, e)
        self.session.flush()  # necessary otherwise add_or_get below doesn;t work
        assert new_added is True and e_ is not None

        e_, new_added = add_or_get(self.session, e)
        assert new_added is False and e_ is not None

        e.latitude = 67.0546
        e_, new_added = add_or_get(self.session, e, 'latitude')
        # THIS STILL RETURNS e! we updated the same object!!
        assert new_added is False and e_ is not None

        # now re-initialize e WITH THE SAME ARGUMENT AS ABOVE AT THE BEGINNING:
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7, magnitude=56)
        # THIS RETURNS a new object, as e has the latitude set above so an element with
        # latitude-89.5 is NOT found on the db with the query
        e_, new_added = add_or_get(self.session, e, 'latitude')
        assert new_added is True and e_ is not None

        # now we should get an error cause we added two elements with same id:
        with pytest.raises(IntegrityError):
            self.session.flush()
            self.session.commit()
        self.session.rollback()


    def test_event_sta_channel_seg(self):
        dc= models.DataCenter(station_query_url="345635434246354765879685432efbdfnrhytwfesdvfbgfnyhtgrefs")
        self.session.add(dc)
        self.session.flush()

        id = '__abcdefghilmnopq'
        utcnow = datetime.datetime.utcnow()
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)

        e, added = add_or_get(self.session, e)

        s = models.Station(network='sdf', station='_', latitude=90, longitude=-45,
                           datacenter_id = dc.id)

        c = models.Channel(location= 'tyu', channel='rty', sample_rate=6)

        s.channels.append(c)
        
        s, added = add_or_get(self.session, s)
        
        self.session.flush()
        self.session.commit()
        
        id = '__abcdefghilmnopq'
        utcnow = datetime.datetime.utcnow()
        e = models.Event(id=id, time=utcnow)

        e, added = add_or_get(self.session, e)

        s = models.Station(network='sdf', station='_', latitude=90, longitude=-45)

        c = models.Channel(location= 'tyu', channel='rty')

        # 
        # self.session.flush()
        s, added = add_or_get(self.session, s, 'network', 'station')
        self.session.flush()

        s.channels.append(c)  # that's an error. test it:
        with pytest.raises(IntegrityError):
            self.session.flush()
        self.session.rollback()
        
#         c, added = add_or_get2(self.session, c)
#         
#         seg = models.Segment(start_time=datetime.datetime.utcnow(),
#                              end_time=datetime.datetime.utcnow(),
#                              event_distance_deg=9,
#                              arrival_time=datetime.datetime.utcnow(),
#                              data=b'')
# 
#         self.session.flush()
#         e.segments.append(seg)
#         self.session.flush()
#         c.segments.append(seg)
#         self.session.flush()


#         id = '__abcdefghilmnopq'
#         utcnow = datetime.datetime.utcnow()
#         e = models.Event(id=id, time=utcnow)
# 
#         e, added = add_or_get2(self.session, e)
#         
#         s = models.Station(network='sdf', station='_', latitude=90, longitude=-45)
#         
#         s, added = add_or_get2(self.session, s, flush=True)
#         
#         c = models.Channel(location= 'tyu', channel='rty')
#         
#         s.channels.append(c)
#         self.session.flush()
#         c, added = add_or_get2(self.session, c)
#         
#         seg = models.Segment(start_time=datetime.datetime.utcnow(),
#                              end_time=datetime.datetime.utcnow(),
#                              event_distance_deg=9,
#                              arrival_time=datetime.datetime.utcnow(),
#                              data=b'')
# 
#         self.session.flush()
#         e.segments.append(seg)
#         self.session.flush()
#         c.segments.append(seg)
#         self.session.flush()
        
        
    def test_harmonize_columns(self):

        id = 'abcdefghilmnopq'
        utcnow = datetime.datetime.utcnow()
        e = models.Event(id=id, time=utcnow)
        df = pd.DataFrame(columns=models.Event.get_col_names(),
                          data=[[None for _ in models.Event.get_col_names()]])
        # add int and bool fields
        df.insert(0, 'int', 1)
        df.insert(0, 'bool', 1)

        colnames, df2 = _harmonize_columns(models.Event, df)

        df2types = df2.dtypes
        # check if a class is datetime is cumbersome innumpy. See here:
        # http://stackoverflow.com/questions/23063362/consistent-way-to-check-if-an-np-array-is-datetime-like
        # so we do:
        assert 'datetime64' in str(df2types[models.Event.time.key])
        # other stuff works fine with normal check:
        assert df2types['int'] == np.int64
        assert df2types['bool'] == np.int64
        assert df2types[models.Event.latitude.key] == np.float64
        assert df2types[models.Event.longitude.key] == np.float64
        assert df2types[models.Event.depth_km.key] == np.float64
        assert df2types[models.Event.magnitude.key] == np.float64
        # assert also other fields are objects (not all of them, just two):
        assert df2types[models.Event.event_location_name.key] == object
        assert df2types[models.Event.author.key] == object
        
        df3 = harmonize_columns(models.Event, df2)[colnames] # this calls _harmonize_columns above
        
        assert len(df3.columns) == len(df.columns)-2
        
        
    def test_context_man(self):
        pass
#     def dontrunthis_pdsql_utils(self): 
#         event_rows = df_to_table_rows(models.Event, df3)
#         
#         assert len(event_rows) == 1
#         
#         df5 = pd.DataFrame(index=[0,1], columns = df3.columns)
#         utcnow1 = datetime.datetime.utcnow()
#         utcnow2 = datetime.datetime.utcnow()
#         df5.loc[0, models.Event.time.key] = utcnow1
#         df5.loc[1, models.Event.time.key] = utcnow2
#         df5.loc[0, models.Event.latitude.key] = '80.5'
#         df5.loc[1, models.Event.id.key] = 'id'
#         
#         event_rows = df_to_table_rows(models.Event, df5)
#         assert len(event_rows) == 2
#         assert event_rows[0].time == utcnow1
#         assert event_rows[1].time == utcnow2
#         assert event_rows[0].latitude == 80.5
#         assert event_rows[1].id == 'id'
        

#         types = pd.Series(index=e2.get_col_names(), data= [pd.object
# time                   datetime64[ns]
# latitude                      float64
# longitude                     float64
# depth_km                      float64
# author                         object
# catalog                        object
# contributor                    object
# contributor_id                 object
# mag_type                       object
# magnitude                     float64
# mag_author                     object
# event_location_name            object
# dtype: object])
        g = 9
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()