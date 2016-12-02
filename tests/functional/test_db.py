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
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from stream2segment.s2sio.db.pd_sql_utils import _harmonize_columns, harmonize_columns,\
    get_or_add_iter
from stream2segment.s2sio.dataseries import dumps_inv, loads_inv
from sqlalchemy.orm.exc import FlushError


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
        except SQLAlchemyError as _:
            pass
            # self.session.rollback()
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

    def test_inventory_io(self):
        from obspy.core.inventory.inventory import Inventory
        e = models.Station(network='abc', station='f')

        parentdir = os.path.dirname(os.path.dirname(__file__))
        invname = os.path.join(parentdir, "data", "inventory_GE.APE.xml")
        with open(invname, 'rb') as opn:
            data = opn.read()

        dumped_inv = dumps_inv(data,  compression='gzip', compresslevel=9)

        assert len(dumped_inv) < len(data)
        e.inventory_xml = dumped_inv

        self.session.commit()
        inv_xml = loads_inv(e.inventory_xml)

        assert isinstance(inv_xml, Inventory)
        
        
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
        e = models.Station(network='abc', station='f')
        assert e.id is None
        self.session.add(e)
        # we do not have specified all non-null fields:
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # now test auto id
        e = models.Station(network='abc', station='f', latitude='89.5', longitude='56')
        assert e.id is None
        self.session.add(e)
        # we do not have specified all non-null fields:
        self.session.commit()
        assert e.id == "abc.f"

        # test unique constraints by changing only network
        sta = models.Station(network='a', station='f', latitude='89.5', longitude='56')
        self.session.add(sta)
        # we do not have specified all non-null fields:
        self.session.commit()
        assert sta.id == "a.f"

        # now re-add it. Unique constraint failed
        sta = models.Station(network='a', station='f', latitude='189.5', longitude='156')
        self.session.add(sta)
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # test stations channels relationship:
        sta = models.Station(network='ax', station='f', latitude='89.5', longitude='56')
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
        e = models.Station(id="a.b", network='a', station='b', latitude=56, longitude=78)
        self.session.add(e)
        self.session.commit()
#         
        df = pd.DataFrame(columns=models.Station.get_col_names(), data=[[None for _ in models.Station.get_col_names()]])
        df.loc[0, 'id'] = id + '.j'
        df.loc[0, 'network'] = id
        df.loc[0, 'station'] = 'j'
        df.loc[0, 'latitude'] = 43
        df.loc[0, 'longitude'] = 56.7
        # df.loc[0, 'datacenter_id'] = dc.id
        
        df.to_sql(e.__table__.name, self.engine, if_exists='append', index=False)
         
        # same id as above, but check that data exist (i.e., error)
        df.loc[0, 'id'] = id
        with pytest.raises(IntegrityError):
            df.to_sql(e.__table__.name, self.engine, if_exists='append', index=False)

    def get_or_add(self, session, model_instance, model_cols_or_colnames=None, on_add='flush'):
        for inst, isnew in get_or_add_iter(session, [model_instance], model_cols_or_colnames,
                                           on_add):
            return inst, isnew

    def test_add_or_get(self):
        id = 'some_id_not_used_elsewhere'
        utcnow = datetime.datetime.utcnow()
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7, magnitude=56)
        e_, new_added = self.get_or_add(self.session, e)
        # we should have flushed, as it is the default argument
        # of the function above. Without flush get_or_add below doesn't work
        assert new_added is True and e_ is not None

        e_, new_added = self.get_or_add(self.session, e)
        assert new_added is False and e_ is not None

        e.latitude = 67.0546  # auto updated, in fact if we do:
        e_, new_added = self.get_or_add(self.session, e, 'latitude')
        # we didn't add any new object
        assert new_added is False and e_ is not None and e_.latitude == e.latitude

        # now re-initialize e WITH THE SAME ARGUMENT AS ABOVE AT THE BEGINNING:
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7, magnitude=56)
        # and do NOT flush by default
        e_, new_added = self.get_or_add(self.session, e, 'latitude', on_add=None)
        assert new_added is True and e_ is not None
        # However, now we should get an error cause we added two elements with same id:
        with pytest.raises(IntegrityError):
            self.session.flush()
            self.session.commit()
        self.session.rollback()


    def test_event_sta_channel_seg(self):
        dc= models.DataCenter(station_query_url="345635434246354765879685432efbdfnrhytwfesdvfbgfnyhtgrefs")
        self.session.add(dc)

        utcnow = datetime.datetime.utcnow()

        run = models.Run(run_time=utcnow)
        self.session.add(run)

        id = '__abcdefghilmnopq'
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        self.session.add(e)

        s = models.Station(network='sdf', station='_', latitude=90, longitude=-45)
        self.session.add(s)

        c = models.Channel(location= 'tyu', channel='rty', sample_rate=6)
        s.channels.append(c)
        self.session.commit()
        
        id = '__abcdefghilmnopq'
        utcnow = datetime.datetime.utcnow()
        e = models.Event(id=id, time=utcnow)
        e, added = self.get_or_add(self.session, e)
        assert added == False

        s = models.Station(network='sdf', station='_', latitude=90, longitude=-45)
        s, added = self.get_or_add(self.session, s, 'network', 'station')
        assert added == False
        
        self.session.commit()  # harmless
        
#         c, added = add_or_get2(self.session, c)
#
        seg = models.Segment(start_time=datetime.datetime.utcnow(),
                             end_time=datetime.datetime.utcnow(),
                             event_distance_deg=9,
                             arrival_time=datetime.datetime.utcnow(),
                             data=b'')

        self.session.add(seg)

        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()
        
        # set necessary attributes
        seg.event_id = e.id
        seg.datacenter_id = dc.id
        seg.run_id = run.id
        seg.channel_id = c.id
        # and now it will work:
        self.session.add(seg)
        self.session.commit()
        
        # Create a copy of the instance, with same value. We should excpect a UniqueConstraint
        # but we actually get a FlushErro (FIXME: check difference). Anyway the exception is:
        # FlushError: New instance <Segment at 0x11295cf10> with identity key
        # (<class 'stream2segment.s2sio.db.models.Segment'>, (1,)) conflicts with persistent
        # instance <Segment at 0x11289bb50>
        seg_ = seg.copy()
        self.session.add(seg_)
        with pytest.raises(FlushError):
            self.session.commit()
        self.session.rollback()
        
        # It seems that qlalchemy keeps track
        # of the instance object linked to a db row.
        # Take the seg object (already added and committed) change ITS id to force even more
        # the "malformed" case, but we won't have any exceptions
        s1 = self.session.query(models.Segment).all()
        seg.id += 1
        self.session.add(seg)
        self.session.commit()
        s2 = self.session.query(models.Segment).all()
        assert len(s1) == len(s2)
        
        
        # anyway, now add a new segment
        seg_ = seg.copy()
        seg_.id = None  # autoincremented
        seg_.end_time += datetime.timedelta(seconds=1) # safe unique constraints
        self.session.add(seg_)
        self.session.commit()
        assert len(self.session.query(models.Segment).all()) == 2


        # select segments which do not have processings (all)
        s = self.session.query(models.Segment).filter(~models.Segment.processings.any()).all()  # @UndefinedVariable
        assert len(s) == 2
        
        seg1, seg2 = s[0], s[1]
        pro = models.Processing(run_id=run.id)
        pro.segment = seg1
        assert pro.segment_id != seg1.id
        self.session.flush()  # updates id or foreign keys related to the relation above
        assert pro.segment_id == seg1.id

        # check changing segment and segment id and see if the other gets updated
        
        # what happens if we change segment_id?
#         pro.segment_id = seg2.id
#         assert pro.segment_id != seg2.id and pro.segment.id == seg1.id
#         self.session.flush()
#         self.session.commit()
# #         d=9
#         
#         # this does not raise error, it just updates seg.channel_id
#         seg = models.Segment()
#         seg.channel = c
#         self.session.flush()
#         assert seg.channel_id == c.id
#         # we would have troubles doing this (there are null fields in seg):
#         with pytest.raises(IntegrityError):
#             self.session.commit()
#         self.session.rollback()
#         f = 9
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
        
    def test_toattrdict(self):
#         e = models.Event(id='abc')
#         
#         s = models.Station(network='n', station='s')
        
        c = models.Channel(location='l', channel='c')
#         c.station = s
        
        adict = c.toattrdict()
        assert 'id' not in adict and 'station_id' not in adict
        
        adict = c.toattrdict(primary_keys=True)
        assert 'id' in adict and 'station_id' not in adict
        
        adict = c.toattrdict(foreign_keys=True)
        assert 'id' not in adict and 'station_id' in adict
        
        adict = c.toattrdict(primary_keys=True, foreign_keys=True)
        assert 'id' in adict and 'station_id' in adict
        
        
        
    def test_harmonize_columns(self):

        id = 'abcdefghilmnopq'
        utcnow = datetime.datetime.utcnow()

        df = pd.DataFrame(columns=models.Event.get_col_names(),
                          data=[[None for _ in models.Event.get_col_names()]])


        # add a column which is NOT on the table:
        colx = 'iassdvgdhrnjynhnt_________'
        df.insert(0, colx, 1)

        colnames, df2 = _harmonize_columns(models.Event, df)

        # colx is not part of the Event model:
        assert colx not in colnames

        # df2 has been modified in place actually:
        assert (df.dtypes == df2.dtypes).all()

        df2types = df2.dtypes
        # checking if a class is datetime is cumbersome innumpy. See here:
        # http://stackoverflow.com/questions/23063362/consistent-way-to-check-if-an-np-array-is-datetime-like
        # so we do:
        assert 'datetime64' in str(df2types[models.Event.time.key])
        # other stuff works fine with normal check:
        assert df2types[models.Event.latitude.key] == np.float64
        assert df2types[models.Event.longitude.key] == np.float64
        assert df2types[models.Event.depth_km.key] == np.float64
        assert df2types[models.Event.magnitude.key] == np.float64
        # assert also other fields are objects (not all of them, just two):
        assert df2types[models.Event.event_location_name.key] == object
        assert df2types[models.Event.author.key] == object
        
        
        df3 = harmonize_columns(models.Event, df2)[colnames] # this calls _harmonize_columns above
        
        assert colx not in df3.columns
        
        
        
        # now try to see with invalid values for floats
        dfx = pd.DataFrame(columns=models.Event.get_col_names(),
                          data=[["a" for _ in models.Event.get_col_names()]])
        
        _harmonize_columns(models.Event, dfx)
        
        # df2 and dfx should have the same dtypes:
        assert (dfx.dtypes == df2[colnames].dtypes).all()
        
        # fast check: datetimes and a float field
        assert pd.isnull(dfx.loc[0, models.Event.time.key])
        assert pd.isnull(dfx.loc[0, models.Event.longitude.key])
        
        
        dfx = pd.DataFrame(columns=models.Event.get_col_names(),
                          data=[["a" for _ in models.Event.get_col_names()]])
        
        dfx.loc[0, models.Event.time.key] = utcnow
        dfx.loc[0, models.Event.latitude.key] = 6.5
        
        _harmonize_columns(models.Event, dfx)
        # fast check: datetimes and a float field
        assert pd.notnull(dfx.loc[0, models.Event.time.key])
        assert pd.isnull(dfx.loc[0, models.Event.longitude.key])
        assert pd.notnull(dfx.loc[0, models.Event.latitude.key])
        
        
        
        
        
        g = 9
        
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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()