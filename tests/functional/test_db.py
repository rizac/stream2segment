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
from stream2segment.process.utils import dcname

class Test(unittest.TestCase):
    
    engine = None

    @classmethod
    def setUpClass(cls):
        file = os.path.dirname(__file__)
        filedata = os.path.join(file,"..","data")
        url = os.path.join(filedata, "_test.sqlite")
        cls.dbfile = url
        cls.deletefile()
        
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        cls.engine = create_engine('sqlite:///'+url)
        Base.metadata.drop_all(cls.engine)
        Base.metadata.create_all(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.deletefile()
        Base.metadata.drop_all(cls.engine)
    
    @classmethod
    def deletefile(cls):
        if os.path.isfile(cls.dbfile):
            os.remove(cls.dbfile)

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

    def test_run(self):
        r = models.Run()
        assert r.run_time is None and r.warnings is None and r.errors is None and r.id is None
        self.session.add(r)
        self.session.commit()
        assert r.run_time is not None and r.warnings == 0 and r.errors == 0 and r.id is not None

    def test_inventory_io(self):
        from obspy.core.inventory.inventory import Inventory
        e = models.Station(network='abcwerwre', station='gwdgafsf',
                           datacenter_id=self.session.query(models.DataCenter)[0].id,
                           latitude=3,
                           longitude=3)

        parentdir = os.path.dirname(os.path.dirname(__file__))
        invname = os.path.join(parentdir, "data", "inventory_GE.APE.xml")
        with open(invname, 'rb') as opn:
            data = opn.read()

        dumped_inv = dumps_inv(data,  compression='gzip', compresslevel=9)

        assert len(dumped_inv) < len(data)
        e.inventory_xml = dumped_inv

        self.session.add(e)
        self.session.commit()
        inv_xml = loads_inv(e.inventory_xml)

        assert isinstance(inv_xml, Inventory)
        
        inv_count = self.session.query(models.Station).filter(models.Station.inventory_xml != None).count()
        stationsc = self.session.query(models.Station).count()
        
        # test what happens deleting it:
        ret = self.session.query(models.Station).\
            filter(models.Station.inventory_xml!=None).\
            update({models.Station.inventory_xml: None})
        assert ret == inv_count
        
        self.session.commit()
        # assert we did not delete stations, but also their inventories:
        assert self.session.query(models.Station).count() == stationsc
        
        # test what happens deleting it (DANGER: WE ARE NOT DELETING ONLY invenotry_xml, SEE BELOW):
        ret = self.session.query(models.Station.inventory_xml).delete()
        assert ret == stationsc
        self.session.commit()
        
        # SHIT< WE DELETED ALL STATIONS IN THE COMMAND ABOVE, NOT ONLY inventory_xml!!
        # now delete only nonnull, should return zero:
        assert self.session.query(models.Station).count() == 0
        

    def test_dcanter_name(self):
        
        dc = models.DataCenter(station_query_url='abc')
        assert dcname(dc) == ''
        
        dc = models.DataCenter(station_query_url='http://www.orfeus-eu.org/fdsnws/station/1/query')
        assert dcname(dc) == 'www.orfeus-eu.org'
        
        dc = models.DataCenter(station_query_url='http://eida.ethz.ch/fdsnws/station/1/query')
        assert dcname(dc) == 'eida.ethz.ch'
        
        dc = models.DataCenter(station_query_url='http://geofon.gfz-potsdam.de/fdsnws/station/1/query')
        assert dcname(dc) == 'geofon.gfz-potsdam.de'
        
        
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
        cnames = list(colnames(run_row.__class__))
        assert len(cnames) > 0

        assert object_session(run_row) is not self.session
        # test id is auto added:
        self.session.add_all([run_row])
        # assert we have a session
        assert object_session(run_row) is self.session
        
        # self.session.flush()
        self.session.commit()
        
        # assert we still have a session
        assert object_session(run_row) is self.session
        
        assert run_row.id is not None

        # assert run_row.run_time is not None:
        assert run_row.run_time
        
        # check if we can add a new run_row safely. According to the server_default specified
        # in models.py that would
        # fail cause within the same transaction the db issues always the same value
        # BUT we specified also a python default which should make the trick:
        self.session.add(models.Run())
        # self.session.flush()
        self.session.commit()
        runz = self.session.query(models.Run).all()
        assert len(runz) == 2
        
        # assert the two timestamps are equal cause issued within the same session:
        # (https://www.ibm.com/developerworks/community/blogs/SQLTips4DB2LUW/entry/current_timestamp?lang=en)
        assert runz[0].run_time == runz[1].run_time
        
        
        # now pass a utcdatetime and see if we keep that value:
        utcnow = datetime.utcnow()
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
        e = models.Event(id='abc', time=datetime.utcnow(),
                         latitude=val, longitude=78, magnitude=56, depth_km=45)
        assert e.latitude != float(val)
        assert object_session(e) == None
        self.session.add(e)
        assert object_session(e) is self.session
        self.session.commit()
        assert e.latitude == float(val)

        # create a datacenter WITHOUT the two fields stations and dataselect
        dc = models.DataCenter(station_query_url='abc')
        with pytest.raises(IntegrityError):
            self.session.add(dc)
            self.session.flush()
            
        self.session.rollback()
        
        # now add it properly:
        dc = models.DataCenter(station_query_url='abc', dataselect_query_url='edf')
        self.session.add(dc)
        self.session.commit()

        # test stations auto id (concat):
        # first test non-specified non-null fields datacenter_id (should reaise an IntegrityError)
        e = models.Station(network='abc', station='f')
        assert e.id == "abc.f"
        self.session.add(e)
        # we do not have specified all non-null fields:
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # now test auto id
        e = models.Station(network='abc', datacenter_id=dc.id, station='f', latitude='89.5', longitude='56')
        assert e.id == "abc.f"
        self.session.add(e)
        # we do not have specified all non-null fields:
        self.session.commit()
        assert e.id == "abc.f"

        # test unique constraints by changing only network
        sta = models.Station(network='a', datacenter_id=dc.id, station='f', latitude='89.5', longitude='56')
        self.session.add(sta)
        # we do not have specified all non-null fields:
        self.session.commit()
        assert sta.id == "a.f"

        # now re-add it. Unique constraint failed
        sta = models.Station(network='a', datacenter_id=dc.id, station='f', latitude='189.5', longitude='156')
        self.session.add(sta)
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # test stations channels relationship:
        sta = models.Station(network='ax', datacenter_id=dc.id, station='f', latitude='89.5', longitude='56')
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
        e = models.Event(time = datetime.utcnow(), latitude = 6, longitude=8, depth_km=6,
                         magnitude=56)
        events = len(self.session.query(models.Event).all())
        
        self.session.add(e)
        with pytest.raises(IntegrityError):
            self.session.flush()
        self.session.rollback()  # necessary for the query below (all()). Try to comment and see
        assert len(self.session.query(models.Event).all()) == events
        
        # single flush after two add: rollback rolls back BOTH
        e2 = models.Event(id = 'Column(String, primary_key=True', 
                         time = datetime.utcnow(), latitude = 6, longitude=8, depth_km=6,
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
        dc = models.DataCenter(station_query_url='awergedfbvdbfnhfsnsbstndggf ',
                               dataselect_query_url='edf')
        self.session.add(dc)
        self.session.commit()
        
        id = 'abcdefghilmnopq'
        utcnow = datetime.utcnow()
        e = models.Station(id="a.b", network='a', datacenter_id=dc.id, station='b', latitude=56, longitude=78)
        self.session.add(e)
        self.session.commit()

        stacolnames = list(colnames(models.Station))
        df = pd.DataFrame(columns=stacolnames, data=[[None for _ in stacolnames]])
        df.loc[0, 'id'] = id + '.j'
        df.loc[0, 'network'] = id
        df.loc[0, 'datacenter_id'] = dc.id
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
        utcnow = datetime.utcnow()
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
        dc= models.DataCenter(station_query_url="345fbgfnyhtgrefs", dataselect_query_url='edfawrefdc')
        self.session.add(dc)

        utcnow = datetime.utcnow()

        run = models.Run(run_time=utcnow)
        self.session.add(run)

        id = '__abcdefghilmnopq'
        e = models.Event(id=id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        self.session.add(e)
        
        self.session.commit()  # refresh datacenter id (alo flush works)

        s = models.Station(network='sdf', datacenter_id=dc.id, station='_', latitude=90, longitude=-45)
        self.session.add(s)

        c = models.Channel(location= 'tyu', channel='rty', sample_rate=6)
        s.channels.append(c)
        self.session.commit()
        
        id = '__abcdefghilmnopq'
        utcnow = datetime.utcnow()
        e = models.Event(id=id, time=utcnow)
        e, added = self.get_or_add(self.session, e)
        assert added == False

        s = models.Station(network='sdf', datacenter_id=dc.id, station='_', latitude=90, longitude=-45)
        s, added = self.get_or_add(self.session, s, 'network', 'station')
        assert added == False
        
        self.session.commit()  # harmless
        
#         c, added = add_or_get2(self.session, c)
#
        seg = models.Segment(start_time=datetime.utcnow(),
                             end_time=datetime.utcnow(),
                             event_distance_deg=9,
                             arrival_time=datetime.utcnow(),
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

        # let's do some assertion about relationship segment <-> station
        cha__ = self.session.query(models.Channel).filter(models.Channel.id==seg.channel_id).first()
        sta__ = self.session.query(models.Station).filter(models.Station.id==cha__.station_id).first()
        assert seg.station.id == sta__.id        
        segs__ = sta__.segments.all()
        assert len(segs__)==1 and segs__[0].id == seg.id


        # test segment-class associations:
        clabel1 = models.Class(label='class1')
        clabel2 = models.Class(label='class2')
        self.session.add_all([clabel1, clabel2])
        self.session.commit()
        
        # FUNCTION TO CREATE A DEEPCOPY OF AN INSTANCE
        # NOTE: THE FUNCTION BELOW WAS A BOUND METHOD TO THE Base Class.
        # (we kept the self argument for that reason) AND COPIES 'COLUMNS' (including primary keys)
        # BUT ALSO RELATIONSHIPS (everything that is an InstrumentedAtrribute)
        # This method might be handy but needs more investigation especially on two subjects:
        # what detached elements, and relationships
        # (http://stackoverflow.com/questions/20112850/sqlalchemy-clone-table-row-with-relations?lq=1
        #  http://stackoverflow.com/questions/14636192/sqlalchemy-modification-of-detached-object)
        # so let's move it here for the moment:
        def COPY(self):
            cls = self.__class__
            mapper = inspect(cls)
            # http://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper.mapped_table
            table = mapper.mapped_table
            return cls(**{c: getattr(self, c) for c in mapper.columns.keys()})

        seg__1 = COPY(seg)
        # make it unique
        seg__1.id=None
        seg__1.end_time += timedelta(seconds=1)
        seg__2 = COPY(seg)
        # make it unique
        seg__2.id=None
        seg__2.end_time += timedelta(seconds=2)
        
        self.session.add_all([seg__1, seg__2])
        self.session.commit()
        
        labelings = [models.ClassLabelling(class_id=clabel1.id, segment_id=seg__1.id),
               models.ClassLabelling(class_id=clabel2.id, segment_id=seg__1.id),
               models.ClassLabelling(class_id=clabel1.id, segment_id=seg__2.id)
        ]
        self.session.add_all(labelings)
        self.session.commit()

        assert not seg.classes.all()
        assert len(seg__1.classes.all()) == 2
        assert len(seg__2.classes.all()) == 1
        
        # test on update and on delete:
        old_labellings = sorted([labelings[0].class_id, labelings[1].class_id])
        assert sorted(c.id for c in seg__1.classes.all()) == old_labellings
        # NOTE: DOING THIS WITH SQLITE MSUT HAVE PRAGMA foreign key = ON issued
        #THIS IS DONE BY DEFAULT IN models.py, BUT WATCH OUT!!:
        old_clabel1_id = clabel1.id
        clabel1.id=56
        self.session.commit()
        # this still equal (sqlalachemy updated also labellings)
        assert sorted(c.id for c in seg__1.classes.all()) == sorted([labelings[0].class_id, labelings[1].class_id])
        assert sorted(c.id for c in seg__1.classes.all()) != old_labellings
        assert 56 in [c.id for c in seg__1.classes.all()]
        assert old_clabel1_id not in [c.id for c in seg__1.classes.all()]

        # Create a copy of the instance, with same value.
        # We should excpect a UniqueConstraint
        # but we actually get a FlushErro (FIXME: check difference). Anyway the exception is:
        # FlushError: New instance <Segment at 0x11295cf10> with identity key
        # (<class 'stream2segment.s2sio.db.models.Segment'>, (1,)) conflicts with persistent
        # instance <Segment at 0x11289bb50>
        # See this:
        # http://stackoverflow.com/questions/14636192/sqlalchemy-modification-of-detached-object
        k = str(seg)
        seg_ = COPY(seg)  # seg.copy()
        self.session.add(seg_)
        with pytest.raises(FlushError):
            self.session.commit()
        self.session.rollback()

        # keep track of all segments added:
        s1 = self.session.query(models.Segment).all()

        # Add a new segment. We cannot do like this
        # (http://stackoverflow.com/questions/14636192/sqlalchemy-modification-of-detached-object):
        seg.id += 1110
        seg.start_time += timedelta(seconds=-25)  # this is just to avoid unique constraints
        self.session.add(seg)
        self.session.commit()
        # a new segment wasn't added. Try:
        s2 = self.session.query(models.Segment).all()
        assert len(s1) == len(s2)
    

        # Ok go on: now add a new segment. Use the hack of COPY, but there are better ways
        # (see link above)
        seg_ = COPY(seg)
        seg_.id = None  # autoincremented
        seg_.end_time += timedelta(seconds=1) # safe unique constraints
        self.session.add(seg_)
        self.session.commit()
        assert len(self.session.query(models.Segment).all()) == len(s1)+1


        # select segments which do not have processings (all)
#         s = self.session.query(models.Segment).filter(~models.Segment.processings.any()).all()  # @UndefinedVariable
#         assert len(s) == 2

#         seg1, seg2 = s[0], s[1]
#         pro = models.Processing(run_id=run.id)
#         pro.segment = seg1
#         assert pro.segment_id != seg1.id
#         self.session.flush()  # updates id or foreign keys related to the relation above
#         assert pro.segment_id == seg1.id

        self.tst_get_cols(seg)


    def tst_get_cols(self, seg):
        
        clen = len(seg.__class__.__table__.columns)
        
        cols = seg.__table__.columns
        c = list(colnames(seg.__class__))  # or models.Segment
        assert len(c) == clen

        c = list(colnames(seg.__class__, pkey=False))
        assert len(c) == clen - 1

        c = list(colnames(seg.__class__, pkey=True))
        assert len(c) == 1

        c = list(colnames(seg.__class__, fkey=False))
        assert len(c) == clen - 4

        c = list(colnames(seg.__class__, fkey=True))
        assert len(c) == 4

        c = list(colnames(seg.__class__, nullable=True))
        assert len(c) == 1

        c = list(colnames(seg.__class__, nullable=False))
        assert len(c) == clen-1

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
        
#     def test_toattrdict(self):
#         
#         c = models.Channel(location='l', channel='c')
#         
#         adict = c.toattrdict()
#         assert 'id' not in adict and 'station_id' not in adict
#         
#         adict = c.toattrdict(primary_keys=True)
#         assert 'id' in adict and 'station_id' not in adict
#         
#         adict = c.toattrdict(foreign_keys=True)
#         assert 'id' not in adict and 'station_id' in adict
#         
#         adict = c.toattrdict(primary_keys=True, foreign_keys=True)
#         assert 'id' in adict and 'station_id' in adict
        
        
        
    def test_harmonize_columns(self):

        id = 'abcdefghilmnopq'
        utcnow = datetime.utcnow()

        eventcolnames = list(colnames(models.Event))
        df = pd.DataFrame(columns=eventcolnames,
                          data=[[None for _ in eventcolnames]])


        # add a column which is NOT on the table:
        colx = 'iassdvgdhrnjynhnt_________'
        df.insert(0, colx, 1)

        cnames, df2 = _harmonize_columns(models.Event, df)

        # colx is not part of the Event model:
        assert colx not in cnames

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
        
        
        df3 = harmonize_columns(models.Event, df2)[cnames] # this calls _harmonize_columns above
        
        assert colx not in df3.columns
        
        
        
        
        
        # now try to see with invalid values for floats
        evcolnames = list(colnames(models.Event))
        dfx = pd.DataFrame(columns=evcolnames,
                          data=[["a" for _ in evcolnames]])
        
        _harmonize_columns(models.Event, dfx)
        
        # df2 and dfx should have the same dtypes:
        assert (dfx.dtypes == df2[cnames].dtypes).all()
        
        # fast check: datetimes and a float field
        assert pd.isnull(dfx.loc[0, models.Event.time.key])
        assert pd.isnull(dfx.loc[0, models.Event.longitude.key])
        
        # check harmonize rows: invalid rows should be removed (we have 1 invalid row)
        oldlen = len(dfx)
        dfrows = harmonize_rows(models.Event, dfx, inplace=False)
        assert len(dfrows) ==0 and len(dfx) == oldlen
        # check inplace = True
        dfrows = harmonize_rows(models.Event, dfx, inplace=True)
        assert len(dfrows) == len(dfx) == 0
        
        # go on by checking harmonize_columns. FIXME: what are we doing here below?
        dfx = pd.DataFrame(columns=evcolnames,
                          data=[["a" for _ in evcolnames]])
        
        dfx.loc[0, models.Event.time.key] = utcnow
        dfx.loc[0, models.Event.latitude.key] = 6.5
        
        _harmonize_columns(models.Event, dfx)
        # fast check: datetimes and a float field
        assert pd.notnull(dfx.loc[0, models.Event.time.key])
        assert pd.isnull(dfx.loc[0, models.Event.longitude.key])
        assert pd.notnull(dfx.loc[0, models.Event.latitude.key])
        
        
        
        
        
        g = 9
        
    def test_query_get(self):
        # query.get() is special in that it provides direct access to the identity map of the owning
        # Session. If the given primary key identifier is present in the local identity map,
        # the object is returned directly from this collection and no SQL is emitted,
        # unless the object has been marked fully expired. If not present, a SELECT is performed
        # in order to locate the object.
        n = 'tyyugibib'
        s = 'lbibjfd'
        dc= models.DataCenter(station_query_url="345fbg666tgrefs", dataselect_query_url='edfawrptojfh')
        self.session.add(dc)
        assert dc.id == None
        self.session.flush()  # or commit()
        assert dc.id != None
        
        sta = models.Station(station=s, network=n, datacenter_id=dc.id,
                             latitude=5, longitude=9)
        q = self.session.query(models.Station)
        
        staq = q.get(n+"."+s)
        assert staq is None
        self.session.add(sta)
        staq = q.get(n+"."+s)
        assert staq is not None
        
        # works also with tuples, in case of primary key(s)
        staq = q.get((n+"."+s,))
        assert staq is not None
        
        # what if id is None? what does it return? well it should be None right?
        staq = q.get((None,))
        assert staq is None

        
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