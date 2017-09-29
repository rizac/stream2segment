#@PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''
from builtins import str
from builtins import range
import pytest, os
import unittest
import numpy as np
import os
from stream2segment.io.db.models import Base, Event, WebService, Channel, Station, \
    DataCenter, Segment, Class, Download, ClassLabelling, withdata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, load_only
import pandas as pd
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, DataError
from stream2segment.io.db.pd_sql_utils import _harmonize_columns, harmonize_columns, \
    harmonize_rows, colnames, dbquery2df
from stream2segment.io.utils import dumps_inv, loads_inv
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from datetime import datetime, timedelta
from sqlalchemy.orm.session import object_session
from sqlalchemy.sql.expression import func, bindparam
import time

class Test(unittest.TestCase):
    
#     engine = None
# 
#     @classmethod
#     def setUpClass(cls):
#         url = os.getenv("DB_URL", "sqlite:///:memory:")
#         # an Engine, which the Session will use for connection
#         # resources
#         # some_engine = create_engine('postgresql://scott:tiger@localhost/')
#         cls.engine = create_engine(url)
#         Base.metadata.drop_all(cls.engine)
#         Base.metadata.create_all(cls.engine)
#         
# #         file = os.path.dirname(__file__)
# #         filedata = os.path.join(file,"..","data")
# #         url = os.path.join(filedata, "_test.sqlite")
# #         cls.dbfile = url
# #         cls.deletefile()
# #         
# #         # an Engine, which the Session will use for connection
# #         # resources
# #         # some_engine = create_engine('postgresql://scott:tiger@localhost/')
# #         cls.engine = create_engine('sqlite:///'+url)
# #         Base.metadata.drop_all(cls.engine)
# #         Base.metadata.create_all(cls.engine)
# 
#     @classmethod
#     def tearDownClass(cls):
#         # cls.deletefile()
#         Base.metadata.drop_all(cls.engine)
#     
# #     @classmethod
# #     def deletefile(cls):
# #         if os.path.isfile(cls.dbfile):
# #             os.remove(cls.dbfile)
# 
#     def setUp(self):
#         
#         # create a configured "Session" class
#         Session = sessionmaker(bind=self.engine)
#         # create a Session
#         self.session = Session()
# 
#     def tearDown(self):
#         try:
#             self.session.flush()
#             self.session.commit()
#         except SQLAlchemyError as _:
#             pass
#             # self.session.rollback()
#         self.session.close()
#         # self.DB.drop_all()

    def setUp(self):
        url = os.getenv("DB_URL", "sqlite:///:memory:")
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        self.engine = create_engine(url)
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

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
        Base.metadata.drop_all(self.engine)
    
    @property
    def is_sqlite(self):
        return str(self.engine.url).startswith("sqlite:///")
    
    @property
    def is_postgres(self):
        return str(self.engine.url).startswith("postgresql://")
    
#     def testParseCol(self):
#         e = Event()
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
        r = Download()
        assert r.run_time is None and r.warnings is None and r.errors is None and r.id is None
        self.session.add(r)
        self.session.commit()
        assert r.run_time is not None and r.warnings == 0 and r.errors == 0 and r.id is not None

    def test_inventory_io(self):
        
        dc = DataCenter(station_url='http://www.orfeus-eu.org/fdsnws/station/1/query')
        self.session.add(dc)
        self.session.commit()
        
        from obspy.core.inventory.inventory import Inventory
        e = Station(network='abcwerwre', station='gwdgafsf',
                           datacenter_id=dc.id,
                           latitude=3,
                           longitude=3, start_time=datetime.utcnow())

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
        
        inv_count = self.session.query(Station).filter(Station.inventory_xml != None).count()
        stationsc = self.session.query(Station).count()
        
        # test what happens deleting it:
        ret = self.session.query(Station).\
            filter(Station.inventory_xml!=None).\
            update({Station.inventory_xml: None})
        assert ret == inv_count
        
        self.session.commit()
        # assert we did not delete stations, but also their inventories:
        assert self.session.query(Station).count() == stationsc
        
        # test what happens deleting it (DANGER: WE ARE NOT DELETING ONLY invenotry_xml, SEE BELOW):
        ret = self.session.query(Station.inventory_xml).delete()
        assert ret == stationsc
        self.session.commit()
        
        # SHIT< WE DELETED ALL STATIONS IN THE COMMAND ABOVE, NOT ONLY inventory_xml!!
        # now delete only nonnull, should return zero:
        assert self.session.query(Station).count() == 0
        

    def test_dcanter_name(self):
        
        dc = DataCenter(station_url='abc')
        assert dc.netloc == 'abc'
        
        dc = DataCenter(station_url='http://www.orfeus-eu.org/fdsnws/station/1/query')
        assert dc.netloc == 'www.orfeus-eu.org'
        
        dc = DataCenter(station_url='http://eida.ethz.ch/fdsnws/station/1/query')
        assert dc.netloc == 'eida.ethz.ch'
        
        dc = DataCenter(station_url='http://geofon.gfz-potsdam.de/fdsnws/station/1/query')
        assert dc.netloc == 'geofon.gfz-potsdam.de'
    
    # Note that the naming here RUNS this test FIRST!!!
    def test_010_difference_supplying_autoinc_id(self):
        ID  =1 # put a very high number to be sure is unique
        # add a webservice specifying the id (which is autocincrement)
        ws = WebService(id=ID, url = 'asd_________')  
        self.session.add(ws)
        self.session.commit()
        
        # that's REALLY IMPORTANT!:
        # we added above a Webservice row SPECIFYING THE id explicitly!
        # from https://stackoverflow.com/questions/40280158/postgres-sqlalchemy-auto-increment-not-working:
        # With Postgres if you happen to supply the id field when you insert a new record,
        # the sequence of the table is not used. Upon further insertion if you don't specify the id,
        # the sequence table is not used and hence you have duplication.
        # For details see also:
        # https://stackoverflow.com/questions/37970743/unique-violation-7-error-duplicate-key-value-violates-unique-constraint-users/37972960#37972960
        if not self.is_postgres:
            ws = WebService(url='webservicASDKeurl')
            self.session.add(ws)
            self.session.commit()
        else:
            with pytest.raises(IntegrityError):
                ws = WebService(url='webservicASDKeurl')
                self.session.add(ws)
                self.session.commit()
            self.session.rollback()
            # NOW THIS WORKS:
            ws = WebService(url='webservicASDKeurl', id=ID+1)
            self.session.add(ws)
            self.session.commit()
            
        self.session.query(WebService).delete()
        assert not self.session.query(WebService).all()
        
    def testSqlAlchemy(self):
        #run_cols = Download.__table__.columns.keys()
        #run_cols.remove('id')  # remove id (auto set in the model)
        #d = pd.DataFrame(columns=run_cols, data=[[None for _ in run_cols]])
        #records = d.to_dict('records') # records(index=False)
        # record = records[0]

        # pass a run_id without id and see if it's updated as utcnow:
        run_row = Download()
        assert run_row.id is None

        run_row = Download(id=None)
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
        # in py that would
        # fail cause within the same transaction the db issues always the same value
        # BUT we specified also a python default which should make the trick:
        self.session.add(Download())
        # self.session.flush()
        self.session.commit()
        runz = self.session.query(Download).all()
        assert len(runz) == 2
        
        # assert the two timestamps are equal cause issued within the same session:
        # (https://www.ibm.com/developerworks/community/blogs/SQLTips4DB2LUW/entry/current_timestamp?lang=en)
        # this is true in sqlite!!!
        if self.is_sqlite:
            assert runz[0].run_time == runz[1].run_time
        else:
            assert runz[0].run_time <= runz[1].run_time
        
        # now pass a utcdatetime and see if we keep that value:
        utcnow = datetime.utcnow()
        run_row = Download(run_time=utcnow)
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
        e = Class(id=val)
        assert e.id != int(val)
        self.session.add(e)
        self.session.commit()
        assert e.id == int(val)

        # test types. string floats are parsed automatically as int? YES if INT in sqlite
        # so this is IntegrityError in sqlite (for what we just stated) and No on postgres (but raises a DataError instead)
        val = '5.2'
        e = Class(id=val)
        assert e.id != float(val)
        self.session.add(e)

        error = IntegrityError if self.is_sqlite else DataError
        with pytest.raises(error):
            self.session.commit()
        # necessary after failure? FIXME: check!
        self.session.rollback()
            
        # this is YES in sqlite, not in postgres:
        val = '5.0'
        e = Class(id=val)
        assert e.id != int(float(val))
        
        if self.is_sqlite:
            self.session.add(e)
            self.session.commit()
            assert e.id == int(float(val))
        else:
            with pytest.raises(DataError):
                self.session.add(e)
                self.session.commit()
            self.session.rollback()

        # test types. String floats are parsed automatically? YES
        
        ws = WebService(url = 'asd')
        self.session.add(ws)
        self.session.commit()
        
        val = '6.7'
        e = Event(eventid='abc', webservice_id=ws.id, time=datetime.utcnow(),
                         latitude=val, longitude=78, magnitude=56, depth_km=45)
        assert e.latitude != float(val)
        assert object_session(e) == None
        self.session.add(e)
        assert object_session(e) is self.session
        self.session.commit()
        assert e.latitude == float(val)

        # create a datacenter WITHOUT the two fields stations and dataselect
        dc = DataCenter()
        with pytest.raises(IntegrityError):
            self.session.add(dc)
            self.session.flush()
            
        self.session.rollback()
        
        # now add it properly:
        dc = DataCenter(station_url='abc', dataselect_url='edf')
        self.session.add(dc)
        self.session.commit()

        # test stations auto id (concat):
        # first test non-specified non-null fields datacenter_id (should reaise an IntegrityError)
        e = Station(network='abc', station='f')
        # assert e.id == "abc.f"
        self.session.add(e)
        # we do not have specified all non-null fields:
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        start_time = datetime.utcnow()
        # now test auto id
        e = Station(network='abc', datacenter_id=dc.id, station='f', latitude='89.5', longitude='56',
                           start_time = start_time)
        assert e.id == None
        self.session.add(e)
        self.session.commit()
        assert e.id is not None

        # test unique constraints by changing only network
        sta = Station(network='a', datacenter_id=dc.id, station='f', latitude='89.5', longitude='56',
                             start_time = start_time)
        self.session.add(sta)
        self.session.commit()
        assert sta.id != e.id
        # now re-add it. Unique constraint failed
        sta = Station(network='a', datacenter_id=dc.id, station='f', latitude='189.5', longitude='156',
                             start_time = start_time)
        self.session.add(sta)
        with pytest.raises(IntegrityError):
            self.session.commit()
        self.session.rollback()

        # test stations channels relationship:
        sta = Station(network='ax', datacenter_id=dc.id, station='f', latitude='89.5', longitude='56',
                             start_time = start_time)
        # write channels WITHOUT foreign key
        cha1 = Channel(location='l', channel='HHZ', sample_rate=56)
        cha2 = Channel(location='l', channel='HHN', sample_rate=12)
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
        sta = self.session.query(Station).filter((Station.network == 'a') &
                                                        (Station.station == 'f')).first()
        cha1 = Channel(location='l2', channel='HHZ', sample_rate=56)
        cha2 = Channel(location='l', channel='HHW', sample_rate=56)
        sta.channels.append(cha1)
        sta.channels.append(cha2)
        assert cha1.station_id != sta.id
        assert cha2.station_id != sta.id
        self.session.flush()
        # foreign keys are auto updated!!! TEST IT:
        assert cha1.station_id == sta.id
        assert cha2.station_id == sta.id
        
        k = self.session.query(Event).all()
        assert len(k) == 1

            

#     def get_or_add(self, session, model_instance, model_cols_or_colnames=None, on_add='flush'):
#         for inst, isnew in get_or_add_iter(session, [model_instance], model_cols_or_colnames,
#                                            on_add):
#             return inst, isnew

#     def test_add_or_get(self):
#         id = 'some_id_not_used_elsewhere'
#         utcnow = datetime.utcnow()
#         e = Event(id=id, time=utcnow, latitude=89.5, longitude=6,
#                          depth_km=7, magnitude=56)
#         e_, new_added = self.get_or_add(self.session, e)
#         # we should have flushed, as it is the default argument
#         # of the function above. Without flush get_or_add below doesn't work
#         assert new_added is True and e_ is not None
# 
#         e_, new_added = self.get_or_add(self.session, e)
#         assert new_added is False and e_ is not None
# 
#         e.latitude = 67.0546  # auto updated, in fact if we do:
#         e_, new_added = self.get_or_add(self.session, e, 'latitude')
#         # we didn't add any new object
#         assert new_added is False and e_ is not None and e_.latitude == e.latitude
# 
#         # now re-initialize e WITH THE SAME ARGUMENT AS ABOVE AT THE BEGINNING:
#         e = Event(id=id, time=utcnow, latitude=89.5, longitude=6,
#                          depth_km=7, magnitude=56)
#         # and do NOT flush by default
#         e_, new_added = self.get_or_add(self.session, e, 'latitude', on_add=None)
#         assert new_added is True and e_ is not None
#         # However, now we should get an error cause we added two elements with same id:
#         with pytest.raises(IntegrityError):
#             self.session.flush()
#             self.session.commit()
#         self.session.rollback()


    def test_event_sta_channel_seg(self):
        dc= DataCenter(station_url="345fbgfnyhtgrefs", dataselect_url='edfawrefdc')
        self.session.add(dc)

        utcnow = datetime.utcnow()

        run = Download(run_time=utcnow)
        self.session.add(run)
        
        ws = WebService(url='webserviceurl')
        self.session.add(ws)
        self.session.commit()
            
        id = '__abcdefghilmnopq'
        e = Event(eventid=id, webservice_id=ws.id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        self.session.add(e)
        e2 = Event(eventid=id+'5', webservice_id=ws.id, time=utcnow, latitude=49.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        self.session.add(e2)
        e3 = Event(eventid=id+'5_', webservice_id=ws.id, time=utcnow, latitude=49.5, longitude=67,
                         depth_km=7.1, magnitude=56)
        self.session.add(e3)
        e4 = Event(eventid=id+'5_werger', webservice_id=ws.id, time=utcnow, latitude=49.5, longitude=67.6,
                         depth_km=7.1, magnitude=56)
        self.session.add(e4)
        
        
        self.session.commit()  # refresh datacenter id (alo flush works)

        d = datetime.utcnow()
        
        s = Station(network='sdf', datacenter_id=dc.id, station='_', latitude=90, longitude=-45,
                    start_time=d)
        self.session.add(s)

        c = Channel(location= 'tyu', channel='rty', sample_rate=6)
        s.channels.append(c)
        self.session.commit()
        
        id = '__abcdefghilmnopq'
        utcnow = datetime.utcnow()
        e = Event(id=e.id, eventid=id, webservice_id=ws.id, time=utcnow)
#         e, added = self.get_or_add(self.session, e)
#         assert added == False

        with pytest.raises(IntegrityError):
            self.session.add(e)
            self.session.commit()
        self.session.rollback()
        
        s = Station(network='sdf', datacenter_id=dc.id, station='_', latitude=90, longitude=-45,
                    start_time=d)
#         s, added = self.get_or_add(self.session, s, 'network', 'station')
#         assert added == False
        with pytest.raises(IntegrityError):
            self.session.add(s)
            self.session.commit()
        self.session.rollback()
        
        self.session.commit()  # harmless
        
#         c, added = add_or_get2(self.session, c)
#
        seg = Segment(start_time=datetime.utcnow(),
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
        seg.download_id = run.id
        seg.channel_id = c.id
        # and now it will work:
        self.session.add(seg)
        self.session.commit()


        # let's try something, to see what is faster:
        N = 100
        t = time.time()
        # THIS IS FASTER:
        for i in range(N):
            _ = ".".join((seg.station.network, seg.station.station, seg.channel.location, seg.channel.channel))
        el1 = time.time() - t
        
        # THIS IS SLOWER:
        t = time.time()
        for i in range(N):
            tup = self.session.query(Station.network, Station.station, Channel.location, Channel.channel).select_from(Segment).join(Channel,Station).filter(Segment.id==seg.id).first()
            _ = ".".join(tup)
        el2 = time.time() - t
        
        assert el2 > el1


        # let's do some assertion about relationship segment <-> station
        cha__ = self.session.query(Channel).filter(Channel.id==seg.channel_id).first()
        sta__ = self.session.query(Station).filter(Station.id==cha__.station_id).first()
        assert seg.station.id == sta__.id        
        segs__ = sta__.segments.all()
        assert len(segs__)==1 and segs__[0].id == seg.id


        # test segment-class associations:
        clabel1 = Class(label='class1')
        clabel2 = Class(label='class2')
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
            return cls(**{c: getattr(self, c) for c in list(mapper.columns.keys())})

        seg__1 = COPY(seg)
        # make it unique
        seg__1.id=None
        seg__1.end_time += timedelta(seconds=1)
        seg__1.event_id = e2.id

        seg__2 = COPY(seg)
        # make it unique
        seg__2.id=None
        seg__2.end_time += timedelta(seconds=2)
        seg__2.event_id = e3.id
        
        self.session.add_all([seg__1, seg__2])
        self.session.commit()
        
        labelings = [ClassLabelling(class_id=clabel1.id, segment_id=seg__1.id),
               ClassLabelling(class_id=clabel2.id, segment_id=seg__1.id),
               ClassLabelling(class_id=clabel1.id, segment_id=seg__2.id)
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
        #THIS IS DONE BY DEFAULT IN py, BUT WATCH OUT!!:
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
        # (<class 'stream2segment.s2sio.db.Segment'>, (1,)) conflicts with persistent
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
        s1 = self.session.query(Segment).all()

        # Add a new segment. We cannot do like this
        # (http://stackoverflow.com/questions/14636192/sqlalchemy-modification-of-detached-object):
        seg.id += 1110
        seg.start_time += timedelta(seconds=-25)  # this is just to avoid unique constraints
        self.session.add(seg)
        self.session.commit()
        # a new segment wasn't added. Try:
        s2 = self.session.query(Segment).all()
        assert len(s1) == len(s2)
    

        # Ok go on: now add a new segment. Use the hack of COPY, but there are better ways
        # (see link above)
        seg_ = COPY(seg)
        seg_.id = None  # autoincremented
        seg_.end_time += timedelta(seconds=1) # safe unique constraints
        seg.event_id = e4.id
        self.session.add(seg_)
        self.session.commit()
        assert len(self.session.query(Segment).all()) == len(s1)+1


        # select segments which do not have processings (all)
#         s = self.session.query(Segment).filter(~Segment.processings.any()).all()  # @UndefinedVariable
#         assert len(s) == 2

#         seg1, seg2 = s[0], s[1]
#         pro = Processing(run_id=run.id)
#         pro.segment = seg1
#         assert pro.segment_id != seg1.id
#         self.session.flush()  # updates id or foreign keys related to the relation above
#         assert pro.segment_id == seg1.id

        assert len([x for x in seg.station.segments]) == 4
        assert len([x for x in seg.station.segments.filter(withdata(Segment.data))]) == 0
        assert seg.station.segments.count() == 4
        
        flt = withdata(Segment.data)
        qry = self.session.query(Station).join(Station.segments).\
            filter(flt)
        
        assert len(qry.all()) == 0
        
        seg.data=b'asd'
        self.session.commit()
        
        qry = self.session.query(Station).options(load_only('id')).join(Station.segments).\
            filter(flt)
        
        stationz = qry.all()
        assert len(stationz) == 1
        
        segz = stationz[0].segments.filter(flt).all()
        assert len(segz) == 1
        assert segz[0] is seg
        
        
        qry = self.session.query(Station).filter(Station.segments.any(flt)).all()
        assert len(qry) == 1
        
        self.tst_hybrid_atts()
       
        
        self.tst_get_cols(seg)


    def tst_hybrid_atts(self):
        # when called from the method above, this is the situation now:
        
        # segments:
        #       id has_data
        #  0  1111 False
        #  1  1112 False
        #  2     2 True
        #  3     3 False

        # classes:
        #      id
        #  0    5
        #  1    6
        #  2    8
        #  3   56
        
        # class_labellings:
        #     segment_id  class_id
        #  0           2         8
        #  1           2        56
        #  2           3        56
        
        # test has_data as query argument
        q1 = self.session.query(withdata(Segment.data))
        q2 = self.session.query(Segment.has_data)
        # string are equal except "AS anon_1" which is "has_data"
        assert str(q1) == str(q2).replace('has_data', 'anon_1')
        
        # test has_data as filter argument
        seg1 = self.session.query(Segment.id).filter(Segment.has_data)
        seg2 = self.session.query(Segment.id).filter(withdata(Segment.data))
        assert str(seg1) == str(seg2)
        assert sorted(x[0] for x in seg1.all()) == sorted(x[0] for x in seg2.all())
        assert len(seg1.all()) == 1
        # test hybrid on instances:
        segz = self.session.query(Segment).all()
        assert sum(x.has_data for x in segz) == len(seg1.all())
        
        # test has_inventory
        stas1 = self.session.query(Station.id).filter(Station.has_inventory)
        stas2 = self.session.query(Station.id).filter(withdata(Station.inventory_xml))
        assert str(stas1) == str(stas2)
        assert sorted(x[0] for x in stas1.all()) == sorted(x[0] for x in stas2.all())
        
        # we might test in the future hybrid attributes on relationships
        # REMEMBER: relationships have the .any() and .has() methods which can avoid join,
        # BUT: they might be more time consuming:
        # https://stackoverflow.com/questions/33809066/difference-between-join-and-has-any-in-sqlalchemy
        
        
        # NOTe however that join returns dupes:
        qry1 = sorted([x[0] for x in self.session.query(Segment.id).join(Segment.classes).filter(Segment.has_class()).all()])
        qry2 = sorted([x[0] for x in self.session.query(Segment.id).filter(Segment.has_class()).all()])
        
        assert len(qry1) ==3
        assert len(qry2) == 2

        # we should do like this
        qry1b = sorted([x[0] for x in self.session.query(Segment.id).join(Segment.classes).filter(Segment.has_class()).distinct().all()])
        
        assert len(qry1b) == 2
        assert qry1b == qry2
        
        # Thus: we have basically 3 types of query:
        # stations with data: use query, not hybrid attrs (in download.main)
        # segments with data, stations with inventory data: use hybrid attrs (in process.main)
        # segments with classes: any, none: use query, not hybrid attrs (in gui)
        
    def tst_get_cols(self, seg):
        
        clen = len(seg.__class__.__table__.columns)
        
        cols = seg.__table__.columns
        c = list(colnames(seg.__class__))  # or Segment
        assert len(c) == clen

        c = list(colnames(seg.__class__, pkey=False))
        assert len(c) == clen - 1

        c = list(colnames(seg.__class__, pkey=True))
        assert len(c) == 1

        c = list(colnames(seg.__class__, fkey=False))
        assert len(c) == clen - 4

        c = list(colnames(seg.__class__, fkey=True))
        assert len(c) == 4

        expected_nullables = 5
        c = list(colnames(seg.__class__, nullable=True))
        assert len(c) == expected_nullables

        
        c = list(colnames(seg.__class__, nullable=False))
        assert len(c) == clen -  expected_nullables

        # check changing segment and segment id and see if the other gets updated

        # what happens if we change segment_id?
#         pro.segment_id = seg2.id
#         assert pro.segment_id != seg2.id and pro.segment.id == seg1.id
#         self.session.flush()
#         self.session.commit()
# #         d=9
#         
#         # this does not raise error, it just updates seg.channel_id
#         seg = Segment()
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
#         e = Event(id=id, time=utcnow)
# 
#         e, added = add_or_get2(self.session, e)
#         
#         s = Station(network='sdf', station='_', latitude=90, longitude=-45)
#         
#         s, added = add_or_get2(self.session, s, flush=True)
#         
#         c = Channel(location= 'tyu', channel='rty')
#         
#         s.channels.append(c)
#         self.session.flush()
#         c, added = add_or_get2(self.session, c)
#         
#         seg = Segment(start_time=datetime.datetime.utcnow(),
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
#         c = Channel(location='l', channel='c')
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

        eventcolnames = list(colnames(Event))
        df = pd.DataFrame(columns=eventcolnames,
                          data=[[None for _ in eventcolnames]])


        # add a column which is NOT on the table:
        colx = 'iassdvgdhrnjynhnt_________'
        df.insert(0, colx, 1)

        cnames, df2 = _harmonize_columns(Event, df)

        # colx is not part of the Event model:
        assert colx not in cnames

        # df2 has been modified in place actually:
        assert (df.dtypes == df2.dtypes).all()

        df2types = df2.dtypes
        # checking if a class is datetime is cumbersome innumpy. See here:
        # http://stackoverflow.com/questions/23063362/consistent-way-to-check-if-an-np-array-is-datetime-like
        # so we do:
        assert 'datetime64' in str(df2types[Event.time.key])
        # other stuff works fine with normal check:
        assert df2types[Event.latitude.key] == np.float64
        assert df2types[Event.longitude.key] == np.float64
        assert df2types[Event.depth_km.key] == np.float64
        assert df2types[Event.magnitude.key] == np.float64
        # assert also other fields are objects (not all of them, just two):
        assert df2types[Event.event_location_name.key] == object
        assert df2types[Event.author.key] == object
        assert df2types[Event.id.key] == np.float64
        assert df2types[Event.webservice_id.key] == np.float64
        
        # last two columns where coerced to float cause we had None's. Now try to see
        # if by supplying good values they are coerced to int
        df2bis = df2.copy()
        df2bis.loc[:, Event.id.key] = 64.0
        df2bis.loc[:, Event.webservice_id.key] = 164.0
        cnames, df2bis = _harmonize_columns(Event, df2bis)
        df2types = df2bis.dtypes
        assert df2types[Event.id.key] == np.int64
        assert df2types[Event.webservice_id.key] == np.int64
        
        
        
        df3 = harmonize_columns(Event, df2)[cnames] # this calls _harmonize_columns above
        
        assert colx not in df3.columns
        
        
        
        
        
        # now try to see with invalid values for floats
        evcolnames = list(colnames(Event))
        dfx = pd.DataFrame(columns=evcolnames,
                          data=[["a" for _ in evcolnames]])
        
        _harmonize_columns(Event, dfx)
        
        # df2 and dfx should have the same dtypes:
        assert (dfx.dtypes == df2[cnames].dtypes).all()
        
        # fast check: datetimes and a float field
        assert pd.isnull(dfx.loc[0, Event.time.key])
        assert pd.isnull(dfx.loc[0, Event.longitude.key])
        
        # check harmonize rows: invalid rows should be removed (we have 1 invalid row)
        oldlen = len(dfx)
        dfrows = harmonize_rows(Event, dfx, inplace=False)
        assert len(dfrows) ==0 and len(dfx) == oldlen
        # check inplace = True
        dfrows = harmonize_rows(Event, dfx, inplace=True)
        assert len(dfrows) == len(dfx) == 0
        
        # go on by checking harmonize_columns. FIXME: what are we doing here below?
        dfx = pd.DataFrame(columns=evcolnames,
                          data=[["a" for _ in evcolnames]])
        
        dfx.loc[0, Event.time.key] = utcnow
        dfx.loc[0, Event.latitude.key] = 6.5
        
        _harmonize_columns(Event, dfx)
        # fast check: datetimes and a float field
        assert pd.notnull(dfx.loc[0, Event.time.key])
        assert pd.isnull(dfx.loc[0, Event.longitude.key])
        assert pd.notnull(dfx.loc[0, Event.latitude.key])
        

# ====================================================================================================
# THE TESTS BELOW ARE ACTUALLY TO GET INSIGHTS ON SQLALCHEMY, or they tests code not existing anymore.
# THUS we do not maintain them anymore and WE CALL THEM TST_*


    def tst_query_get(self):
        # query.get() is special in that it provides direct access to the identity map of the owning
        # Session. If the given primary key identifier is present in the local identity map,
        # the object is returned directly from this collection and no SQL is emitted,
        # unless the object has been marked fully expired. If not present, a SELECT is performed
        # in order to locate the object.
        n = 'tyyugibib'
        s = 'lbibjfd'
        dc= DataCenter(station_url="345fbg666tgrefs", dataselect_url='edfawrptojfh')
        self.session.add(dc)
        assert dc.id == None
        self.session.flush()  # or commit()
        assert dc.id != None
        
        id = 1000000
        sta = Station(id=id, station=s, network=n, datacenter_id=dc.id,
                             latitude=5, longitude=9)
        q = self.session.query(Station)
        
        staq = q.get(id)
        assert staq is None
        self.session.add(sta)
        staq = q.get(id)
        assert staq is not None
        
        # works also with tuples, in case of multiple primary key(s)
        staq = q.get((id))
        assert staq is not None
        
        # what if id is None? what does it return? well it should be None right?
        staq = q.get((None,))
        assert staq is None


    def tst_add_and_flush(self):
        # bad event entry (no id):
        e = Event(time = datetime.utcnow(), latitude = 6, longitude=8, depth_km=6,
                         magnitude=56)
        events = len(self.session.query(Event).all())
        
        self.session.add(e)
        with pytest.raises(IntegrityError):
            self.session.flush()
        self.session.rollback()  # necessary for the query below (all()). Try to comment and see
        assert len(self.session.query(Event).all()) == events
        
        # single flush after two add: rollback rolls back BOTH
        e2 = Event(id = 'Column(String, primary_key=True', 
                         time = datetime.utcnow(), latitude = 6, longitude=8, depth_km=6,
                         magnitude=56)
        self.session.add(e)
        self.session.add(e2)
        with pytest.raises(IntegrityError):
            self.session.flush()

        self.session.rollback()  # necessary for the query below (all()). Try to comment and see        
        assert len(self.session.query(Event).all()) == events

        # CORRECT WAY TO DO:
        # two flushes after two adds: rollback rolls back only those that failed:
        self.session.add(e)
        with pytest.raises(IntegrityError):
            self.session.flush()
        self.session.rollback()  # necessary for the query below (all()). Try to comment and see        
        self.session.add(e2)
        
        assert len(self.session.query(Event).all()) == events+1



    # WHAT DO WE DO HERE BELOW??? test some pandas to_sql and then?
    # in any case, this method does not test ANY of our code anymore, so skip it        
    def tst_pd_to_sql(self):
        dc = DataCenter(station_url='awergedfbvdbfnhfsnsbstndggf ',
                               dataselect_url='edf')
        self.session.add(dc)
        self.session.commit()
        
        id = 'abcdefghilmnopq'
        utcnow = datetime.utcnow()
        e = Station(id="a.b", network='a', datacenter_id=dc.id, station='b', latitude=56, longitude=78)
        self.session.add(e)
        self.session.commit()
    
        stacolnames = list(colnames(Station))
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()