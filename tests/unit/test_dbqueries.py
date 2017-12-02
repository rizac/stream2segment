#@PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''
from builtins import zip
from builtins import str
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
from stream2segment.io.db.pdsql import _harmonize_columns, harmonize_columns, \
    harmonize_rows, colnames, dbquery2df
from stream2segment.io.utils import dumps_inv, loads_inv
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.inspection import inspect
from datetime import datetime, timedelta
from sqlalchemy.orm.session import object_session
from sqlalchemy.sql.expression import func, bindparam
import time

from stream2segment.io.db.queries import getallcomponents

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
        self.initdb()

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
    
    def initdb(self):
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
        
        self.session.commit()  # refresh datacenter id (alo flush works)

        d = datetime.utcnow()
        
        s = Station(network='network', station='station', datacenter_id=dc.id, latitude=90, longitude=-45,
                    start_time=d)
        self.session.add(s)

    def test_query4gui(self):
        s = self.session.query(Station).first()
        e = self.session.query(Event).first()
        dc = self.session.query(DataCenter).first()
        run = self.session.query(Download).first()

        channels = [
            Channel(location= '00', channel='HHE', sample_rate=6),
            Channel(location= '00', channel='HHN', sample_rate=6),
            Channel(location= '00', channel='HHZ', sample_rate=6),
            Channel(location= '00', channel='HHW', sample_rate=6),
            
            Channel(location= '10', channel='HHE', sample_rate=6),
            Channel(location= '10', channel='HHN', sample_rate=6),
            Channel(location= '10', channel='HHZ', sample_rate=6),
            
            Channel(location= '', channel='HHE', sample_rate=6),
            Channel(location= '', channel='HHN', sample_rate=6),
            
            Channel(location= '30', channel='HHZ', sample_rate=6)]
        # expected lengths when querying for gui below. CHANGE THIS
        # IF YOU CHANGE THE PREVIOUS channels VARIABLE
        expected_lengths = [4,4,4,4,3,3,3,2,2,1]
        
        s.channels.extend(channels)
        self.session.commit()
        
        
        args = dict(request_start=datetime.utcnow(),
                     request_end=datetime.utcnow(),
                     event_distance_deg=9,
                     arrival_time=datetime.utcnow(),
                     data=b'',
                     event_id = e.id,
                     datacenter_id = dc.id,
                     download_id = run.id,
                     )
        segments = []
        # and now it will work:
        for c in channels:
            segments.append(Segment(channel_id=c.id, **args))
        
        self.session.add_all(segments)
        self.session.commit()


        for leng, segment in zip(expected_lengths, segments):
            # do query for GUI:
            assert getallcomponents(self.session, segment.id).count() == leng


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()