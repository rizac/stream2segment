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
from stream2segment.process.utils import enhancesegmentclass

class Test(object):
    
    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init:
        db.reinit(to_file=False)
        
        dc= DataCenter(station_url="345fbgfnyhtgrefs", dataselect_url='edfawrefdc')
        db.session.add(dc)

        utcnow = datetime.utcnow()

        run = Download(run_time=utcnow)
        db.session.add(run)
        
        ws = WebService(url='webserviceurl')
        db.session.add(ws)
        db.session.commit()
            
        id = '__abcdefghilmnopq'
        e = Event(event_id=id, webservice_id=ws.id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        db.session.add(e)
        
        db.session.commit()  # refresh datacenter id (alo flush works)

        d = datetime.utcnow()
        
        s = Station(network='network', station='station', datacenter_id=dc.id, latitude=90, longitude=-45,
                    start_time=d)
        db.session.add(s)

    def test_query4gui(self, db):
        s = db.session.query(Station).first()
        e = db.session.query(Event).first()
        dc = db.session.query(DataCenter).first()
        run = db.session.query(Download).first()

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
        db.session.commit()
        
        
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
        
        db.session.add_all(segments)
        db.session.commit()

        with enhancesegmentclass():
            for leng, segment in zip(expected_lengths, segments):
                # assert the other segments are the expected lengh. Note that leng INCLUDES current
                # segment whereas siblings DOES NOT. So compare to leng-1:
                assert segment.siblings(colname='id').count() == leng-1
                # assert getallcomponents(db.session, segment.id).count() == leng


# if __name__ == "__main__":
#     #import sys;sys.argv = ['', 'Test.testName']
#     unittest.main()