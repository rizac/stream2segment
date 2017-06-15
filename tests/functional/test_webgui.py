#@PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''
import pytest, os
import unittest
import numpy as np
import os
from io import BytesIO
from stream2segment.io.db.models import Base, Event, WebService, Channel, Station, \
    DataCenter, Segment, Class, Run, ClassLabelling, withdata
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
from sqlalchemy.sql.expression import func, bindparam, and_
import time
from itertools import izip, product
from stream2segment.io.db.queries import getallcomponents
from obspy.core.stream import read
from stream2segment.utils import load_source, yaml_load
from stream2segment.gui.webapp.plotviews import PlotManager, exec_function
from obspy.io.stationtxt.core import all_components
from mock.mock import patch
from stream2segment.gui.webapp import create_app, get_session
import json

class Test(unittest.TestCase):

    def setUp(self):
        self.addCleanup(Test.cleanup, self)
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
        # self.session = Session()
        self.initdb()
        self.pymodule = load_source(os.path.join(os.path.dirname(__file__), '..', '..',
                                                 'stream2segment',
                                                  'resources', 'templates',
                                               'gui.py'))
        self.config = yaml_load(os.path.join(os.path.dirname(__file__), '..', '..',
                                             'stream2segment',
                                                  'resources', 'templates',
                                               'gui.yaml'))
        
        self._app = create_app(url, self.pymodule, self.config)
        
        # app.config.from_object('webapp.config.Testing')
        # db.init_app(app)
        self.app = self._app.test_client()

    @property
    def session(self):
        return get_session()

    @staticmethod
    def cleanup(me):
        if me.engine:
#             if me.session:
#                 try:
#                     me.session.rollback()
#                     me.session.close()
#                 except:
#                     pass
            try:
                Base.metadata.drop_all(me.engine)
            except:
                pass

#     def tearDown(self):
#         try:
#             self.session.flush()
#             self.session.commit()
#         except SQLAlchemyError as _:
#             pass
#             # self.session.rollback()
#         self.session.close()
#         Base.metadata.drop_all(self.engine)
    
    @property
    def is_sqlite(self):
        return str(self.engine.url).startswith("sqlite:///")
    
    @property
    def is_postgres(self):
        return str(self.engine.url).startswith("postgresql://")
    
    def initdb(self):
        # create a configured "Session" class
        Session = sessionmaker(bind=self.engine)
        # create a Session
        session = Session()
        
        dc= DataCenter(station_url="345fbgfnyhtgrefs", dataselect_url='edfawrefdc')
        session.add(dc)

        utcnow = datetime.utcnow()

        run = Run(run_time=utcnow)
        session.add(run)
        
        ws = WebService(url='webserviceurl')
        session.add(ws)
        session.commit()
            
        id = 'firstevent'
        e1 = Event(eventid='event1', webservice_id=ws.id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        e2 = Event(eventid='event2', webservice_id=ws.id, time=utcnow + timedelta(seconds=5),
                  latitude=89.5, longitude=6, depth_km=7.1, magnitude=56)
        
        session.add_all([e1, e2])
        
        session.commit()  # refresh datacenter id (alo flush works)

        d = datetime.utcnow()
        
        s = Station(network='network', station='station', datacenter_id=dc.id, latitude=90,
                    longitude=-45,
                    start_time=d)
        session.add(s)
        
        channels = [
            Channel(location= '01', channel='HHE', sample_rate=6),
            Channel(location= '01', channel='HHN', sample_rate=6),
            Channel(location= '01', channel='HHZ', sample_rate=6),
            Channel(location= '01', channel='HHW', sample_rate=6),
            
            Channel(location= '02', channel='HHE', sample_rate=6),
            Channel(location= '02', channel='HHN', sample_rate=6),
            Channel(location= '02', channel='HHZ', sample_rate=6),
            
            Channel(location= '03', channel='HHE', sample_rate=6),
            Channel(location= '03', channel='HHN', sample_rate=6),
            
            Channel(location= '04', channel='HHZ', sample_rate=6),
            
            Channel(location= '05', channel='HHE', sample_rate=6),
            Channel(location= '05gap_merged', channel='HHN', sample_rate=6),
            Channel(location= '05err', channel='HHZ', sample_rate=6),
            Channel(location= '05gap_unmerged', channel='HHZ', sample_rate=6)
            ]
        
        s.channels.extend(channels)
        session.commit()
        
        fixed_args = dict(datacenter_id = dc.id,
                     run_id = run.id,
                     )
        
        folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        with open(os.path.join(folder, "GE.FLT1..HH?.mseed"), 'rb') as opn:
            data_gaps_unmerged = opn.read()  # unmerged cause we have three traces of different channels
        with open(os.path.join(folder, "IA.BAKI..BHZ.D.2016.004.head"), 'rb') as opn:
            data_gaps_merged = opn.read()
        
        
        obspy_trace = read(BytesIO(data_gaps_unmerged))[0]
        # write data_ok is actually bytes data of 3 traces, write just the first one, we have
        # as it is it would be considered a trace with gaps, wwe have
        # another trace with gaps
        b = BytesIO()
        obspy_trace.write(b, format='MSEED')
        data_ok = b.getvalue()
        data_err = data_ok[:5]  # whatever slice should be ok
             
        for ev, c in product([e1, e2], channels):
            val = int(c.location[:2])
            mseed = data_gaps_merged if "gap_merged" in c.location else \
                data_err if "err" in c.location else data_gaps_unmerged if 'gap_unmerged' in c.location else data_ok
            seg = Segment(start_time = ev.time+timedelta(seconds=val),
                          arrival_time = ev.time+timedelta(seconds=2*val),
                          end_time = ev.time+timedelta(seconds=5*val),
                          data = mseed,
                          seed_identifier = obspy_trace.get_id() if mseed == data_ok else None,
                          event_distance_deg = val,
                          event_id=ev.id,
                          **fixed_args)
            c.segments.append(seg)
        
        session.commit()
        
        session.close()
        
        # set inventory
        with open(os.path.join(folder, "GE.FLT1.xml"), 'rb') as opn:
            self.inventory = loads_inv(opn.read())
    
    def test_root(self):
        with self._app.test_request_context():
            # test first with classes:
            # WHY THIS DOES NOT WORK??!!!
            # the config set on the aopp IS NOT self.config!! why??!!
            #self.config['class_labels'] = {'wtf': 'wtfd'}
            # this on the other hand works:
            self._app.config['CONFIG.YAML']['class_labels'] = {'wtf': 'wtfd'}
            clz = self.session.query(Class).count()
            assert clz == 0
            rv = self.app.get('/')
            expected_str = """var __SETTINGS = {"segment_orderby": ["event.time-", "segment.event_distance_deg"], "segment_select": {"has_data": "true"}, "spectra": {"arrival_time_shift": 0, "signal_window": [0.1, 0.9]}};"""
            assert expected_str in rv.data
            expected_str = """<div ng-show='plots[2].visible' data-plotindex=2 class='plot'></div>
                        
                        <div ng-show='plots[3].visible' data-plotindex=3 class='plot'></div>"""
            clz = self.session.query(Class).all()
            assert len(clz) == 1 and clz[0].label == 'wtf' and clz[0].description == 'wtfd'
            
            # change description:
            self._app.config['CONFIG.YAML']['class_labels'] = {'wtf': 'abc'}
            rv = self.app.get('/')
            expected_str = """var __SETTINGS = {"segment_orderby": ["event.time-", "segment.event_distance_deg"], "segment_select": {"has_data": "true"}, "spectra": {"arrival_time_shift": 0, "signal_window": [0.1, 0.9]}};"""
            assert expected_str in rv.data
            expected_str = """<div ng-show='plots[2].visible' data-plotindex=2 class='plot'></div>
                        
                        <div ng-show='plots[3].visible' data-plotindex=3 class='plot'></div>"""
            clz = self.session.query(Class).all()
            assert len(clz) == 1 and clz[0].label == 'wtf' and clz[0].description == 'abc'
            
            self.session.query(Class).delete()
            self.session.commit()
            clz = self.session.query(Class).count()
            assert clz == 0
            
            # delete entry 'class_labels' and test when not provided
            del self._app.config['CONFIG.YAML']['class_labels'] 
            rv = self.app.get('/')
            expected_str = """var __SETTINGS = {"segment_orderby": ["event.time-", "segment.event_distance_deg"], "segment_select": {"has_data": "true"}, "spectra": {"arrival_time_shift": 0, "signal_window": [0.1, 0.9]}};"""
            assert expected_str in rv.data
            expected_str = """<div ng-show='plots[2].visible' data-plotindex=2 class='plot'></div>
                        
                        <div ng-show='plots[3].visible' data-plotindex=3 class='plot'></div>"""
            clz = self.session.query(Class).count()
            # assert nothing has changed (same as previous assert):
            assert clz == 0

    def test_get_segs(self):
        rv = self.app.post("/get_segments", data=json.dumps(dict(segment_select={'has_data':'true'},
                                           segment_orderby=None, metadata=True, classes=True)),
                           headers={'Content-Type': 'application/json'})
        #    rv = self.app.get("/get_segments")
        data = json.loads(rv.data)
        assert len(data['segment_ids']) == 28
        assert any(x[0] == 'has_data' for x in data['metadata'])
        assert not data['classes']
    
    def test_toggle_class_id(self):
        with self._app.test_request_context():
            segid = 1
            segment = self.session.query(Segment).filter(Segment.id == segid).first()
            c = Class(label='label')
            self.session.add(c)
            self.session.commit()
            cid = c.id
            assert len(segment.classes.all()) == 0
            rv = self.app.post("/toggle_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid}),
                                   headers={'Content-Type': 'application/json'})
            data = json.loads(rv.data)
            
            assert len(segment.classes.all()) == 1
            assert segment.classes.all()[0].id == cid
            
            # toggle again:
            rv = self.app.post("/toggle_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid}),
                                   headers={'Content-Type': 'application/json'})
            assert len(segment.classes.all()) == 0
            
            # toggle again and run test_get_seg with a class set
            rv = self.app.post("/toggle_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid}),
                                   headers={'Content-Type': 'application/json'})
            assert len(segment.classes.all()) == 1
            self._tst_get_seg()
    
    def test_get_seg(self):
        with self._app.test_request_context():
            self._tst_get_seg()
    
    def _tst_get_seg(self):
        # does pytest.mark.parametrize work with unittest?
        # seems not. So:
        has_labellings = self.session.query(ClassLabelling).count() > 0
        for _ in product([[0,1,2], [], [0]], [True, False], [True, False], [True, False], [True, False]):
            plot_indices, filtered, metadata, classes, all_components = _
        
            d = dict(seg_id = 1,
                     filtered = filtered,
                     # zooms = data['zooms']
                     plot_indices = plot_indices,  # data['plotIndices']
                     metadata = metadata,
                     classes=classes,
                     all_components = all_components)
                     #conf = data.get('config', {})
                     #plotmanager = current_app.config['PLOTMANAGER']
    #         if conf:
    #             current_app.config['CONFIG.YAML'].update(conf)
                
            rv = self.app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
            #    rv = self.app.get("/get_segments")
            data = json.loads(rv.data)
            assert len(data['plots']) == len(d['plot_indices'])
            assert bool(len(data['metadata'])) == metadata
            assert bool(len(data['classes'])) == (classes and has_labellings)
            if 0 in plot_indices:
                traces_in_first_plot = len(data['plots'][plot_indices.index(0)][1])
                assert (traces_in_first_plot == 1 and not all_components) or traces_in_first_plot >= 1
            # we should add a a test for the filtered case also, but we should inspect the plotmanager defined as app['PLOTMANAGER'] 
            # we should add a test for the zooms, too

    def test_segment_sa_session_not_expired(self):
        plot_indices = [0]
        metadata = False
        classes = False
        
        with self._app.test_request_context():
            d = dict(seg_id = 1,
                     filtered = False,
                     # zooms = data['zooms']
                     plot_indices = plot_indices,  # data['plotIndices']
                     metadata = metadata,
                     classes=classes,
                     all_components = all_components)
            rv = self.app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
            
            
        # Now we exited the session, we try to load the filtered traces which make use
        # of session.event. we should get anyway an error cause we didn't load any inventory
        with self._app.test_request_context():
            d = dict(seg_id = 1,
                     filtered = True,
                     # zooms = data['zooms']
                     plot_indices = plot_indices,  # data['plotIndices']
                     metadata = metadata,
                     classes=classes,
                     all_components = all_components)
            rv = self.app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
        
        # assert we have exceptions:
        # FIXME: check why we have to access _app for global vars and not app (see setup method)
        pm = self._app.config['PLOTMANAGER']
        filtered_views = pm._fviews
        for view in filtered_views.itervalues():
            data = view.data
            for d in data.itervalues():
                _, plots, _2 = d
                for p in plots:
                    if p is None: # not calculated, skip
                        continue
                    assert "No matching response information found" in p.warning
                    
        # now reset the filtered views:
        pm._fviews = {}
        # set an inventory matching the segments we have:
        # set inventory
        folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        with open(os.path.join(folder, "GE.FLT1.xml"), 'rb') as opn:
            inventory = loads_inv(opn.read())
        # re-set internal dicts:
        newdic = {k: inventory for k in pm._inv_cache}
        pm._inv_cache = newdic
        newdic = {k: inventory for k in pm.segid2inv}
        pm.segid2inv = newdic
        # and now test again
        with self._app.test_request_context():
            d = dict(seg_id = 1,
                     filtered = True,
                     # zooms = data['zooms']
                     plot_indices = plot_indices,  # data['plotIndices']
                     metadata = metadata,
                     classes=classes,
                     all_components = all_components)
            rv = self.app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
        
        # assert we do NOT have exceptions:
        # FIXME: check why we have to access _app for global vars and not app (see setup method)
        pm = self._app.config['PLOTMANAGER']
        filtered_views = pm._fviews
        for view in filtered_views.itervalues():
            data = view.data
            for d in data.itervalues():
                _, plots, _2 = d
                for p in plots:
                    if p is None: # not calculated, skip
                        continue
                    assert "" == p.warning
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()