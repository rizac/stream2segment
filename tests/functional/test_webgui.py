# @PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''
from builtins import str
import pytest, os
import unittest
import numpy as np
import os
from io import BytesIO
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
from sqlalchemy.sql.expression import func, bindparam, and_
import time
from itertools import product
from obspy.core.stream import read
from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load, get_templates_fpaths
from stream2segment.gui.webapp.mainapp.plots.core import PlotManager
from obspy.io.stationtxt.core import all_components
from mock.mock import patch
from stream2segment.gui.webapp import get_session
from stream2segment.gui.main import create_main_app
import json
import tempfile
import shutil

class Test(object):

     # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.reinit(to_file=True)
        
        self.pyfile, self.configfile = get_templates_fpaths('processing.py',
                                                            'processing.yaml')
        
        self.app = create_main_app(db.dburl, self.pyfile, self.configfile)

        with self.app.app_context():
            # create a configured "Session" class
            # Session = sessionmaker(bind=self.engine)
            # create a Session
            # session = Session()
            
            session = self.session
            
            dc = DataCenter(station_url="345fbgfnyhtgrefs", dataselect_url='edfawrefdc')
            session.add(dc)
    
            utcnow = datetime.utcnow()
    
            run = Download(run_time=utcnow)
            session.add(run)
            
            ws = WebService(url='webserviceurl')
            session.add(ws)
            session.commit()
                
            id = 'firstevent'
            e1 = Event(event_id='event1', webservice_id=ws.id, time=utcnow, latitude=89.5, longitude=6,
                             depth_km=7.1, magnitude=56)
            e2 = Event(event_id='event2', webservice_id=ws.id, time=utcnow + timedelta(seconds=5),
                      latitude=89.5, longitude=6, depth_km=7.1, magnitude=56)
            
            session.add_all([e1, e2])
            
            session.commit()  # refresh datacenter id (alo flush works)
    
            d = datetime.utcnow()
            
            s = Station(network='network', station='station', datacenter_id=dc.id, latitude=90,
                        longitude=-45,
                        start_time=d)
            session.add(s)
            
            channels = [
                Channel(location='01', channel='HHE', sample_rate=6),
                Channel(location='01', channel='HHN', sample_rate=6),
                Channel(location='01', channel='HHZ', sample_rate=6),
                Channel(location='01', channel='HHW', sample_rate=6),
                
                Channel(location='02', channel='HHE', sample_rate=6),
                Channel(location='02', channel='HHN', sample_rate=6),
                Channel(location='02', channel='HHZ', sample_rate=6),
                
                Channel(location='03', channel='HHE', sample_rate=6),
                Channel(location='03', channel='HHN', sample_rate=6),
                
                Channel(location='04', channel='HHZ', sample_rate=6),
                
                Channel(location='05', channel='HHE', sample_rate=6),
                Channel(location='05gap_merged', channel='HHN', sample_rate=6),
                Channel(location='05err', channel='HHZ', sample_rate=6),
                Channel(location='05gap_unmerged', channel='HHZ', sample_rate=6)
                ]
            
            s.channels.extend(channels)
            session.commit()
            
            fixed_args = dict(datacenter_id=dc.id,
                         download_id=run.id,
                         )
                
            data_gaps_unmerged = data.read("GE.FLT1..HH?.mseed")
            data_gaps_merged = data.read("IA.BAKI..BHZ.D.2016.004.head")
            
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
                seg = Segment(request_start=ev.time + timedelta(seconds=val),
                              arrival_time=ev.time + timedelta(seconds=2 * val),
                              request_end=ev.time + timedelta(seconds=5 * val),
                              data=mseed,
                              data_seed_id=obspy_trace.get_id() if mseed == data_ok else None,
                              event_distance_deg=val,
                              event_id=ev.id,
                              **fixed_args)
                c.segments.append(seg)
            
            session.commit()
            
            session.close()
            
            # set inventory
#             with open(os.path.join(folder, "GE.FLT1.xml"), 'rb') as opn:
#                 self.inventory = loads_inv(opn.read())

            # set inventory
            self.inventory = data.read_inv("GE.FLT1.xml")

    @property
    def session(self):
        '''returns the db session by using the same function used from the Flask app
        i.e., DO NOT CALL `db.session` in the tests methods but `self.session`'''
        return get_session(self.app)

    def jsonloads(self, _data, encoding='utf8'):  
        # do not use data as argument as it might conflict with the data fixture
        # defined in conftest.py
        
        # IMPORTANT: python 3.5 and 3.6 behave differently, seems that the latter accepts bytes
        # and decodes them automatically, whereas in 3.5 (and below?) it doesn't
        # This method decodes bytes data and then returns json.loads
        # For info see thread (last post seems to confirm what we said):
        # https://bugs.python.org/issue10976
        if isinstance(_data, bytes):
            _data = _data.decode('utf8')
        return json.loads(_data)
    
    def test_root(self, db):  # db is a fixture (see conftest.py). Even if not used, it will
        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            app = self.app.test_client()
            # test first with classes:
            # WHY THIS DOES NOT WORK??!!!
            # the config set on the aopp IS NOT self.config!! why??!!
            # self.config['class_labels'] = {'wtf': 'wtfd'}
            # this on the other hand works:
            self.app.config['CONFIG.YAML']['class_labels'] = {'wtf': 'wtfd'}
            clz = self.session.query(Class).count()
            assert clz == 0

            rv = app.get('/')
            clz = self.session.query(Class).all()

            # assert global yaml config vars are injected as javascript from jinja rendering:
            # (be relaxed, if we change the template yaml file we do not want to fail)
            expected_str = """var __SETTINGS = {"config": {"""
            # https://github.com/pallets/flask/issues/716 is bytes in python3. Fix for both 2 and 3:
            response_data = rv.data.decode('utf-8')
            assert expected_str in response_data

            expected_str = ["""<div class='plot-wrapper' ng-show='plots[{0:d}].visible'>""",
                            """<div data-plot='time-series' data-plotindex={0:d} class='plot'></div>"""]
            # In the default processing, we implemented 6 plots, assure they are there:
            for plotindex in range(6):
                assert "<div id='plot-{0:d}' class='plot'".format(plotindex) in response_data
            assert len(clz) == 1 and clz[0].label == 'wtf' and clz[0].description == 'wtfd'
            
            # change description:
            self.app.config['CONFIG.YAML']['class_labels'] = {'wtf': 'abc'}
            rv = app.get('/')
#             expected_str = """var __SETTINGS = {"segment_orderby": ["event.time-", "segment.event_distance_deg"], "segment_select": {"has_data": "true"}, "spectra": {"arrival_time_shift": 0, "signal_window": [0.1, 0.9]}};"""
#             assert expected_str in rv.data
#             expected_str = """<div ng-show='plots[2].visible' data-plotindex=2 class='plot'></div>
#                         
#                         <div ng-show='plots[3].visible' data-plotindex=3 class='plot'></div>"""
            clz = self.session.query(Class).all()
            assert len(clz) == 1 and clz[0].label == 'wtf' and clz[0].description == 'abc'
            
            self.session.query(Class).delete()
            self.session.commit()
            clz = self.session.query(Class).count()
            assert clz == 0
            
            # delete entry 'class_labels' and test when not provided
            del self.app.config['CONFIG.YAML']['class_labels'] 
            rv = app.get('/')
#             expected_str = """var __SETTINGS = {"segment_orderby": ["event.time-", "segment.event_distance_deg"], "segment_select": {"has_data": "true"}, "spectra": {"arrival_time_shift": 0, "signal_window": [0.1, 0.9]}};"""
#             assert expected_str in rv.data
#             expected_str = """<div ng-show='plots[2].visible' data-plotindex=2 class='plot'></div>
#                         
#                         <div ng-show='plots[3].visible' data-plotindex=3 class='plot'></div>"""
            clz = self.session.query(Class).count()
            # assert nothing has changed (same as previous assert):
            assert clz == 0

    def test_get_segs(self, db):  # db is a fixture (see conftest.py). Even if not used, it will
        # assure this function is run once for each given dburl
        with self.app.app_context():
            app = self.app.test_client()
            # test your app context code
            rv = app.post("/get_segments", data=json.dumps(dict(segment_select={'has_data':'true'},
                                               segment_orderby=None, metadata=True, classes=True)),
                               headers={'Content-Type': 'application/json'})
            #    rv = app.get("/get_segments")
            data = self.jsonloads(rv.data)
            assert len(data['segment_ids']) == 28
            assert any(x[0] == 'has_data' for x in data['metadata'])
            assert not data['classes']

    def test_toggle_class_id(self, db):  # db is a fixture (see conftest.py). Even if not used, it will
        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            app = self.app.test_client()
            segid = 1
            segment = self.session.query(Segment).filter(Segment.id == segid).first()
            c = Class(label='label')
            self.session.add(c)
            self.session.commit()
            cid = c.id
            assert len(segment.classes) == 0
            rv = app.post("/set_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid,
                                                               'value':True}),
                                   headers={'Content-Type': 'application/json'})
            data = self.jsonloads(rv.data)
            
            assert len(segment.classes) == 1
            assert segment.classes[0].id == cid
            
            # toggle value (now False):
            rv = app.post("/set_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid,
                                                               'value':False}),
                                   headers={'Content-Type': 'application/json'})
            assert len(segment.classes) == 0
            
            # toggle again and run test_get_seg with a class set
            rv = app.post("/set_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid,
                                                               'value': True}),
                                   headers={'Content-Type': 'application/json'})
            assert len(segment.classes) == 1
            self._tst_get_seg(app)
    
    def test_get_seg(self, db):  # db is a fixture (see conftest.py). Even if not used, it will
        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            app = self.app.test_client()
            self._tst_get_seg(app)
    
    def _tst_get_seg(self, app):
        # does pytest.mark.parametrize work with unittest?
        # seems not. So:
        has_labellings = self.session.query(ClassLabelling).count() > 0
        for _ in product([[0, 1, 2], [], [0]], [True, False], [True, False], [True, False], [True, False]):
            plot_indices, preprocessed, metadata, classes, all_components = _
        
            d = dict(seg_id=1,
                     pre_processed=preprocessed,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=all_components)
                     # conf = data.get('config', {})
                     # plotmanager = current_app.config['PLOTMANAGER']
    #         if conf:
    #             current_app.config['CONFIG.YAML'].update(conf)
                
            rv = app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
            # https: 
            data = self.jsonloads(rv.data)
            assert len(data['plots']) == len(d['plot_indices'])
            assert bool(len(data['metadata'])) == metadata
            assert bool(len(data['classes'])) == (classes and has_labellings)

            if 0 in plot_indices:
                traces_in_first_plot = len(data['plots'][plot_indices.index(0)][1])
                assert (traces_in_first_plot == 1 and not all_components) or traces_in_first_plot >= 1
            # we should add a a test for the pre_processed case also, but we should inspect the plotmanager defined as app['PLOTMANAGER'] 
            # we should add a test for the zooms, too

    def test_segment_sa_station_inv_errors_in_preprocessed_traces(self, db):
        # db is a fixture (see conftest.py). Even if not used, it will
        # assure this function is run once for each given dburl
        ''''''
        plot_indices = [0]
        metadata = False
        classes = False
        
        with self.app.test_request_context():
            app = self.app.test_client()
            d = dict(seg_id=1,
                     pre_processed=False,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=all_components)
            rv = app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
            
            
        # Now we exited the session, we try with pre_processed=True
        with self.app.test_request_context():
            app = self.app.test_client()
            d = dict(seg_id=1,
                     pre_processed=True,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=all_components)
            rv = app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
        
        # assert we have exceptions:
        # FIXME: check why we have to access _app for global vars and not app (see setup method)
        pm = self.app.config['PLOTMANAGER']
        for plotlists in pm.values():
            plots = plotlists[1]
            if plots is None:  # not calculated, skip
                continue
            for i in plot_indices:
                assert "Station inventory (xml) error" in plots[i].warnings[0]
