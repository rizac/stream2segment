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
        self.session = Session()
        self.initdb()
        self.pymodule = load_source(os.path.join(os.path.dirname(__file__), '..', '..',
                                                 'stream2segment',
                                                  'resources', 'templates',
                                               'gui.py'))
        self.config = yaml_load(os.path.join(os.path.dirname(__file__), '..', '..',
                                             'stream2segment',
                                                  'resources', 'templates',
                                               'gui.yaml'))
        

    @staticmethod
    def cleanup(me):
        if me.engine:
            if me.session:
                try:
                    me.session.rollback()
                    me.session.close()
                except:
                    pass
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
        dc= DataCenter(station_url="345fbgfnyhtgrefs", dataselect_url='edfawrefdc')
        self.session.add(dc)

        utcnow = datetime.utcnow()

        run = Run(run_time=utcnow)
        self.session.add(run)
        
        ws = WebService(url='webserviceurl')
        self.session.add(ws)
        self.session.commit()
            
        id = 'firstevent'
        e1 = Event(eventid='event1', webservice_id=ws.id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        e2 = Event(eventid='event2', webservice_id=ws.id, time=utcnow + timedelta(seconds=5),
                  latitude=89.5, longitude=6, depth_km=7.1, magnitude=56)
        
        self.session.add_all([e1, e2])
        
        self.session.commit()  # refresh datacenter id (alo flush works)

        d = datetime.utcnow()
        
        s = Station(network='network', station='station', datacenter_id=dc.id, latitude=90,
                    longitude=-45,
                    start_time=d)
        self.session.add(s)
        
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
        self.session.commit()
        
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
        
        self.session.commit()
        
        # set inventory
        with open(os.path.join(folder, "GE.FLT1.xml"), 'rb') as opn:
            self.inventory = loads_inv(opn.read())


    def test_view_other_comps(self):
        m = PlotManager(self.pymodule, self.config)
        
        prevlen = len(m._views)
        components_count = {} # group_id -> num expected components
        # where group_id is the tuple (event_id, channel.location)
        for s in self.session.query(Segment): 
            group_id = (s.event_id, s.channel.location)
            if group_id not in components_count:
                # we should have created views also for the other components. To get
                # other components, use the segment channel and event id
                other_comps_count = self.session.query(Segment).join(Segment.channel).\
                    filter(and_(Segment.event_id == s.event_id, Channel.location == s.channel.location)).count()
                
                components_count[group_id] = other_comps_count

        
        for s in self.session.query(Segment):
            expected_components_count = components_count[(s.event_id, s.channel.location)]
            
            idxs = [0]
            all_components = True
            # s_id_was_in_views = s.id in m._views
            plots = m.getplots(self.session, s.id, True, *idxs)
            # assert returned plot has the correct time-series:
            assert len(plots[0].data) == expected_components_count
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not m._fviews  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            viewmanager = m._views[s.id]
            calculated_plots=0
            for sid in viewmanager.data:
                calculated_plots += sum(plot is not None for plot in viewmanager.data[sid][1])
            assert calculated_plots == expected_components_count
        
    def test_view_inv_err(self):
        m = PlotManager(self.pymodule, self.config)
        
        prevlen = len(m._views)
        components_count = {} # group_id -> num expected components
        # where group_id is the tuple (event_id, channel.location)
        for s in self.session.query(Segment): 
            group_id = (s.event_id, s.channel.location)
            if group_id not in components_count:
                # we should have created views also for the other components. To get
                # other components, use the segment channel and event id
                other_comps_count = self.session.query(Segment).join(Segment.channel).\
                    filter(and_(Segment.event_id == s.event_id, Channel.location == s.channel.location)).count()
                
                components_count[group_id] = other_comps_count

        
        for s in self.session.query(Segment):
            expected_components_count = components_count[(s.event_id, s.channel.location)]
            
            idxs = []
            all_components = False
            # s_id_was_in_views = s.id in m._views
            plots = m.getplots(self.session, s.id, False, *idxs)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not m._fviews  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            viewmanager = m._views[s.id]
            calculated_plots = sum(plot is not None for plot in viewmanager.data[s.id][1])
            assert calculated_plots == len(plots)

            # assert we associated the viewmanager to expected_components_count keys:
            assert sum(viewmanager is vm for vm in  m._views.itervalues()) == expected_components_count
            # assert we have the inventory for the segment:
            # FIXME: test that we don't have it if we set inventory=False in the config
            assert s.id in  m.segid2inv
            
            idxs = [0, 1]
            plots = m.getplots(self.session, s.id, False, *idxs)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not m._fviews  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            viewmanager = m._views[s.id]
            calculated_plots = sum(plot is not None for plot in viewmanager.data[s.id][1])
            assert calculated_plots == len(plots)

            # assert we associated the viewmanager to expected_components_count keys:
            assert sum(viewmanager is vm for vm in  m._views.itervalues()) == expected_components_count
            # assert we have the inventory for the segment:
            # FIXME: test that we don't have it if we set inventory=False in the config
            assert s.id in  m.segid2inv

            # calculate all indices
            idxs = range(len(m.functions))
            plots = m.getplots(self.session, s.id, False, *idxs)
            calculated_plots = sum(plot is not None for plot in viewmanager.data[s.id][1])
            
            custom_plots = plots[1:]
            plots = plots[:1]
            if 'err' in s.channel.location:
                assert all('Unknown format' in p.warning for p in custom_plots)
                assert all('Unknown format' in p.warning for p in plots)
                # mseed errors do not check for inventory errors, thus:
                assert all('\n' not in p.warning for p in custom_plots)
                assert all('\n' not in p.warning for p in plots)
            elif 'gap_merged'  in s.channel.location:
                assert all('Gaps/overlaps (merged)' in p.warning for p in custom_plots)
                assert all('Gaps/overlaps (merged)' in p.warning for p in plots)
                assert all('\n' in p.warning for p in custom_plots)
                assert all('Inventory error' in p.warning for p in custom_plots)
                assert all('Inventory error' in p.warning for p in plots)
            elif 'gap_unmerged'  in s.channel.location:
                assert all('Gaps/overlaps (unmergeable)' in p.warning for p in custom_plots)
                assert all('Gaps/overlaps (unmergeable)' in p.warning for p in plots)
                # mseed errors (unmergeable gaps are treated as errors)
                # do not check for inventory errors, thus:
                assert all('\n' not in p.warning for p in custom_plots)
                assert all('\n' not in p.warning for p in plots)
            else:
                # for segments ok, we should have only a warning concerning the inventory:
                assert all('\n' not in p.warning for p in custom_plots)
                try:
                    assert all('Inventory error' in p.warning for p in custom_plots)
                except:
                    print "------------------"
                    print [p.warning for p in custom_plots]
                    print "------------------"
                assert all('Inventory error' in p.warning for p in plots)
            
    @patch('stream2segment.gui.webapp.plotviews.get_inventory')
    @patch('stream2segment.gui.webapp.plotviews.exec_function')
    def test_view_inv(self, mock_exec_func, mock_get_inv):
        mock_exec_func.side_effect=lambda *a, **v: exec_function(*a, **v)
        mock_get_inv.side_effect=lambda *a, **v: self.inventory 
        
        m = PlotManager(self.pymodule, self.config)
        
        prevlen = len(m._views)
        components_count = {} # group_id -> num expected components
        # where group_id is the tuple (event_id, channel.location)
        for s in self.session.query(Segment): 
            group_id = (s.event_id, s.channel.location)
            if group_id not in components_count:
                # we should have created views also for the other components. To get
                # other components, use the segment channel and event id
                other_comps_count = self.session.query(Segment).join(Segment.channel).\
                    filter(and_(Segment.event_id == s.event_id, Channel.location == s.channel.location)).count()
                
                components_count[group_id] = other_comps_count

        numplots = len(m.functions)
        for s in self.session.query(Segment):
            expected_components_count = components_count[(s.event_id, s.channel.location)]
             
            all_components = True
            idxs = range(numplots)
            plots = m.getplots(self.session, s.id, all_components, *idxs)
            assert len(plots) == len(idxs)
            # as long as idxs[0] == 0 and 0 refers to the 'main' trace plot, we can do like this:
            custom_plots = plots[1:]
            plots = plots[:1]
            
            assert not m._fviews  # no filtering calculated
             
            if 'err' in s.channel.location:
                assert all('Unknown format' in p.warning for p in custom_plots)
                assert all('Unknown format' in p.warning for p in plots)
                # assert no other warning:
                assert all('\n' not in p.warning for p in custom_plots)
            elif 'gap_merged'  in s.channel.location:
                assert all('Gaps/overlaps (merged)' in p.warning for p in custom_plots)
                assert all('Gaps/overlaps (merged)' in p.warning for p in plots)
                # assert no other warning:
                assert all('\n' not in p.warning for p in custom_plots)
            elif 'gap_unmerged'  in s.channel.location:
                assert all('Gaps/overlaps (unmergeable)' in p.warning for p in custom_plots)
                assert all('Gaps/overlaps (unmergeable)' in p.warning for p in plots)
                # assert no other warning:
                assert all('\n' not in p.warning for p in custom_plots)
            else:
                # for segments ok, we should have no warnings:
                assert all('' in p.warning for p in custom_plots)
                assert all('' in p.warning for p in plots)
        
        assert mock_get_inv.call_count == 1
        # assert we called exec_function the correct number of times
        assert mock_exec_func.call_count == self.session.query(Segment).count()*numplots
        # and assure all plots are non-none:
        for v in m._views.itervalues():
            for t, plots, _ in v.data.itervalues():
                assert all(plot is not None for plot in plots)
        
        mock_exec_func.reset_mock()
        mock_get_inv.reset_mock()
        for s in self.session.query(Segment):
            idxs = range(numplots)  # range(len(viewmanager.customfunctions))
            expected_components_count = components_count[(s.event_id, s.channel.location)]
            plots = m.getfplots(self.session, s.id, True, *idxs)
            assert len(plots) == len(idxs)
            # as long as idxs[0] == 0 and 0 refers to the 'main' trace plot, we can do like this:
            custom_plots = plots[1:]
            plots = plots[:1]
            if 'err' in s.channel.location:
                assert all('Unknown format' in p.warning for p in custom_plots)
                assert all('Unknown format' in p.warning for p in plots)
                # assert no other warning:
                assert all('\n' not in p.warning for p in custom_plots)
            elif 'gap_merged'  in s.channel.location:
                assert all('No matching response information found' in p.warning for p in custom_plots)
                assert all('No matching response information found' in p.warning for p in plots)
                # assert no other warning:
                assert all('\n' not in p.warning for p in custom_plots)
            elif 'gap_unmerged'  in s.channel.location:
                assert all('Gaps/overlaps (unmergeable)' in p.warning for p in custom_plots)
                assert all('Gaps/overlaps (unmergeable)' in p.warning for p in plots)
                # assert no other warning:
                assert all('\n' not in p.warning for p in custom_plots)
            else:
                # for segments ok, we should have no warnings:
                assert all('' in p.warning for p in custom_plots)
                assert all('' in p.warning for p in plots)
        
        assert mock_get_inv.call_count == 0  # already called
        # assert we called exec_function the correct number of times
        assert mock_exec_func.call_count == self.session.query(Segment).count()*numplots
        # and assure all plots are non-none:
        for v in m._fviews.itervalues():
            for t, plots, _ in v.data.itervalues():
                assert all(plot is not None for plot in plots)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()