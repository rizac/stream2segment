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
from itertools import product
from stream2segment.io.db.queries import getallcomponents
from obspy.core.stream import read, Stream
from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load
from stream2segment.gui.webapp.plotviews import PlotManager, get_plot
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

    @staticmethod
    def plotslen(plotmanager):
        return sum(1 if p[0] else 0 for p in plotmanager._plotscache.values())
    
    @staticmethod
    def pplotslen(plotmanager):
        return sum(1 if p[1] else 0 for p in plotmanager._plotscache.values())

    @staticmethod
    def getplots(plotmanager, seg_id, allcomponents=False):
        data = plotmanager._plotscache[seg_id][0].data
        if not data:
            return []
        if allcomponents:
            return list(data[segid]['plots'] for segid in data)
        else:
            return data[seg_id]['plots']
    
    @staticmethod
    def getpplots(plotmanager, seg_id, allcomponents=False):
        data = plotmanager._plotscache[seg_id][1].data
        if not data:
            return []
        if allcomponents:
            return list(data[segid]['plots'] for segid in data)
        else:
            return data[seg_id]['plots']

    @staticmethod
    def computedplotslen(plotmanager, seg_id, allcomponents=False):
        data = Test.getplots(plotmanager, seg_id, allcomponents)
        if not allcomponents:
            data = [data]
        s = 0
        for d in data:
            s += sum(1 if _ else 0 for _ in d)
        return s
    
    @staticmethod
    def computedpplotslen(plotmanager, seg_id, allcomponents=False):
        data = Test.getpplots(plotmanager, seg_id, allcomponents)
        if not allcomponents:
            data = [data]
        s = 0
        for d in data:
            s += sum(1 if _ else 0 for _ in d)
        return s
    
    @staticmethod
    def tracecount(plotmanager, seg_id, allcomponents=False):
        allsegids = plotmanager._plotscache[seg_id][0].data.keys()
        total = 0
        for sid in allsegids:
            streamorexc = plotmanager._plotscache[sid][0].get_cache(sid, 'stream')
            num = len(streamorexc) if isinstance(streamorexc, Stream) else 1 if isinstance(streamorexc, Exception) else 0
            if not allcomponents and sid == seg_id:
                return num
            total += num
        return total

    @staticmethod
    def ptracecount(plotmanager, seg_id, allcomponents=False):
        allsegids = plotmanager._plotscache[seg_id][1].data.keys()
        total = 0
        for sid in allsegids:
            streamorexc = plotmanager._plotscache[sid][1].get_cache(sid, 'stream')
            num = len(streamorexc) if isinstance(streamorexc, Stream) else 1 if isinstance(streamorexc, Exception) else 0
            if not allcomponents and sid == seg_id:
                return num
            total += num
        return total

    def test_view_other_comps(self):
        m = PlotManager(self.pymodule, self.config)
        
        prevlen = self.plotslen(m)
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
            # s_id_was_in_views = s.id in m._plots
            plots = m.getplots(self.session, s.id, idxs, False, all_components)
            # assert returned plot has the correct number of time/line-series:
            # note that plots[0] might be generated from a stream with gaps
            expected_lineseries = self.tracecount(m, s.id, all_components)
            assert len(plots[0].data) == expected_lineseries

            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.pplotslen(m)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(m, s.id, allcomponents=False) == len(idxs)
            assert self.computedplotslen(m, s.id, allcomponents=True) == expected_components_count
        
    def tst_view_inv_err(self):
        m = PlotManager(self.pymodule, self.config)
        
        prevlen = self.plotslen(m)
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
            # s_id_was_in_views = s.id in m._plots
            plots = m.getplots(self.session, s.id, idxs, False)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.pplotslen(m)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            plotscache = m._plotscache[s.id][0]
            calculated_plots = sum(plot is not None for plot in plotscache.data[s.id][1])
            assert calculated_plots == len(plots)

            # assert we associated the plotscache to expected_components_count keys:
            assert sum(plotscache is vm for vm in  m._plots.values()) == expected_components_count
            # assert we have the inventory for the segment:
            # FIXME: test that we don't have it if we set inventory=False in the config
            assert s.id in  m.segid2inv
            
            idxs = [0, 1]
            plots = m.getplots(self.session, s.id, idxs, False)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.pplotslen(m)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            plotscache = m._plotscache[s.id][0]
            calculated_plots = sum(plot is not None for plot in plotscache.data[s.id][1])
            assert calculated_plots == len(plots)

            # assert we associated the plotscache to expected_components_count keys:
            assert sum(plotscache is vm for vm in  m._plots.values()) == expected_components_count
            # assert we have the inventory for the segment:
            # FIXME: test that we don't have it if we set inventory=False in the config
            assert s.id in  m.segid2inv

            # calculate all indices
            idxs = list(range(len(m.functions)))
            plots = m.getplots(self.session, s.id, idxs, False)
            calculated_plots = sum(plot is not None for plot in plotscache.data[s.id][1])
            
#             custom_plots = plots[1:]
#             plots = plots[:1]
            
            # assert warnings are as expected. See plotmanager.get_warnings for a details of
            # warnings. Note that the warning messages might change in the future, so the assert
            # statements below try to check what most likely will not be changed, to avoid
            # recurring failing tests because of a message text changed in the future
            
            warnings = m.get_warnings(s.id, False)
            # we have inventory errors in any case:
            assert any('inventory n/a' in _.lower() for _ in warnings)
            if 'err' in s.channel.location:
                # error 
                assert any('sn-windows n/a: ' in _.lower() for _ in warnings)
            elif 'gap_merged'  in s.channel.location:
                # gaps merged is not anymore merged by default, so this equals the if below
                # gaps merged is not anymore merged by default, so check we should heve these warnings:
                assert any('gaps/overlaps' in _.lower() for _ in warnings)
                # also, the stream has more traces, so we cannot calculate sn windows:
                assert any('sn-windows n/a: ' in _.lower() for _ in warnings)
            elif 'gap_unmerged'  in s.channel.location:
                # gaps unmerged is not anymore merged by default, so this equals the if above
                assert any('gaps/overlaps' in _.lower() for _ in warnings)
                # also, the stream has more traces, so we cannot calculate sn windows
                assert any('sn-windows n/a: ' in _.lower() for _ in warnings)
                   
            else:
                # for segments ok, we should have only a warning concerning the inventory:
                assert len(warnings) == 1
                
            
    @patch('stream2segment.gui.webapp.plotviews.get_inventory')
    @patch('stream2segment.gui.webapp.plotviews.get_plot')
    def tst_view_inv(self, mock_get_plot, mock_get_inv):
        mock_get_plot.side_effect=lambda *a, **v: exec_function(*a, **v)
        mock_get_inv.side_effect=lambda *a, **v: self.inventory 
        
        m = PlotManager(self.pymodule, self.config)
        
        prevlen = len(m._plotscache)
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


        def assert_warnings(plotsmanager, segment, preprocessed):
            '''asserts the correct warnings for the given segment. Called after plots are
            calculated and after filtered plots to assure the warnings are the same'''
            
            # assert warnings are as expected. See plotmanager.get_warnings for a details of
            # warnings. Note that the warning messages might change in the future, so the assert
            # statements below try to check what most likely will not be changed, to avoid
            # recurring failing tests because of a message text changed in the future
            
            warnings = plotsmanager.get_warnings(segment.id, preprocessed)
            # we have inventory errors in any case:
            assert not any('inventory n/a' in _.lower() for _ in warnings)
            if 'err' in s.channel.location:
                # error reading stream
                assert any('sn-windows n/a: ' in _.lower() for _ in warnings)
            elif 'gap_merged'  in s.channel.location:
                # gaps merged is not anymore merged by default, so this equals the if below
                # gaps merged is not anymore merged by default, so check we should heve these warnings:
                assert any('gaps/overlaps' in _.lower() for _ in warnings)
                # also, the stream has more traces, so we cannot calculate sn windows:
                assert any('sn-windows n/a: ' in _.lower() for _ in warnings)
            elif 'gap_unmerged'  in s.channel.location:
                # gaps unmerged is not anymore merged by default, so this equals the if above
                assert any('gaps/overlaps' in _.lower() for _ in warnings)
                # also, the stream has more traces, so we cannot calculate sn windows:
                assert any('sn-windows n/a: ' in _.lower() for _ in warnings)
                   
            else:
                # for segments ok, we should have only a warning concerning the inventory:
                assert len(warnings) == 0
            
        numplots = len(m.functions)
        for s in self.session.query(Segment):
            expected_components_count = components_count[(s.event_id, s.channel.location)]
             
            all_components = True
            idxs = list(range(numplots))
            plots = m.getplots(self.session, s.id, idxs, all_components)
            assert len(plots) == len(idxs)
            # as long as idxs[0] == 0 and 0 refers to the 'main' trace plot, we can do like this:
            custom_plots = plots[1:]
            plots = plots[:1]
            
            assert not self.pplotslen(m)  # no filtering calculated
            assert_warnings(m, s, False)
        
        assert mock_get_inv.call_count == 1
        # assert we called exec_function the correct number of times
        assert mock_get_plot.call_count == self.session.query(Segment).count()*numplots
        # and assure all plots are non-none:
        for plotscache in m._plotscache.values():
            for s, plots, sn_warnings in plotscache.data.values():
                assert all(plot is not None for plot in plots)
        
        mock_get_plot.reset_mock()
        mock_get_inv.reset_mock()
        for s in self.session.query(Segment):
            idxs = list(range(numplots))  # range(len(viewmanager.customfunctions))
            expected_components_count = components_count[(s.event_id, s.channel.location)]
            # get pre-processed plots:
            plots = m.getplots(self.session, s.id, idxs, True, True)
            assert len(plots) == len(idxs)
            # as long as idxs[0] == 0 and 0 refers to the 'main' trace plot, we can do like this:
            
            assert_warnings(m, s, True)

        assert mock_get_inv.call_count == 0  # already called
        # assert we called exec_function the correct number of times
        assert mock_get_plot.call_count == self.session.query(Segment).count()*numplots
        # and assure all plots are non-none:
        for plotscache_ in m._plotscache.values():
            plotscache = plotscache_[1]
            for s, plots, sn_warnings in plotscache.data.values():
                assert all(plot is not None for plot in plots)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()