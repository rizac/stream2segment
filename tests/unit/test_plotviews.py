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
from sqlalchemy.sql.expression import func, bindparam, and_
import time
from itertools import product
from stream2segment.io.db.queries import getallcomponents
from obspy.core.stream import read, Stream
from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load
from stream2segment.gui.webapp.processing.plots.core import PlotManager
from mock.mock import patch

from stream2segment.utils.postdownload import get_inventory as original_get_inventory, get_stream as original_get_stream
from obspy.core.utcdatetime import UTCDateTime


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
                                               'processing.py'))
        self.config = yaml_load(os.path.join(os.path.dirname(__file__), '..', '..',
                                             'stream2segment',
                                                  'resources', 'templates',
                                               'processing.yaml'))
        

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

        run = Download(run_time=utcnow)
        self.session.add(run)
        
        ws = WebService(url='webserviceurl')
        self.session.add(ws)
        self.session.commit()
            
        id = 'firstevent'
        e1 = Event(eventid='event1', webservice_id=ws.id, time=utcnow, latitude=89.5, longitude=6,
                         depth_km=7.1, magnitude=56)
        # note: e2 not used, store in db here anyway...
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
            
#             Channel(location= '03', channel='HHE', sample_rate=6),
#             Channel(location= '03', channel='HHN', sample_rate=6),
            
            Channel(location= '04', channel='HHZ', sample_rate=6),
            
            Channel(location= '05', channel='HHE', sample_rate=6),
            Channel(location= '05gap_merged', channel='HHN', sample_rate=6),
            Channel(location= '05err', channel='HHZ', sample_rate=6),
            Channel(location= '05gap_unmerged', channel='HHZ', sample_rate=6)
            ]
        
        s.channels.extend(channels)
        self.session.commit()
        
        fixed_args = dict(datacenter_id = dc.id,
                          download_id = run.id,
                          )
        
        # Note: data_gaps_merged is a stream where gaps can be merged via obspy.Stream.merge
        # data_gaps_unmerged is a stream where gaps cannot be merged (is a stream of three different channels
        # of the same event)
        folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        with open(os.path.join(folder, "GE.FLT1..HH?.mseed"), 'rb') as opn:
            data_gaps_unmerged = opn.read()  # unmerged cause we have three traces of different channels
        with open(os.path.join(folder, "IA.BAKI..BHZ.D.2016.004.head"), 'rb') as opn:
            data_gaps_merged = opn.read()
        with open(os.path.join(folder, "GE.FLT1..HH?.mseed"), 'rb') as opn:
            data_ok = opn.read()
        # create an 'ok' and 'error' Stream, the first by taking the first trace of "GE.FLT1..HH?.mseed",
        # the second by maipulating it
        obspy_stream = read(BytesIO(data_ok))
        obspy_trace = obspy_stream[0]
        
        # write data_ok is actually bytes data of 3 traces, write just the first one, we have
        # as it is it would be considered a trace with gaps, wwe have
        # another trace with gaps
        b = BytesIO()
        obspy_trace.write(b, format='MSEED')
        data_ok = b.getvalue()
        data_err = data_ok[:5]  # whatever slice should be ok
        
        seedid_ok = seedid_err = obspy_trace.get_id()
        seedid_gaps_unmerged = None
        seedid_gaps_merged = read(BytesIO(data_gaps_merged))[0].get_id()
        
        
        
        for ev, c in product([e1], channels):
            val = int(c.location[:2])
            mseed = data_gaps_merged if "gap_merged" in c.location else \
                data_err if "err" in c.location else data_gaps_unmerged if 'gap_unmerged' in c.location else data_ok
            seedid = seedid_gaps_merged if "gap_merged" in c.location else \
                seedid_err if 'err' in c.location else seedid_gaps_unmerged  if 'gap_unmerged' in c.location else seedid_ok
            
            # set times. For everything except data_ok, we set a out-of-bounds time:
            start_time = ev.time - timedelta(seconds=5)
            arrival_time = ev.time - timedelta(seconds=4)
            end_time = ev.time - timedelta(seconds=1)
            
            if "gap_merged" not in c.location and not 'err' in c.location and not \
                'gap_unmerged' in c.location:
                start_time = obspy_trace.stats.starttime.datetime
                arrival_time = (obspy_trace.stats.starttime + (obspy_trace.stats.endtime - obspy_trace.stats.starttime)/2).datetime
                end_time = obspy_trace.stats.endtime.datetime
            
            seg = Segment(request_start = start_time,
                          arrival_time = arrival_time,
                          request_end = end_time,
                          data = mseed,
                          data_identifier = seedid,
                          event_distance_deg = val,
                          event_id=ev.id,
                          **fixed_args)
            c.segments.append(seg)
        
        self.session.commit()
        
        # set inventory
        with open(os.path.join(folder, "GE.FLT1.xml"), 'rb') as opn:
            self.inventory_bytes = opn.read()
        self.inventory = loads_inv(self.inventory_bytes)

    @staticmethod
    def plotslen(plotmanager, preprocessed):
        '''total number of non-null plots'''
        i = 1 if preprocessed else 0
        return sum(1 if p[i] else 0 for p in plotmanager.values())

    @staticmethod
    def computedplotslen(plotmanager, seg_id, preprocessed, allcomponents=False):
        '''total number of non-null plots for a given segment'''
        i = 1 if preprocessed else 0
        segplotlists = [plotmanager[seg_id][i]]
        if allcomponents:
            for segid in segplotlists[0].oc_segment_ids:
                segplotlists.append(plotmanager[segid][i])
        n = 0
        for segplotlist in segplotlists:
            for plot in segplotlist:
                if plot is not None:
                    n += 1
        return n
    
    @staticmethod
    def traceslen(plotmanager, seg_id, preprocessed, allcomponents=False, count_exceptions_as_one_series=True):
        '''total number of traces to be plot for a given segment
        count_exceptions_as_one_series means that a stream() raising exception is counted as
        one series (the default). Otherwise counts as 0
        '''
        i = 1 if preprocessed else 0
        segplotlists = [plotmanager[seg_id][i]]
        if allcomponents:
            for segid in segplotlists[0].oc_segment_ids:
                segplotlists.append(plotmanager[segid][i])
        n = 0
        for segplotlist in segplotlists:
            streamorexc = segplotlist.data['stream']
            num = len(streamorexc) if isinstance(streamorexc, Stream) else 1 \
                if (isinstance(streamorexc, Exception) and count_exceptions_as_one_series) else 0
            n += num
        return n

    @patch('stream2segment.utils.postdownload.get_inventory')
    @patch('stream2segment.utils.postdownload.get_stream')
    def test_view_other_comps(self, mock_get_stream, mock_get_inv):
        
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

        def assert_(segplotlist, segment, preprocessed, is_invalidated=False):
            '''does some assertion
            :preprocessed: if the segplotlist refers toa  preprocessed segment
            :is_invalidated: if the segplotlist has been invalidated. Valid only
            if preprocessed=False (otherwise segplotlist should be None)
            This is used for checking that segplotlist.data['sn_windows'] is None
            '''
            # raw:
            # gap: stream: ok, sn_spectra exc, sn_windows: none
            # err: stream exc, sn_spectra: exc, sn_windows: none
            # other: stream ok, sn_spectra:ok, sn_windows: non-none
            # preprocessed
            iserr = 'err' in segment.channel.location
            hasgaps = 'gap' in segment.channel.location
            isinverr = isinstance(segplotlist.data['stream'], Exception) and \
                "inventory" in str(segplotlist.data['stream'])
            if preprocessed:
                if iserr or hasgaps or isinverr:
                    assert len("".join(segplotlist[0].warnings))
                    assert isinstance(segplotlist.data['stream'], Exception)
                    # if stream has an exception, as we use the stream for the sn_windows, assert
                    # the exception is the same:
                    assert segplotlist.data['sn_windows'] == segplotlist.data['stream']
                    if segplotlist[1] is not None:
                        assert len("".join(segplotlist[1].warnings))
                else:
                    assert not len("".join(segplotlist[0].warnings))
                    assert isinstance(segplotlist.data['stream'], Stream)
                    # assert sn_windows are correct:
                    sn_wdw = segplotlist.data['sn_windows']
                    assert len(sn_wdw) == 2
                    assert all(isinstance(_, UTCDateTime) for _ in list(sn_wdw[0]) + list(sn_wdw[1]))
                    if segplotlist[1] is not None:
                        assert not len("".join(segplotlist[1].warnings))
            else:
                # test sn_windows first:
                if is_invalidated:
                    assert segplotlist.data['sn_windows'] is None  # reset from invalidation
                elif iserr:
                    assert isinstance(segplotlist.data['sn_windows'], Exception)
                    # assert also that it is the same exception raised from stream:
                    assert segplotlist.data['stream'] == segplotlist.data['sn_windows']
                elif 'gap' in segment.channel.location:
                    # sn_windows should raise, as we have more than one trace:
                    assert isinstance(segplotlist.data['sn_windows'], Exception)
                    assert "gap" in str(segplotlist.data['sn_windows'])
                else:  # good segment
                    # if the stream is unprocessed, and it was successfully loaded, assert
                    # sn_windows are correct:
                    sn_wdw = segplotlist.data['sn_windows']
                    assert len(sn_wdw) == 2
                    assert all(isinstance(_, UTCDateTime) for _ in list(sn_wdw[0]) + list(sn_wdw[1]))
                # test other stuff:
                if iserr:
                    assert len("".join(segplotlist[0].warnings))
                    assert isinstance(segplotlist.data['stream'], Exception)
                    if segplotlist[1] is not None:
                        assert len("".join(segplotlist[1].warnings))
                else:
                    assert isinstance(segplotlist.data['stream'], Stream)
                    if "gap_unmerged" in segment.channel.location:
                        assert "different seed ids" in "".join(segplotlist[0].warnings)    
                    else:
                        assert not len("".join(segplotlist[0].warnings)) 
                    if segplotlist[1] is not None:
                        if hasgaps:
                            assert len("".join(segplotlist[1].warnings))
                        else:
                            assert not len("".join(segplotlist[1].warnings))
            
            
        for s in self.session.query(Segment):
            
            m = PlotManager(self.pymodule, self.config)
            
            expected_components_count = components_count[(s.event_id, s.channel.location)]

            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            mock_get_stream.side_effect = original_get_stream
            mock_get_inv.side_effect = original_get_inventory
            allcomponents = True
            preprocessed = False
            idxs = [0]
            # s_id_was_in_views = s.id in m._plots
            plots = m.get_plots(self.session, s.id, idxs, preprocessed, allcomponents)
#             # assert returned plot has the correct number of time/line-series:
#             # note that plots[0] might be generated from a stream with gaps
            assert len(plots[0].data) == self.traceslen(m, s.id, preprocessed, allcomponents, count_exceptions_as_one_series=True)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.plotslen(m, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents=False) == len(idxs)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents) == expected_components_count
            # assert SegmentWrapper function calls:
            assert not mock_get_inv.called  # preprocess=False
            assert mock_get_stream.call_count == expected_components_count
            # assert we did not calculate any useless stream:
            assert_(m[s.id][0], s, preprocessed=False)
            assert m[s.id][1] is None
            
            
            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            allcomponents = True
            preprocessed = False
            idxs = [0, 1]
            # s_id_was_in_views = s.id in m._plots
            plots = m.get_plots(self.session, s.id, idxs, preprocessed, allcomponents)
#           # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.plotslen(m, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents=False) == len(idxs)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents) == expected_components_count+1
            # assert SegmentWrapper function calls:
            assert not mock_get_inv.called  # preprocess=False
            assert not mock_get_stream.called  # already computed
            # assert we did not calculate any useless stream:
            assert_(m[s.id][0], s, preprocessed=False)
            assert m[s.id][1] is None
            
            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            allcomponents = False
            preprocessed = True
            idxs = [0, 1]
            # s_id_was_in_views = s.id in m._plots
            plots = m.get_plots(self.session, s.id, idxs, preprocessed, allcomponents)
#           # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert self.plotslen(m, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents=False) == len(idxs)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents) == len(idxs)
            # assert SegmentWrapper function calls:
            if 'err' not in s.channel.location and not 'gap' in s.channel.location:
                assert mock_get_inv.called  # preprocess=False
            else:
                assert not mock_get_inv.called
            assert not mock_get_stream.called  # already computed
            # assert we did not calculate any useless stream:
            assert_(m[s.id][0], s, preprocessed=False)
            assert_(m[s.id][1], s, preprocessed=True)

            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            allcomponents = True
            preprocessed = True
            idxs = [0, 1]
            # s_id_was_in_views = s.id in m._plots
            plots = m.get_plots(self.session, s.id, idxs, preprocessed, allcomponents)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert self.plotslen(m, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents=False) == len(idxs)
            assert self.computedplotslen(m, s.id, preprocessed, allcomponents) == expected_components_count+1
            # assert SegmentWrapper function calls:
            assert not mock_get_inv.called  # already called
            assert not mock_get_stream.called  # already computed
            # assert all titles are properly set, with the given prefix
            seedid = s.seed_identifier
            assert all(p is None or p.title.startswith(seedid) for p in m[s.id][0])
            assert all(p is None or p.title.startswith(seedid) for p in m[s.id][1])
            # check plot titles and warnings:
            stream = m[s.id][0].data['stream']
            preprocessed_stream = m[s.id][1].data['stream']
            if 'err' in s.channel.location:
                assert isinstance(stream, Exception) and isinstance(preprocessed_stream, Exception) and \
                    'MiniSeed error' in str(preprocessed_stream)
                for i in idxs:
                    plot, pplot = m[s.id][0][i], m[s.id][1][i]
                    assert len(plot.data) == 1 # only one (fake) trace
                    assert plot.warnings
                    assert len(pplot.data) == 1 # only one (fake) trace
                    assert pplot.warnings
                    
            elif 'gap' in s.channel.location:
                assert isinstance(stream, Stream) and isinstance(preprocessed_stream, Exception) and \
                    'gaps/overlaps' in str(preprocessed_stream)
                for i in idxs:
                    plot, pplot = m[s.id][0][i], m[s.id][1][i]
                    # if idx=1, plot has 1 series (due to error in gaps/overlaps) otherwise matches stream traces count:
                    assert len(plot.data) == 1 if i==1 else len(stream)
                    if 'gap_unmerged' in s.channel.location:
                        assert 'different seed ids' in "".join(plot.warnings) if i == 0 \
                            else 'gaps/overlaps' in pplot.warnings[0]
                    else:
                        assert not plot.warnings if i == 0 else \
                            'gaps/overlaps' in pplot.warnings[0]  # gaps /overlaps are simply shown as lineseries, no warnings
                    assert len(pplot.data) == 1 # only one (fake) trace
                    assert pplot.warnings and 'gaps/overlaps' in pplot.warnings[0]  # gaps /overlaps
            else:
                assert isinstance(stream, Stream) and isinstance(preprocessed_stream, Exception) and \
                        'Station inventory (xml) error: unknown url type' in str(preprocessed_stream)
                for i in idxs:
                    plot, pplot = m[s.id][0][i], m[s.id][1][i]
                    # if idx=1, plot has 2 series (noie/signal) otherwise matches stream traces count:
                    assert len(plot.data) == 2 if i==1 else len(stream)
                    assert not plot.warnings  # gaps /overlaps
                    assert len(pplot.data) == 1 # only one (fake) trace
                    assert pplot.warnings and 'inventory' in pplot.warnings[0]  # gaps /overlaps
            # assert we did not calculate any useless stream:
            assert_(m[s.id][0], s, preprocessed=False)
            assert_(m[s.id][1], s, preprocessed=True)
            
            # now check update config:
            # store the s_stream to compare later:
            # we need now to store the proper inventory for the 'ok' segment in order to let
            # the preprocess function work properly (so that get_inventory) does not raise:
            # Don't know why, but side_effect does not work:
#             mock_get_inv.reset_mock()
#             def ginv(*a, **v):
#                 return self.inventory
#             mock_get_inv.side_effect = ginv
            # so we manually set the inventory on the db, discarding it afterwards:
            s.station.inventory_xml = self.inventory_bytes
            self.session.commit()
            assert s.station.inventory_xml
            # re-initialize a new PlotManager to assure everything is re-calculated
            # this also sets all cache to None, including m.inv_cache:
            m = PlotManager(self.pymodule, self.config)
            # calculate plots
            idxs = [0, 1]
            m.get_plots(self.session, s.id, idxs, preprocessed=False, all_components_in_segment_plot=True)
            m.get_plots(self.session, s.id, idxs, preprocessed=True, all_components_in_segment_plot=True)
            # and store their values for later comparison
            SN_INDEX = 1
            sn_plot_unprocessed = m[s.id][0][SN_INDEX].data
            sn_plot_preprocessed = m[s.id][1][SN_INDEX].data
            # shift back the arrival time. 1 second is still within the stream time bounds for the 'ok'
            # stream:
            sn_windows = dict(m.config['sn_windows'])
            sn_windows['arrival_time_shift']  -= 1
            m.update_config(sn_windows=sn_windows)
            # assert we restored streams that have to be invalidated, and we kept those not to invalidate:
            assert_(m[s.id][0], s, preprocessed=False, is_invalidated=True)
            assert m[s.id][1] is None
            # and run again the get_plots: with preprocess=False
            idxs = [0, 1]
            plots = m.get_plots(self.session, s.id, idxs, preprocessed=False, all_components_in_segment_plot=True)
            assert_(m[s.id][0], s, preprocessed=False)
            assert m[s.id][1] is None
            sn_plot_unprocessed_new = m[s.id][0][SN_INDEX].data
            # we changed the arrival time, BUT: the signal noise depends on the cumulative, thus
            # changing the arrival time does not change the signal window s_stream
            # Conversely, n_stream should change BUT only for the 'ok' stream (no 'gap' or 'err' in s.channel.location)
            # as for the other we explicitly set a miniseed starttime, endtime BEFORE the event time
            # which should result in noise stream all padded with zeros regardless of the arrival time shift
            if len(sn_plot_unprocessed) == 1:
                #there was an error in sn ratio (e.g., gaps, overlaps in source stream):
                assert len(sn_plot_unprocessed_new) == 1
            else:
                # no singal window is the same (index 0. Then index 2 cause it is the np.array of the data)
                assert np.allclose(sn_plot_unprocessed_new[0][2], sn_plot_unprocessed[0][2], equal_nan=True)
                # the n window does not:
                assert not np.allclose(sn_plot_unprocessed_new[1][2], sn_plot_unprocessed[1][2], equal_nan=True)

            # now run again with preprocessed=True.
            plots = m.get_plots(self.session, s.id, idxs, preprocessed=True, all_components_in_segment_plot=True)
            sn_plot_preprocessed_new = m[s.id][1][SN_INDEX].data
            # assert the s_stream differs from the previous, as we changed the signal/noise arrival time shift
            # this must hold only for the 'ok' stream (no 'gap' or 'err' in s.channel.location)
            # as for the other we explicitly set a miniseed starttime, endtime BEFORE the event time
            # (thus by shifting BACK the arrival time we should not see changes in the s/n stream windows)
            if len(sn_plot_preprocessed) == 1:
                #there was an error in sn ratio (e.g., gaps, overlaps in source stream):
                assert len(sn_plot_preprocessed_new) == 1
            else:
                # no singal window is the same (index 0. Then index 2 cause it is the np.array of the data)
                assert np.allclose(sn_plot_preprocessed_new[0][2], sn_plot_preprocessed[0][2], equal_nan=True)
                # the n window does not:
                assert not np.allclose(sn_plot_preprocessed_new[1][2], sn_plot_preprocessed[1][2], equal_nan=True)
            
            assert_(m[s.id][1], s, preprocessed=True)
            # re-set the inventory_xml to None:
            s.station.inventory_xml = None
            self.session.commit()
            assert not s.station.inventory_xml


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()