'''
Created on Jul 15, 2016

@author: riccardo
'''
from builtins import str, range
import os
import sys
from contextlib import contextmanager
from io import BytesIO
from datetime import datetime, timedelta
from itertools import product
from mock.mock import patch

import pytest
import numpy as np
from sqlalchemy.sql.expression import and_
from obspy.core.stream import read, Stream
from obspy.core.utcdatetime import UTCDateTime

from stream2segment.io.db.models import Event, WebService, Channel, Station, \
    DataCenter, Segment, Download
from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load, get_templates_fpaths
from stream2segment.gui.webapp.mainapp.plots.core import PlotManager, LimitedSizeDict, \
    InventoryCache, _default_size_limits
from stream2segment.process.db import get_inventory as original_get_inventory,\
    get_stream as original_get_stream


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False, process=True)

        # init db:
        session = db.session

        dct = DataCenter(station_url="345fbgfnyhtgrefs", dataselect_url='edfawrefdc')
        session.add(dct)

        utcnow = datetime.utcnow()

        dwl = Download(run_time=utcnow)
        session.add(dwl)

        ws = WebService(url='webserviceurl')
        session.add(ws)
        session.commit()

        # id = 'firstevent'
        ev1 = Event(event_id='event1', webservice_id=ws.id, time=utcnow, latitude=89.5,
                    longitude=6, depth_km=7.1, magnitude=56)
        # note: e2 not used, store in db here anyway...
        ev2 = Event(event_id='event2', webservice_id=ws.id, time=utcnow + timedelta(seconds=5),
                    latitude=89.5, longitude=6, depth_km=7.1, magnitude=56)

        session.add_all([ev1, ev2])

        session.commit()  # refresh datacenter id (alo flush works)

        d = datetime.utcnow()

        s = Station(network='network', station='station', datacenter_id=dct.id, latitude=90,
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

            Channel(location='04', channel='HHZ', sample_rate=6),

            Channel(location='05', channel='HHE', sample_rate=6),
            Channel(location='05gap_merged', channel='HHN', sample_rate=6),
            Channel(location='05err', channel='HHZ', sample_rate=6),
            Channel(location='05gap_unmerged', channel='HHZ', sample_rate=6)
            ]

        s.channels.extend(channels)
        session.commit()

        fixed_args = dict(datacenter_id=dct.id, download_id=dwl.id)

        # Note: data_gaps_merged is a stream where gaps can be merged via obspy.Stream.merge
        # data_gaps_unmerged is a stream where gaps cannot be merged (is a stream of three
        # different channels of the same event)
        data_gaps_unmerged = data.read("GE.FLT1..HH?.mseed")
        data_gaps_merged = data.read("IA.BAKI..BHZ.D.2016.004.head")
        data_ok = data.read("GE.FLT1..HH?.mseed")

        # create an 'ok' and 'error' Stream, the first by taking the first trace of
        # "GE.FLT1..HH?.mseed", the second by maipulating it
        obspy_stream = data.read_stream("GE.FLT1..HH?.mseed")  # read(BytesIO(data_ok))
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

        for evt, cha in product([ev1], channels):
            val = int(cha.location[:2])
            mseed = data_gaps_merged if "gap_merged" in cha.location else \
                data_err if "err" in cha.location else \
                data_gaps_unmerged if 'gap_unmerged' in cha.location else data_ok
            seedid = seedid_gaps_merged if "gap_merged" in cha.location else \
                seedid_err if 'err' in cha.location else \
                seedid_gaps_unmerged if 'gap_unmerged' in cha.location else seedid_ok

            # set times. For everything except data_ok, we set a out-of-bounds time:
            start_time = evt.time - timedelta(seconds=5)
            arrival_time = evt.time - timedelta(seconds=4)
            end_time = evt.time - timedelta(seconds=1)

            if "gap_merged" not in cha.location and 'err' not in cha.location and \
                    'gap_unmerged' not in cha.location:
                start_time = obspy_trace.stats.starttime.datetime
                arrival_time = (obspy_trace.stats.starttime +
                                (obspy_trace.stats.endtime -
                                 obspy_trace.stats.starttime)/2).datetime
                end_time = obspy_trace.stats.endtime.datetime

            seg = Segment(request_start=start_time,
                          arrival_time=arrival_time,
                          request_end=end_time,
                          data=mseed,
                          data_seed_id=seedid,
                          event_distance_deg=val,
                          event_id=evt.id,
                          **fixed_args)
            cha.segments.append(seg)

        session.commit()

        self.inventory_bytes = data.read("GE.FLT1.xml")
        self.inventory = data.read_inv("GE.FLT1.xml")

        pfile, cfile = get_templates_fpaths('paramtable.py', 'paramtable.yaml')
        self.pymodule = load_source(pfile)
        self.config = yaml_load(cfile)

        # remove segment_select, we use all segments here:
        self.config.pop('segment_select', None)

    @staticmethod
    def plotslen(plotmanager, preprocessed):
        '''total number of non-null plots'''
        i = 1 if preprocessed else 0
        return sum(1 if p[i] else 0 for p in plotmanager.values())

    @staticmethod
    def computedplotslen(plotmanager, seg_id, preprocessed, allcomponents=False):
        '''total number of non-null plots for a given segment'''
        i = 1 if preprocessed else 0
        plotlists = [plotmanager[seg_id][i]]
        if allcomponents:
            for segid in plotlists[0].oc_segment_ids:
                if plotmanager[segid][i] is not None:  # is preprocessed = True, it might be None
                    plotlists.append(plotmanager[segid][i])
        n = 0
        for plotlist in plotlists:
            for plot in plotlist:
                if plot is not None:
                    n += 1
        return n

    @staticmethod
    def traceslen(plotmanager, seg_id, preprocessed, allcomponents=False):
        '''total number of traces to be plot for a given segment
        '''
        i = 1 if preprocessed else 0
        plotlists = [plotmanager[seg_id][i]]
        has_components = False
        if allcomponents:
            for segid in plotlists[0].oc_segment_ids:
                plotlists.append(plotmanager[segid][i])
                has_components = True
        n = 0
        for plotlist in plotlists:
            streamorexc = plotlist.data['stream']
            num = len(streamorexc) if isinstance(streamorexc, Stream) else 0 if \
                (allcomponents and has_components) else 1
            n += num
        return n

    @patch('stream2segment.process.db.get_inventory')
    @patch('stream2segment.process.db.get_stream')
    def test_view_other_comps(self, mock_get_stream, mock_get_inv, db):

        components_count = {}  # group_id -> num expected components
        # where group_id is the tuple (event_id, channel.location)
        for sess in db.session.query(Segment):
            group_id = (sess.event_id, sess.channel.location)
            if group_id not in components_count:
                # we should have created views also for the other components. To get
                # other components, use the segment channel and event id
                other_comps_count = db.session.query(Segment).join(Segment.channel).\
                    filter(and_(Segment.event_id == sess.event_id,
                                Channel.location == sess.channel.location)).count()

                components_count[group_id] = other_comps_count

        def assert_(plotlist, segment, preprocessed, is_invalidated=False):
            '''does some assertion
            :preprocessed: if the plotlist refers toa  preprocessed segment
            :is_invalidated: if the plotlist has been invalidated. Valid only
            if preprocessed=False (otherwise plotlist should be None)
            This is used for checking that plotlist.data['sn_windows'] is None
            '''
            # raw:
            # gap: stream: ok, sn_spectra exc, sn_windows: none
            # err: stream exc, sn_spectra: exc, sn_windows: none
            # other: stream ok, sn_spectra:ok, sn_windows: non-none
            # preprocessed
            iserr = 'err' in segment.channel.location
            hasgaps = 'gap' in segment.channel.location
            isinverr = isinstance(plotlist.data['stream'], Exception) and \
                "inventory" in str(plotlist.data['stream'])
            if preprocessed:
                if iserr or hasgaps or isinverr:
                    assert len("".join(plotlist[0].warnings))
                    assert isinstance(plotlist.data['stream'], Exception)
                    # if stream has an exception, as we use the stream for the sn_windows, assert
                    # the exception is the same:
                    assert plotlist.data['sn_windows'] == plotlist.data['stream']
                    if plotlist[1] is not None:
                        assert len("".join(plotlist[1].warnings))
                else:
                    assert not len("".join(plotlist[0].warnings))
                    assert isinstance(plotlist.data['stream'], Stream)
                    # assert sn_windows are correct:
                    sn_wdw = plotlist.data['sn_windows']
                    assert len(sn_wdw) == 2
                    assert all(isinstance(_, UTCDateTime)
                               for _ in list(sn_wdw[0]) + list(sn_wdw[1]))
                    if plotlist[1] is not None:
                        assert not len("".join(plotlist[1].warnings))
            else:
                # test sn_windows first:
                if is_invalidated:
                    assert plotlist.data['sn_windows'] is None  # reset from invalidation
                    assert all(p is None for p in plotlist)
                    return
                elif iserr:
                    assert isinstance(plotlist.data['sn_windows'], Exception)
                    # assert also that it is the same exception raised from stream:
                    assert plotlist.data['stream'] == plotlist.data['sn_windows']
                elif 'gap' in segment.channel.location:
                    # sn_windows should raise, as we have more than one trace:
                    assert isinstance(plotlist.data['sn_windows'], Exception)
                    assert "gap" in str(plotlist.data['sn_windows'])
                else:  # good segment
                    # if the stream is unprocessed, and it was successfully loaded, assert
                    # sn_windows are correct:
                    sn_wdw = plotlist.data['sn_windows']
                    assert len(sn_wdw) == 2
                    assert all(isinstance(_, UTCDateTime)
                               for _ in list(sn_wdw[0]) + list(sn_wdw[1]))
                # test other stuff:
                if iserr:
                    assert len("".join(plotlist[0].warnings))
                    assert isinstance(plotlist.data['stream'], Exception)
                    if plotlist[1] is not None:
                        assert len("".join(plotlist[1].warnings))
                else:
                    assert isinstance(plotlist.data['stream'], Stream)
                    if "gap_unmerged" in segment.channel.location:
                        # assert that traces labels (d[-1]) are displayed with their seed_id.
                        # To prove that,
                        # assert that we didn't named each trace as "chunk1", "chunk2" etcetera:
                        assert all("chunk" not in d[-1] for d in plotlist[0].data)
                    elif hasgaps:
                        assert "gaps/overlaps" in "".join(plotlist[0].warnings)
                        # assert that we display all traces with "chunk1", "cunk2" etcetera:
                        assert all("chunk" in d[-1] for d in plotlist[0].data)
                    else:
                        assert not len("".join(plotlist[0].warnings))
                    if plotlist[1] is not None:
                        if hasgaps:
                            assert len("".join(plotlist[1].warnings))
                        else:
                            assert not len("".join(plotlist[1].warnings))

        for sess in db.session.query(Segment):

            pmg = PlotManager(self.pymodule, self.config)

            expected_components_count = components_count[(sess.event_id, sess.channel.location)]

            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            mock_get_stream.side_effect = original_get_stream
            mock_get_inv.side_effect = original_get_inventory
            allcomponents = True
            preprocessed = False
            idxs = [0]
            # s_id_was_in_views = sess.id in pmg._plots
            plots = pmg.get_plots(db.session, sess.id, idxs, preprocessed, allcomponents)
#             # assert returned plot has the correct number of time/line-series:
#             # note that plots[0] might be generated from a stream with gaps
            assert len(plots[0].data) == self.traceslen(pmg, sess.id, preprocessed, allcomponents)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.plotslen(pmg, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents=False) == \
                len(idxs)
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents) == \
                expected_components_count
            # assert SegmentWrapper function calls:
            assert not mock_get_inv.called  # preprocess=False
            assert mock_get_stream.call_count == expected_components_count
            # assert we did not calculate any useless stream:
            assert_(pmg[sess.id][0], sess, preprocessed=False)
            assert pmg[sess.id][1] is None

            # from here on, try to calculate the plots for 3 types: main plot (index 0)
            # index of cumulative, and index of spectra
            CUMUL_INDEX, SN_INDEX, DERIVCUM2_INDEX = [None] * 3  # pylint: disable=invalid-name
            for i, p in enumerate(pmg.userdefined_plots):
                if p['name'] == 'cumulative':
                    CUMUL_INDEX = p['index']
                elif p['name'] == 'sn_spectra':
                    SN_INDEX = p['index']
                elif p['name'] == 'derivcum2':
                    DERIVCUM2_INDEX = p['index']

            if CUMUL_INDEX is None or SN_INDEX is None or DERIVCUM2_INDEX is None:
                raise Exception('either the test function names have to be changed, or '
                                'the processing file needs to implement "cumulative" and '
                                '"sn_spectra" and "derivcum2"')
            idxs = [0, SN_INDEX, CUMUL_INDEX, DERIVCUM2_INDEX]

            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            allcomponents = True
            preprocessed = False
            # s_id_was_in_views = sess.id in pmg._plots
            plots = pmg.get_plots(db.session, sess.id, idxs, preprocessed, allcomponents)
#           # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert not self.plotslen(pmg, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents=False) == \
                len(idxs)
            # if we calculate all components, we should have expected components count PLUS
            # all plots which are not the main plot (index 0):
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents) == \
                expected_components_count + sum(_ != 0 for _ in idxs)
            # assert SegmentWrapper function calls:
            assert not mock_get_inv.called  # preprocess=False
            assert not mock_get_stream.called  # already computed
            # assert we did not calculate any useless stream:
            assert_(pmg[sess.id][0], sess, preprocessed=False)
            assert pmg[sess.id][1] is None

            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            allcomponents = False
            preprocessed = True
            # s_id_was_in_views = sess.id in pmg._plots
            plots = pmg.get_plots(db.session, sess.id, idxs, preprocessed, allcomponents)
#           # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert self.plotslen(pmg, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents=False) == \
                len(idxs)
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents) == len(idxs)
            # assert SegmentWrapper function calls:
            if 'err' not in sess.channel.location and 'gap' not in sess.channel.location:
                assert mock_get_inv.called  # preprocess=False
            else:
                assert not mock_get_inv.called
            assert not mock_get_stream.called  # already computed
            # assert we did not calculate any useless stream:
            assert_(pmg[sess.id][0], sess, preprocessed=False)
            assert_(pmg[sess.id][1], sess, preprocessed=True)

            mock_get_stream.reset_mock()
            mock_get_inv.reset_mock()
            allcomponents = True
            preprocessed = True
            # s_id_was_in_views = sess.id in pmg._plots
            plots = pmg.get_plots(db.session, sess.id, idxs, preprocessed, allcomponents)
            # asssert the returned value match the input:
            assert len(plots) == len(idxs)
            assert self.plotslen(pmg, preprocessed=True)  # assert no filtering calculated
            # assert we did not calculate other components (all_components=False)
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents=False) == \
                len(idxs)
            # regardless whether allcomponents is true or false, we compute only the main plot
            assert self.computedplotslen(pmg, sess.id, preprocessed, allcomponents) == len(idxs)
            # assert SegmentWrapper function calls:
            assert not mock_get_inv.called  # already called
            assert not mock_get_stream.called  # already computed
            # assert all titles are properly set, with the given prefix
            seedid = sess.seed_id
            assert all(p is None or p.title.startswith(seedid) for p in pmg[sess.id][0])
            assert all(p is None or p.title.startswith(seedid) for p in pmg[sess.id][1])
            # check plot titles and warnings:
            stream = pmg[sess.id][0].data['stream']
            preprocessed_stream = pmg[sess.id][1].data['stream']
            if 'err' in sess.channel.location:
                assert isinstance(stream, Exception) and \
                    isinstance(preprocessed_stream, Exception) and \
                    'MiniSeed error' in str(preprocessed_stream)
                for i in idxs:
                    plot, pplot = pmg[sess.id][0][i], pmg[sess.id][1][i]
                    assert len(plot.data) == 1  # only one (fake) trace
                    assert plot.warnings
                    assert len(pplot.data) == 1  # only one (fake) trace
                    assert pplot.warnings

            elif 'gap' in sess.channel.location:
                assert isinstance(stream, Stream) and \
                    isinstance(preprocessed_stream, Exception) and \
                    'gaps/overlaps' in str(preprocessed_stream)
                for i in idxs:
                    plot, pplot = pmg[sess.id][0][i], pmg[sess.id][1][i]
                    # if idx=1, plot has 1 series (due to error in gaps/overlaps) otherwise
                    # matches stream traces count:
                    assert len(plot.data) == 1 if i == 1 else len(stream)
                    if i != 0:  # we are iterating over the spectra plots
                        assert "gaps/overlaps" in plot.warnings[0]
                        assert "gaps/overlaps" in pplot.warnings[0]
                    elif i == 0:  # we are iterating over the streams plots
                        if 'gap_unmerged' in sess.channel.location:
                            # assert that we display all traces with their seed_id. To prove that,
                            # assert that we didn't named each trace as "chunk1", "cunk2" etcetera:
                            assert all("chunk" not in d[-1] for d in plot.data)
                        else:
                            assert 'gaps/overlaps' in pplot.warnings[0]
                            # assert that we display all traces with "chunk1", "cunk2" etcetera:
                            assert all("chunk" in d[-1] for d in plot.data)
                    assert len(pplot.data) == 1  # only one (fake) trace
                    assert pplot.warnings and 'gaps/overlaps' in pplot.warnings[0]  # gaps / olaps
            else:
                assert isinstance(stream, Stream) and \
                    isinstance(preprocessed_stream, Exception) and \
                    'Station inventory (xml) error: no data' in str(preprocessed_stream)
                for i in idxs:
                    plot, pplot = pmg[sess.id][0][i], pmg[sess.id][1][i]
                    # if idx=SN_INDEX, plot has 2 series (noie/signal) otherwise matches
                    # vstream traces count:
                    assert len(plot.data) == 2 if i == SN_INDEX else len(stream)
                    assert not plot.warnings  # gaps /overlaps
                    assert len(pplot.data) == 1  # only one (fake) trace
                    assert pplot.warnings and 'inventory' in pplot.warnings[0]  # gaps /overlaps

            # assert we did not calculate any useless stream:
            assert_(pmg[sess.id][0], sess, preprocessed=False)
            assert_(pmg[sess.id][1], sess, preprocessed=True)

            # so we manually set the inventory on the db, discarding it afterwards:
            sess.station.inventory_xml = self.inventory_bytes
            db.session.commit()
            assert sess.station.inventory_xml
            # re-initialize a new PlotManager to assure everything is re-calculated
            # this also sets all cache to None, including pmg.inv_cache:
            pmg = PlotManager(self.pymodule, self.config)

            # calculate plots
            pmg.get_plots(db.session, sess.id, idxs, preprocessed=False,
                          all_components_in_segment_plot=True)
            pmg.get_plots(db.session, sess.id, idxs, preprocessed=True,
                          all_components_in_segment_plot=True)
            # and store their values for later comparison
            sn_plot_unprocessed = pmg[sess.id][0][SN_INDEX].data
            sn_plot_preprocessed = pmg[sess.id][1][SN_INDEX].data
            # shift back the arrival time. 1 second is still within the stream time bounds for
            # the 'ok' stream:
            sn_windows = dict(pmg.config['sn_windows'])
            sn_windows['arrival_time_shift'] -= 1
            pmg.update_config(sn_windows=sn_windows)
            # assert we restored streams that have to be invalidated, and we kept those not to
            # invalidate:
            assert_(pmg[sess.id][0], sess, preprocessed=False, is_invalidated=True)
            assert pmg[sess.id][1] is None
            # and run again the get_plots: with preprocess=False
            plots = pmg.get_plots(db.session, sess.id, idxs, preprocessed=False,
                                  all_components_in_segment_plot=True)
            assert_(pmg[sess.id][0], sess, preprocessed=False)
            assert pmg[sess.id][1] is None
            sn_plot_unprocessed_new = pmg[sess.id][0][SN_INDEX].data
            # we changed the arrival time and both the signal and noise depend on the cumulative,
            # thus changing the arrival time does change them signal window s_stream
            # Conversely, n_stream should change BUT only for the 'ok' stream (no 'gap' or 'err'
            # in sess.channel.location) as for the other we explicitly set a miniseed starttime,
            # endtime BEFORE the event time which should result in noise stream all padded with
            # zeros regardless of the arrival time shift
            if len(sn_plot_unprocessed) == 1:
                # there was an error in sn ratio (e.g., gaps, overlaps in source stream):
                assert len(sn_plot_unprocessed_new) == 1
            else:
                # both signal and noise plots are different. Check it:
                sig_array_new, sig_array_old = \
                    sn_plot_unprocessed_new[0][2], sn_plot_unprocessed[0][2]
                noi_array_new, noi_array_old = \
                    sn_plot_unprocessed_new[1][2], sn_plot_unprocessed[1][2]

                assert len(sig_array_new) != len(sig_array_old)
                assert len(noi_array_new) != len(noi_array_old) or \
                    not np.allclose(noi_array_new, noi_array_old, equal_nan=True)

            # now run again with preprocessed=True.
            plots = pmg.get_plots(db.session, sess.id, idxs, preprocessed=True,
                                  all_components_in_segment_plot=True)
            sn_plot_preprocessed_new = pmg[sess.id][1][SN_INDEX].data
            # assert the s_stream differs from the previous, as we changed the signal/noise
            # arrival time shift this must hold only for the 'ok' stream (no 'gap' or 'err'
            # in sess.channel.location) as for the other we explicitly set a miniseed starttime,
            # endtime BEFORE the event time (thus by shifting BACK the arrival time we should
            # not see changes in the sess/n stream windows)
            if len(sn_plot_preprocessed) == 1:
                # there was an error in sn ratio (e.g., gaps, overlaps in source stream):
                assert len(sn_plot_preprocessed_new) == 1
            else:
                # both signal and noise plots are different. Check it:
                sig_array_new, sig_array_old = \
                    sn_plot_unprocessed_new[0][2], sn_plot_unprocessed[0][2]
                noi_array_new, noi_array_old = \
                    sn_plot_unprocessed_new[1][2], sn_plot_unprocessed[1][2]

                assert len(sig_array_new) != len(sig_array_old)
                assert len(noi_array_new) != len(noi_array_old) or \
                    not np.allclose(noi_array_new, noi_array_old, equal_nan=True)

            assert_(pmg[sess.id][1], sess, preprocessed=True)
            # re-set the inventory_xml to None:
            sess.station.inventory_xml = None
            db.session.commit()
            assert not sess.station.inventory_xml


def test_limited_size_dict():

    @contextmanager
    def setup(limited_size_dict_instance):
        with patch.object(LimitedSizeDict, '_size_limit_popped',
                          wraps=limited_size_dict_instance._size_limit_popped) as _mock_popitem:
            yield limited_size_dict_instance, _mock_popitem

    # test LimitedSizeDict with no size_limit arg (no size limit):
    with setup(LimitedSizeDict()) as (lsd, mock_popitem):
        for a in range(10000):
            lsd[a] = 5
        assert not mock_popitem.called

    with setup(LimitedSizeDict(size_limit=50)) as (lsd, mock_popitem):
        for a in range(50):
            lsd[a] = 5
        assert not mock_popitem.called

    with setup(LimitedSizeDict(size_limit=50)) as (lsd, mock_popitem):
        for a in range(51):
            lsd[a] = 5
        assert mock_popitem.call_count == 1

    # test wrong argument in update:
    with pytest.raises(TypeError):
        with setup(LimitedSizeDict(size_limit=50)) as (lsd, mock_popitem):
            lsd.update(*[{str(_): 1} for _ in range(101)])

    # test update and setdefault:
    with setup(LimitedSizeDict(size_limit=50)) as (lsd, mock_popitem):
        lsd.update({str(_): 1 for _ in range(101)})
        assert mock_popitem.call_count == 101-50

    # lsd = LimitedSizeDict(size_limit=50)
    with setup(LimitedSizeDict(size_limit=50)) as (lsd, mock_popitem):
        lsd.update(**{str(_): 1 for _ in range(101)})
        assert mock_popitem.call_count == 101-50

    # lsd = LimitedSizeDict(size_limit=50)
    with setup(LimitedSizeDict(size_limit=50)) as (lsd, mock_popitem):
        for _ in range(101):
            lsd.setdefault(str(_), 1)
        assert mock_popitem.call_count == 101-50


def test_inv_cache(data):

    class Segment(object):

        def __init__(self, id_, staid):
            if staid is not None:
                self.station = Segment(staid, None)
            else:
                self.station = None
            self.id = id_

    def_size_limit = _default_size_limits()[1]

    inventory = data.read_inv("GE.FLT1.xml")

    @contextmanager
    def setup(inv_cache_instance):
        with patch.object(InventoryCache, '_size_limit_popped',
                       wraps=inv_cache_instance._size_limit_popped) as _mock_popitem:
            yield inv_cache_instance, _mock_popitem

    # test LimitedSizeDict with no size_limit arg (no size limit):
    with setup(InventoryCache()) as (inv, mock_popitem):
        for i in range(def_size_limit):
            inv[Segment(i, 1)] = inventory
        assert not mock_popitem.called

    # test supplying always the same inventory
    with setup(InventoryCache()) as (inv, mock_popitem):
        # same station id (1) for all segments:
        # it does not matter: keys are removed if they have the same value
        # (via 'is' keyword), thus all keys will be removed
        for i in range(def_size_limit+1):
            inv[Segment(i, 1)] = inventory
        assert mock_popitem.call_count == 1
        assert len(inv) == 0
    with setup(InventoryCache()) as (inv, mock_popitem):
        # different station ids (0,1,2,..) for all segments:
        # it does not matter: keys are removed if they have the same
        # value (via 'is' keyword), thus all keys will be removed
        mock_popitem.reset_mock()
        for i in range(def_size_limit+1):
            inv[Segment(i, i)] = inventory
        assert mock_popitem.call_count == 1
        assert len(inv) == 0
    with setup(InventoryCache()) as (inv, mock_popitem):
        # now provide different objects, we remove only one element
        mock_popitem.reset_mock()
        for i in range(def_size_limit+1):
            inv[Segment(i, i)] = ValueError('a')
        assert mock_popitem.call_count == 1
        assert len(inv) == def_size_limit
    with setup(InventoryCache()) as (inv, mock_popitem):
        # now provide same object again, we are again with 0 items
        v = ValueError('a')
        mock_popitem.reset_mock()
        for i in range(def_size_limit+1):
            inv[Segment(i, i)] = v
        assert mock_popitem.call_count == 1
        assert len(inv) == 0

