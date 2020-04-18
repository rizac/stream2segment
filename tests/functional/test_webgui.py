# @PydevCodeAnalysisIgnore
'''
Created on Jul 15, 2016

@author: riccardo
'''

import os
import sys
from io import BytesIO
import json
from itertools import product
from datetime import datetime, timedelta

from mock.mock import patch
import pytest
import numpy as np
from obspy.core.stream import read

from stream2segment.io.db.models import Event, WebService, Channel, Station, \
    DataCenter, Segment, Class, Download, ClassLabelling
from stream2segment.utils.resources import get_templates_fpaths

# from stream2segment.gui.webapp import get_session
from stream2segment.gui.main import create_s2s_show_app
from stream2segment.gui.webapp.mainapp import db as db_module, core as core_module


class Test(object):
    pyfile, configfile = get_templates_fpaths("paramtable.py", "paramtable.yaml")

     # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=True)
        self.app = create_s2s_show_app(db.dburl, self.pyfile, self.configfile)
        
        with self.app.app_context():
            
            # hack to set a scoped session on pur db
            db._session = self.session = db_module.get_session()
        
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

            data_gaps_unmerged = data.to_segment_dict("GE.FLT1..HH?.mseed")
            data_gaps_merged = data.to_segment_dict("IA.BAKI..BHZ.D.2016.004.head")
            obspy_trace = read(BytesIO(data_gaps_unmerged['data']))[0]
            # write data_ok is actually bytes data of 3 traces, write just the first one, we have
            # as it is it would be considered a trace with gaps, wwe have
            # another trace with gaps
            b = BytesIO()
            obspy_trace.write(b, format='MSEED')
            start, end = obspy_trace.stats.starttime.datetime, obspy_trace.stats.endtime.datetime
            data_ok = dict(data_gaps_unmerged, data=b.getvalue(),
                           start_time=start,
                           end_time=end,
                           arrival_time=start + (end-start)/3)
            data_err = dict(data_ok, data=data_ok['data'][:5])

            for ev, c in product([e1, e2], channels):
                val = int(c.location[:2])
                data_atts = data_gaps_merged if "gap_merged" in c.location else \
                    data_err if "err" in c.location else data_gaps_unmerged \
                    if 'gap_unmerged' in c.location else data_ok
                seg = Segment(event_distance_deg=val,
                              event_id=ev.id,
                              datacenter_id=dc.id,
                              download_id=run.id,
                              **data_atts
                              )
                c.segments.append(seg)

            session.commit()

            # set inventory
            self.inventory = data.read_inv("GE.FLT1.xml")


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
    
    def index_of(self, segment_id):
        '''Returns the index of the segment with given id. 
        The web application gets data from segment index'''

        segsids = db_module.get_segment_ids(core_module.get_select_conditions(),
                                            limit=1000, offset=0)
        for i, _ in enumerate(segsids):
            if _ == segment_id:
                return i
        raise ValueError('segment id %d not found' % segment_id)

    def test_root_no_config_and_pyfile_and_classes(self,
                                                   # fixtures:
                                                   db):

        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            assert not self.session.query(Class).all()
            app = self.app.test_client()
            resp = app.get('/')
            assert resp.status_code == 200
            assert not self.session.query(Class).all()
            # https://github.com/pallets/flask/issues/716 is bytes in python3. Fix for both 2 and 3:
            response_data = resp.data.decode('utf-8')
            assert '"hasPreprocessFunc": true' in response_data \
                or "'hasPreprocessFunc': true" in response_data
            assert '"config": {}' not in response_data and "'config': {}" not in response_data

        # ------------------------
        # TEST NOW WITH NO CONFIG:
        # ------------------------
        core_module._reset_global_config()
        core_module._reset_global_functions()

        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            assert not self.session.query(Class).all()
            app = self.app.test_client()
            resp = app.get('/')
            assert resp.status_code == 200
            assert not self.session.query(Class).all()
            response_data = resp.data.decode('utf-8')
            assert '"hasPreprocessFunc": false' in response_data \
                or "'hasPreprocessFunc': false" in response_data
            # we do not inject the config in the html anymore:
            assert "config:" not in response_data
            
            resp = app.post("/set_selection",
                          data=json.dumps({}),
                          headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            assert not data['error_msg'] and data['num_segments']
            
            
            resp = app.post("/get_config",
                          data=json.dumps({'asstr': True}),
                          headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            assert not data['error_msg'] and not data['data']
            
            d = dict(seg_index=1,
                     pre_processed=True,
                     # zooms = data['zooms']
                     plot_indices=[0],  # data['plotIndices']
                     metadata=True,
                     classes=True,
                     all_components=True)
            resp = app.post("/get_segment", data=json.dumps(d),
                          headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            # https: 
            data = self.jsonloads(resp.data)
            assert ['classes', 'metadata', 'plot_types', 'plots',
                    'seg_id', 'sn_windows'] == sorted(data.keys())

            # assert '"config": {}' in response_data or "'config': {}" in response_data


    def test_root(self,
                  # fixtures:
                  db):

        core_module.g_config['class_labels'] = {'wtf': 'abc'}
        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            app = self.app.test_client()
            clz = self.session.query(Class).count()
            assert clz == 0

            resp = app.get('/')
            clz = self.session.query(Class).all()
            assert len(clz) == 1 and clz[0].label == 'wtf' and clz[0].description == 'abc'
            assert resp.status_code == 200
            response_data = resp.data.decode('utf-8')
            # we do not inject the config in the html anymore:
#             # assert global yaml config vars are injected as javascript from jinja rendering:
#             # (be relaxed, if we change the template yaml file we do not want to fail)
#             expected_str = """var __SETTINGS = {"config": {"""
#             # https://github.com/pallets/flask/issues/716 is bytes in python3. Fix for both 2 and 3:
#             assert expected_str in response_data

            # simple test:
            assert '"hasPreprocessFunc": true' in response_data \
                or "'hasPreprocessFunc': true" in response_data

            expected_str = ["""<div class='plot-wrapper' ng-show='plots[{0:d}].visible'>""",
                            """<div data-plot='time-series' data-plotindex={0:d} class='plot'></div>"""]
            # In the default processing, we implemented 6 plots, assure they are there:
            for plotindex in range(6):
                assert "<div id='plot-{0:d}' class='plot'".format(plotindex) in response_data

            self.session.query(Class).delete()
            self.session.commit()
            clz = self.session.query(Class).count()
            assert clz == 0
 
            # now the plotmanager does not set any class:
            del core_module.g_config['class_labels']
            resp = app.get('/')
            assert resp.status_code == 200
            clz = self.session.query(Class).count()
            # assert nothing has changed (same as previous assert):
            assert clz == 0

    def test_get_segs(self, db):  # db is a fixture (see conftest.py). Even if not used, it will
        # assure this function is run once for each given dburl
        with self.app.app_context():
            app = self.app.test_client()
            app.get('/')  # initializes plot manager
            # test your app context code
            resp = app.post("/set_selection",
                          data=json.dumps(dict(segment_select={'has_data':'true'})),
                          headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            assert data['num_segments'] == 28
            assert data['error_msg'] == ''
            assert sorted(data.keys()) == ['error_msg', 'num_segments']

            # test no selection:
            resp = app.post("/set_selection",
                          data=json.dumps(dict(segment_select={'id':'-100000'})),
                          headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            assert data['num_segments'] == 0
            assert data['error_msg'] == ''
            
            # test no selection (but due to selection error, e.g. int overflow)
            resp = app.post("/set_selection",
                          data=json.dumps(dict(segment_select={'id':'9' * 10000})),
                          headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            assert data['num_segments'] == 0
            # the overflow error is raised only in sqlite:
            if db.is_sqlite:
                assert data['error_msg'] != ''

    def test_toggle_class_id(self,
                             # fixtures:
                             db):
        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            app = self.app.test_client()
            app.get('/')  # initializes plot manager
            app.post("/set_selection", data=json.dumps(dict(segment_select={'has_data':'true'})),
                               headers={'Content-Type': 'application/json'})
            segid = 1
            segment = self.session.query(Segment).filter(Segment.id == segid).first()
            c = Class(label='label')
            self.session.add(c)
            self.session.commit()
            cid = c.id
            assert len(segment.classes) == 0

            resp = app.post("/set_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid,
                                                               'value':True}),
                                   headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            # need to remove the session and query again from a new one (WHY?):
            self.session.remove()
            segment = self.session.query(Segment).filter(Segment.id == segid).first()
            # check the segment has classes:
            assert len(segment.classes) == 1
            assert segment.classes[0].id == cid

            # toggle value (now False):
            resp = app.post("/set_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid,
                                                               'value':False}),
                                   headers={'Content-Type': 'application/json'})
            # need to remove the session and query again from a new one (WHY?):
            self.session.remove()
            segment = self.session.query(Segment).filter(Segment.id == segid).first()
            # check the segment has no classes:
            assert len(segment.classes) == 0
            assert resp.status_code == 200

            # toggle again and run test_get_seg with a class set
            resp = app.post("/set_class_id", data=json.dumps({'segment_id':segid, 'class_id':cid,
                                                               'value': True}),
                                   headers={'Content-Type': 'application/json'})
            # need to remove the session and query again from a new one (WHY?):
            self.session.remove()
            segment = self.session.query(Segment).filter(Segment.id == segid).first()
            # check the segment has classes:
            assert len(segment.classes) == 1
            assert resp.status_code == 200
            self._tst_get_seg(app)

    def test_get_seg(self,
                     # fixtures:
                     db):
        # assure this function is run once for each given dburl
        with self.app.test_request_context():
            app = self.app.test_client()
            app.get('/')
            # change selection to be more relaxed (by default it should have
            # maxgap_numsamples within (-0.5, 0.5) in the selection,
            # but we built a test dataset with all maxgap_numsamples = None,
            # thus keep only has_data':'true' in the selection):
            app.post("/set_selection", data=json.dumps(dict(segment_select={'has_data':'true'})),
                               headers={'Content-Type': 'application/json'})
            self._tst_get_seg(app)

    def _tst_get_seg(self, app):

        has_labellings = self.session.query(ClassLabelling).count() > 0        
            
        for _ in product([[0, 1, 2], [], [0]], [True, False], [True, False], [True, False], [True, False]):
            plot_indices, preprocessed, metadata, classes, all_components = _

            d = dict(seg_index=self.index_of(1),
                     pre_processed=preprocessed,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=all_components)

            resp = app.post("/get_segment", data=json.dumps(d),
                          headers={'Content-Type': 'application/json'})
            # https: 
            data = self.jsonloads(resp.data)
            assert len(data['plots']) == len(d['plot_indices'])
            assert bool(len(data['metadata'])) == metadata
            assert bool(len(data['classes'])) == (classes and has_labellings)

            if 0 in plot_indices:
                traces_in_first_plot = len(data['plots'][plot_indices.index(0)][1])
                assert (traces_in_first_plot == 1 and not all_components) or traces_in_first_plot >= 1
            # we should add a a test for the pre_processed case also, but we should inspect the plotmanager defined as app['PLOTMANAGER'] 
            # we should add a test for the zooms, too

    def test_segment_sa_station_inv_errors_in_preprocessed_traces(self,
                                                                  # fixtures:
                                                                  db):
        ''''''
        plot_indices = [0]
        metadata = False
        classes = False

        with self.app.test_request_context():
            app = self.app.test_client()
            
            # change selection to be more relaxed (by default it should have
            # maxgap_numsamples within (-0.5, 0.5) in the selection,
            # but we built a test dataset with all maxgap_numsamples = None,
            # thus keep only has_data':'true' in the selection):
            app.post("/set_selection",
                     data=json.dumps(dict(segment_select={'has_data':'true'})),
                     headers={'Content-Type': 'application/json'})
            
            d = dict(seg_index=self.index_of(1),
                     pre_processed=False,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=True)
            resp = app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            plots = self.jsonloads(resp.data)['plots']
            # each plot is the jsonified version of a plot, i.e. a list of
            # [title, data, warnings, is_timeserie]
            assert not any(len(p[2]) for p in plots)

        # Now we exited the session, we try with pre_processed=True
        with self.app.test_request_context():
            app = self.app.test_client()
            d = dict(seg_index=self.index_of(1),
                     pre_processed=True,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=True)
            resp = app.post("/get_segment", data=json.dumps(d),
                               headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            plots = self.jsonloads(resp.data)['plots']
            # each plot is the jsonified version of a plot, i.e. a list of
            # [title, data, warnings, is_timeserie]
            assert all("Station inventory (xml) error" in p[2] for p in plots)


    @pytest.mark.parametrize('calculate_sn_spectra', [True, False])
    def test_change_config(self, calculate_sn_spectra,
                           # fixtures:
                           db):
        '''test a change in the config from within the GUI'''
        plot_indices = [0]
        index_of_sn_spectra = None
        if calculate_sn_spectra:
            for ud in core_module.userdefined_plots:
                if ud['name'] == 'sn_spectra':
                    index_of_sn_spectra = ud['index']
                    plot_indices.append(index_of_sn_spectra)
                    break
        metadata = False
        classes = False

        with self.app.test_request_context():
            app = self.app.test_client()
            
            # change selection to be more relaxed (by default it should have
            # maxgap_numsamples within (-0.5, 0.5) in the selection,
            # but we built a test dataset with all maxgap_numsamples = None,
            # thus keep only has_data':'true' in the selection):
            app.post("/set_selection",
                     data=json.dumps(dict(segment_select={'has_data':'true'})),
                     headers={'Content-Type': 'application/json'})
            
            d = dict(seg_index=self.index_of(1),
                     pre_processed=False,
                     # zooms = data['zooms']
                     plot_indices=plot_indices,  # data['plotIndices']
                     metadata=metadata,
                     classes=classes,
                     all_components=False)
            resp1 = app.post("/get_segment", data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})

            # now change the config:
            config = core_module.get_config(asstr=False)
            config['sn_windows']['arrival_time_shift'] += .2  # shift by .2 second
            d['config'] = config
#             # the dict passed from the client to the server has only strings, thus:
#             for key in list(d['config'].keys()):
#                 d['config'][key] = json.dumps(d['config'][key])
            resp2 = app.post("/get_segment", data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})

            try:
                data1 = json.loads(resp1.data)
                data2 = json.loads(resp2.data)
            except TypeError:
                # avoid TypeError: the JSON object must be str, not 'bytes'
                # in py 3.5:
                data1 = json.loads(resp1.data.decode('utf8'))
                data2 = json.loads(resp2.data.decode('utf8'))

            assert len(data1['sn_windows']) == 2
            assert len(data2['sn_windows']) == 2
            for wdw1, wdw2 in zip(data1['sn_windows'], data2['sn_windows']):
                assert wdw1 != wdw2

            plots1 = data1['plots']
            plots2 = data2['plots']
            assert len(plots1) == len(plots2) == len(plot_indices)

            for index in range(len(plot_indices)):
                # each plots* is:
                # [plotdata[0], plotdata[2], ...]
                # where each plotdata is:
                # [plot.title or '', data, "\n".join(plot.warnings), plot.is_timeseries]
                # each data is [x0, dx, y, label] all numeric except 'label'
                # see class jsplot
                # get each 'data' 
                plot1data = plots1[index][1]
                plot2data = plots2[index][1]
                plotindex = plot_indices[index]
                expected_lineseries_num = 1 if plotindex != index_of_sn_spectra else 2
                assert len(plot1data) == len(plot2data) == expected_lineseries_num
                for lineseriesindex in range(len(plot1data)):
                    # plots are returned as a list of lineseries:
                    # [name, [startime, step, data, ...]]
                    # ... is other stuff we do not test here (lineseries name)
                    x0a, dxa, ya, labela = plot1data[lineseriesindex]
                    x0b, dxb, yb, labelb = plot2data[lineseriesindex]
                    assert x0a == x0b
                    # the number of points should be equal also for frequencies, as it
                    # depends on the time resolution, which is always the same:
                    assert len(ya) == len(yb)
                    if plotindex != index_of_sn_spectra:
                        assert dxa == dxb
                        assert np.allclose(ya, yb)
                    else:
                        # the delta frequency changes if we move the arrival time shif, so
                        # this might be due to a change in pts of the original time series
                        # too hard to test, skip this:
                        # assert dxa == dxb
                        assert not np.allclose(ya, yb)
