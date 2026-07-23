# @PydevCodeAnalysisIgnore
"""
Created on Jul 15, 2016

@author: riccardo
"""
from io import BytesIO
import json
from itertools import product
from datetime import datetime, timedelta
import re

from unittest.mock import patch
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func
import pytest
import numpy as np
from obspy.core.stream import read

# from stream2segment.process import get_default_segments_selection
from stream2segment.process.db.models import (Event, WebService, Channel, Station,
                                              DataCenter, Segment, Class, Download,
                                              ClassLabelling)
from stream2segment.process.gui.introspection import load_source
from stream2segment.process.db import get_session
from stream2segment.resources import get_templates_fpaths
from stream2segment.io import yaml_load
from stream2segment.process.db.models import get_stream as original_get_stream

from stream2segment.process.gui.main import create_s2s_show_app
from stream2segment.process.gui.webapp.mainapp import core as core_module
from stream2segment.process.gui.webapp.mainapp import db as db_module


@pytest.fixture
def config_dict():
    """global fixture returning the dict from paramtable.yaml"""
    def func(**overridden_pars):
        with open(get_templates_fpath('paramtable.yaml')) as _:
            return {**yaml.safe_load(_), **overridden_pars}
    return func


event_time_with_data = datetime.fromisoformat('2019-02-01T00:00:00')

@pytest.fixture
def db_engine(db_url, test_data_dir):
    """Call `db.create` and then populates the database with the data for
    processing tests
    """
    # re-init a sqlite database (no-op if the db is not sqlite):
    engine = create_engine(db_url, check_db_existence=False)
    from stream2segment.io.db.models import Base
    if database_exists(engine):
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    session = sessionmaker(bind=engine)()

    # Populate the database:
    dwl = DownloadRun()
    session.add(dwl)
    session.commit()

    wsv = WebService(id=1, url='eventws')
    session.add(wsv)
    session.commit()

    now = datetime.now(UTC).replace(tzinfo=None)

    # set up an event:
    ev1 = Event(
        id=1,
        webservice_id=wsv.id,
        eventid='abc1',
        latitude=8,
        longitude=9,
        magnitude=5,
        depth_km=4,
        time=event_time_with_data
    )
    ev2 = Event(
        id=2,
        webservice_id=wsv.id,
        eventid='abc2',
        latitude=8,
        longitude=9,
        magnitude=5,
        depth_km=4,
        time=now + timedelta(seconds=1)
    )
    ev3 = Event(
        id=3,
        webservice_id=wsv.id,
        eventid='abc3',
        latitude=8,
        longitude=9,
        magnitude=5,
        depth_km=4,
        time=now + timedelta(seconds=1)
    )

    session.add_all([ev1, ev2, ev3])
    session.commit()

    dtc = WebService(url='https://domain/fdsnws/1/dataselect/query')
    session.add(dtc)
    session.commit()

    # s_ok stations have lat and lon > 11, other stations do not
    with open(test_data_dir / "inventory_GE.APE.xml", 'rb') as f:
        inv_xml = f.read()
    s_ok = StationXML(data=inv_xml)
    s_no = StationXML(data=None)
    session.add_all([s_ok, s_no])
    session.commit()

    with open(test_data_dir / 'GE.APE..HHE.mseed', 'rb') as f:
        trace_ok_E = f.read()

    trace_ok_Z = re.sub(rb'HHE', b'HHZ', trace_ok_E)
    trace_ok_N = re.sub(rb'HHE', b'HHN', trace_ok_E)

    assert read(BytesIO(trace_ok_N), format='MSEED')[0].stats.channel == 'HHN'
    assert read(BytesIO(trace_ok_Z), format='MSEED')[0].stats.channel == 'HHZ'

    with open(test_data_dir / 'IA.BAKI..BHZ.D.2016.004.head', 'rb') as f:
        trace_gap = f.read()

    c_oks = {o: Channel(
        station_id=s_ok.id,
        data_webservice_id=dtc.id,
        latitude=11,
        longitude=12,
        network_code='GE',
        station_code='APE',
        location_code='',
        # legacy channel_code="ok" becomes:
        band_code='H',
        instrument_code='H',
        orientation_code=o,
        start_time=now,
        sample_rate=56.7
    ) for o in 'ENZ'}
    session.add_all(c_oks.values())
    session.commit()

    net, sta, loc, cha = read(BytesIO(trace_gap), format='MSEED')[0].get_id().split('.')

    c_gap = Channel(
        latitude=-31,
        longitude=-32,
        station_id=s_no.id,
        data_webservice_id=dtc.id,
        network_code=net,
        station_code=sta,
        location_code=loc,
        # legacy channel_code="no" becomes:
        band_code=cha[0],
        instrument_code=cha[1],
        orientation_code=cha[2],
        start_time=now,
        sample_rate=56.7
    )
    session.add(c_gap)
    session.commit()

    atts = {
        'event_distance_km': 35*110,
        'gap_score_percent': 0,
        'noise_window_s': 10,
        'signal_window_s': 30,
    }
    for ev in (ev1, ev2):
        s_gap = Segment(
            channel_id=c_gap.id, event_id=ev.id, station_id=s_no.id, **atts
        )
        s_skip = SkippedSegment(
            channel_id=list(c_oks.values())[0].id, event_id=ev.id, download_code=204
        )
        session.add_all([s_gap, s_skip])
        session.commit()
        ms_gap = MiniSeed(data=trace_gap, id=s_gap.id)
        session.add_all([ms_gap])
        session.commit()

        # add segments with orientation:
        s_ok_E = Segment(
            channel_id=c_oks['E'].id, event_id=ev.id, station_id=s_ok.id, **atts
        )
        s_ok_N = Segment(
            channel_id=c_oks['N'].id, event_id=ev.id, station_id=s_ok.id, **atts
        )
        s_ok_Z = Segment(
            channel_id=c_oks['Z'].id, event_id=ev.id, station_id=s_ok.id, **atts
        )
        session.add_all([s_ok_E, s_ok_N, s_ok_Z])
        session.commit()

        m_ok_E = MiniSeed(data=trace_ok_E, id=s_ok_E.id)
        m_ok_N = MiniSeed(data=trace_ok_N, id=s_ok_N.id)
        m_ok_Z = MiniSeed(data=trace_ok_Z, id=s_ok_Z.id)
        session.add_all([m_ok_E, m_ok_N, m_ok_Z])
        session.commit()

        if ev == ev2:
            session.delete(m_ok_Z)
            session.delete(s_ok_Z)
            session.commit()

    # atts_ok = dict(data.to_segment_dict('GE.APE..HHE.mseed'))
    # atts_gap = data.to_segment_dict('IA.BAKI..BHZ.D.2016.004.head')
    # atts_none = dict(atts_ok, data=b'')

    # for ch_ in (c_ok, c_none):
    #     # ch_.location  below reflects if the station has inv
    #     atts = dict(atts_ok, data_seed_id='%s.ok' % ch_.location, download_code=200)
    #     sg1 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev1.id,
    #                       download_id=dwl.id, event_distance_deg=35, **atts)
    #     atts = dict(atts_gap, data_seed_id='%s.gap' % ch_.location, download_code=200)
    #     sg2 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev2.id,
    #                       download_id=dwl.id, event_distance_deg=35, **atts)
    #     atts = dict(atts_none, data_seed_id='%s.no' % ch_.location, download_code=204)
    #     sg3 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev3.id,
    #                       download_id=dwl.id, event_distance_deg=35, **atts)
    #     session.add_all([sg1, sg2, sg3])
    #     session.commit()

    with patch(
        "stream2segment.process.main.create_engine",
        return_value=engine,
    ):
        yield engine


@patch('stream2segment.process.gui.webapp.mainapp.views.core.get_segment_id')
def test_root_no_config_and_pyfile_and_classes(
    mock_get_segment_id, db_engine
):

    # test some combinations of plots. Return always the same segment,
    # so mock the function returning a segment from a given index:
    def _(*a, **v):
        return self.segment_id
    mock_get_segment_id.side_effect = _

    # assure this function is run once for each given dburl
    with self.app.test_request_context():
        assert not self.session.query(Class).all()
        app = self.app.test_client()
        resp = app.get('/')
        assert resp.status_code == 200
        assert not self.session.query(Class).all()
        # https://github.com/pallets/flask/issues/716 is bytes in python3. Fix for both 2 and 3:
        response_data = resp.data.decode('utf-8')
        assert '"config": {}' not in response_data and "'config': {}" not in response_data

    # ------------------------
    # TEST NOW WITH NO CONFIG:
    # ------------------------
    core_module.reset_global_vars({}, {})
    core_module._reset_global_functions()

    # assure this function is run once for each given dburl
    with self.app.test_request_context():
        assert not self.session.query(Class).all()
        app = self.app.test_client()
        resp = app.get('/')
        assert resp.status_code == 200
        assert not self.session.query(Class).all()
        response_data = resp.data.decode('utf-8')
        # we do not inject the config in the html anymore:
        assert "config:" not in response_data

        resp = app.post("/set_selection",
                        data=json.dumps({}),
                        headers={'Content-Type': 'application/json'})
        assert resp.status_code == 200
        assert int(resp.data) == 28  # not data['error_msg'] and data['num_segments']

        resp = app.post("/get_config",
                      data=json.dumps({'asstr': True}),
                      headers={'Content-Type': 'application/json'})
        assert resp.status_code == 200
        data = self.jsonloads(resp.data)
        assert not data  # empty dict

        d = dict(seg_index=1,  # whatever, not used
                 seg_count=1,  # whatever, not used
                 pre_processed=True,
                 # zooms = data['zooms']
                 plot_names=[""],  # data['plotIndices']
                 metadata=True,
                 classes=True,
                 all_components=True)
        resp = app.post("/get_segment_data", data=json.dumps(d),
                        headers={'Content-Type': 'application/json'})
        assert resp.status_code == 200
        # https:
        data = self.jsonloads(resp.data)
        assert ['attributes', 'classes', 'description',
                'plotData', 'plotLayout'] == sorted(data.keys())


def test_init(db_engine):

    # attach a fake method to Segment where the type is unknown:
    defval = 'a'
    Segment._fake_method = \
        hybrid_property(lambda self: defval,
                        expr=lambda cls: func.substr(cls.download_code, 1, 1))

    with self.app.test_request_context():
        app = self.app.test_client()
        # resp = app.post("/init",
        #                 data=json.dumps({'metadata': True, 'classes': True}),
        #                headers={'Content-Type': 'application/json'})
        # assert resp.status_code == 200
        # data = self.jsonloads(resp.data)
        data = core_module.get_init_data(metadata=True, classes=True)

        a2 = data['metadata']

        # a2 = None  get_metadata(db.session, None)
        # Station.inventory_xml, Segment.data, Download.log,
        # Download.config, Download.errors, Download.warnings,
        # Download.program_version, Class.description
        for excluded in ['station.inventory_xml', 'data', 'download.log',
                         'download.config', 'download.errors', 'download.warnings',
                         'download.program_version', 'class.description']:
            assert not any(_['label'] == excluded for _ in a2)
        assert sum(_['label'].startswith('download.') for _ in a2) > 1
        assert sum(_['label'].startswith('channel.') for _ in a2) > 1
        assert sum(_['label'].startswith('event.') for _ in a2) > 1
        assert sum(_['label'].startswith('station.') for _ in a2) > 1
        assert sum(_['label'].startswith('classes.') for _ in a2) == 0  # no classes selection allowed
        # fake method does not have a python type, not returned:
        assert not any(_['label'] == '_fake_method' for _ in a2)

        # too long to count how many attributes should be missing, launched a test and put the
        # number here (17):
        seg = db.session.query(Segment).first()
        b = db_module.get_metadata(1)
        # Check we do not have relationships. E.g.:
        assert sum("station" == _['label'] for _ in b) == 0
        assert sum("class" in _['label'] for _ in b) == 1  # just classlabels_count (see below)
        assert any(_['label'] == 'classlabels_count' for _ in b)
        assert any(_['label'] == '_fake_method' and _['value'] == 'a' for _ in b)

        delattr(Segment, "_fake_method")
        assert not hasattr(Segment, '_fake_method')


def test_root(db_engine):

    # assure this function is run once for each given dburl
    with self.app.test_request_context():
        app = self.app.test_client()
        clz = self.session.query(Class).count()
        assert clz == 0

        resp = app.get('/')

        assert resp.status_code == 200
        response_data = resp.data.decode('utf-8')

        # In the default processing, we implemented 3 plots, assure they are there:
        assert len(list(re.finditer(r"\sdata-plot=['\"]\[", response_data))) == 3


def test_get_segs(db_engine):
    # assure this function is run once for each given dburl
    with self.app.app_context():
        app = self.app.test_client()
        app.get('/')  # initializes plot manager
        # test your app context code
        resp = app.post("/set_selection",
                        data=json.dumps({'has_data': 'true'}),
                        headers={'Content-Type': 'application/json'})
        assert resp.status_code == 200
        data = self.jsonloads(resp.data)
        assert data == 28

        # test no selection:
        resp = app.post("/set_selection",
                        data=json.dumps({'id': '-100000'}),
                        headers={'Content-Type': 'application/json'})
        assert resp.status_code == 500
        data = self.jsonloads(resp.data)
        assert 'no segment' in data['message'].lower()

        # test no selection (but due to selection error, e.g. int overflow)
        resp = app.post("/set_selection",
                      data=json.dumps({'id': '9' * 10000}),
                      headers={'Content-Type': 'application/json'})
        assert resp.status_code == 500
        data = self.jsonloads(resp.data)
        assert 'ValueError' in data['message']


def test_toggle_class_id(
    self, db_engine
):
    # assure this function is run once for each given dburl
    with self.app.test_request_context():
        app = self.app.test_client()
        # app.get('/')  # initializes plot manager
        resp = app.post("/set_selection",
                        data=json.dumps({'has_data': 'true'}),
                        headers={'Content-Type': 'application/json'})

        num_segments = int(json.loads(resp.data))
        resp = app.post("/get_segment_data",
                        data=json.dumps({'seg_index': 0, 'seg_count': num_segments,
                                         'attributes': True}),
                        headers={'Content-Type': 'application/json'})
        segid = [_ for _ in json.loads(resp.data)['attributes'] if _['label'] == 'id'][0]['value']

        segment = self.session.query(Segment).filter(Segment.id == segid).first()
        c = Class(label='label')  # noqa
        self.session.add(c)
        self.session.commit()
        cid = c.id
        assert segment.classlabels_count == 0

        resp = app.post("/set_class_id",
                        data=json.dumps({
                            'seg_index': 0, 'seg_count': num_segments,
                            'class_id': cid, 'value':True}),
                        headers={'Content-Type': 'application/json'})
        assert resp.status_code == 200
        data = self.jsonloads(resp.data)
        # need to remove the session and query again from a new one (WHY?):
        self.session.remove()
        segment = self.session.query(Segment).filter(Segment.id == segid).first()
        # check the segment has classes:
        assert segment.classlabels_count == 1
        assert segment.classes[0].id == cid

        # toggle value (now False):
        resp = app.post("/set_class_id",
                        data=json.dumps({
                            'seg_index': 0, 'seg_count': num_segments,
                            'class_id': cid, 'value': False
                        }),
                        headers={'Content-Type': 'application/json'})
        # need to remove the session and query again from a new one (WHY?):
        self.session.remove()
        segment = self.session.query(Segment).filter(Segment.id == segid).first()
        # check the segment has no classes:
        assert segment.classlabels_count == 0
        assert resp.status_code == 200


@pytest.mark.parametrize('has_labellings', [True, False])
@patch('stream2segment.process.db.models.get_stream',
       side_effect=original_get_stream)
@patch('stream2segment.process.gui.webapp.mainapp.views.core.get_segment_id')
def test_get_segment(
    mock_get_segment_id,
    mock_get_stream,
    has_labellings,
    db_engine
):
    # assure this function is run once for each given dburl
    # with self.app.test_request_context():
    with self.app.app_context():
        app = self.app.test_client()

        resp = app.post("/set_selection",
                        data=json.dumps({'has_data': 'true'}),
                        headers={'Content-Type': 'application/json'})

        if has_labellings:
            c = Class(label='label')  # noqa
            self.session.add(c)
            self.session.commit()
            cid = c.id

            cl = ClassLabelling(class_id=cid, segment_id=self.segment_id)  # noqa
            self.session.add(cl)
            self.session.commit()

        # test some combinations of plots. Return always the same segment,
        # so mock the function returning a segment from a given index:
        def _(*a, **v):
            return self.segment_id
        mock_get_segment_id.side_effect = _

        # do not use pytest parametrize, as it causes hundreds of db creation
        # destruction (see init) and is inefficient. Also  postgres complains about
        # too many connections (but FIXME: this should never happen,
        # as the session should be removed after each request)
        func_names = [_ for _ in core_module.g_functions if _]
        for plot_names, preprocessed, attributes, classes, all_components in \
            product([["", func_names[0], func_names[1]], [], [""]],
                         [True, False],
                         [True, False],
                         [True, False],
                         [True, False]):
            mock_get_stream.reset_mock()
            d = dict(seg_index=1,  # whatever, not used (see patch above)
                     seg_count=1,  # whatever, not used
                     pre_processed=preprocessed,
                     # zooms = data['zooms']
                     plot_names=plot_names,
                     attributes=attributes,
                     classes=classes,
                     all_components=all_components)

            resp = app.post("/get_segment_data", data=json.dumps(d),
                            headers={'Content-Type': 'application/json'})
            expected_stream_call_count = 1 if len(plot_names) else 0
            if "" in plot_names and all_components:
                expected_stream_call_count += 3  # we should actually check
                # if we have components on the db, we actually have for the
                # segment id 1 so it's fine
            assert mock_get_stream.call_count == expected_stream_call_count
            # https:
            assert resp.status_code == 200
            data = self.jsonloads(resp.data)
            assert len(data['plotData']) == len(d['plot_names'])
            assert bool(len(data['attributes'])) == attributes
            assert bool(len(data['classes'])) == (classes and has_labellings)

            if "" in plot_names:
                traces_in_first_plot = len(data['plotData'][""])
                assert (traces_in_first_plot == 1 and not all_components) \
                       or traces_in_first_plot >= 1
            # we should add a test for the pre_processed case also
            # we should add a test for the zooms, too
            db.session.remove()

@patch('stream2segment.process.gui.webapp.mainapp.views.core.get_segment_id')
def test_segment_sa_station_inv_errors_in_preprocessed_traces(
    mock_get_segment_id, db_engine
):
    """"""

    # test some combinations of plots. Return always the same segment,
    # so mock the function returning a segment from a given index:
    def _(*a, **v):
        return self.segment_id

    mock_get_segment_id.side_effect = _

    plot_names = [""]
    attributes = False
    classes = False

    with self.app.test_request_context():
        app = self.app.test_client()

        # change selection to be more relaxed (by default it should have
        # maxgap_numsamples within (-0.5, 0.5) in the selection,
        # but we built a test dataset with all maxgap_numsamples = None,
        # thus keep only has_data':'true' in the selection):
        app.post("/set_selection",
                 data=json.dumps({'has_data': 'true'}),
                 headers={'Content-Type': 'application/json'})

        d = dict(seg_index=1, # whatever, not used
                 seg_count=1,  # whatever, not used
                 pre_processed=False,
                 plot_names=plot_names,
                 attributes=attributes,
                 classes=classes,
                 all_components=True)
        resp = app.post("/get_segment_data",
                        data=json.dumps(d),
                        headers={'Content-Type': 'application/json'})
        assert resp.status_code == 200
        plots = self.jsonloads(resp.data)['plotData']
        assert all(len(p['y']) for p in plots[""])

    # Now we exited the session, we try with pre_processed=True
    # store empty inventory xml in segment
    sess = db.session()
    sta = sess.query(Segment).filter(Segment.id == self.segment_id).one().station
    inv_xml = sta.inventory_xml
    sta.inventory_xml = b''
    sess.commit()
    try:
        with self.app.test_request_context():
            app = self.app.test_client()
            d = dict(seg_index=1,  # whatever, not used
                     seg_count=1,  # whatever, not used
                     pre_processed=True,
                     plot_names=plot_names,
                     attributes=attributes,
                     classes=classes,
                     all_components=True)
            resp = app.post("/get_segment_data",
                            data=json.dumps(d),
                            headers={'Content-Type': 'application/json'})
            assert resp.status_code == 200
            plots = self.jsonloads(resp.data)['plotData']
            assert isinstance(plots[""], str) \
                   and "Station inventory (xml) error" in plots[""]
    finally:
        sta.inventory_xml = inv_xml
        sess.commit()

@pytest.mark.parametrize('calculate_sn_spectra', [True, False])
@patch('stream2segment.process.gui.webapp.mainapp.views.core.get_segment_id')
def test_change_config(mock_get_segment_id,
                       calculate_sn_spectra,
                       # fixtures:
                       db):
    """test a change in the config from within the GUI"""
    plot_names = [""]
    spectra_plot_func = 'sn_spectra'
    if calculate_sn_spectra:
        plot_names.append(spectra_plot_func)
    attributes = False
    classes = False

    # test some combinations of plots. Return always the same segment,
    # so mock the function returning a segment from a given index:
    def _(*a, **v):
        return self.segment_id

    mock_get_segment_id.side_effect = _

    with self.app.test_request_context():
        app = self.app.test_client()

        app.post("/set_selection",
                 data=json.dumps({'has_data': 'true'}),
                 headers={'Content-Type': 'application/json'})

        d = dict(seg_index=1,  # whatever, not used
                 seg_count=1,  # whatever, not used
                 pre_processed=False,
                 plot_names=plot_names,
                 attributes=attributes,
                 classes=classes,
                 all_components=False)
        resp1 = app.post("/get_segment_data", data=json.dumps(d),
                         headers={'Content-Type': 'application/json'})

        # now change the config:
        config = core_module.get_config(as_str=False)
        config['sn_windows']['arrival_time_shift'] += .2  # shift by .2 second
        d['config'] = config
        resp2 = app.post("/get_segment_data", data=json.dumps(d),
                         headers={'Content-Type': 'application/json'})

        try:
            data1 = json.loads(resp1.data)
            data2 = json.loads(resp2.data)
        except TypeError:
            # avoid TypeError: the JSON object must be str, not 'bytes'
            # in py 3.5:
            data1 = json.loads(resp1.data.decode('utf8'))
            data2 = json.loads(resp2.data.decode('utf8'))

        plots1 = data1['plotData']
        plots2 = data2['plotData']
        assert len(plots1) == len(plots2) == len(plot_names)

        for name in plot_names:
            plot1data = plots1[name]
            plot2data = plots2[name]
            expected_lineseries_num = 1 if name != spectra_plot_func else 2
            assert len(plot1data) == len(plot2data) == expected_lineseries_num
            for lineseriesindex in range(len(plot1data)):
                # plots are returned as a list of lineseries:
                # [name, [startime, step, data, ...]]
                # ... is other stuff we do not test here (lineseries name)
                x0a, dxa, ya, labela = plot1data[lineseriesindex]['x0'], \
                    plot1data[lineseriesindex]['dx'], \
                    plot1data[lineseriesindex]['y'], \
                    plot1data[lineseriesindex]['name']
                x0b, dxb, yb, labelb = plot2data[lineseriesindex]['x0'], \
                    plot2data[lineseriesindex]['dx'], \
                    plot2data[lineseriesindex]['y'], \
                    plot2data[lineseriesindex]['name']
                assert x0a == x0b
                # Consider that plotindices is either 0 or 2
                # 0: time series plot
                # 2: Spectra
                if name != spectra_plot_func:
                    assert dxa == dxb
                    assert len(ya) == len(yb)
                    assert np.allclose(ya, yb)
                else:
                    # spectra are not assured to have the same number of points
                    # because ya and yb are the result of the same segment
                    # but different arrival time shifts (which in turn affects the
                    # spectra length) and also the downsampling (for plot
                    # visualization) applies only after a certain density of points.

                    # So, as rule of thumb, we can just check that the two spectra
                    # are somehow similar. First take first N points for both:
                    _minlen = min(len(ya), len(yb))
                    ya, yb = np.array(ya[:_minlen]), np.array(yb[:_minlen])

                    # Then, check that the mean and median abs
                    # difference of the arrays is below 0.1 (10% difference)
                    # (they are around 0.04 ~= 4%, so 0.1 should be a roughly good
                    # value to check, whilst avoiding false positives):
                    diffs = np.abs(ya - yb) / np.abs(yb)
                    assert np.nanmedian(diffs) < 0.1
                    assert np.nanmean(diffs) < 0.1


def test_all_basehtml_are_the_same():
    """Test that the base.html in all template directories is the same. We could put
    a single base.html file in the "resources" dir but this would mean beoing able to
    load HTML templates from several directories, which is not supported in flask
    """
    import os
    name = 'stream2segment'
    root = os.path.abspath(__file__)
    while os.path.basename(root) != name:
        root2 = os.path.dirname(root)
        if not root2 or root2 == root:
            raise ValueError('%s directory not found' % name)
        root = root2

    root = os.path.join(root, name)
    if not os.path.isdir(root):
        raise ValueError('%s directory not found' % root)

    from pathlib import Path
    html_content = None
    for fle in Path(root).rglob('templates/base.html'):
        with open(fle, 'r') as _:
            _html_content = _.read()
        assert html_content is None or _html_content == html_content
        html_content = _html_content