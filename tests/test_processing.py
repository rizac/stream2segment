"""
test processing routine
"""
# Feb 14, 2017
import os
import re
from dataclasses import fields
from datetime import datetime, UTC, timedelta
from io import BytesIO
from itertools import product
from unittest.mock import patch
import pandas as pd
import pytest
import yaml
from obspy import read
from pandas._testing import assert_frame_equal
from pandas.errors import EmptyDataError
from sqlalchemy.orm import sessionmaker

from stream2segment.io.db import create_engine, database_exists, is_sqlite, is_postgres
from stream2segment.io.db.models import (
    DownloadRun, WebService, Channel, StationXML, MiniSeed, SkippedSegment,Event,
    Segment
)
from stream2segment.process import imap, SegmentMetadata, build_where_clause
from stream2segment.process.main import get_segments
from stream2segment.process.segments_selection import SelectFields, WhereFields, \
    get_orderby_columns, build_select
from stream2segment.resources import get_templates_fpath
from stream2segment.process.main import process


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


# @pytest.fixture
# def log_capture(tmp_path: Path):
#     """
#     Intercept the log file Path created in tested functions to be accessible in tests
#     """
#
#     log_file = tmp_path / 'test.log'
#
#     def wrapper(logger, log_file_path, verbose):
#         return original_start_logging(logger, log_file, verbose)
#
#     with patch(
#         "stream2segment.process.main.start_logging", side_effect=wrapper
#     ):
#         yield log_file

class patches:
    # paths container for class-level patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    # get_session = 'stream2segment.process.main.get_session'
    # close_session = 'stream2segment.process.main.close_session'
    run_process = 'stream2segment.process.main.process'
    configlog4processing = 'stream2segment.process.main.configlog4processing'


# ======== ACTUAL TESTS: ================================


def test_simple_run_no_outfile_provided(
    # fixtures:
    capsys, db_engine, config_dict
):
    """test a case where save inventory is True, and that we saved inventories"""
    from stream2segment.resources.templates import paramtable
    _ = process(
        dburl=str(db_engine.url), pyfunc=paramtable.main,
        # segments_selection={'has_data': 'true'},
        segments_selection={},
        config=config_dict(snr_threshold=0),
        verbose=True, logfile=''
    )

    # assert "Output file:  n/a" in result output:
    _ = capsys.readouterr()
    output, error = _.out, _.err
    assert not error
    assert re.search('Output file:\\s+n/a', output)

    for subs in ["Processing function: "]:
        idx = output.find(subs)
        assert idx > -1


@pytest.mark.parametrize(
    "file_extension, options",
    product(['.h5', '.csv'], [
        # {},
        {'chunksize': 1},
        {'chunksize': 1, 'multi_process': True},
        {'chunksize': 1, 'multi_process': True, 'group_components': True},
        # {'multi_process': True},
        # {'chunksize': 1, 'multi_process': 1},
    ])
)
def test_simple_run_retDict_complex_select(
    file_extension, options,
    # fixtures:
    capsys, db_engine, config_dict, tmp_path
):
    """test a case where we have a more complex select involving joins"""

    from stream2segment.resources.templates import paramtable
    filename = tmp_path / f'output{file_extension}'
    logfile = tmp_path / 'test.log'

    if filename.is_file():
        filename.unlink()

    if logfile.is_file():
        logfile.unlink()

    segs_selection = {
        # 'has_data': 'true',
        'event_time': '<=%s' % (event_time_with_data.isoformat()),
        'network_code': 'GE',
        'orientation_code': 'Z'
    }
    _ = process(
        dburl=str(db_engine.url),
        pyfunc=paramtable.main,
        segments_selection=segs_selection,
        config=config_dict(snr_threshold=0),
        outfile=filename,
        verbose=True,
        logfile=logfile,
        **options
    )

    # check file has been correctly written:
    if file_extension == '.csv':
        dfr = pd.read_csv(filename)
        assert len(dfr) == 1
        assert 8.877e-5 < float(dfr['PGA'][0]) < 8.878e-5
        # assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
    else:
        dfr = pd.read_hdf(filename)
        assert len(dfr) == 1
        assert 8.877e-5 < dfr['PGA'][0] < 8.878e-5
        # assert dfr.iloc[0][SEGMENT_ID_COLNAME] == expected_first_row_seg_id

    log_content = logfile.read_text()

    segs = sum(1 for _ in get_segments(db_engine, segs_selection, segments_only=True))
    assert f"{segs} segment(s) found to process" in log_content
    assert f"1 of {segs} segment(s) successfully processed" in log_content
    assert (
        f"{segs-1} of {segs} segment(s) skipped with error message reported "
        f"in the log file"
    ) in log_content


def test_simple_run_retDict_high_snr_threshold(
    # fixtures:
    capsys, db_engine, tmp_path, config_dict
):
    """same as `test_simple_run_retDict_saveinv` above
    but with a very high snr threshold => no rows processed"""


    from stream2segment.resources.templates import paramtable
    options = {}
    file_extension = ".csv"
    filename = tmp_path / ('test' + file_extension)
    log_file = tmp_path / 'test.log'

    _ = process(
        dburl=str(db_engine.url),
        pyfunc=paramtable.main,
        # segments_selection={'has_data': 'true'},
        config=config_dict(snr_threshold=3),
        outfile=filename,
        verbose=True,
        logfile=log_file,
        **options
    )

    # no file written (see next comment for details). Check outfile is empty:
    with pytest.raises(EmptyDataError):
        csv1 = pd.read_csv(filename)

    log_content = log_file.read_text()


    sentence = '4 traces (probably gaps/overlaps)'
    # assert we find 2 times the sentence above in log content:
    idx1 = log_content.find(sentence)
    assert idx1 > -1
    idx2 = log_content[idx1+1:].find(sentence)
    assert idx2 > -1

    assert 'low snr' in log_content
    assert 'no station inventory provided' in log_content

    assert "0 of 4 segment(s) successfully processed" in log_content
    assert (
        "4 of 4 segment(s) skipped with error message reported in the log file"
        in log_content
    )


@pytest.mark.parametrize(
    "file_extension, options",
    product(['.h5', '.csv'], [
        # {},
        # {'chunksize': 1},
        # {'chunksize': 1, 'multi_process': True},
        {'chunksize': 1, 'multi_process': True, 'group_components': True},
        # {'multi_process': True},
        # {'chunksize': 1, 'multi_process': 1},
    ])
)
def test_simple_groupby_components(
    file_extension, options,
    # fixtures:
    capsys, db_engine, config_dict, tmp_path
):
    """test a case where we have a more complex select involving joins"""

    from stream2segment.resources.templates import paramtable
    filename = tmp_path / f'output{file_extension}'
    logfile = tmp_path / 'test.log'

    if filename.is_file():
        filename.unlink()

    if logfile.is_file():
        logfile.unlink()

    segs_selection = {
        # 'has_data': 'true',
        # 'event_time': '<=%s' % (event_time_with_data.isoformat()),
        # 'network_code': 'GE',
    }
    _ = process(
        dburl=str(db_engine.url),
        pyfunc=paramtable.main,
        segments_selection=segs_selection,
        config=config_dict(snr_threshold=0),
        outfile=filename,
        verbose=True,
        logfile=logfile,
        **options
    )

    # check file has NOT been written (all traces with gaps):
    # this means that files are not readable (empty r with few bytes):
    with pytest.raises(ValueError):
        if file_extension == '.csv':
            dfr = pd.read_csv(filename)
        else:
            dfr = pd.read_hdf(filename)
        assert filename.stat().st_size <= 1024

    log_content = logfile.read_text()

    segs = sum(1 for _ in get_segments(db_engine, segs_selection, segments_only=True))
    assert f"{segs} segment(s) found to process" in log_content
    assert f"0 of {segs} segment(s) successfully processed" in log_content
    assert (
        f"{segs} of {segs} segment(s) skipped with error message reported "
        f"in the log file"
    ) in log_content


@pytest.mark.parametrize(
    "err_type", [ImportError, AttributeError, TypeError]
)
def test_errors_process_not_run(
    err_type,
    # fixtures:
    capsys, tmp_path, db_engine, config_dict
):
    """
    test processing in case of several 'critical' errors (which do not launch the
    process. None means simply a bad argument (funcname missing)
    """

    main = None
    if err_type == ImportError:
        def main(segment, station, event, config):
            import asdbasdabsdabsdasdbasdb

    elif err_type == AttributeError:
        def main(segment, station, event, config):
            return segment.___attribute_that_does_not_exist___()

    elif err_type == TypeError:
        def main(segment, config):
            return {}

    options = {}
    # seg_sel = {'has_data': 'true'}
    file_extension = ".csv"
    out_file = tmp_path / ("test" + file_extension)
    log_file = tmp_path / "test.log"
    with pytest.raises(Exception) as excinfo:
        _ = process(
            dburl=str(db_engine.url),
            pyfunc=main,
            # segments_selection=seg_sel,
            config=config_dict(snr_threshold=0),
            outfile=out_file,
            verbose=True,
            logfile=log_file,
            **options
        )

    stdout, stderr = capsys.readouterr()
    # we did open the output file:
    assert os.path.isfile(out_file)
    # and we never wrote on it:
    assert os.stat(out_file).st_size == 0
    outputs = [stdout]
    # check correct outputs, in both log and output:
    outputs.append(log_file.read_text())

    for output in outputs:
        assert 'Traceback' in output
        assert ' line ' in output
        err_name = err_type.__name__
        if err_name not in output and err_type == ImportError:
            # Check that the err_type name is in the output traceback. But note that
            # ImportError is "ModuleNotFoundError" in recent versions of Python (3.9?),
            # so:
            err_name = 'ModuleNotFoundError'
        assert err_name in output


@pytest.mark.parametrize('hdf', [True, False])
def test_append(
    hdf,
    # fixtures:
    capsys, tmp_path, db_engine, config_dict
):
    """test a typical case where we supply the append option"""
    options = {'append': True}
    config = config_dict(snr_threshold=0)

    out_file = tmp_path / ("test" + ('.hdf' if hdf else '.csv'))
    log_file = tmp_path / "test.log"

    from stream2segment.resources.templates import paramtable
    main = paramtable.main

    _ = process(
        dburl=str(db_engine.url),
        pyfunc=main,
        config=config,
        outfile=out_file,
        verbose=True,
        logfile=log_file,
        **options
    )

    if hdf:
        processing_df1 = pd.read_hdf(out_file)
    else:
        processing_df1 = pd.read_csv(out_file)
    assert len(processing_df1) == 1

    log_text1 = log_file.read_text()
    assert "4 segment(s) found to process" in log_text1
    assert "Skipping 1 already processed segment(s)" not in log_text1
    assert "1 of 4 segment(s) successfully processed" in log_text1
    out, err = capsys.readouterr()
    assert "'append' ignored" in out

    # now test a second call, the same as before:
    log_file = tmp_path / "test2.log"
    _ = process(
        dburl=str(db_engine.url),
        pyfunc=main,
        config=config,
        outfile=out_file,
        verbose=True,
        logfile=log_file,
        **options
    )
    # check file has been correctly written:
    if hdf:
        processing_df2 = pd.read_hdf(out_file)
    else:
        processing_df2 = pd.read_csv(out_file)
    assert len(processing_df2) == 2

    log_text2 = log_file.read_text()
    assert "4 segment(s) found to process" in log_text1
    assert "Skipping 1 already processed segment(s)" not in log_text2
    assert "1 of 4 segment(s) successfully processed" in log_text2
    # assert two rows are equal:
    assert_frame_equal(
        processing_df2[:1].reset_index(drop=True),
        processing_df1.reset_index(drop=True),
        check_dtype=True
    )
    assert_frame_equal(
        processing_df2[1:2].reset_index(drop=True),
        processing_df1.reset_index(drop=True),
        check_dtype=True
    )
    out2, err2 = capsys.readouterr()
    assert "write mode: append" in out2

    # last try: no append (also set no-prompt to test that we did not
    # prompt the user)
    log_file = tmp_path / "test3.log"
    options_ = {**options, 'append': False}
    _ = process(
        dburl=str(db_engine.url),
        pyfunc=main,
        config=config,
        outfile=out_file,
        verbose=True,
        logfile=log_file,
        **options_
    )
    # check file has been correctly written:
    if hdf:
        processing_df3 = pd.read_hdf(out_file)
    else:
        processing_df3 = pd.read_csv(out_file)
    assert len(processing_df3) == 1
    assert_frame_equal(
        processing_df3.reset_index(drop=True),
        processing_df1.reset_index(drop=True),
        check_dtype=True
    )
    log_text3 = log_file.read_text()
    assert "4 segment(s) found to process" in log_text3
    assert "Skipping 1 already processed segment(s)" not in log_text3
    assert "1 of 4 segment(s) successfully processed" in log_text3
    out3, err = capsys.readouterr()
    assert "write mode: overwrite" in out3


def test_process_verbosity(
    capsys, tmp_path, db_engine, config_dict
):

    if not is_sqlite(str(db_engine.url)):
        pytest.skip("Skipping postgres test (only sqlite memory used)")

    def main(segment, staation, event, config):
        """no-op processing function"""
        return None

    log_file = tmp_path / 'test.log'

    for l, v in product([True, False], [True, False]):
        out_file = None

        # run verbosity = True, with output file. This configures a logger
        # to log file and a logger stdout
        config = config_dict()
        _ = process(
            dburl=str(db_engine.url),
            pyfunc=main,
            config=config,
            logfile=log_file if l else '',
            verbose=v,
            outfile=out_file,
        )
        out, err = capsys.readouterr()  # also resets capsys
        assert log_file.exists() == l
        if l:
            log_file.unlink()
        assert (len(out) > 0) == v


@pytest.mark.parametrize('mp', [True, False])
def test_suppress_output(
    mp,
    # fixtures
    capsys, tmp_path, db_engine, config_dict
):
    class CustomProcessingError(Exception):
        pass

    x = 0
    log_file = tmp_path / 'test.log'

    def worker_task(segment, station, event, config):
        # ctx = redirect_both() if suppress else nullcontext()
        os.write(1, f"[pid={os.getpid()}] noisy C stdout for task {x}\n".encode())
        os.write(2, f"[pid={os.getpid()}] noisy C stderr for task {x}\n".encode())
        print(f"[pid={os.getpid()}] a plain Python print(), task {x}")
        if segment[0].stats.segment_metadata.event_id == 3:
            raise CustomProcessingError(f"task {x} hit a known bad-data condition")
        return os.getpid(), x * x

    for _ in imap(
        dburl=str(db_engine.url),
        pyfunc=worker_task,
        config={},
        logfile=log_file,
        multi_process=mp,
        verbose=True,
    ):
        pass

    out, err = capsys.readouterr()
    assert True


def test_query(db_engine):
    """test that iour query does not use a b-tree for sorting (performant)"""
    if not is_sqlite(str(db_engine.url)):
        pytest.skip("Postgres not supported")
    orderby_cols = get_orderby_columns(db_engine)
    stmt = build_select(
        db_engine, {}, True
    ).order_by(*orderby_cols.values())

    with db_engine.connect() as conn:
        # res = conn.execute(stmt.prefix_with("EXPLAIN QUERY PLAN ")).all()

        compiled = stmt.compile(
            dialect=conn.dialect,
            compile_kwargs={"literal_binds": True},
        )

        plan1 = conn.exec_driver_sql(
            f"EXPLAIN QUERY PLAN {compiled}"
        ).fetchall()
    assert not any('USE TEMP B-TREE FOR ORDER BY' in p for p in plan1)

    orderby_cols={
        'network_code': Channel.network_code,
        'station_code': Channel.station_code,
        'event_id': Segment.event_id,
        'id': Segment.id
    }
    stmt = build_select(
        db_engine, {}, True
    ).order_by(*orderby_cols.values())

    with db_engine.connect() as conn:
        # res = conn.execute(stmt.prefix_with("EXPLAIN QUERY PLAN ")).all()

        compiled = stmt.compile(
            dialect=conn.dialect,
            compile_kwargs={"literal_binds": True},
        )

        plan2 = conn.exec_driver_sql(
            f"EXPLAIN QUERY PLAN {compiled}"
        ).fetchall()
        assert any('USE TEMP B-TREE FOR ORDER BY' in p for p in plan2)


def test_fields():
    select_fnames = {_.name for _ in fields(SelectFields)}
    where_fnames = {_.name for _ in fields(WhereFields)}

    # fields that I need to fetch but do not want in where clause:
    assert select_fnames - where_fnames == {'data'}
    assert where_fnames - select_fnames == {
        'band_code', 'channel_code', 'event_distance_deg', 'event_distance_km',
        'gap_score_percent', 'instrument_code', 'location_code', 'network_code',
        'orientation_code', 'station_code'
    }

    metadata_fnames = {_.name for _ in fields(SegmentMetadata)}
    metadata_props = {
        name
        for name, value in vars(SegmentMetadata).items()
        if isinstance(value, property)
    }
    metadata_all = metadata_fnames | metadata_props

    # fields that I want in where clause but do not expose in processing
    # or that are defined as properties
    assert where_fnames - metadata_fnames == {
        'band_code', 'event_distance_deg', 'event_distance_km',
        'instrument_code', 'orientation_code', 'gap_score_percent',
    }
    assert metadata_fnames - where_fnames == {'arrival_time'}

    assert metadata_all - where_fnames == {'arrival_time'}
    assert where_fnames - metadata_all == {'gap_score_percent'}

    # FIXME try to get all metadata from all DBs
    # FIXME gap_score test across download and process

    with patch('stream2segment.process.segments_selection.is_legacy_db') as is_leg_db:
        for leg_db in [True, False]:
            is_leg_db.return_value = leg_db
            cols = get_orderby_columns(None)  # <- arg is irrelevant
            assert list(cols.keys())[-1] == 'id'
            assert len(cols) > 1
            for c in cols:
                try:
                    assert (c in metadata_all)
                    assert (c in where_fnames)
                except AssertionError:
                    raise