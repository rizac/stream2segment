"""
Created on Feb 14, 2017

@author: riccardo
"""
import os
import re
from pathlib import Path
from datetime import datetime, UTC, timedelta
from itertools import product
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
import pytest
import yaml
from pandas._testing import assert_frame_equal
from pandas.errors import EmptyDataError
from sqlalchemy.orm import sessionmaker
from tables import HDF5ExtError, NaturalNameWarning

from stream2segment.io.db import create_engine, database_exists, is_postgres, is_sqlite
from stream2segment.io.db.models import (
    DownloadRun, WebService, Channel, StationXML, MiniSeed, SkippedSegment,Event,
    Segment
)
from stream2segment.process.main import get_segments
from stream2segment.process import SkipSegment
from stream2segment.resources import get_templates_fpath
from stream2segment.process.main import process
# from stream2segment.process.writers import SEGMENT_ID_COLNAME, BaseWriter


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
    session.add(s_ok)
    session.commit()

    c_ok = Channel(
        stationxml_id=s_ok.id,
        data_webservice_id=dtc.id,
        latitude=11,
        longitude=12,
        network_code='ok',
        station_code='ok',
        location_code='ok',
        # legacy channel_code="ok" becomes:
        band_code="o",
        instrument_code="k",
        orientation_code="",
        start_time=now,
        sample_rate=56.7
    )
    session.add(c_ok)
    session.commit()

    c_none = Channel(
        latitude=-31,
        longitude=-32,
        data_webservice_id=dtc.id,
        network_code='no',
        station_code='no',
        location_code='no',
        # legacy channel_code="no" becomes:
        band_code="n",
        instrument_code="o",
        orientation_code="",
        start_time=now,
        sample_rate=56.7
    )
    session.add(c_none)
    session.commit()

    with open(test_data_dir / 'trace_GE.APE.mseed', 'rb') as f:
        trace_ok = f.read()
    with open(test_data_dir / 'IA.BAKI..BHZ.D.2016.004.head', 'rb') as f:
        trace_gap = f.read()

    atts = {
        'event_distance_km': 35*110,
        'gap_score_percent': 0,
        'noise_window_s': 10,
        'signal_window_s': 30,
    }
    for ch_ in (c_ok, c_none):
        sg1 = Segment(
            channel_id=ch_.id, event_id=ev1.id, **atts
        )
        sg2 = Segment(
            channel_id=ch_.id, event_id=ev2.id, **atts
        )
        sg3 = SkippedSegment(
            channel_id=ch_.id, event_id=ev3.id, download_code=204
        )
        session.add_all([sg1, sg2, sg3])
        session.commit()

        ms_ok = MiniSeed(data=trace_ok, id=sg1.id)
        ms_gap = MiniSeed(data=trace_gap, id= sg2.id)
        session.add_all([ms_ok, ms_gap])
        session.commit()

    # atts_ok = dict(data.to_segment_dict('trace_GE.APE.mseed'))
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

# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
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

    # # advanced_settings, cmdline_opts = options
    # session = db.session
    # # select the event times for the segments with data:
    # etimes = sorted(_[1] for _ in session.query(Segment.id, Event.time).
    #                 join(Segment.event).filter(Segment.has_data))
    #
    # _seg = db.segments(with_inventory=True, with_data=True, with_gap=False).one()
    # expected_first_row_seg_id = _seg.id
    # station_id_whose_inventory_is_saved = _seg.station.id

    from stream2segment.resources.templates import paramtable
    filename = tmp_path / f'output{file_extension}'
    logfile = tmp_path / 'test.log'

    if filename.is_file():
        filename.unlink()

    if logfile.is_file():
        logfile.unlink()

    segs_selection = {
        # 'has_data': 'true',
        'event_time': '<=%s' % (event_time_with_data.isoformat())
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
    assert f"{segs} segment(s) to process found" in log_content
    assert f"1 of {segs} segment(s) successfully processed" in log_content
    assert (
        f"{segs-1} of {segs} segment(s) skipped with error message reported "
        f"in the log file"
    ) in log_content


# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
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

    idx1 = log_content.find('4 traces (probably gaps/overlaps)')
    assert idx1 > -1
    idx2 = log_content[idx1+1:].find('4 traces (probably gaps/overlaps)')
    assert idx2 > idx1
    idx3 = log_content[idx2 + 1:].find('4 traces (probably gaps/overlaps)')
    assert idx3 == -1

    assert 'low snr' in log_content
    assert 'no onventory provided' in log_content

    assert "0 of 4 segment(s) successfully processed" in log_content
    assert (
        "4 of 4 segment(s) skipped with error message reported in the log file"
        in log_content
    )


# Even though we are not interested here to check what is there on the created db,
# because we test errors,
# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
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


# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
@pytest.mark.parametrize('hdf', [True, False])
# @patch('stream2segment.cli.click.confirm', return_value=True)
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


# @pytest.mark.parametrize(
#     "file_extension, options",
#     product(['.h5', '.csv'], [
#         # {},
#         {'chunksize': 1},
#         {'chunksize': 1, 'multi_process': True},
#         {'chunksize': 1, 'multi_process': True, 'group_components': True},
#         # {'multi_process': True},
#         # {'chunksize': 1, 'multi_process': 1},
#     ])
# )
# def test_simple_run_retDict_complex_select_group_components(
#     file_extension, options,
#     # fixtures:
#     capsys, db_engine, config_dict, tmp_path
# ):
#     """test a case where we have a more complex select involving joins"""
#
#     # # advanced_settings, cmdline_opts = options
#     # session = db.session
#     # # select the event times for the segments with data:
#     # etimes = sorted(_[1] for _ in session.query(Segment.id, Event.time).
#     #                 join(Segment.event).filter(Segment.has_data))
#     #
#     # _seg = db.segments(with_inventory=True, with_data=True, with_gap=False).one()
#     # expected_first_row_seg_id = _seg.id
#     # station_id_whose_inventory_is_saved = _seg.station.id
#
#     from stream2segment.resources.templates import paramtable
#     filename = tmp_path / f'output{file_extension}'
#     logfile = tmp_path / 'test.log'
#
#     if filename.is_file():
#         filename.unlink()
#
#     if logfile.is_file():
#         logfile.unlink()
#
#     segs_selection = {
#         # 'has_data': 'true',
#         'event_time': '<=%s' % (event_time_with_data.isoformat())
#     }
#     _ = process(
#         dburl=str(db_engine.url),
#         pyfunc=paramtable.main,
#         segments_selection=segs_selection,
#         config=config_dict(snr_threshold=0),
#         outfile=filename,
#         verbose=True,
#         logfile=logfile,
#         **options
#     )
#
#     # check file has been correctly written:
#     if file_extension == '.csv':
#         dfr = pd.read_csv(filename)
#         assert len(dfr) == 1
#         assert 8.877e-5 < float(dfr['PGA'][0]) < 8.878e-5
#         # assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
#     else:
#         dfr = pd.read_hdf(filename)
#         assert len(dfr) == 1
#         assert 8.877e-5 < dfr['PGA'][0] < 8.878e-5
#         # assert dfr.iloc[0][SEGMENT_ID_COLNAME] == expected_first_row_seg_id
#
#     log_content = logfile.read_text()
#
#     segs = sum(1 for _ in get_segments(db_engine, segs_selection, segments_only=True))
#     assert f"{segs} segment(s) to process found" in log_content
#     assert f"1 of {segs} segment(s) successfully processed" in log_content
#     assert (
#         f"{segs-1} of {segs} segment(s) skipped with error message reported "
#         f"in the log file"
#     ) in log_content
