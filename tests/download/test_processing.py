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
    capsys, pytestdir, db, config_dict
):
    """same as `test_simple_run_retDict_saveinv` above
    but with a very high snr threshold => no rows processed"""
    session = db.session

    from stream2segment.resources.templates import paramtable
    options = {}
    file_extension = ".csv"
    filename = pytestdir.newfile(file_extension)
    logfile = pytestdir.newfile('.log')
    _ = process(dburl=db.dburl, pyfunc=paramtable.main,
                segments_selection={'has_data': 'true'},
                config=config_dict(snr_threshold=3),
                outfile=filename,
                verbose=True, logfile=logfile, **options)

    # no file written (see next comment for details). Check outfile is empty:
    with pytest.raises(EmptyDataError):
        csv1 = readcsv(filename)

    with open(logfile, 'r') as _:
        logcontent = _.read()
    segs = session.query(Segment.id).filter(Segment.has_data).all()
    # snr value might change (rounding problems). Catch it:
    snr_value = re.search(r"1\.35\d*", logcontent).group()
    # now check that log message is correct"
    assert (f"""4 segment(s) found to process

segment (id=1): low snr {snr_value}
segment (id=2): 4 traces (probably gaps/overlaps)
segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

0 of 4 segment(s) successfully processed
4 of 4 segment(s) skipped with error message reported in the log file""") in logcontent


# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
@pytest.mark.parametrize('select_with_data, seg_chunk',
                         [(True, None), (True, 1), (False, None), (False, 1)])
def test_simple_run_retDict_seg_select_empty_and_err_segments(
    select_with_data, seg_chunk,
    # fixtures:
    capsys, pytestdir, db, config_dict
):
    """test a segment selection that takes only non-processable segments"""
    from stream2segment.resources.templates import paramtable
    options = {}
    if seg_chunk is not None:
        options['chunksize'] = seg_chunk
    seg_sel = {'station.latitude': '<10', 'station.longitude': '<10'}
    if select_with_data:
        seg_sel['has_data'] = 'true'
    file_extension = ".csv"
    filename = pytestdir.newfile(file_extension)
    logfile = pytestdir.newfile('.log')
    _ = process(dburl=db.dburl, pyfunc=paramtable.main,
                segments_selection=seg_sel,
                config=config_dict(snr_threshold=0),
                outfile=filename,
                verbose=True, logfile=logfile, **options)

    # check file has not been written (no data):
    with pytest.raises(EmptyDataError):
        csv1 = readcsv(filename)

    with open(logfile, 'r') as _:
        logcontent = _.read()

    if select_with_data:
        # selecting only with data means out of the three candidate segments, one
        # is discarded prior to processing:
        assert ("""2 segment(s) found to process

segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)

0 of 2 segment(s) successfully processed
2 of 2 segment(s) skipped with error message reported in the log file""") in logcontent
    else:
        assert ("""3 segment(s) found to process

segment (id=4): Station inventory (xml) error: no data
segment (id=5): 4 traces (probably gaps/overlaps)
segment (id=6): MiniSeed error: no data

0 of 3 segment(s) successfully processed
3 of 3 segment(s) skipped with error message reported in the log file""") in logcontent


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
    capsys, pytestdir, db, config_dict
):
    """
    test processing in case of several 'critical' errors (which do not launch the process
    None means simply a bad argument (funcname missing)
    """

    main = None
    if err_type == ImportError:
        def main(segment, config):
            import asdbasdabsdabsdasdbasdb

    elif err_type == AttributeError:
        def main(segment, config):
            return segment.___attribute_that_does_not_exist___()

    elif err_type == TypeError:
        def main(segment, config, wrong_argument):
            return {}

    options = {}
    seg_sel = {'has_data': 'true'}
    file_extension = ".csv"
    filename = pytestdir.newfile(file_extension)
    logfile = pytestdir.newfile('.log')
    with pytest.raises(Exception) as excinfo:
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection=seg_sel,
                    config=config_dict(snr_threshold=0),
                    outfile=filename,
                    verbose=True, logfile=logfile, **options)

    stdout, stderr = capsys.readouterr()
    # we did open the output file:
    assert os.path.isfile(filename)
    # and we never wrote on it:
    assert os.stat(filename).st_size == 0
    # check correct outputs, in both log and output:
    with open(logfile) as _:
        logfilecontent = _.read()
    outputs = [stdout, logfilecontent]
    for output in outputs:
        # Check that the err_type name is in the output traceback. But note that
        # ImportError is "ModuleNotFoundError" in recent versions of Python (3.9?),
        # so:
        err_names = [err_type.__name__]
        if err_type == ImportError:
            err_names.append('ModuleNotFoundError')
        # Try to loosely assert the messages is on standard output:
        assert any(e in output and 'Traceback' in output and ' line ' in output
                   for e in err_names)


@pytest.mark.parametrize("err_type", [None, SkipSegment])
def test_errors_process_completed(
    err_type,
    # fixtures:
    capsys, pytestdir, db, config_dict
):
    """test processing in case of non 'critical' errors i.e., which do not prevent the process
      to be completed. None means we do not override SEG_SEL_STR which, with the current
      templates, causes no segment to be selected"""
    from stream2segment.resources.templates import paramtable
    if err_type == SkipSegment:
        seg_sel = {'has_data': 'true'}
        def main2(segment, config):
            raise SkipSegment(ValueError("invalid literal for .* with base 10: '4d'"))
    else:
        seg_sel = {'maxgap_numsamples': '[-0.5, 0.5]', 'has_data': 'true'}
        def main2(segment, config):
            return paramtable.main(segment, config)

    options = {}
    file_extension = ".csv"
    filename = pytestdir.newfile(file_extension)
    logfile = pytestdir.newfile('.log')
    _ = process(dburl=db.dburl, pyfunc=main2,
                segments_selection=seg_sel,
                config=config_dict(),
                outfile=filename,
                verbose=True, logfile=logfile, **options)

    output, error = capsys.readouterr()
    with open(logfile) as _:
        logcontent = _.read()

    assert not error
    # we did open the output file:
    assert os.path.isfile(filename)
    # and we never wrote on it:
    assert os.stat(filename).st_size == 0
    # check correct outputs, in both log and output:
    if err_type is None:  # no segments processed
        # we want to check that a particular string (str2check) is in the stdout
        # But consider that string changes according to py versions so use regex:
        str2check = \
            (r"0 segment\(s\) found to process\n"
             r"\n+"
             r"0 of 0 segment\(s\) successfully processed\n"
             r"0 of 0 segment\(s\) skipped with error message reported in the log file")
        assert re.search(str2check, output)
        assert re.search(str2check, logcontent)
    else:
        # we want to check that a particular string (str2check) is in the stdout
        # But consider that string changes according to py versions so use regex:
        str2check = \
            (r'4 segment\(s\) found to process\n'
             r'\n+'
             r'0 of 4 segment\(s\) successfully processed\n'
             r'4 of 4 segment\(s\) skipped with error message reported in the log file')
        assert re.search(str2check, output)

        str2check = \
            (r"4 segment\(s\) found to process\n"
             r"\n+"
             r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
             r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
             r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
             r"segment \([^\)]+\)\: invalid literal for .* with base 10: '4d'\n"
             r"\n+"
             r"0 of 4 segment\(s\) successfully processed\n"
             r"4 of 4 segment\(s\) skipped with error message reported in the log file")
        try:
            assert re.search(str2check, logcontent)
        except AssertionError:
            asd =9

# appending to file:

# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
@pytest.mark.parametrize('hdf', [True, False])
@pytest.mark.parametrize("output_file_empty", [False, True])
@pytest.mark.parametrize('processing_py_return_list', [True, False])
def test_append_on_badly_formatted_outfile(
    processing_py_return_list, output_file_empty, hdf,
    # fixtures:
    capsys, pytestdir, db, config_dict
):
    """test a case where we append on an badly formatted output file
    (no segment id column found)"""
    if processing_py_return_list and hdf:
        # hdf does not support returning lists
        return
    seg_sel = {'has_data': 'true'}
    config = config_dict(snr_threshold=0)
    options = {'append': True}

    from stream2segment.resources.templates import paramtable
    if processing_py_return_list:
        def main(segment, config):
            return list(paramtable.main(segment, config).values())
    else:
        main = paramtable.main

    outfilepath = pytestdir.newfile('.hdf' if hdf else '.csv', create=True)
    if not output_file_empty:
        if hdf:
            pd.DataFrame(columns=['-+-', '[[['], data=[[1, 'a']]).to_hdf(outfilepath,
                                                                         format='t',
                                                                         key='f')
        else:
            with open(outfilepath, 'wt') as _:
                _.write('asdasd')

    logfile = pytestdir.newfile('.log')

    # this are the cases where the append is ok:
    should_be_ok = processing_py_return_list or \
                   (not hdf and output_file_empty)
    try:
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection=seg_sel,
                    config=config,
                    outfile=outfilepath,
                    verbose=True, logfile=logfile, **options)
        assert should_be_ok
    except (HDF5ExtError, BaseWriter._SEGID_NOTFOUND_ERR.__class__) as exc:
        if isinstance(exc, HDF5ExtError):
            assert hdf
        assert not should_be_ok

    output, error = capsys.readouterr()

    if not should_be_ok:
        # if hdf and output file is empty, the error is a HDF error
        # (because an emopty file cannot be opened as HDF, a CSV apparently
        # can)
        is_empty_hdf_file = hdf and output_file_empty
        if not is_empty_hdf_file:
            # otherwise, it's a s2s error where we could not find the
            # segment id column:
            assert ("TypeError: Cannot append to file, segment_id column " \
                    "name not found") in output
        return

    with open(logfile) as _:
        logtext = _.read()
    assert len(logtext) > 0
    assert "Appending results to existing file" in logtext


# Recall: we have 6 segments, issued from all combination of
# station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
# use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
# those segments in case. For info see db4process in conftest.py
@pytest.mark.parametrize('hdf', [True, False])
@pytest.mark.parametrize('processing_py_return_list', [True, False])
# @patch('stream2segment.cli.click.confirm', return_value=True)
def test_append(
    processing_py_return_list, hdf,
    # fixtures:
    capsys, pytestdir, db, config_dict
):
    """test a typical case where we supply the append option"""
    if processing_py_return_list and hdf:
        # hdf does not support returning lists
        pytest.skip("Python function cannot return lists when output is HDF")

    # legacy code needs to test also with legacy segment id 'Segment.db.id', but
    # this generates a table warning that we want to suppress, so:
    if '.' in segment_id_colname:
        warnings.filterwarnings('ignore', category=NaturalNameWarning)

    with patch('stream2segment.process.writers.SEGMENT_ID_COLNAME',
          segment_id_colname):
        options = {'append': True}
        config = config_dict(snr_threshold=0)

        _seg = db.segments(with_inventory=True, with_data=True, with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        session = db.session

        outfilepath = pytestdir.newfile('.hdf' if hdf else '.csv')
        logfile = pytestdir.newfile('.log')

        from stream2segment.resources.templates import paramtable
        main = paramtable.main
        if processing_py_return_list:
            def main(segment, config):
                return list(paramtable.main(segment, config).values())

        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    verbose=True, logfile=logfile, **options)

        processing_df1 = read_processing_output(outfilepath,
                                                header=not processing_py_return_list)
        assert len(processing_df1) == 1
        segid_column = segment_id_colname if hdf else processing_df1.columns[0]
        assert processing_df1.loc[0, segid_column] == expected_first_row_seg_id
        with open(logfile) as _:
            logtext1 = _.read()
        assert "4 segment(s) found to process" in logtext1
        assert "Skipping 1 already processed segment(s)" not in logtext1
        assert "Ignoring `append` functionality: output file does not exist or not provided" \
            in logtext1
        assert "1 of 4 segment(s) successfully processed" in logtext1

        # now test a second call, the same as before:
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    verbose=True, logfile=logfile, **options)
        # check file has been correctly written:
        processing_df2 = read_processing_output(outfilepath,
                                                header=not processing_py_return_list)
        assert len(processing_df2) == 1
        segid_column = segment_id_colname if hdf else processing_df1.columns[0]
        assert processing_df2.loc[0, segid_column] == expected_first_row_seg_id
        with open(logfile) as _:
            logtext2 = _.read()
        assert "3 segment(s) found to process" in logtext2
        assert "Skipping 1 already processed segment(s)" in logtext2
        assert "Appending results to existing file" in logtext2
        assert "0 of 3 segment(s) successfully processed" in logtext2
        # assert two rows are equal:
        assert_frame_equal(processing_df1, processing_df2, check_dtype=True)

        # change the segment id of the written segment
        seg = session.query(Segment).filter(Segment.id == expected_first_row_seg_id).\
            first()
        new_seg_id = seg.id * 100
        seg.id = new_seg_id
        session.commit()

        # now test a second call, the same as before:
        logfile = pytestdir.newfile('.log')
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    verbose=True, logfile=logfile, **options)
        # check file has been correctly written:
        processing_df3 = read_processing_output(outfilepath,
                                                header=not processing_py_return_list)
        assert len(processing_df3) == 2
        segid_column = segment_id_colname if hdf else processing_df1.columns[0]
        assert processing_df3.loc[0, segid_column] == expected_first_row_seg_id
        assert processing_df3.loc[1, segid_column] == new_seg_id
        with open(logfile) as _:
            logtext3 = _.read()
        assert "4 segment(s) found to process" in logtext3
        assert "Skipping 1 already processed segment(s)" in logtext3
        assert "Appending results to existing file" in logtext3
        assert "1 of 4 segment(s) successfully processed" in logtext3
        # assert two rows are equal:
        assert_frame_equal(processing_df1, processing_df3[:1], check_dtype=True)

        # last try: no append (also set no-prompt to test that we did not
        # prompt the user)
        logfile = pytestdir.newfile('.log')
        options_ = {**options, 'append': False}
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    verbose=True, logfile=logfile, **options_)
        # check file has been correctly written:
        processing_df4 = read_processing_output(outfilepath,
                                                header=not processing_py_return_list)
        assert len(processing_df4) == 1
        segid_column = segment_id_colname if hdf else processing_df1.columns[0]
        assert processing_df4.loc[0, segid_column] == new_seg_id
        with open(logfile) as _:
            logtext4 = _.read()
        assert "4 segment(s) found to process" in logtext4
        assert "Skipping 1 already processed segment(s)" not in logtext4
        assert "Appending results to existing file" not in logtext4
        assert "1 of 4 segment(s) successfully processed" in logtext4
        assert 'Overwriting existing output file' in logtext4

    # reset warning (maybe not needed, for safety):
    warnings.filterwarnings('always', category=NaturalNameWarning)


def test_process_verbosity(
    capsys, pytestdir, db, config_dict
):

    if not db.is_sqlite:
        pytest.skip("Skipping postgres test (only sqlite memory used)")

    # mock configlog4processing:
    from stream2segment.process.log import \
        configlog4processing as original_config_log
    # store stuff in this dict when running configlog below:
    logvar = {'numloggers': 0, 'logfilepath': None}
    # define mocking function configlog:
    def configlog(logger, logfilebasepath, verbose):
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # config logger as usual, but redirects to a temp file
        # that will be deleted by pytest, instead of polluting the program
        # package:
        original_config_log(logger,
                            pytestdir.newfile('.log') if logfilebasepath else None,
                            verbose)

        logvar['numloggers'] = len(logger.handlers)
        logvar['logfilepath'] = None
        try:
            logvar['logfilepath'] = logger.handlers[0].baseFilename
        except (IndexError, AttributeError):
            pass

    with patch(patches.configlog4processing, side_effect=configlog) as mock_configlog:

        def main(segment, config):
            """no-op processing function"""
            return None

        # run verbosity = True, with output file. This configures a logger
        # to log file and a logger stdout
        config = config_dict()
        outfilepath = pytestdir.newfile()
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    verbose=True, logfile=True)
        out, err = capsys.readouterr()
        assert len(out)
        assert logvar['numloggers'] == 2
        with open(logvar['logfilepath']) as _opn:  # noqa
            expected_out = _opn.read()
        # assert out ==- expected_out, but ignore spaces/newlines which might differ:
        assert re.sub(r'\s+', ' ', expected_out.strip()) == \
               re.sub(r'\s+', ' ', out.strip())

        # run verbosity = False, with output file. This configures a logger
        # to log file
        config = config_dict()
        outfilepath = pytestdir.newfile()
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    logfile=False, verbose=False)
        out, err = capsys.readouterr()
        assert not out
        assert logvar['numloggers'] == 0

        # run verbosity = False, with output file. This configures a logger
        # to log file
        config = config_dict()
        outfilepath = pytestdir.newfile()
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=outfilepath,
                    logfile=True, verbose=False)
        out, err = capsys.readouterr()
        assert not out
        assert logvar['numloggers'] == 1

        # run verbosity = True, with no output file. This configures a
        # logger stderr and a logger stdout
        config = config_dict()
        # outfilepath = pytestdir.newfile()
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=None,
                    logfile=True, verbose=True)
        out, err = capsys.readouterr()
        assert out
        assert logvar['numloggers'] == 1

        # run verbosity = False, with no output file. This configures a
        # logger stderr but no logger
        config = config_dict()
        # outfilepath = pytestdir.newfile()
        _ = process(dburl=db.dburl, pyfunc=main,
                    segments_selection={'has_data': 'true'},
                    config=config,
                    outfile=None,
                    logfile=False, verbose=False)
        out, err = capsys.readouterr()
        assert not out
        assert logvar['logfilepath'] is None
        assert logvar['numloggers'] == 0


def read_processing_output(filename, header=True):  # <- header only for csv
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.hdf':
        return pd.read_hdf(filename).reset_index(drop=True, inplace=False)  # noqa
    elif ext == '.csv':
        return pd.read_csv(filename, header=None) if not header \
            else pd.read_csv(filename)
    else:
        raise ValueError('Unrecognized extension %s' % ext)