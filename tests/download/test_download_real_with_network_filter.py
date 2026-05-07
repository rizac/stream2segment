"""
Created on Feb 4, 2016

@author: riccardo
"""
import sys

from click.testing import CliRunner
from sqlalchemy.testing.plugin.plugin_base import FixtureFunctions

from stream2segment.download.modules.utils import NothingToDownload
from io import StringIO
from unittest.mock import patch
import socket

import pandas as pd
import pytest

from stream2segment.cli import cli
from stream2segment.io.db.models import WebService


def no_connection():
    try:
        # 8.8.8.8 → Google Public DNS
        # Port 53 is almost always reachable if internet works.
        # No DNS resolution needed → faster and avoids local resolver issues.
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return False
    except OSError:
        return True


download_save_segments_path = 'stream2segment.download.main.download_and_save'
get_events_df_path = 'stream2segment.download.main.get_events'
mock_merge_event_stations_path = 'stream2segment.download.main.merge_events_stations'


@pytest.mark.skipif(no_connection(), reason="no internet connection")
@pytest.mark.skipif(sys.version_info < (3,7), reason="requires python3.7+")
@patch(download_save_segments_path)
@patch(get_events_df_path)
# @patch(patches.get_post_data)  # FIXME REMOVE?
def test_real_run_old_buggy_network_filter(
    mock_get_events_df, mock_download_save_segments,
    # fixtures:
    db, log_capture, test_data_dir
):
    """This tess a REAL download run with an OLD bug when providing filtering on network
    and stations with negations only. We just test that the correct 'NothingToDownload'
    messages are issued. The download of segments and inventories (the time-consuming
    part) is mocked and raises NothingToDownload (we just want to test stations and
    network)
    """
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    ws_id = 1
    with db.engine.begin() as conn:
        conn.execute(
            WebService.__table__.insert(),
            {'url': 'http://www.isc.ac.uk/fdsnws/event/1/query', 'ws_id': ws_id}
        )

    # mock just one event downloaded. The event below is a RELa event (we took  the
    # 1st one only):
    d = pd.read_csv(StringIO("""event_id,time,latitude,longitude,depth_km,author,catalog,contributor,contributor_id,mag_type,magnitude,mag_author,event_location_name,event_type,webservice_id,id
750359 P,2000-01-03T18:28:35,42.2585,2.5413,6.9,MDD,ISC,ISC,1750359 P,mb,4.3,MDD,yrenees,,1,1"""), sep=',')
    d['time'] = pd.to_datetime(d['time'])
    d['event_type'] = d['event_type'].astype(str)
    d['webservice_id'].at[0] = ws_id

    mock_get_events_df.return_value = d

    # mock download save segments: raise NothingToDownload to speed up things:
    def func_(*a, **kw):
        raise NothingToDownload("custom message")
    mock_download_save_segments.side_effect = func_

    cfg_file = test_data_dir / "download-network-filter.yaml"

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )

    # do it again (test what's been written):
    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url,
            '--net', 'BE,16,DK,1B,1C,3D,BK', '--sta', 'MEM,MG09,NUUG,KARA,ANR,MM01,BRIB'
        ]
    )

    result = CliRunner().invoke(
        cli, [
            'download', '-c', str(cfg_file), '--dburl', db.url
        ]
    )
    assert not result.exit_code == 0
    assert 'No station found' in result.output

    from stream2segment.download.inputvalidation import extract_download_args
    with patch("stream2segment.download.inputvalidation.extract_download_args") as mock_load_config:
        mock_load_config.side_effect = lambda *a, **kw: dict()

@pytest.mark.skipif(no_connection(),
                    reason="no internet connection")
@pytest.mark.skipif(sys.version_info < (3,7),
                    reason="requires python3.7+")

# @patch(patches.close_session)
@patch(mock_merge_event_stations_path)
@patch(get_events_df_path)
def test_real_run(mock_get_events_df,
                  # fixtures:
                  db, clirunner, pytestdir, data):
    """This tess a REAL download run providing filtering on network and stations
    The download of segments and inventories (the time consuming part) is mocked
    and raises NothingToDownload (we just want to test stations and netowrk)
    """
    if db.is_postgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    db.create(to_file=False)

    ws = WebService(name='isc', type='event', url='http://www.isc.ac.uk/fdsnws/event/1/query')
    db.session.add(ws)
    db.session.commit()
    ws_id = ws.id

    # mock just one event downloaded. The event below is a REAL event (we took  the
    # 1st one only):
    d = pd.read_csv(StringIO("""event_id,time,latitude,longitude,depth_km,author,catalog,contributor,contributor_id,mag_type,magnitude,mag_author,event_location_name,event_type,webservice_id,id
750359 P,2000-01-03T18:28:35,42.2585,2.5413,6.9,MDD,ISC,ISC,1750359 P,mb,4.3,MDD,yrenees,,1,1"""), sep=',')
    d['time'] = pd.to_datetime(d['time'])
    d['event_type'] = d['event_type'].astype(str)
    d['webservice_id'].at[0] = ws_id

    mock_get_events_df.return_value = d

    mock_get_session.return_value=db.session
    # (close_session is ignored, as we will close the session with the db ficture)
    # Now define the mock for the config4download option
    logfilepath = pytestdir.newfile('.log')
    def c4d(logger, logfilebasepath, verbose):
        # config logger as usual, but redirects to a temp file
        # that will be deleted by pytest, instead of polluting the program
        # package:
        ret = configlog4download(logger, logfilepath, verbose)
        return ret

    mock_config4download.side_effect = c4d

    # mock the first function after channels are saved to skip useless stuff
    # raise NothingToDownload to speed up things:
    def func_(*a, **kw):
        raise NothingToDownload('YES')
    mock_merge_event_stations.side_effect = func_

    cfg_file = data.path("download-network-filter.yaml")

    result = clirunner.invoke(cli, ['download',
                                    '-c', cfg_file,
                                    '--dburl', db.dburl,
                                    ])
    assert clirunner.ok(result)
    # test we have downloaded some networks (not included in the negation filters):
    # WARNING: THIS TEST MIGHT RAISE A FALSE POSITIVE, I.E. WHEN TESTS FAIL
    # DUE TO CONNECTION ERRORS
    assert db.session.query(Station).filter((Station.network.in_(['CH', 'FR', 'IV']))).all()
