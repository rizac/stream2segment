'''
Created on Feb 4, 2016

@author: riccardo
'''
from __future__ import print_function

from datetime import datetime

from stream2segment.download.exc import NothingToDownload

try:
    from cStringIO import StringIO  # python2.x
except ImportError:
    from io import StringIO, BytesIO

from mock import patch
import pandas as pd

import pytest

from stream2segment.cli import cli
from stream2segment.download.log import configlog4download
from stream2segment.download.db.models import Segment, Download, Station, WebService
from stream2segment.download.modules.utils import s2scodes
from stream2segment.download.url import urlread
from stream2segment.download.modules.channels import get_post_data as origi_get_post_data


def no_connection():
    try:
        urlread("https://geofon.gfz-potsdam.de/")
        return False
    except:
        return True


@pytest.mark.skipif(no_connection(),
                    reason="no internet connection")

class patches(object):
    # paths container for class-level patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    # urlopen = 'stream2segment.download.url.urlopen'
    get_session = 'stream2segment.download.inputvalidation.get_session'
    close_session = 'stream2segment.download.main.close_session'
    # yaml_load = 'stream2segment.download.inputvalidation.yaml_load'
    # ThreadPool = 'stream2segment.download.url.ThreadPool'
    configlog4download = 'stream2segment.download.main.configlog4download'
    download_save_segments = 'stream2segment.download.main.download_save_segments'
    get_events_df = 'stream2segment.download.main.get_events_df'
    get_post_data = 'stream2segment.download.modules.channels.get_post_data'
    mock_merge_event_stations = 'stream2segment.download.main.merge_events_stations'


@patch(patches.get_session)
@patch(patches.close_session)
@patch(patches.configlog4download)
@patch(patches.download_save_segments)
@patch(patches.get_events_df)
@patch(patches.get_post_data)
def test_real_run_old_buggy_network_filter(mock_get_post_data,
                                           mock_get_events_df,
                                           mock_download_save_segments,
                                           mock_config4download,
                                           mock_close_session, mock_get_session,
                                           # fixtures:
                                           db, clirunner, pytestdir, data):
    """This tess a REAL download run with an OLD bug when providing filtering on network
    and stations with negations only. We just test that the correct 'NothingToDownload'
    messages are issued. The download of segments and inventories (the time consuming
    part) is mocked and raises NothingToDownload (we just want to test stations and
    network)
    """
    if db.is_potgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    db.create(to_file=False)

    ws = WebService(name='isc', type='event', url='http://www.isc.ac.uk/fdsnws/event/1/query')
    db.session.add(ws)
    db.session.commit()
    ws_id = ws.id

    # mock just one event downloaded. The event below is a RELa event (we took  the
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

    def mock_get_post_data_side_effect(*a, **kw):
        ret = origi_get_post_data(*a, **kw)
        return ret.replace('*', '')
    mock_get_post_data.side_effect = mock_get_post_data_side_effect

    # mock download save segments: raise NothingToDownload to speed up things:
    def func_(*a, **kw):
        raise NothingToDownload()
    mock_download_save_segments.side_effect = func_

    cfg_file = data.path("download-network-filter.yaml")

    result = clirunner.invoke(cli, ['download',
                                    '-c', cfg_file,
                                    '--dburl', db.dburl,
                                    ])
    assert not clirunner.ok(result)
    assert 'No station found' in result.output


@patch(patches.get_session)
@patch(patches.close_session)
@patch(patches.configlog4download)
@patch(patches.mock_merge_event_stations)
@patch(patches.get_events_df)
def test_real_run(mock_get_events_df, mock_merge_event_stations, mock_config4download,
                  mock_close_session, mock_get_session,
                  # fixtures:
                  db, clirunner, pytestdir, data):
    """This tess a REAL download run providing filtering on network and stations
    The download of segments and inventories (the time consuming part) is mocked
    and raises NothingToDownload (we just want to test stations and netowrk)
    """
    if db.is_potgres:
        # THIS TEST IS JUST ENOUGH WITH ONE DB (USE SQLITE BECAUSE POSTGRES MIGHT NOT BE
        # SETUP FOR TESTS)
        return

    db.create(to_file=False)

    ws = WebService(name='isc', type='event', url='http://www.isc.ac.uk/fdsnws/event/1/query')
    db.session.add(ws)
    db.session.commit()
    ws_id = ws.id

    # mock just one event downloaded. The event below is a RELa event (we took  the
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
