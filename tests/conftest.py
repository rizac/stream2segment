"""
Basic conftest.py defining fixtures to be accessed during tests
"""
# Created on 3 May 2018
import sys
import os
from collections import namedtuple
from io import BytesIO, StringIO
import urllib
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

# import pandas as pd
# from obspy.core.stream import read as read_stream
# from obspy.core.inventory.inventory import read_inventory
# from click.testing import CliRunner

from stream2segment.io.db import is_postgres, is_sqlite


def pytest_addoption(parser):
    """ add command line option db_url"""
    parser.addoption(
        "--db_url",
        action="append",
        default=[None],
    )


def pytest_generate_tests(metafunc):
    """parametrize all tests containing the fixture db_url with all passed --db_url"""
    if "db_url" in metafunc.fixturenames:
        metafunc.parametrize(
            "db_url",
            metafunc.config.getoption("--db_url"),
            # treat the parameter as input to a fixture instead , pass it into the
            # fixture 'db_url' as request.param instead (see below):
            indirect=True,
        )


@pytest.fixture
def db_url(request, tmp_path):
    db_url = request.param

    if db_url is None:
        db_file = tmp_path / "test.db"
        return f"sqlite:///{db_file}"

    return db_url


@pytest.fixture
def online_only():
    try:
        urllib.request.urlopen("https://example.com", timeout=2)
    except Exception:
        pytest.skip("no internet connection")


@pytest.fixture(scope="session")
def test_data_dir():  # n
    return Path(__file__).parent / "data"


# @pytest.fixture
# def db4process(db, data):
#     """This fixture basically extends the `db` fixture and returns an object with all the
#     method of the `db` object (db.dburl, db.session) plus:
#     db4process.segments(self, with_inventory, with_data, with_gap)
#     """
#     class _ProcessDB:
#         """So, no easy way to override easily with pytest from the object returned by the
#         `db` fixture. The best way is to pass `db` as argument above, which also assures
#         that functions/methods having `db4process` as arguments will behave as those
#         having `db` (i.e., they will be called iteratively for any database url passed in
#         the command line). Drawback: we cannot override the class returned by `db`, so we
#         provide a different class which mimics inheritance by forwarding to db each
#         attribute not found (__getatttr__). Whether there might be a better way to
#         achieve this, it wasn't clear from pytest docs
#         """
#         def __getattr__(self, name):
#             """Lets the user call db4process.dburl, db4process.session,..."""
#             return getattr(db, name)
#
#         def segments(self, with_inventory, with_data, with_gap):
#             """Return the segment ids matching the given criteria"""
#             data_seed_id = 'ok.' if with_inventory else 'no.'
#             data_seed_id += 'ok' if with_data else ('gap' if with_gap else 'no')
#             return self.session.query(dbp.Segment).\
#                 filter(dbp.Segment.data_seed_id == data_seed_id)
#
#         def create(self, to_file=False):
#             """Call `db.create` and then populates the database with the data for
#             processing tests
#             """
#             # re-init a sqlite database (no-op if the db is not sqlite):
#             db.create(to_file, True)
#             # init db:
#             session = db.session
#
#             # Populate the database:
#             dwl = db.Download()
#             session.add(dwl)
#             session.commit()
#
#             wsv = dbp.WebService(id=1, url='eventws')
#             session.add(wsv)
#             session.commit()
#
#             # setup an event:
#             ev1 = dbp.Event(id=1, webservice_id=wsv.id, event_id='abc1', latitude=8, longitude=9,
#                         magnitude=5, depth_km=4, time=datetime.utcnow())
#             ev2 = dbp.Event(id=2, webservice_id=wsv.id, event_id='abc2', latitude=8, longitude=9,
#                         magnitude=5, depth_km=4, time=datetime.utcnow())
#             ev3 = dbp.Event(id=3, webservice_id=wsv.id, event_id='abc3', latitude=8, longitude=9,
#                         magnitude=5, depth_km=4, time=datetime.utcnow())
#
#             session.add_all([ev1, ev2, ev3])
#             session.commit()
#
#             dtc = dbp.DataCenter(station_url='asd', dataselect_url='sdft')
#             session.add(dtc)
#             session.commit()
#
#             # s_ok stations have lat and lon > 11, other stations do not
#             inv_xml = data.read("inventory_GE.APE.xml")
#             s_ok = dbp.Station(datacenter_id=dtc.id, latitude=11, longitude=12, network='ok',
#                            station='ok', start_time=datetime.utcnow(),
#                            inventory_xml=inv_xml)
#             session.add(s_ok)
#             session.commit()
#
#             s_none = dbp.Station(datacenter_id=dtc.id, latitude=-31, longitude=-32, network='no',
#                              station='no', start_time=datetime.utcnow())
#             session.add(s_none)
#             session.commit()
#
#             c_ok = dbp.Channel(station_id=s_ok.id, location='ok', channel="ok", sample_rate=56.7)
#             session.add(c_ok)
#             session.commit()
#
#             c_none = dbp.Channel(station_id=s_none.id, location='no', channel="no", sample_rate=56.7)
#             session.add(c_none)
#             session.commit()
#
#             atts_ok = dict(data.to_segment_dict('trace_GE.APE.mseed'))
#             atts_gap = data.to_segment_dict('IA.BAKI..BHZ.D.2016.004.head')
#             atts_none = dict(atts_ok, data=b'')
#
#             for ch_ in (c_ok, c_none):
#
#                 # ch_.location  below reflects if the station has inv
#                 atts = dict(atts_ok, data_seed_id='%s.ok' % ch_.location, download_code=200)
#                 sg1 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev1.id,
#                               download_id=dwl.id, event_distance_deg=35, **atts)
#                 atts = dict(atts_gap, data_seed_id='%s.gap' % ch_.location, download_code=200)
#                 sg2 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev2.id,
#                               download_id=dwl.id, event_distance_deg=35, **atts)
#                 atts = dict(atts_none, data_seed_id='%s.no' % ch_.location, download_code=204)
#                 sg3 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev3.id,
#                               download_id=dwl.id, event_distance_deg=35, **atts)
#                 session.add_all([sg1, sg2, sg3])
#                 session.commit()
#
#     return _ProcessDB()
