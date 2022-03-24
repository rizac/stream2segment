"""
Created on Feb 14, 2017

@author: riccardo
"""
import os
from unittest.mock import patch
import pandas as pd
import pytest

from stream2segment.resources import get_templates_fpath
from stream2segment.process.db.models import (get_inventory, get_stream,
                                              Event, Station, Segment,
                                              Channel, Download, DataCenter)
from stream2segment.process.main import query4process
from stream2segment.process.log import configlog4processing as o_configlog4processing


@pytest.fixture
def yamlfile(pytestdir):
    """global fixture wrapping pytestdir.yamlfile"""
    def func(**overridden_pars):
        return pytestdir.yamlfile(get_templates_fpath('paramtable.yaml'), **overridden_pars)

    return func


def readcsv(filename, header=True):
    return pd.read_csv(filename, header=None) if not header else pd.read_csv(filename)


class Test:

    pyfile = get_templates_fpath("paramtable.py")

    @property
    def logfilecontent(self):
        assert os.path.isfile(self._logfilename)
        with open(self._logfilename) as opn:
            return opn.read()

    # The class-level `init` fixture is marked with autouse=true which implies that all test
    # methods in the class will use this fixture without a need to state it in the test
    # function signature or with a class-level usefixtures decorator. For info see:
    # https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, pytestdir, db4process):
        db4process.create(to_file=True)
        session = db4process.session

        class patches:
            # paths container for class-level patchers used below. Hopefully
            # will mek easier debug when refactoring/move functions
            get_session = 'stream2segment.process.main.get_session'
            close_session = 'stream2segment.process.main.close_session'
            configlog4processing = 'stream2segment.process.main.configlog4processing'

        # sets up the mocked functions: db session handling (using the already created
        # session) and log file handling:
        with patch(patches.get_session, return_value=session):
            with patch(patches.close_session, side_effect=lambda *a, **v: None):
                with patch(patches.configlog4processing) as mock2:

                    def clogd(logger, logfilebasepath, verbose):
                        # config logger as usual, but redirects to a temp file
                        # that will be deleted by pytest, instead of polluting the program
                        # package:
                        o_configlog4processing(logger,
                                               pytestdir.newfile('.log') \
                                               if logfilebasepath else None,
                                               verbose)

                        self._logfilename = logger.handlers[0].baseFilename

                    mock2.side_effect = clogd

                    yield

    def inlogtext(self, string):
        '''Checks that `string` is in log text.
        The assertion `string in self.logfilecontent` fails in py3.5, although the differences
        between characters is the same position is zero. We did not find any better way than
        fixing it via this cumbersome function'''
        logtext = self.logfilecontent
        i = 0
        while len(logtext[i:i+len(string)]) == len(string):
            if (sum(ord(a)-ord(b) for a, b in zip(string, logtext[i:i+len(string)]))) == 0:
                return True
            i += 1
        return False

# ## ======== ACTUAL TESTS: ================================

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @patch('stream2segment.process.db.models.get_inventory', side_effect=get_inventory)
    @patch('stream2segment.process.db.models.get_stream', side_effect=get_stream)
    def test_segwrapper(self, mock_getstream, mock_getinv,
                        # fixtures:
                        db4process, data):
        session = db4process.session
        segids = query4process(session, {}).all()
        seg_with_inv = \
            db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        sta_with_inv_id = seg_with_inv.station.id
        invcache = {}

        def read_stream(segment, reload=False):
            '''calls segment.stream(reload) asserting that if segment has no
            data it raises. This function never raises'''
            if segment.data:
                segment.stream(reload)
            else:
                with pytest.raises(Exception) as exc:  # all inventories are None
                    segment.stream(reload)

        prev_staid = None
        for segid in [_[0] for _ in segids]:
            segment = session.query(Segment).filter(Segment.id == segid).first()
            sta = segment.station
            staid = sta.id
            assert prev_staid is None or staid >= prev_staid
            staequal = prev_staid is not None and staid == prev_staid
            prev_staid = staid
            segment.station._inventory = invcache.get(sta.id, None)

            mock_getinv.reset_mock()
            if sta.id != sta_with_inv_id:
                with pytest.raises(Exception):  # all inventories are None
                    segment.inventory()
                assert mock_getinv.called
                # re-call it and assert we raise the previous Exception:
                ccc = mock_getinv.call_count
                with pytest.raises(Exception):  # all inventories are None
                    segment.inventory()
                assert mock_getinv.call_count == ccc
                # re-call it with reload=True and assert we raise the previous
                # exception, and that we called get_inv:
                with pytest.raises(Exception):  # all inventories are None
                    segment.inventory(True)
                assert mock_getinv.call_count == ccc + 1
            else:
                invcache[sta.id] = segment.inventory()
                if staequal:
                    assert not mock_getinv.called
                else:
                    assert mock_getinv.called
                assert len(segment.station.inventory_xml) > 0
                # re-call it with reload=True and assert we raise the previous
                # exception, and that we called get_inv:
                ccc = mock_getinv.call_count
                segment.inventory(True)
                assert mock_getinv.call_count == ccc + 1

                assert type(segment.inventory(format='stationxml')) == bytes
                assert type(segment.inventory(format='stationtxt')) == str

            # call segment.stream
            assert not mock_getstream.called
            read_stream(segment)
            assert mock_getstream.call_count == 1
            read_stream(segment)
            assert mock_getstream.call_count == 1
            # with reload flag:
            read_stream(segment, True)
            assert mock_getstream.call_count == 2
            mock_getstream.reset_mock()
            
            segs = segment.siblings().all()
            # as channel's channel is either 'ok' or 'err' we should never have
            # other components
            assert len(segs) == 0

            segs = segment.siblings(include_self=True).all()  # include the segment
            assert len(segs) == 1


        # NOW TEST OTHER ORIENTATION PROPERLY. WE NEED TO ADD WELL FORMED SEGMENTS WITH CHANNELS
        # WHOSE ORIENTATION CAN BE DERIVED:
        staid = session.query(Station.id).first()[0]
        dcid = session.query(DataCenter.id).first()[0]
        eid = session.query(Event.id).first()[0]
        dwid = session.query(Download.id).first()[0]
        # add channels
        c_1 = Channel(station_id=staid, location='ok', channel="AB1", sample_rate=56.7)
        c_2 = Channel(station_id=staid, location='ok', channel="AB2", sample_rate=56.7)
        c_3 = Channel(station_id=staid, location='ok', channel="AB3", sample_rate=56.7)
        session.add_all([c_1, c_2, c_3])
        session.commit()
        # add segments. Create attributes (although not strictly necessary to have bytes data)
        atts = data.to_segment_dict('trace_GE.APE.mseed')
        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c_1.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        sg2 = Segment(channel_id=c_2.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        sg3 = Segment(channel_id=c_3.id, datacenter_id=dcid, event_id=eid, download_id=dwid,
                      event_distance_deg=35, **atts)
        session.add_all([sg1, sg2, sg3])
        session.commit()
        # start testing:
        segids = query4process(session, {}).all()

        for segid in [_[0] for _ in segids]:
            segment = session.query(Segment).filter(Segment.id == segid).first()
            # staid = segment.station.id
            segs = segment.siblings()
            if segs.all():
                assert segment.id in (sg1.id, sg2.id, sg3.id)
                assert len(segs.all()) == 2
