"""
This Python file is a foreseen development of the update metadata command, which should be
able to re-download metadata and/or inventory of an already existing database
"""
from io import StringIO
from unittest.mock import patch

import pandas as pd
from stream2segment.cli import cli
from stream2segment.download.db.models import Segment, Download, Station, DataCenter

from stream2segment.download.modules.mseedlite import unpack
from stream2segment.io.db.pdsql import insertdf, updatedf, dbquery2df

# Old test in test_downloads, when there was the "only" option for the update_metadata
# parameter. As reference:


@patch('stream2segment.download.main.get_events_df')
@patch('stream2segment.download.main.get_datacenters_df')
@patch('stream2segment.download.main.get_channels_df')
@patch('stream2segment.download.main.save_inventories')
@patch('stream2segment.download.main.download_save_segments')
@patch('stream2segment.download.modules.segments.mseedunpack')
@patch('stream2segment.io.db.pdsql.insertdf')
@patch('stream2segment.io.db.pdsql.updatedf')
def tst_cmdline_inv_only(self, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                          mock_download_save_segments, mock_save_inventories,
                          mock_get_channels_df,
                          mock_get_datacenters_df, mock_get_events_df,
                          # fixtures:
                          db, clirunner, pytestdir):
    mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v)
    mock_get_datacenters_df.side_effect = \
        lambda *a, **v: self.get_datacenters_df(None, *a, **v)
    mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a,
                                                                            **v)
    mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a,
                                                                              **v)
    mock_download_save_segments.side_effect = \
        lambda *a, **v: self.download_save_segments(None, *a, **v)
    # mseed unpack is mocked by accepting only first arg (so that time bounds are not
    # considered)
    mock_mseed_unpack.side_effect = lambda *a, **v: unpack(a[0])
    mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
    mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
    # prevlen = len(db.session.query(Segment).all())

    # NOTE: NOT SPECIFYING inventory uses the configfile value. IN THESE
    # TESTS, HOWEVER, SETS THE inventory FLAG AS FALSE (see line 130)
    result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                    '--dburl', db.dburl,
                                    '--start', '2016-05-08T00:00:00',
                                    '--end', '2016-05-08T09:00:00'])
    assert clirunner.ok(result)

    # assert we called logger config with log2file rg not None (a file):
    config_logger_args = self.mock_config4download.call_args_list[-1][0]
    assert config_logger_args[1] is not None  # 2nd arg is none

    # test with loading station inventories only

    download_id = max(_[0] for _ in db.session.query(Download.id).all())
    # we should not have inventories saved:
    stainvs = db.session.query(Station).filter(Station.has_inventory).all()
    assert len(stainvs) == 0
    # calculate the expected stations:
    expected_invs_to_download_ids = \
        [x[0] for x in db.session.query(Station.id).filter((~Station.has_inventory) &
                                                           (Station.segments.any(
                                                               Segment.has_data))).all()]  # noqa

    # test that we have data, but also errors
    num_expected_inventories_to_download = len(expected_invs_to_download_ids)
    assert num_expected_inventories_to_download == 2  # just in order to set the value below
    # and be more safe about the fact that we will have only ONE station inventory saved
    inv_urlread_ret_val = [self._inv_data, URLError('a')]
    mock_save_inventories.side_effect = \
        lambda *a, **v: self.save_inventories(inv_urlread_ret_val, *a, **v)

    mock_download_save_segments.reset_mock()
    old_log_msg = self.log_msg()
    # ok run now:
    result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                    '--dburl', db.dburl,
                                    '--start', '2016-05-08T00:00:00',
                                    '--end', '2016-05-08T09:00:00',
                                    '--update-metadata', 'only',
                                    '--inventory'])
    assert clirunner.ok(result)
    assert not mock_download_save_segments.called
    assert "STEP 3 of 3: Downloading 2 station inventories" in result.output
    # But assert the log message does contain the warning
    # (self.log_msg() is from a logger configured additionaly for these tests)
    new_log_msg = self.log_msg()[len(old_log_msg):]
    assert "Inventory download error" in new_log_msg
    stainvs = db.session.query(Station).filter(Station.has_inventory).all()
    assert len(stainvs) == 1
    ix = \
        db.session.query(Station.id, Station.inventory_xml).filter(
            Station.has_inventory).all()
    num_downloaded_inventories_first_try = len(ix)
    assert len(ix) == num_downloaded_inventories_first_try
    staid, invdata = ix[0][0], ix[0][1]
    expected_invs_to_download_ids.remove(staid)  # remove the saved inventory
    assert not invdata.startswith(b'<?xml ')  # assert we compressed data
    assert mock_save_inventories.called
    download_id_ = max(_[0] for _ in db.session.query(Download.id).all())
    assert download_id_ == download_id + 1
    mock_save_inventories.reset_mock()

    # Now write also to the second station inventory (the one
    # which raised before)
    mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories([b"x"], *a,
                                                                              **v)

    result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                    '--dburl', db.dburl,
                                    '--start', '2016-05-08T00:00:00',
                                    '--end', '2016-05-08T09:00:00',
                                    '--update-metadata', 'only',
                                    '--inventory'])
    assert clirunner.ok(result)
    stainvs = db.session.query(Station).filter(Station.has_inventory).all()
    # assert we still have one station (the one we saved before):
    assert len(stainvs) == num_downloaded_inventories_first_try + 1
    assert mock_save_inventories.called
    download_id_ = max(_[0] for _ in db.session.query(Download.id).all())
    assert download_id_ == download_id + 2
    mock_save_inventories.reset_mock()

    # And now assert we do not have anything to update anymore (update_metadata is false)
    result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                    '--dburl', db.dburl,
                                    '--start', '2016-05-08T00:00:00',
                                    '--end', '2016-05-08T09:00:00',
                                    '--update-metadata', 'false',
                                    '--inventory'])
    assert clirunner.ok(result)
    stainvs = db.session.query(Station).filter(Station.has_inventory).all()
    # assert we still have one station (the one we saved before):
    assert len(stainvs) == num_downloaded_inventories_first_try + 1
    assert 'No station inventory to download' in result.output
    assert not mock_save_inventories.called
    download_id_ = max(_[0] for _ in db.session.query(Download.id).all())
    assert download_id_ == download_id + 3

    # Now, we want to simulate the case where we provide a different
    # data center returning an already saved station. The saved station has
    # been downloaded from a different data center

    # id datacenter_id inventory_xml
    # 1 1              b'...'
    # 2 1              None
    # 3 2              b'...'
    # 4 2              None

    # which is equivalent to the dataframe returned by this function
    def get_stadf():
        return pd.read_csv(
            StringIO(
                dbquery2df(
                    db.session.query(
                        Station.id,
                        Station.datacenter_id,
                        Station.inventory_xml,
                        Station.network,
                        Station.station,
                        Station.start_time)
                ).to_string(index=False)
            ),
            sep=r'\s+').sort_values(by=['id']).reset_index(drop=True)

    # i.e., this dataframe:
    sta_df = get_stadf()

    # run a download with no update metadata and no inventory download
    mock_save_inventories.reset_mock()
    new_dataselect = 'http://abc/fdsnws/dataselect/1/query'
    result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                    '--dburl', db.dburl,
                                    '--start', '2016-05-08T00:00:00',
                                    '--end', '2016-05-08T09:00:00',
                                    '-ds', new_dataselect,
                                    '--update-metadata', 'false',
                                    '--inventory'])
    assert clirunner.ok(result)
    assert not mock_save_inventories.called
    new_stadf = get_stadf()
    pd.testing.assert_frame_equal(sta_df, new_stadf)

    # run a download with update metadata and no inventory download
    result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                    '--dburl', db.dburl,
                                    '--start', '2016-05-08T00:00:00',
                                    '--end', '2016-05-08T09:00:00',
                                    '-ds', new_dataselect,
                                    '--update-metadata', 'true'])
    assert clirunner.ok(result)
    assert not mock_save_inventories.called
    # according to self._sta_urlread_sideeffect (see above), the first
    # station response is the GEOFON response, which now it is bound to
    # another data center
    # let's test it:
    new_stadf = get_stadf()
    expected_new_datacenter_id = sta_df.datacenter_id.max() + 1
    ids_changed = sta_df[sta_df.datacenter_id == 1].id
    assert (new_stadf[new_stadf.id.isin(ids_changed)].datacenter_id ==
            expected_new_datacenter_id).all()
    pd.testing.assert_frame_equal(
        sta_df[sta_df.datacenter_id != 1],
        new_stadf[new_stadf.datacenter_id != expected_new_datacenter_id]
    )

    # now run a final download with update metadata and inventory donwload
    # Create a new Datacenter and set all stations and segments with
    # foreign key that datacenter:
    dcn = DataCenter(station_url='http://zzz/fdsnws/station/1/query',
                     dataselect_url='http://zzz/fdsnws/dataselect/1/query')
    # the id must be set manually in postgres (why? do not know, see here
    # for details: https://stackoverflow.com/a/40281835)
    fake_dc_id = 1 + max(_[0] for _ in db.session.query(DataCenter.id))
    dcn.id = fake_dc_id
    db.session.add(dcn)
    db.session.commit()

    def reset_dc_ids():
        stas = db.session.query(Station).all()
        segs = db.session.query(Segment).all()
        for sta in stas:
            sta.datacenter_id = fake_dc_id
        for seg in segs:
            seg.datacenter_id = fake_dc_id
        db.session.commit()

    fake_dc_id = dcn.id
    # now run the tests:
    for param in ['false', 'true', 'only']:
        reset_dc_ids()
        mock_get_events_df.reset_mock()
        mock_get_datacenters_df.reset_mock()
        mock_get_channels_df.reset_mock()
        mock_save_inventories.reset_mock()
        mock_download_save_segments.reset_mock()
        result = clirunner.invoke(cli, ['download', '-c', self.configfile,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T09:00:00',
                                        '-ds', new_dataselect,
                                        '--update-metadata', param,
                                        '--inventory'])
        assert clirunner.ok(result)
        assert mock_get_events_df.called == (param != 'only')
        assert mock_get_datacenters_df.called
        assert mock_get_channels_df.called
        assert mock_download_save_segments.called == (param != 'only')
        alist = self.mock_urlopen.call_args_list
        stainvs = []
        # get all the urls used to fetch the inventories:
        for call_ in alist:
            arg0 = call_[0][0]
            if b'level=response' in getattr(arg0, 'data', b'') \
                    and hasattr(arg0, 'full_url'):
                stainvs.append(arg0.full_url)
        # now check those urls:
        if param == 'false':
            # no call to station inventories:
            assert not stainvs
            # assert we did not change any datacenter:
            assert all(_.datacenter_id == fake_dc_id for _ in db.session.query(Station))
            assert all(_.datacenter_id == fake_dc_id for _ in db.session.query(Segment))
        else:
            assert any(new_dataselect.replace('dataselect', 'station')
                       in _ for _ in stainvs)
            assert any(_.datacenter_id == fake_dc_id for _ in db.session.query(Station))
            assert any(_.datacenter_id != fake_dc_id for _ in db.session.query(Station))
            if param == 'only':
                assert all(
                    _.datacenter_id == fake_dc_id for _ in db.session.query(Segment))
            else:
                assert any(
                    _.datacenter_id == fake_dc_id for _ in db.session.query(Segment))
                assert any(
                    _.datacenter_id != fake_dc_id for _ in db.session.query(Segment))
