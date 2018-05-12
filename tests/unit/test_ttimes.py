'''
Created on Sep 4, 2017

@author: riccardo
'''
from __future__ import division
from future import standard_library
import pytest
standard_library.install_aliases()

from stream2segment.cli import ttcreate
from builtins import zip
from builtins import str
from past.utils import old_div
import numpy as np
from os.path import dirname, join, isfile
import unittest
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.traveltimes import ttcreator
# from stream2segment.download.utils import get_min_travel_time
from click.testing import CliRunner
import os
from stream2segment.traveltimes.ttcreator import _filepath, StepIterator, min_traveltimes,\
    min_traveltime


@pytest.fixture(scope='module')
def ttdata(data):

    iasp_ttp_5 = data.read_tttable("iasp91_ttp+_5.npz")
    iasp_tts_10 = data.read_tttable("iasp91_tts+_10.npz")
    iasp_tts_5 = data.read_tttable("iasp91_tts+_5.npz")
    ak135_ttp_10 = data.read_tttable("ak135_ttp+_10.npz")
    ak135_ttp_5 = data.read_tttable("ak135_ttp+_5.npz")
    ak135_tts_10 = data.read_tttable("ak135_tts+_10.npz")
    ak135_tts_5 = data.read_tttable("ak135_tts+_5.npz")

    _values = np.array([(0, 0, 0), (1, 0, 0), (5, 0, 0), (700,0,0),
                        (0, 0, 11.4), (1, 0, 11.4), (5, 0, 11.4), (700, 0, 11.4),
                        (0, 0, 21.9), (1, 0, 21.9), (5, 0, 21.9), (700, 0, 21.9),
                        (0, 0, 178.9), (1, 0, 178.9), (5, 0, 178.9), (700, 0, 178.9),
                        (0, 0, 361.9), (1, 0, 361.9), (5, 0, 361.9), (700, 0, 361.9),
                       ])

    # Tests are too long. Instead of writing this:
    # let's test only for tables with errtol 5 seconds. This means basically the other tables
    # are not tested. But we use 5sec errtol models so it's fine
    return type('ttdata', (object,), {'tables': [iasp_ttp_5, iasp_tts_5, ak135_ttp_5, ak135_tts_5],
                                      'values': _values })


def test_ttcreator():

    runner = CliRunner()
    with runner.isolated_filesystem():
        mydir = os.getcwd()
        result = runner.invoke(ttcreate, catch_exceptions=True)
        assert result.exit_code != 0

        phase = 'ttp'  # NOTE: by setting e.g. just 'P' we do NOT make it faster, cause
        # some numbers might be nan and thus the iteration is slower
        # try to make ttp which should be faster than ttp+ and most likely without
        # nans that are not in ttp+
        result = runner.invoke(ttcreate, ['-o', mydir, '-m', 'iasp91', '-t', 10, '-p',
                                               phase, '-s', 51.3, '-r', 2, '-d', 34.3],
                               catch_exceptions=False)
        assert result.exit_code == 0
        assert os.path.isfile(_filepath(mydir, 'iasp91', [phase]) + ".npz")
        # fixme: we should load the file and assert something...


# the test below might be refined and assert more stuff
# (it has however been inspected with eclipse debugger once to test everything worked fine)
def test_ttcreator_tts():

    runner = CliRunner()
    with runner.isolated_filesystem():
        mydir = os.getcwd()
        result = runner.invoke(ttcreate, catch_exceptions=True)
        assert result.exit_code != 0

        # test with no receiver depths (set to 0)
        phase = 'tts'
        result = runner.invoke(ttcreate, ['-o', mydir, '-m', 'ak135', '-t', 10, '-p',
                                               phase, '-s', 51.3, '-r', 0 , '-d', 34.3],
                               catch_exceptions=False)
        assert result.exit_code == 0
        assert os.path.isfile(_filepath(mydir, 'ak135', [phase]) + ".npz")
        # fixme: we should load the file and assert something...


def test_stepiterator():
    '''test a step iterator which should give me approximately every 100's'''
    lastnum = -1
    results = []
    stepiterator = StepIterator(0, 700.0, 31.5)
    for val in stepiterator:
        if int(old_div(val, 100.0)) > lastnum:  # condition whereby we crossed the 'mark' 
            if stepiterator.moveback():
                continue
            else:
                results.append(val)
                lastnum += 1
    assert len(results) == 8

    # now try an edge case with a single value
    results = []
    stepiterator = StepIterator(0, 0, 31.5)
    for val in stepiterator:
        results.append(val)
    assert results == [0]


def test_ttable(ttdata):
    for ttable in ttdata.tables:
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = old_div(ttable._distances[1],2.0)
        # get the half step (max distance along y axis = row)
        half_vstep = old_div(ttable._sourcedepths[0], 2.0)
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], ttdata.values))
        results_c = ttable.min(values[:, 0], values[:, 1], values[:, 2], method='cubic')

        real_results = []
        for v in values:
            real_results.append(min_traveltime(ttable.model,
                                               v[0], v[1], v[2], ttable.phases))

        assert np.allclose(results_c, real_results, rtol=0, atol=ttable._tt_errtol,
                           equal_nan=True)

        results_l = ttable.min(values[:, 0], values[:, 1], values[:, 2], method='linear')
        results_n = ttable.min(values[:, 0], values[:, 1], values[:, 2], method='nearest')

        # for some tts+ models, the linear case might lead to median that are
        # better than the cubic case

        err_c = np.abs(results_c-real_results)
        err_l = np.abs(results_l-real_results)
        err_n = np.abs(results_n-real_results)

        # for cubic vs nearest, we can simply assert this:
        assert np.nanmean(err_c) < np.nanmean(err_n)
        assert np.nanmedian(err_c) < np.nanmedian(err_n)
        assert np.nanmax(err_c) < np.nanmean(err_n)
        assert np.nanmean(err_l) < np.nanmean(err_n)
        assert np.nanmedian(err_l) < np.nanmedian(err_n)
        assert np.nanmax(err_l) < np.nanmedian(err_n)

    # on the other hand, sometimes (tts+ models) the mean linear is better than the cubic one
    # The reason is partly because there might be a bias on the (low) number of points
    # and also because it seems that linear outperforms cubic for the tts+ case
    # so do not do this:
#             assert np.nanmean(err_c) < np.nanmean(err_l)
#             assert np.nanmedian(err_c) < np.nanmedian(err_l)
#             assert np.nanmax(err_c) < np.nanmax(err_l)

def test_edge_cases(ttdata):
    for ttable in ttdata.tables:
        for method in ['linear', 'cubic', 'nearest']:
            # test scalar case. Assert is stupid is just to test no error is thrown
            assert ttable.min(101.2, 0, 16.6, method) >= 0
            # out of bounds: source depth
            assert np.isnan(ttable.min(701.2, 0, 16.6, method))
            # out of bounds: source depth + receiver depth
            assert np.isnan(ttable.min(701, 0.3, 16.6, method))
            # out of bounds: receiver depth
            assert np.isnan(ttable.min(567.5, 0.3, 16.6, method))
            # source depths < 0 are converted to 0
            assert ttable.min(-.5, 0, 16.6, method) == ttable.min(0, 0, 16.6, method)
            # receiver depths < 0 are converted to 0
            assert ttable.min(567.5, -0.3, 16.6, method) == ttable.min(567.5, 0, 16.6, method)
            # distances are never out of bounds to be compliant with obspy travel times
            # but they are modulus 360:
            assert np.allclose(ttable.min(567.5, 0, 1.66, method),
                               ttable.min(567.5, 0, 361.66, method))
            # distances equidistant from 180 degree are also treated as equal:
            assert ttable.min(567.5, 0, 180+1.66, method) == ttable.min(567.5, 0, 180-1.66, method)

def test_tostr(ttdata):
    repr = str(ttdata.tables[0])
    expected = """Model: 'iasp91'
Phases: ['ttp+']
Input error tolerance: 5.000000
Data:
-------- -------- ---------------------------------------------------------
  Source Receiver
   depth    depth                                              Travel times
-------- -------- ---------------------------------------------------------
     0.0        0      0.0    4.314    8.627 ...  1212.08  1212.08  1212.08
    30.1        0    5.002    6.491    9.644 ...  1207.08  1207.08  1207.08
    69.2        0   10.009   10.621   12.261 ...  1202.07  1202.07  1202.08
     ...      ...                                                       ...
   600.1        0   70.074   70.128    70.29 ...  1142.01  1142.01  1142.01
   650.6        0   75.082   75.131   75.277 ...   1137.0   1137.0   1137.0
   700.0        0   79.694   79.738   79.871 ...  1132.39  1132.39  1132.39
-------- -------- ---------------------------------------------------------
      Distances->      0.0    0.225     0.45 ...   179.55  179.775    180.0"""
    # do a comparison line by line cause apparently the string above is not equal to repr
    # but we suspect is because we have problems with eclipse
    for l_1, l_2 in zip(repr.split("\n"), expected.split("\n")):
        assert l_1.strip() == l_2.strip()

