'''
Created on Sep 4, 2017

@author: riccardo
'''
import numpy as np
from os.path import dirname, join, isfile
import unittest
from stream2segment.download.traveltimes.ttloader import TTTable
from stream2segment.download.utils import get_min_travel_time
from click.testing import CliRunner
import os
from stream2segment.download.traveltimes.ttcreator import _filepath


class Test(unittest.TestCase):

    def setUp(self):
        pass
#         self.ttable_10sec = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_errtol=10sec.npz"))
#         self.ttable_04sec = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_errtol=04sec.npz"))
#         self.ttable_03sec = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_errtol=03sec.npz"))
# 
#         self._values = np.array([(0,0,0), (1, 0, 0), (5, 0, 0), (700,0,0),
#                            (0, 0, 11.4), (1, 0, 11.4), (5, 0, 11.4), (700, 0, 11.4),
#                            (0, 0, 21.9), (1, 0, 21.9), (5, 0, 21.9), (700, 0, 21.9),
#                            (0, 0, 178.9), (1, 0, 178.9), (5, 0, 178.9), (700, 0, 178.9),
#                            (0, 0, 361.9), (1, 0, 361.9), (5, 0, 361.9), (700, 0, 361.9),
#                            ])
    def tearDown(self):
        pass

    def test_ttcreator(self):
        from stream2segment.download.traveltimes import ttcreator
        runner = CliRunner()
        with runner.isolated_filesystem():
            mydir = os.getcwd()
            result = runner.invoke(ttcreator.run)
            assert result.exit_code != 0

            result = runner.invoke(ttcreator.run, ['-o', mydir, '-m', 'iasp91', '-t', 100, '-p',
                                                   'ttp+'])
            assert result.exit_code == 0
            assert os.path.isfile(_filepath(mydir, 'iasp91', ['ttp+']) + ".npz")
            # fixme: we should load the file and assert something...
            

    def tst_err10sec(self):
        ttable = self.ttable_10sec
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = ttable._distances_deg[1]/2.0
        # get the half step (max distance along y axis = row)
        half_vstep = ttable._sdrd_pairs[1][0] / 2.0
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], self._values))
        results = ttable.min(values[:, 0], values[:, 1], values[:, 2])
        p = ttable._phases
        real_results = []
        for v in values:
            real_results.append(get_min_travel_time(v[0], v[2], list(p),
                                v[1], 'iasp91'))

        assert np.allclose(results, real_results, rtol=0, atol=ttable._tt_errtol,
                           equal_nan=True)
        max = np.max(np.abs(results-real_results))
        # assert the maximum error is what we expect:
        assert max == np.abs(results-real_results)[0]
        h = 9

    def tst_err04sec(self):
        ttable = self.ttable_04sec
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = ttable._distances_deg[1]/2.0
        # get the half step (max distance along y axis = row)
        half_vstep = ttable._sdrd_pairs[1][0] / 2.0
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], self._values))
        results = ttable.min(values[:,0], values[:,1], values[:,2])
        p = ttable._phases
        real_results = []
        for v in values:
            real_results.append(get_min_travel_time(v[0], v[2], list(p),
                                v[1], 'iasp91'))

        assert np.allclose(results, real_results, rtol=0, atol=ttable._tt_errtol,
                           equal_nan=True)
        max = np.max(np.abs(results-real_results))
        # assert the maximum error is what we expect:
        assert max == np.abs(results-real_results)[0]
        h = 9

    def tst_04err_vs_10err(self):
        ttable04 = self.ttable_04sec
        ttable10 = self.ttable_10sec
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = ttable04._distances_deg[1]/2.0
        # get the half step (max distance along y axis = row)
        half_vstep = ttable04._sdrd_pairs[1][0] / 2.0
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], self._values))
        results04 = ttable04.min(values[:, 0], values[:, 1], values[:, 2])
        results10 = ttable10.min(values[:, 0], values[:, 1], values[:, 2])
        p = ttable04._phases
        real_results04 = []
        for v in values:
            real_results04.append(get_min_travel_time(v[0], v[2], list(p),
                                                      v[1], 'iasp91'))

        if (ttable10._phases != ttable04._phases).any():
            real_results10 = []
            for v in values:
                real_results10.append(get_min_travel_time(v[0], v[2], list(p),
                                                          v[1], 'iasp91'))
        else:
            real_results10 = real_results04

        err04 = np.abs(results04-real_results04)
        err10 = np.abs(results10-real_results10)
        improved = ((err10 - err04) > 0).sum()
        notimproved = ((err10 - err04) < 0).sum()
        assert improved > notimproved
    # test perfs with different steps and see that results are closer
    # test with scalars
    # test with source_depths and receiver depths out of bounds
    
    def tst_03err_vs_10err(self):
        ttable03 = self.ttable_03sec
        ttable10 = self.ttable_10sec
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = ttable03._distances_deg[1]/2.0
        # get the half step (max distance along y axis = row)
        half_vstep = ttable03._sdrd_pairs[1][0] / 2.0
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], self._values))
        results03 = ttable03.min(values[:, 0], values[:, 1], values[:, 2])
        results10 = ttable10.min(values[:, 0], values[:, 1], values[:, 2])
        p = ttable03._phases
        real_results03 = []
        for v in values:
            real_results03.append(get_min_travel_time(v[0], v[2], list(p),
                                                      v[1], 'iasp91'))

        if (ttable10._phases != ttable03._phases).any():
            real_results10 = []
            for v in values:
                real_results10.append(get_min_travel_time(v[0], v[2], list(p),
                                                          v[1], 'iasp91'))
        else:
            real_results10 = real_results03

        err03 = np.abs(results03-real_results03)
        err10 = np.abs(results10-real_results10)
        improved = ((err10 - err03) > 0).sum()
        notimproved = ((err10 - err03) < 0).sum()
        assert improved > notimproved

    def tst_maxerrs(self):
        ttable03 = self.ttable_03sec
        ttable04 = self.ttable_04sec
        ttable10 = self.ttable_10sec
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = ttable03._distances_deg[1]/2.0
        # get the half step (max distance along y axis = row)
        half_vstep = ttable03._sdrd_pairs[1][0] / 2.0
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], self._values))
        results03 = ttable03.min(values[:, 0], values[:, 1], values[:, 2])
        results04 = ttable04.min(values[:, 0], values[:, 1], values[:, 2])
        results10 = ttable10.min(values[:, 0], values[:, 1], values[:, 2])
        p = ttable03._phases
        real_results = []
        for v in values:
            real_results.append(get_min_travel_time(v[0], v[2], list(p),
                                                      v[1], 'iasp91'))


        err03max = np.abs(results03-real_results).max()
        err04max = np.abs(results04-real_results).max()
        err10max = np.abs(results10-real_results).max()
        err03mean = np.abs(results03-real_results).mean()
        err04mean = np.abs(results04-real_results).mean()
        err10mean = np.abs(results10-real_results).mean()
        err03median = np.median(np.abs(results03-real_results))
        err04median = np.median(np.abs(results04-real_results))
        err10median = np.median(np.abs(results10-real_results))
        
        assert True
        assert True
        assert True

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()