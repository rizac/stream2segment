'''
Created on Nov 26, 2016

@author: riccardo
'''
import unittest
from obspy.taup.tau import TauPyModel

import time
from obspy.taup.taup_time import TauPTime


class Test(unittest.TestCase):


    def setUp(self):
        self.model='ak135'
        self.depths = [163.0,
                       2.0,
                       40.0,
                       31.0,
                       60.0,
                       2.0,
                       8.0,
                       1.0,
                       10.0,
                       23.0,
                       2.0,
                       2.0,
                       9.0,
                       9.0,
                       5.0,
                       5.0,
                       95.0,
                       40.0,
                       2.0,
                       2.0,
                       5.0,
                       40.0,
                       108.0,
                       13.0,
                       2.0,
                       5.0,
                       100.0,
                       125.0,
                       1.0,
                       8.0,
                       3.0,
                       10.0,
                       33.0,
                       8.0,
                       5.0,
                       7.0,
                       15.0,
                       2.0,
                       10.0,
                       2.0,
                       12.0,
                       2.0,
                       5.0,
                       10.0,
                       10.0,
                       6.0,
                       7.0,
                       60.0,
                       4.0,
                       13.0,
                       10.0,
                       10.0,
                       7.0,
                       10.0,
                       7.0,
                       20.0,
                       133.0,
                       6.0,
                       14.0,
                       13.0,
                       8.0,
                       2.0,
                       1.0,
                       8.0,
                       2.0,
                       10.0,
                       2.0,
                       30.0,
                       1.0,
                       10.0,
                       1.0,
                       3.0]
        self.distances = [0.423639654834945,
                          0.423639654834945,
                          0.423639654834945,
                          0.614828712976814,
                          0.614828712976814,
                          0.614828712976814,
                          0.091869799713805,
                          0.091869799713805,
                          0.091869799713805,
                          0.820942518261341,
                          0.820942518261341,
                          0.820942518261341,
                          0.830863851001827,
                          0.830863851001827,
                          0.830863851001827,
                          0.745644850860421,
                          0.745644850860421,
                          0.745644850860421,
                          0.423639654834945,
                          0.423639654834945,
                          0.423639654834945,
                          0.614828712976814,
                          0.614828712976814,
                          0.614828712976814,
                          0.091869799713805,
                          0.091869799713805,
                          0.091869799713805,
                          0.820942518261341,
                          0.820942518261341,
                          0.820942518261341,
                          0.830863851001827,
                          0.830863851001827,
                          0.830863851001827,
                          0.745644850860421,
                          0.745644850860421,
                          0.745644850860421,
                          0.656298974931482,
                          0.656298974931482,
                          0.656298974931482,
                          0.656298974931482,
                          0.656298974931482,
                          0.656298974931482,
                          0.521856037266905,
                          0.521856037266905,
                          0.521856037266905,
                          0.521856037266905,
                          0.521856037266905,
                          0.521856037266905,
                          0.172061610011542,
                          0.172061610011542,
                          0.172061610011542,
                          1.03883029291681,
                          1.03883029291681,
                          1.03883029291681,
                          1.03883029291681,
                          1.03883029291681,
                          1.03883029291681,
                          0.831822894298477,
                          0.831822894298477,
                          0.831822894298477,
                          0.831822894298477,
                          0.831822894298477,
                          0.831822894298477,
                          1.13979738166243,
                          1.13979738166243,
                          1.13979738166243,
                          1.13979738166243,
                          1.13979738166243,
                          1.13979738166243,
                          0.195515696210682,
                          0.195515696210682,
                          0.195515696210682,
                          0.195515696210682,
                           ]

        pass


    def tearDown(self):
        pass


    def testTTimes(self):
        
        N = 50
        depths = self.depths[:N+1]
        distances = self.distances[:N+1]
        
        start = time.time()
        for source_depth_in_km, distance_in_degree in zip(depths, distances):
            taupmodel = TauPyModel(self.model)
            tt = taupmodel.get_travel_times(source_depth_in_km, distance_in_degree)
            min_ = tt[0].time
        end = time.time()
        print "S2s arrival_time: %d iterations: % secs" % (10*N, 10*(end - start))

        taupmodel = TauPyModel(self.model)
        phase_list = ("ttall",)
        receiver_depth_in_km = 0.0
        start = time.time()
        for source_depth_in_km, distance_in_degree in zip(depths, distances):
            tt = TauPTime(taupmodel.model, phase_list, source_depth_in_km,
                          distance_in_degree, receiver_depth_in_km)
            tt.run()
            min_ = min(tt.arrivals, key=lambda x: x.time)
            # j = 9
        end = time.time()
        print "S2s arrival_time (optimized?): %d iterations: % secs" % (10*N, 10*(end - start))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testTTimes']
    unittest.main()