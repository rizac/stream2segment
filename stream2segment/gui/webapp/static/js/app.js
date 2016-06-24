var myApp = angular.module('myApp',[]);
 
myApp.controller('myController', ['$scope', '$http', '$window', function($scope, $http, $window) {
	$scope.elements = [];
	$scope.currentIndex = -1;
	$scope.data = [];
	$scope.showFiltered = true;
	
	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.updatePlots();
	};
	
	$scope.setNextIndex = function(){
		var currentIndex = ($scope.currentIndex + 1) % ($scope.elements.length);
		$scope.setCurrentIndex(currentIndex);
	};
	
	$scope.setPreviousIndex = function(){
		var currentIndex = $scope.currentIndex == 0 ? $scope.elements.length - 1 : $scope.currentIndex - 1;
		$scope.setCurrentIndex(currentIndex);
	};
	
	$scope.setCurrentIndex = function(index){
		$http.get("/get_data/" + $scope.elements[index]).then(function(response) {
			$scope.currentIndex = index;
	        $scope.data = response.data;
	        $scope.updatePlots();

	    });
	};
	
	$scope.updatePlots = function(){
	    
	    var key = $scope.showFiltered ? 'trace_bandpass' : 'trace';
        
        $window.mseed1.data.datasets[0].data = $scope.data.data[0][key];
        $window.mseed1.data.datasets[0].label = $scope.data.data[0].id;
        $window.mseed1.data.labels = $scope.data.data[0]['times'];
        $window.mseed1.config.options._arrivalTime = $scope.data.arrival_time;
        $window.mseed1.config.options._snrDtInSec = $scope.data.snr_dt_in_sec;
        
        $window.mseed2.data.datasets[0].data = $scope.data.data[1][key];
        $window.mseed2.data.datasets[0].label = $scope.data.data[1].id;
        $window.mseed2.data.labels = $scope.data.data[1]['times'];
        
        $window.mseed3.data.datasets[0].data = $scope.data.data[2][key];
        $window.mseed3.data.datasets[0].label = $scope.data.data[2].id;
        $window.mseed3.data.labels =  $scope.data.data[2]['times'];
        
        // update snr, env, cum:
        $window.mseed_snr.data.datasets[0].data = $scope.data.data[0]['snr_noise'];
        $window.mseed_snr.data.datasets[1].data = $scope.data.data[0]['snr_sig'];
        $window.mseed_snr.data.datasets[0].label = $scope.data.data[0].id + " (Noise)";
        $window.mseed_snr.data.datasets[1].label = $scope.data.data[0].id + " (Signal)";
        $window.mseed_snr.data.labels = $scope.data.freqs;
        //set also scale min and max, required for the log scale (y axis only, both are not supported):
        //$window.mseed_snr.options.scales.xAxes[0].ticks.min = $scope.data.spectrum_bounds[0];
        //$window.mseed_snr.options.scales.xAxes[0].ticks.max = $scope.data.spectrum_bounds[1];
        
        //$window.mseed_snr.options.scales.yAxes[0].ticks.min = $scope.data.spectrum_bounds[2];
        //$window.mseed_snr.options.scales.yAxes[0].ticks.max = $scope.data.spectrum_bounds[3];
        
        $window.mseed_cum.data.datasets[0].data = $scope.data.data[0]['cum'];
        $window.mseed_cum.data.datasets[0].label = $scope.data.data[0].id + " (Cumulative)";
        $window.mseed_cum.data.labels = $scope.data.data[0]['times'];
        
        $window.mseed_env.data.datasets[0].data = $scope.data.data[0]['env'];
        $window.mseed_env.data.datasets[0].label = $scope.data.data[0].id + " (Envelope)";
        $window.mseed_env.data.labels = $scope.data.data[0]['times'];
        
        $window.mseed1.update();
        $window.mseed2.update();
        $window.mseed3.update();
        $window.mseed_env.update();
        $window.mseed_cum.update();
        $window.mseed_snr.update();
	};
	
	$http.get("/get_elements").then(function(response) {
        $scope.elements = response.data;
        $scope.setCurrentIndex(0);
    });

}]);