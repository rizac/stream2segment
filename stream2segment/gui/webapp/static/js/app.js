var myApp = angular.module('myApp',[]);
 
myApp.controller('myController', ['$scope', '$http', '$window', function($scope, $http, $window) {
	$scope.elements = [];
	$scope.currentIndex = -1;
	$scope.data = {};
	$scope.showFiltered = true;
	$scope.isEditingIndex = false;
	$scope.classes = [];
	$scope.currentSegmentClassIds = [];
	$scope.currentSegmentText = "";

	$scope.init = function(){
		var data = {}; //maybe in the future pass some data
		$http.post("/get_classes", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.classes = response.data.classes;
	        $scope.refresh({});
	    });
		
	};
	
	$scope.refresh = function(data){ 
		//var classIds = data.class_ids;
		$http.post("/get_elements", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.elements = response.data.segment_ids;
	        $scope.setCurrentIndex(0);
	    });
	};
	
	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.updatePlots(true);
	};
	
	$scope.getCurrentSegmentName = function(){
		if (!$scope.data || !$scope.data.metadata){
			return ""
		}
		return "METADATA";
	};
	
	$scope.info2str = function(value){
		if (value && value.startsWith){
			if (value.startsWith("[__TIME__]")){
				value = $scope.cast(value);
				return $scope.splitDateAndTime(value)[1];
			}else if (value.startsWith("[__DATE__]")){
				value = $scope.cast(value);
				return $scope.splitDateAndTime(value)[0];
			}
		}
		return value;
	};
	
	$scope.cast = function(value){
		//converts a value passed from metadata to moment object IF is string and starts
		// with either "[__DATE__]" or "[__TIME__]"
		//in any other case, returns the value
		if (value && value.startsWith){
			if (value.startsWith("[__TIME__]")){
				value = "" + value.substring("[__TIME__]".length, (""+value).length);
				return $window.moment.utc(parseFloat(value));
			}else if (value.startsWith("[__DATE__]")){
				value = "" + value.substring("[__DATE__]".length, (""+value).length);
				return $window.moment.utc(parseFloat(value));
			}
		}
		return value;
	};
	
	$scope.splitDateAndTime = function(momentObj){
		// toISOString seems to consider times as UTC which is what we want. By default,
		// (i.e., on the time axis by specifying time scale) it converts them according to
		// local timezone (but we overridden this behavior to display moment.utc in index.html)
		var ts =  momentObj.toISOString();
		if (ts[ts.length-1] === 'Z'){
			ts = ts.substring(0, ts.length - 1);
		}
		return ts.split("T");
	};

	
	$scope.toggleClassIdSelectionForCurrentSegment = function(classId){
		
		var param = {class_id: classId, segment_id: $scope.elements[$scope.currentIndex]};
	    $http.post("/toggle_class_id", param, {headers: {'Content-Type': 'application/json'}}).
	    success(function(data, status, headers, config) {
	        $scope.classes = data.classes;
	        $scope.currentSegmentClassIds = data.segment_class_ids;
	      }).
	      error(function(data, status, headers, config) {
	        // called asynchronously if an error occurs
	        // or server returns response with an error status.
	      });
	};
	
	$scope.setNextIndex = function(){
		var currentIndex = ($scope.currentIndex + 1) % ($scope.elements.length);
		$scope.setCurrentIndex(currentIndex);
	};
	
	$scope.setPreviousIndex = function(){
		var currentIndex = $scope.currentIndex == 0 ? $scope.elements.length - 1 : $scope.currentIndex - 1;
		$scope.setCurrentIndex(currentIndex);
	};
	
	$scope.setCurrentIndexFromText = function(){
		var index = parseInt($scope.currentSegmentText);
		if (index >0 && index <= $scope.elements.length){
			$scope.setCurrentIndex(index-1);
		}
	}
	
	$scope.setCurrentIndex = function(index){
		$scope.isEditingIndex = false;
		var param = {segId: $scope.elements[index], filteredRemResp: $scope.showFiltered};
		$http.post("/get_data", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.currentIndex = index;
	        $scope.data = response.data;
	        $scope.currentSegmentClassIds = response.data.class_ids;
	        $scope.updatePlots();

	    });
	};
	
	$scope.updatePlots = function(updateFilterOnly){ // update filter only is not actually used anymore
	    
	    var index = $scope.showFiltered ? 1 : 0;
        var SCOPEDATA =  $scope.data.time_data;
        
        var timeLabels = SCOPEDATA.labels;
        var datasets = SCOPEDATA.datasets;
	    $window.mseed1.data.labels = timeLabels;
	    $window.mseed1.data.datasets[0].data = datasets[index].data;
	    $window.mseed1.data.datasets[0].label = datasets[index].label;
	    
	    $window.mseed2.data.labels = timeLabels;
	    $window.mseed2.data.datasets[0].data = datasets[index + 2].data;
	    $window.mseed2.data.datasets[0].label = datasets[index + 2].label;
        
	    $window.mseed3.data.labels = timeLabels;
	    $window.mseed3.data.datasets[0].data = datasets[index + 4].data;
	    $window.mseed3.data.datasets[0].label = datasets[index + 4].label;
        
	    plots2refresh = [$window.mseed1, $window.mseed2, $window.mseed3];
	    
	    if (updateFilterOnly !== true){
		    var SCOPEMETADATA = $scope.data.metadata;
		    
		    var arrivalTime = undefined;
		    var snrWindowInSec = undefined;
		    var cumt5 = undefined;
		    var cumt95 = undefined;
		    var sampleRate = undefined;
		    
		    for (var i in SCOPEMETADATA){
		    	key = SCOPEMETADATA[i][0];
		    	value = SCOPEMETADATA[i][1];
		    	if(key == "arrival_time (+ config delay)"){
		    		arrivalTime = $scope.cast(value);
		    	}else if(key == "noise_fft_start"){
		    		noiseFFtStart = $scope.cast(value);
		    	}else if(key == "cum_t05"){
		    		cumt5 = $scope.cast(value);
		    	}else if(key == "cum_t95"){
		    		cumt95 = $scope.cast(value);
		    	}else if(key == "sample_rate"){
		    		sampleRate = $scope.cast(value);
		    	}
		    }

		    $window.mseed1.config.options.fftWindows = [noiseFFtStart,arrivalTime,cumt5,cumt95];
		    //$window.mseed1.config.options.snrWindowInSec = snrWindowInSec;
		    $window.mseed_cum.config.options.cumT5 = cumt5;
		    $window.mseed_cum.config.options.cumT95 = cumt95;
		    
		    $window.mseed_cum.data.labels = timeLabels;
		    $window.mseed_cum.data.datasets[0].data = datasets[6].data;
		    $window.mseed_cum.data.datasets[0].label = datasets[6].label;
		    
		    $window.mseed_env.data.labels = timeLabels;
		    $window.mseed_env.data.datasets[0].data = datasets[7].data;
		    $window.mseed_env.data.datasets[0].label = datasets[7].label;
		    
		    SCOPEDATA =  $scope.data.freq_data;
		    var freqLabels = SCOPEDATA.labels;
		    datasets = SCOPEDATA.datasets;
		    //set maximum for the x scale, if sampleRate is found
		    if (sampleRate){
		    	$window.mseed_snr.config.options.scales.xAxes[0].ticks.max = Math.log10(parseFloat(sampleRate) / 2);
		    }else{
		    	delete $window.mseed_snr.config.options.scales.xAxes[0].ticks.max;
		    }
		    //$window.mseed_snr.data.labels = freqLabels;
		    $window.mseed_snr.data.datasets[0].data = datasets[0].data;
		    $window.mseed_snr.data.datasets[0].label = datasets[0].label;
		    $window.mseed_snr.data.datasets[1].data = datasets[1].data;
		    $window.mseed_snr.data.datasets[1].label = datasets[1].label;
		    
		    plots2refresh.push($window.mseed_env);
		    plots2refresh.push($window.mseed_cum);
		    //plots2refresh.push($window.mseed_snr);
		    plots2refresh.splice(0, 0, $window.mseed_snr);
		    
	    }
	    
	    for (i in plots2refresh){
	    	plots2refresh[i].update();
	    }
	};
	
	// init our app:
	$scope.init();
	

}]);