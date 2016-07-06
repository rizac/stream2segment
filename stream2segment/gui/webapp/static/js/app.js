var myApp = angular.module('myApp',[]);
 
myApp.controller('myController', ['$scope', '$http', '$window', function($scope, $http, $window) {
	$scope.elements = [];
	$scope.currentIndex = -1;
	$scope.data = {};
	$scope.showFiltered = true;
	$scope.isEditingIndex = false;
	$scope.classesActiveTab = 0;
	$scope.classes = {};

	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.updatePlots(true);
	};
	
	$scope.isClassFilterProperlySet = function(){
		for(var id in $scope.classes){
			if ($scope.classes[id].visible && $scope.classes[id].Count){
				return true;
			}
		}
		return false;
	};
	
	$scope.getCurrentSegmentName = function(){
		if (!$scope.data || !$scope.data.metadata){
			return ""
		}
		return $scope.data.metadata[8][1] + "." + $scope.data.metadata[9][1] + "." + $scope.data.metadata[10][1] + "." + $scope.data.metadata[11][1];
	};
	
	$scope.info2str = function(key, value){
		if (key.toLowerCase().indexOf("time") > -1){
    		return $scope.datetime2iso(value)[1];
    	}else if (key.toLowerCase().indexOf("date") > -1){
    		return $scope.datetime2iso(value)[0];
    	}
		return value;
	};
	
	$scope.datetime2iso = function(timestamp){
		// toISOString seems to consider times as UTC which is what we want. By default,
		// (i.e., on the time axis by specifying time scale) it converts them according to
		// local timezone
		var ts =  $window.moment(timestamp).toISOString();
		if (ts[ts.length-1] === 'Z'){
			ts = ts.substring(0, ts.length - 1);
		}
		return ts.split("T");
	};
	
//	$scope.setClassTab = function(index){
//		$scope.classTabSelIndex = index;
//	};
	
	$scope.setCurrentSegmentClass = function(){
		//note: due to data binding $scope.data.class_id is already updated
    	//i.e., $scope.data.class_id is the new classId
		var param = {segmentId:$scope.elements[$scope.currentIndex], classId: $scope.data.class_id};
	    $http.post("/set_class_id", param, {headers: {'Content-Type': 'application/json'}}).
	    success(function(data, status, headers, config) {
	        // this callback will be called asynchronously
	        // when the response is available
	    	oldClassId = parseInt(data);
	    	if (oldClassId == $scope.data.class_id){
	    		return;
	    	}

	    	$scope.classes[oldClassId].Count -= 1;
	    	$scope.classes[$scope.data.class_id].Count += 1;
	      }).
	      error(function(data, status, headers, config) {
	        // called asynchronously if an error occurs
	        // or server returns response with an error status.
	      });
	};
	
	$scope.init = function(){ //classIds is an Array
		var classIds = [];
		for(var id in $scope.classes){
			if ($scope.classes[id].visible){
				classIds.push(parseInt(id));
			}
		}
	    $scope.refresh({'class_ids': classIds});
	};
	
	$scope.refresh = function(data){ 
		var classIds = data.class_ids;
		$http.post("/get_elements", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.elements = response.data.segment_ids;
	        $scope.classes = {};
	        for (var i in response.data.classes){
	        	var id = response.data.classes[i].Id;
	        	$scope.classes[id] = response.data.classes[i];
	        	$scope.classes[id].visible = (!classIds.length || classIds.indexOf(id) >= 0);
	        }
	        $scope.setCurrentIndex(0);
	    });
	};
	
	$scope.init();
	
	$scope.setNextIndex = function(){
		var currentIndex = ($scope.currentIndex + 1) % ($scope.elements.length);
		$scope.setCurrentIndex(currentIndex);
	};
	
	$scope.setPreviousIndex = function(){
		var currentIndex = $scope.currentIndex == 0 ? $scope.elements.length - 1 : $scope.currentIndex - 1;
		$scope.setCurrentIndex(currentIndex);
	};

	$scope.setCurrentIndex = function(index){
		$scope.isEditingIndex = false;
		var param = {segId: $scope.elements[index]};
		$http.post("/get_data", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.currentIndex = index;
	        $scope.data = response.data;
	        //$scope.currentClassId = response.data.class_id;
	        $scope.updatePlots();

	    });
	};
	
	$scope.updatePlots = function(updateFilterOnly){
	    
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
		    	if(key == "ArrivalTime"){
		    		arrivalTime = value;
		    	}else if(key == "SnrWindow/sec"){
		    		snrWindowInSec = value;
		    	}else if(key == "Cum_time( 5%)"){
		    		cumt5 = value;
		    	}else if(key == "Cum_time(95%)"){
		    		cumt95 = value;
		    	}else if(key == "SampleRate"){
		    		sampleRate = value;
		    	}
		    }

		    $window.mseed1.config.options.arrivalTime = arrivalTime;
		    $window.mseed1.config.options.snrWindowInSec = snrWindowInSec;
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

}]);