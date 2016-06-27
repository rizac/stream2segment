var myApp = angular.module('myApp',[]);
 
myApp.controller('myController', ['$scope', '$http', '$window', function($scope, $http, $window) {
	$scope.elements = [];
	$scope.currentIndex = -1;
	$scope.data = {};
	$scope.showFiltered = true;
	$scope.classTabSelIndex = 0;

	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.updatePlots(true);
	};
	
	$scope.setNextIndex = function(){
		var currentIndex = ($scope.currentIndex + 1) % ($scope.elements.length);
		$scope.setCurrentIndex(currentIndex);
	};
	
	$scope.setPreviousIndex = function(){
		var currentIndex = $scope.currentIndex == 0 ? $scope.elements.length - 1 : $scope.currentIndex - 1;
		$scope.setCurrentIndex(currentIndex);
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
	
	$scope.setClassTab = function(index){
		$scope.classTabSelIndex = index;
	};
	
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
	    	//we need to loop through all classes to update the count
	    	// (a little bit inefficient)
	        for (i in $scope.classes){
	        	if ($scope.classes[i].Id == oldClassId){
	        		$scope.classes[i].Count -= 1;
	        	}else if($scope.classes[i].Id == $scope.data.class_id){
	        		$scope.classes[i].Count += 1;
	        	}
	        }
	      }).
	      error(function(data, status, headers, config) {
	        // called asynchronously if an error occurs
	        // or server returns response with an error status.
	      });
	};
	
	$http.get("/get_elements").then(function(response) {
        $scope.elements = response.data.segment_ids;
        $scope.classes = response.data.classes;
        $scope.setCurrentIndex(0);
    });

	$scope.setCurrentIndex = function(index){
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
        var SCOPEDATA =  $scope.data.data;
        
        var timeLabels = SCOPEDATA[2].labels;
	    $window.mseed2.data.labels = timeLabels;
	    $window.mseed2.data.datasets[0].data = SCOPEDATA[2].datasets[index].data;
	    $window.mseed2.data.datasets[0].label = SCOPEDATA[2].datasets[index].label;
	    
	    timeLabels = SCOPEDATA[3].labels;
	    $window.mseed3.data.labels = timeLabels;
	    $window.mseed3.data.datasets[0].data = SCOPEDATA[3].datasets[index].data;
	    $window.mseed3.data.datasets[0].label = SCOPEDATA[3].datasets[index].label;
        
	    timeLabels = SCOPEDATA[0].labels; // declare at the end cause we might need it later and
	    // does not have to be the timeLabel of the "other-channels" plots
	    
	    $window.mseed1.data.labels = timeLabels;
	    $window.mseed1.data.datasets[0].data = SCOPEDATA[0].datasets[index].data;
	    $window.mseed1.data.datasets[0].label = SCOPEDATA[0].datasets[index].label;
        
	    plots2refresh = [$window.mseed1, $window.mseed2, $window.mseed3];
	    
	    if (updateFilterOnly !== true){
		    var SCOPEMETADATA = $scope.data.metadata;
		    
		    var arrivalTime = undefined;
		    var snrWindowInSec = undefined;
		    var cumt5 = undefined;
		    var cumt95 = undefined;
		    
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
		    	}
		    }

		    $window.mseed1.config.options.arrivalTime = arrivalTime;
		    $window.mseed1.config.options.snrWindowInSec = snrWindowInSec;
		    $window.mseed_cum.config.options.cumT5 = cumt5;
		    $window.mseed_cum.config.options.cumT95 = cumt95;
		    
		    $window.mseed_cum.data.labels = timeLabels;
		    $window.mseed_cum.data.datasets[0].data = SCOPEDATA[0].datasets[2].data;
		    $window.mseed_cum.data.datasets[0].label = SCOPEDATA[0].datasets[2].label;
		    
		    $window.mseed_env.data.labels = timeLabels;
		    $window.mseed_env.data.datasets[0].data = SCOPEDATA[0].datasets[3].data;
		    $window.mseed_env.data.datasets[0].label = SCOPEDATA[0].datasets[3].label;
		    
		    var freqLabels = SCOPEDATA[1].labels;
		    $window.mseed_snr.data.labels = freqLabels;
		    $window.mseed_snr.data.datasets[0].data = SCOPEDATA[1].datasets[0].data;
		    $window.mseed_snr.data.datasets[0].label = SCOPEDATA[1].datasets[0].label;
		    $window.mseed_snr.data.datasets[1].data = SCOPEDATA[1].datasets[1].data;
		    $window.mseed_snr.data.datasets[1].label = SCOPEDATA[1].datasets[1].label;
		    
		    plots2refresh.push($window.mseed_env);
		    plots2refresh.push($window.mseed_cum);
		    plots2refresh.push($window.mseed_snr);
		    
	    }
	    
	    for (i in plots2refresh){
	    	plots2refresh[i].update();
	    }
	};

}]);