var myApp = angular.module('myApp', []);
 
myApp.controller('myController', ['$scope', '$http', '$window', '$timeout', function($scope, $http, $window, $timeout) {
	$scope.selLabels = [];
	$scope.loading=true;
	$scope.data = [];
	$scope.numSegments = 0;

	$scope.init = function(){
		
		// send the current settings as data. settings are written in the main page
		// by means of global js variables
		var data = {settings: $window.__SETTINGS, limit:null, offset:null};
		
		$http.post("/get_data", data, {headers: {'Content-Type': 'application/json'}}).then(
			function(response) {
				var data = response.data.length ? response.data : [];
				$scope.data = data;	
        		$scope.updateMap();
        		// reset counts to zero:
        		$scope.selLabels.forEach(function(elm, idx){
        			elm[2] = 0;
        		});
        		if(!data.length){
        			return;
        		}
        		$scope.numSegments = 0;
        		var startIdx = data[0].length - $scope.selLabels.length;
        		data.forEach(function(element, index){
        			$scope.numSegments += element[4];
        			element.forEach(function(value, idx){
        				if(idx >= startIdx){
        					$scope.selLabels[idx-startIdx][2] += value;
        				}
        			});
        		});
	        	$scope.loading=false;
			},function errorCallback(response) {
				var fgh = 9;
	        	$scope.loading=false;
			}
		);
	};
	
	$scope.updateMap = function(){
		if(window.L){
			window.updateMap($scope.data, $scope.selLabels);
		}
	};
	
	$scope.setPopupContent = function(stationId, circle){
		circle.setPopupContent("Loading info...");
		$http.post("/get_station_data", {'station_id': stationId, 'labels': $scope.selLabels},
				   {headers: {'Content-Type': 'application/json'}}).then(
			function(response) {
				circle.setPopupContent($scope.convert2html(response.data, $scope.selLabels));
			},function errorCallback(response) {
				circle.setPopupContent("ERROR: " + response.data);
			}
		);
	};
	
	$scope.convert2html = function(stationArray, selLabels){
		var html = "<table class='station-info'>\n\t<tr><th>segment.id</th><th>seed_id</th><th>event.id</th><th>start_time</th><th>end_time</th>";
		for (var i=0; i< selLabels.length; i++){
			if (selLabels[i][1]){
				html+="<th>" + selLabels[i][0] + "</th>";
			}
		}
		html += "</tr>";
		for( var i=0; i< stationArray.length; i++){
			html +=  "\n\t<tr><td>" + stationArray[i].join("</td><td>") + "</td></tr>";
		}
		html += "\n</table>";
		return html;
	};
	
	//init labels (and init our app on done)
	$http.post("/get_selectable_labels", {}, {headers: {'Content-Type': 'application/json'}}).then(
		function(response) {
			$scope.selLabels = response.data;
			$scope.init();
		},function errorCallback(response) {
			var fgh = 9;
		}
	);
	

}]);