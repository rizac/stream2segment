var myApp = angular.module('myApp',[]);
 
myApp.controller('myController', ['$scope', '$http', '$window', '$timeout', function($scope, $http, $window, $timeout) {
	$scope.elements = [];
	$scope.currentIndex = -1;
	$scope.data = {}; // the data: it is an array of arrays. Each element has:
	// data title, data x0, data xdelta, data y values
	$scope.plots = new Array(5 + $window._NUM_CUSTOM_PLOTS).fill(undefined); // will be set in configPlots called by refreshElements
	$scope.showFiltered = true;
	$scope.isEditingIndex = false;
	$scope.classes = [];
	$scope.currentSegmentClassIds = [];
	$scope.currentSegmentText = "";
	$scope.loading=true;
	$scope.globalZoom = true;

	$scope.init = function(){  // update classes and elements
		var data = {}; //maybe in the future pass some data
		$http.post("/get_classes", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.classes = response.data.classes;
	        $scope.refreshElements();
	    });
	};
	
	$scope.refreshElements = function(){  // update elements
		var data = {}; //maybe in the future pass some data
		$http.post("/get_elements", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.elements = response.data.segment_ids;
	        $timeout(function () { 
	        	$scope.configPlots(); // this will be called once the dom has rendered
	          }, 0, false);
	    });
	};
	
	$scope.configPlots = function(){
		var plotly = $window.Plotly;
		
		var tSeriesLayout = { //https://plot.ly/javascript/axes/
				margin:{'l':50, 't':30, 'b':40, 'r':15},
				xaxis: {
					autorange: true,
					tickangle: 0,
					type: 'date',
					titlefont: {
					      color: '#df2200'
					}
				},
				yaxis: {
					autorange: true,
					//fixedrange: true
				},
				annotations: [{
				    xref: 'paper',
				    yref: 'paper',
				    x: 0,
				    xanchor: 'left',
				    y: 1,
				    yanchor: 'bottom',
				    text: '',
				    showarrow: false,
				    //bordercolor: '#c7c7c7',
				    //borderwidth: 2,
				    borderpad: 5,
				    bgcolor: 'rgba(31, 119, 180, .1)',  // = '#1f77b4',
				    // opacity: 0.1,
				    font: {
				        // family: 'Courier New, monospace',
				        // size: 16,
				        color: '#000000'
				      },
				  }]
			};
		
		// create a deep copy (assuming we have simple dict-like objects
 		// see http://stackoverflow.com/questions/728360/how-do-i-correctly-clone-a-javascript-object
 		var fftLayout = JSON.parse(JSON.stringify(tSeriesLayout));
		fftLayout.xaxis.type = 'log';
		fftLayout.yaxis.type = 'log';
		
		var divs = [];
		$scope.plots = $scope.plots.map(function(element, i){
			var plotId = 'plot-' + i;
			var div = $window.document.getElementById(plotId);
			divs.push(div);
			var idf = 9;
			var layout = i == 3 ? fftLayout : tSeriesLayout;
			//COPY OBJECTS!!!
			plotly.newPlot(div, [{x0:0, dx:1, y:[0], type:'scatter', 'opacity': 0}], JSON.parse(JSON.stringify(layout)));
			return {
				'div': div,
				'zoom': [null, null],
				'type': div.getAttribute('plot-type')
			};
		});

		//setup listener (we could have done it in the loop above but is more readable like this)
		// create function for notifying zoom. On all plots except
		// other components
		var zoomListenerFunc = function(plotIndex){
			return function(eventdata){
				// check that this function is called from zoom
				// (it is called from any relayout command also)
				var isZoom = 'xaxis.range[0]' in eventdata && 'xaxis.range[1]' in eventdata;
				if(!isZoom){
					return;
				}
				var zoom = [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']];
				var plot = $scope.plots[plotIndex];
				var plotType = plot.type;
				if ($scope.globalZoom){
					$scope.plots.forEach(function(plot){
						if (plot.type === plotType){
							plot.zoom = [zoom[0], zoom[1]];  // copy (for safety)
						}
					});
				}else{
					plot.zoom = [zoom[0], zoom[1]];  // copy (for safety)
				}
				$scope.refreshView();
		    }
		};

		var autoZoomListenerFunc = function(plotIndex){
			return function(eventdata){
				$scope.refreshView(); // zooms are reset after use, so this redraw normal bounds
		    }
		};
		
		divs.forEach(function(div, i){
			div.on('plotly_relayout', zoomListenerFunc(i));
			div.on('plotly_doubleclick', autoZoomListenerFunc(i));
		});
		
		
		// update data (if currentIndex undefined, then set it to zero if we have elements
		// and refresh plots)
		if ($scope.currentIndex < 0){
			if ($scope.elements.length){
				$scope.currentIndex = 0;
			}
		}
		$scope.refreshView();
	}
	
	$scope.setNextSegment = function(){
		var currentIndex = ($scope.currentIndex + 1) % ($scope.elements.length);
		$scope.setSegment(currentIndex);
	};
	
	$scope.setPreviousSegment = function(){
		var currentIndex = $scope.currentIndex == 0 ? $scope.elements.length - 1 : $scope.currentIndex - 1;
        $scope.setSegment(currentIndex);
	};
	
	$scope.refreshView = function(){
		$scope.setSegment($scope.currentIndex);
	}

	$scope.setSegment = function(index){
		$scope.currentIndex = index;
		if (index < 0){
			return;
		}
		$scope.isEditingIndex = false;
		
		var zooms = $scope.plots.map(function(elm){
			zoom = elm.zoom; //2 element array
			//set zoom to zero, otherwise these value are persistent and affect further plots:
			elm.zoom = [null, null];
			return zoom;
		});
		var param = {segId: $scope.elements[index], filteredRemResp: $scope.showFiltered, zooms:zooms};
		$scope.loading = true;
		$http.post("/get_data", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.data = response.data;
	        $scope.currentSegmentClassIds = response.data.class_ids;
	        $scope.redrawPlots();
	    });
	};
	
	$scope.redrawPlots = function(){
		var scopeData = $scope.data;
		var plotly = $window.Plotly;
		for (var i=0; i< Math.min($scope.plots.length, scopeData.length); i++){
			var div = $scope.plots[i].div;
			var plotData = scopeData[i];
			var title = plotData[0];
			var elements = plotData[1];
			var warnings = plotData[2] || "";
			var xrange = plotData[3] || null;
			//http://stackoverflow.com/questions/40673490/how-to-get-plotly-js-default-colors-list
			var colors = Plotly.d3.scale.category20();
			var data = [];
			for (var j=0; j<elements.length; j++){
				var color = colors[j % colors.length];
				var line = elements[j];
				data.push({
					x0: line[0],
					dx: line[1],
					y: line[2],
					name: line[3],
					type: 'scatter',
		            opacity: 0,  // 0.95,
		            line: {
		            	  width: 1,
		            	  color: (i==1 || i==2) ? '#dddddd' : color
		            }
				})
			}
			if (div.layout){
				// hack for setting the title left (so that the tool-bar does not overlap
				// so easily). Comment this:
				// div.layout.title = title;
				// and set the first annotation (provided in configPlots)
				if (div.layout.annotations){
					div.layout.annotations[0].text = title;
				}
				if (div.layout.xaxis){
					div.layout.xaxis.autorange = true;
					if (xrange){
						div.layout.xaxis.autorange = false;
						div.layout.xaxis.range = xrange;
					}
					div.layout.xaxis.title = warnings;
					div.layout.margin['b'] = warnings ? 80 : 40;
				}
			}
			div.data = data;
			plotly.redraw(div);
			
			if (!elements){
				continue;
			}

			//use animation:
			plotly.animate(div, {
			    data: elements.map(function(obj,idx){return {opacity: 0.95};}), // [{opacity: 0.95}],
			    traces: elements.map(function(obj,idx){return idx;}),
			    layout: {}
			  }, {
			    transition: {
			      duration: 500,
			      easing: 'cubic-in-out'
			    }
			  })
			
		}
		$scope.loading=false;
	};
	
	
	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.refreshView();
	};
	
	//$scope.configPlots();
	
	
	//===============================================================================
	
	$scope.setCurrentIndexFromText = function(){
		var index = parseInt($scope.currentSegmentText);
		if (index >0 && index <= $scope.elements.length){
			$scope.setSegment(index-1);
		}
	}
	
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
	
	
	
	
	
	$scope.updatePlots_old = function(updateFilterOnly){ // update filter only is not actually used anymore
	    
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