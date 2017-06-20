var myApp = angular.module('myApp', []);
 
myApp.controller('myController', ['$scope', '$http', '$window', '$timeout', function($scope, $http, $window, $timeout) {
	$scope.segIds = [];  // segment indices
	$scope.segIdx = -1;  // current segment index
	$scope.metadata = []; // array of 3-element arrays [(key, type, expr), ... ] (all elements as string)
						  // example: [('has_data', 'bool', 'true'), ('id', 'int, ''), ('event.id', 'int', ''), ...]
	// selection "window" handling:
	$scope.selection = {
			showForm: false,
			errorMsg: ""
		};
	
	$scope.snWindows = $window.__SETTINGS.sn_windows;
	$scope.snWindows._changed = false;

	$scope.segData = {}; // the segment data (classes, plot data, metadata etc...)
	$scope.plots = $window.plots.map(function(div, index) {
		   return {'visible': index < 3, 'zoom': [null, null], 'div': div}
	});
	// this forwards the plotly 'on' method to all plots defined in $scope.plots: (see when defining zoom event listeners below)
	$scope.plots.on = function(key, callback){
		$scope.plots.forEach(function(elm, index, elms){
			elm.div.on(key, function(eventData){
				callback(elm, index, elms, eventData);
			});
		});
	};
	
	$scope.bottomPlotId = 2;
	$scope.showFiltered = true;
	$scope.showAllComponents = false;
	$scope.classes = [];

	$scope.loading=true;

	$scope.PLOTCONFIGS = {
			otherComponentsColor: '#dddddd',
			// spectra: red the first (noise) green the second
			// https://github.com/plotly/plotly.js/blob/master/src/components/color/attributes.js
			snColors: ['#2ca02c', '#d62728'], // signal, noise
			arrivalTimeLineColor: '#777777'
	};
	
	$scope.err = function(response){
		var msg = (response.data || 'Request failed');
        //$scope.status = response.status;
		return msg;
	}
	
	$scope.init = function(){
		// send the current settings as data. settings are written in the main page
		// by means of global js variables
		var data = {segment_select: $window.__SETTINGS['segment_select'] || null,
					segment_orderby: $window.__SETTINGS['segment_orderby'] || null,
					classes: true, metadata: true};
		// note on data dict above: the server expects also 'metadata' and 'classes' keys which we do provide otherwise
		// they are false by default
		$http.post("/get_segments", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.classes = response.data.classes;
	        $scope.metadata = response.data.metadata;
	        $scope.setSegments(response.data.segment_ids);
	    });
	};
	
	$scope.selectSegments = function(){
		// build the dict for the json request: Currently supported only selection, not order-by
		// (order-by is given in the config once)
		var data = {segment_select: {}};
		$scope.metadata.forEach(function(elm){
			if (elm[2]){
				data['segment_select'][elm[0]] = elm[2];
			}
		});
		$scope.selection.errorMsg = "";
		$scope.loading = true;
		$http.post("/get_segments", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.loading = false;
			segIds = response.data.segment_ids;
	        if (!segIds || (segIds.length < 1)){
	        	$scope.selection.errorMsg = "No segment found with given criteria";
	        	return;
	        }
	        $scope.setSegments(segIds);
	        $scope.selection.showForm = false;  // close window popup, if any
	    }, function(response) {  // error function, print message
	          $scope.selection.errorMsg = $scope.err(response);
	    });
	};
	
	$scope.setSegments = function(segmentIds){
		$scope.segIds = segmentIds;
		if ($scope.segIds.length){
			$scope.segIdx = 0;
		}
        $scope.setSegment($scope.segIdx);
	};
	
	$scope.setNextSegment = function(){
		var currentIndex = ($scope.segIdx + 1) % ($scope.segIds.length);
		$scope.setSegment(currentIndex);
	};
	
	$scope.setPreviousSegment = function(){
		var currentIndex = $scope.segIdx == 0 ? $scope.segIds.length - 1 : $scope.segIdx - 1;
        $scope.setSegment(currentIndex);
	};
	
	$scope.setSegment = function(index){
		$scope.segIdx = index;
		if (index < 0){
			return; // FIXME: better handling!!!!
		}
		
		var zooms = [];
		
		var param = {seg_id: $scope.segIds[index], metadata: true, classes:true, warnings: true};
		$scope.loading = true;
		$http.post("/get_segment", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			var metadata = response.data.metadata || [];
			$scope.segData.classIds = []
			// create segMetadata and set it to $scope.segData.metadata
			// the variable has to be parsed in order to order keys and values according to
			// our choice:
			var segMetadata = {'segment': {}, 'event': {}, 'channel': {}, 'station': {}};
			// note that metadata is an array of 2-element arrays: [[key, value], ...]
			metadata.forEach(function(elm, index){
				var key = elm[0];
				var val = elm[1];
				if (key == 'classes.id'){
					// do not show classes in metadata panel but in a dedicated slot. Thus
					// remove 'classes' and set it to segData.classIds
					$scope.segData.classIds = val;
				}else{
					var elms = key.split(".");
					if (elms.length == 1){
						elms = ["segment", elms[0]];
					}
					if (!(elms[0] in segMetadata)){
						segMetadata[elms[0]] = {};
					}
					segMetadata[elms[0]][elms[1]] = val;
				}
			});
			$scope.segData.metadata = segMetadata;
			$scope.segData.warnings = response.data.warnings;  // FIXME: need to set it up!
			// update plots:
	        $scope.refreshView();
	    });
	};
	
	/** HANDLING ALL EVENTS TOGGLING A PLOT REQUEST **/
	// listen for plotly events. The plotly 'on' function is wrapped by our 'on' function
	// implemented on the plots array (see plotlyconfig.js) and forwards it to the plotly on
	// callback
	// The syntax is the same except that the callback does not take a single eventdata
	// argument, but has four arguments, the first three being the same as Array.map or Array.forEach
	// the fourth being the plotly event:

	$scope.plots.on('plotly_relayout', function(plot, index, plots, eventdata){
		// check that this function is called from zoom
		// (it is called from any relayout command also)
		var isZoom = 'xaxis.range[0]' in eventdata && 'xaxis.range[1]' in eventdata;
		if(!isZoom){
			return;
		}
		var zoom = [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']];
			plot.zoom = [zoom[0], zoom[1]];  // copy (for safety)
		$scope.refreshView([index]);
	});
	
	$scope.plots.on('plotly_doubleclick', function(plot, index, plots, eventdata){
		$scope.refreshView([index]); // zooms are reset after use, so this redraw normal bounds
	});
	
	// this function sets the currently visible bottom plots. Note that by default
	// plots in [0,1] are visible (normal trace + sn-spectra)
	$scope.bottomPlotChanged = function(){
		var bpIdx = parseInt($scope.bottomPlotId);
		$scope.plots.forEach(function(elm, index){
			//set visible if: is main plot or spectra (index < 2)
			// visibleIndex is index (normal case)
			// visibleIndex refers to the components plots (2 and 3)
			if (index > 1){
				elm.visible = (index == bpIdx);
			}
		});
		$scope.refreshView();
	};
	
	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.refreshView();
	};
	
	$scope.toggleAllComponentView = function(){
		//$scope.showAllComponents = !$scope.showAllComponents; THIS IS HANDLED BY ANGULAR!
		// update plots:
		$scope.refreshView([index]);
	}
	
	/** THIS FUNCTION GETS THE CURRENT PLOTS. Puts the data in $scope.segData.plotData **/
	$scope.refreshView = function(indices){
		var index = $scope.segIdx;
		if (index < 0){
			return;
		}
		if (indices === undefined){
			var indices = $scope.getVisiblePlotIndices();
		}
		var zooms = $scope.getAndClearZooms();
		var param = {seg_id: $scope.segIds[index], filtered: $scope.showFiltered, zooms:zooms,
				plot_indices: indices, all_components: $scope.showAllComponents};
		$scope.loading = true;
		// initialize if undefined (as it is the first time we download plots)
		if (!$scope.segData.plotData){
			$scope.segData.plotData = new Array($scope.plots.length);
			$scope.segData.snWindows = [];
		}
		$http.post("/get_segment", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			response.data.plots.forEach(function(elm, idx){
				$scope.segData.plotData[indices[idx]] = elm;
			});
			$scope.segData.snWindows = response.data['sn_windows'];  // might be empty array
			// update plots:
	        $scope.redrawPlots(indices);
	    });
	}
	
	/** functions for getting data for the plot query above **/
	// this function is used for getting the zooms and clearing them:
	$scope.getAndClearZooms = function(){
		return $scope.plots.map(function(elm){
			zoom = [elm.zoom[0], elm.zoom[1]];
			//set zoom to zero, otherwise these value are persistent and affect further plots:
			elm.zoom = [null, null];
			return zoom;
		});
	};
	
	// this function returns the currently visible plot indices
	$scope.getVisiblePlotIndices = function(){
		var plotIndices = [];
		// set which indices should we show:
		$scope.plots.forEach(function(element, index){
			if (element.visible){
				plotIndices.push(index);
				
			}
		});
		return plotIndices;
	};
	
	/** redraws the plots currently downloaded (data is in $scope.segData.plotData) **/
	$scope.redrawPlots = function(indices){
		var plotsData = $scope.segData.plotData;
		var plotly = $window.Plotly;
		for (var i_=0; i_< indices.length; i_++){
			var i = indices[i_];
			var div = $scope.plots[i].div;
			var plotData = plotsData[i];
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
				var elmData = {
					x0: line[0],
					dx: line[1],
					y: line[2],
					name: line[3],
					type: 'scatter',
		            opacity: 0.95,  // set to zero and uncomment the "use animations" below if you wish,
		            line: {
		            	  width: 1
		            }
				};
				// customize colors. Maybe in the future moved to some config (but there should be a way to customize single
				//elemtns of the plot, not difficult but quite long to implement), for the moment hard coded:
				if (i ==2 || i == 3){ // components in gray
					elmData.line.color = $scope.PLOTCONFIGS.otherComponentsColor;
				}else if (i == 1){
					elmData.line.color = $scope.PLOTCONFIGS.snColors[j];
				}
				// push data:
				data.push(elmData);
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
					div.layout.xaxis.title = warnings;
				}
				if (i==0){ // spectra windows
					// https://plot.ly/javascript/shapes/#vertical-and-horizontal-lines-positioned-relative-to-the-axes
					div.layout.shapes = $scope.segData.snWindows.map(function(elm, idx){
						return {type: 'rect',
							    xref: 'x', // x-reference is assigned to the x-values
							    yref: 'paper', // y-reference is assigned to the plot paper [0,1]
							    x0: elm[0],
							    y0: 0,
							    x1: elm[1],
							    y1: 1,
							    fillcolor: $scope.PLOTCONFIGS.snColors[idx],
							    opacity: 0.1,
							    line: {
							        width: 0
							    }
							};
					});
					// append arrival time:
					if (div.layout.shapes && div.layout.shapes.length){ // test for non-empty array (is there a better way?)
						// store arrival time as timestamp. The relative value in
						// segData.metadata.segment.arrival_time cannot be used,
						// see section "Differences in assumed time zone" in
						// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/parse
						div.layout.shapes.push({type: 'line',
						    xref: 'x', // x-reference is assigned to the x-values
						    yref: 'paper', // y-reference is assigned to the plot paper [0,1]
						    x0: div.layout.shapes[1].x1,
						    x1: div.layout.shapes[1].x1,
						    y0: 0,
						    y1: 1,
						    opacity: 1,
						    line: {
						        width: 1,
						        dash: 'dot',
						        color: $scope.PLOTCONFIGS.arrivalTimeLineColor
						    }
						});
					}
					// the noise / signal windows (rectangles in the background) might overflow
					// this is visually misleading, so in case restore the original bounds.
					// get the original bounds: 
					var x00 = data[0].x0;
					var x01 = data[0].x0 + data[0].dx * (data[0].y.length-1);
					// set bounds manually in case of overflow:
					if(div.layout.shapes.some(function (elm) {return elm.x0 < x00 || elm.x1 > x01})){
						div.layout.xaxis.range= [x00, x01];
						div.layout.xaxis.autorange = false;
					}else{
						div.layout.xaxis.autorange = true;
					}
				}
				
			}
			div.data = data;
			plotly.redraw(div);
			
			//use animation: (commented: it takes too much)
//			if (!elements){
//				continue;
//			}			
//			plotly.animate(div, {
//			    data: elements.map(function(obj,idx){return {opacity: 0.95};}), // [{opacity: 0.95}],
//			    traces: elements.map(function(obj,idx){return idx;}),
//			    layout: {}
//			  }, {
//			    transition: {
//			      duration: 500,
//			      easing: 'cubic-in-out'
//			    }
//			  })
			
		}
		$scope.loading=false;
	};
	
	$scope.toggleClassLabelForCurrentSegment = function(classId){
		
		var param = {class_id: classId, segment_id: $scope.segIds[$scope.segIdx]};
	    $http.post("/toggle_class_id", param, {headers: {'Content-Type': 'application/json'}}).
	    success(function(data, status, headers, config) {
	        $scope.segData.classIds = data.segment_class_ids;
	        data.classes.forEach(function(elm, index){
	        	$scope.classes[index]['count'] = elm['count'];  //update count
	        });
	      }).
	      error(function(data, status, headers, config) {
	        // called asynchronously if an error occurs
	        // or server returns response with an error status.
	      });
	};
	
	//visibility of some panels
	$scope.divPanels = {};
	$scope.isDivVisible = function(key){
		if(!(key in $scope.divPanels)){
			$scope.divPanels[key] = true;
		}
		return $scope.divPanels[key];
	}
	$scope.toggleDivVisibility = function(key){
		var status = !$scope.isDivVisible(key); //adds the key if not present
		$scope.divPanels[key] = status;
		return status
	}
	
	$scope.configSpectra = function(){
		var param = $scope.spectraSettings;
		$http.post("/config_spectra", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.spectraSettings._changed = false;
			// update plots:
	        $scope.refreshView([0, 1]); // refresh current segment and spectra only
	    });
	};
	// init our app:
	$scope.init();
	
	// old stuff maybe delete in the future:
	
	// we would like to have the click event setting the arrival time shift, but it is buggy
	// and it works only if we click on a data point (not on the plot)
//	$scope.plots.on('plotly_click', function(plot, index, plots, eventdata){
//		if (index != 0 || $scope.segData._arrivalTimestamp === undefined){
//			return;
//		}
//		var aTime = $scope.segData._arrivalTimestamp;
//		var newATime = Date.parse(eventdata.points[0].x);
//		$scope.spectraSettings.arrivalTimeDelay = parseFloat(newATime - aTime)/1000;
//		$scope.spectraSettings._changed = true;
//		//$scope.refreshView(); // zooms are reset after use, so this redraw normal bounds
//	});
	
	// settings injected in the the main page rendering:
	// $scope.settings = $window.__SETTINGS;
	


//	$scope.bottomPlotId = 3; // 3 components plot, 4: cumulative plot, 5 on: custom one (if any)
//	$scope.bottomPlotChanged = function(){
//		var bpIdx = parseInt($scope.bottomPlotId);
//		$scope.plots.forEach(function(elm, index){
//			//set visible if: is main plot or spectra (index < 2)
//			// visibleIndex is index (normal case)
//			// visibleIndex refers to the components plots (2 and 3)
//			var visible = ((index < 2) || (index == bpIdx) || (bpIdx == 3 && index == 2));
//			elm.visible = visible;
//		});
//		$scope.refreshView(bpIdx == 3 ? [2,3] : [bpIdx]);
//	};

}]);