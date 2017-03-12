var myApp = angular.module('myApp', []);
 
myApp.controller('myController', ['$scope', '$http', '$window', '$timeout', function($scope, $http, $window, $timeout) {
	$scope.segIds = [];  // segment indices
	$scope.segIdx = -1;  // current segment index
	$scope.metadata = []; // array of 2-element arrays [(key, type), ... ] (all elements as string)
	
	$scope.segData = {}; // the segment data (classes, plot data, metadata etc...)
	$scope.plots = $window.plots;
	$scope.showFiltered = true;
	$scope.classes = [];

	$scope.loading=true;
	$scope.globalZoom = true;
	
	$scope.orderBy = [["event.time", "asc"], ["event_distance_deg", "asc"]];
	
	$scope.bottomPlotId = 3; // 3 components plot, 4: cumulative plot, 5 on: custom one (if any)
	$scope.bottomPlotChanged = function(){
		var bpIdx = parseInt($scope.bottomPlotId);
		$scope.plots.forEach(function(elm, index){
			//set visible if: is main plot or spectra (index < 2)
			// visibleIndex is index (normal case)
			// visibleIndex refers to the components plots (2 and 3)
			var visible = ((index < 2) || (index == bpIdx) || (bpIdx == 3 && index == 2));
			elm.visible = visible;
		});
		$scope.refreshView(bpIdx == 3 ? [2,3] : [bpIdx]);
	};
	
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

	$scope.plots.forEach(function(element, i){
		var div = element.div;
		div.on('plotly_relayout', zoomListenerFunc(i));
		div.on('plotly_doubleclick', autoZoomListenerFunc(i));
	});
	
	
	$scope.selection = {
		withDataOnly: true,
		data: {},
		changed: false,
		showForm: false,
		errorMsg: "",
		fireChange: function(newVal, oldVal){
			function differ(val1, val2){
				if (typeof val1 !== typeof val2){
					return true;
				}
				var keys1 = Object.keys(val1);
				var keys2 = Object.keys(val2);
				if (keys1.length !== keys2.length){
					return true;
				}
				if (!keys1.length){
					return typeof val1 === 'object' ? false : val1 !== val2;
				}
				return keys1.some(function(element, index, array){
					return differ(element, keys2[index]);
				});
			}
			this.changed = differ(newVal, oldVal);
		}
	};
    // watch for further changes from now on:
    var watcher = function(newVal, oldVal){
    	$scope.selection.fireChange.call($scope.selection, newVal, oldVal);
    };
    $scope.$watch('selection.data', watcher, true);  // note the true: compares by value, not by ref
	$scope.$watch('selection.withDataOnly', watcher);

	
	$scope.selectSegments = function(){
		var data = {'selection': $scope.selection.data, 'order-by': $scope.orderBy,
					'with-data': $scope.selection.withDataOnly};
		$scope.selection.errorMsg = "";
		$scope.loading = true;
		$http.post("/select_segments", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.loading = false;
			segIds = response.data;
	        if (segIds.length < 1){
	        	$scope.selection.errorMsg = "No segment found with given criteria";
	        	return;
	        }
	        $scope.changed = false;
	        $scope.setSegments(segIds);
	    });
	};

	$scope.init = function(){  // update classes and elements
		var data = {'order-by': $scope.orderBy, 'with-data': $scope.selection.withDataOnly};
		$http.post("/init", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.classes = response.data.classes;
	        $scope.metadata = response.data.metadata;
	        //set selection.data according to metadata:
	        var selectionData = {};
	        $scope.metadata.forEach(function(elm){
	        	selectionData[elm[0]] = undefined;
	        });
	        $scope.selection.data = selectionData;
	        //config plots when dom is rendered (see timeout 0 on google for details):
//	        $timeout(function () { 
//	        	$scope.configPlots(response.data.segment_ids); // this will be called once the dom has rendered
//	          }, 0, false);
	        $scope.setSegments(response.data.segment_ids);
	    });
	};
	
	$scope.setSegments = function(segmentIds){
		$scope.segIds = segmentIds;
		// clear zoom will be handled in setSegments (maybe we have one, it doesn't have to apply to new plots)
        // $scope.getAndClearZooms(); // we simply don't get the zooms, we don't care
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
			return;
		}

		var param = {segId: $scope.segIds[index], filteredRemResp: $scope.showFiltered, zooms:null,
				plotIndices: [], metadataKeys: $scope.metadata.map(function(elm){return elm[0];})};
		$scope.loading = true;
		$http.post("/get_segment_data", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			var metadata = response.data.metadata;
			// do not show classes in metadata panel but in a dedicated slot. Thus
			// remove 'classes' and set it to segData.classIds
			$scope.segData.classIds = metadata['classes.id'] || [];
			delete metadata['classes.id'];  
			// also, set metadata to be a dict of dicts instead of an array
			// (i.e., sorted by 'category': segment, channel etcetera):
			var segMetadata = {};
			for (key in metadata){
				var elms = key.split(".");
				if (elms.length==1){
					elms = ["segment", elms[0]];
				}
				if (!(elms[0] in segMetadata)){
					segMetadata[elms[0]] ={};
				}
				segMetadata[elms[0]][elms[1]] = metadata[key];
			}
			$scope.segData.metadata = segMetadata;
			// update plots:
	        $scope.refreshView();
	    });
	};
	
	$scope.refreshView = function(indices){
		var index = $scope.segIdx;
		if (index < 0){
			return;
		}
		if (indices === undefined){
			var indices =[];
			$scope.plots.forEach(function(elm, index){
				if (elm.visible){
					indices.push(index);
				}
			});
		}
		var zooms = $scope.getAndClearZooms();
		var param = {segId: $scope.segIds[index], filteredRemResp: $scope.showFiltered, zooms:zooms,
				plotIndices: indices, metadataKeys: null};
		$scope.loading = true;
		// initialize if undefined (as it is the first time we download plots)
		if (!$scope.segData.plotData){
			$scope.segData.plotData = new Array($scope.plots.length);
		}
		$http.post("/get_segment_data", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			response.data.plotData.forEach(function(elm, idx){
				$scope.segData.plotData[indices[idx]] = elm;
			});
			// update plots:
	        $scope.redrawPlots(indices);
	    });
	}

	$scope.getAndClearZooms = function(){
		return $scope.plots.map(function(elm){
			zoom = elm.zoom; //2 element array
			// convert to timestamps if the type of plot is timeseries.
			// the server has already converted times to account for local time,
			// so we just need to use Date.parse:
			if (elm.type == 'time-series'){
				zoom = zoom.map(function(z){
					return Date.parse(z);
				});
			}
			//set zoom to zero, otherwise these value are persistent and affect further plots:
			elm.zoom = [null, null];
			return zoom;
		});
	};
	
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
					elmData.line.color = '#dddddd';
				}else if (i == 1){
					// spectra: red the first (noise) green the second
					// https://github.com/plotly/plotly.js/blob/master/src/components/color/attributes.js
					elmData.line.color = j == 0 ? '#d62728' : '#2ca02c';
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
//					if (i == 0){  //FIXME: remove!!!
//						console.log(elements[0][0]);
//						console.log(new Date(elements[0][0]));
//						console.log(div.layout.xaxis.range)
//						console.log("");
//					}
					div.layout.xaxis.title = warnings;
					div.layout.margin['b'] = warnings ? 80 : 40;
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
	
	$scope.extend = function(obj1, obj2){
		for (var key in obj2){
			if (!(key in obj1) || (typeof obj1[key] != typeof obj2[key])){
				obj1[key] = obj2[key];
			}else if (typeof obj2[key] === 'object'){
				$scope.extend(obj1[key], obj2[key])
			}else{
				obj1[key] = obj2[key];
			}
		}
		return obj1;
	}
	
	$scope.toggleFilter = function(){
		//$scope.showFiltered = !$scope.showFiltered; THIS IS HANDLED BY ANGULAR!
		$scope.refreshView();
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
	
	// init our app:
	$scope.init();

}]);