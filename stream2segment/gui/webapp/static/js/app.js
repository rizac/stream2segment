var myApp = angular.module('myApp',[]);
 
myApp.controller('myController', ['$scope', '$http', '$window', '$timeout', function($scope, $http, $window, $timeout) {
	$scope.segIds = [];  // segment indices
	$scope.segIdx = -1;  // current segment index
	$scope.metadata = []; // array of 2-element arrays [(key, type), ... ] (all elements as string)
	
	$scope.segData = {}; // the segment data (classes, plot data, metadata etc...)
	$scope.plots = new Array(5 + $window._NUM_CUSTOM_PLOTS).fill(undefined); // will be set in configPlots called by init below
	$scope.showFiltered = true;
	$scope.classes = [];

	$scope.loading=true;
	$scope.globalZoom = true;
	
	$scope.orderBy = [["event.time", "asc"], ["event_distance_deg", "asc"]];
	
	$scope.selection = {
		data: {},
		empty: function(){ //returns true if the selection criteria accept all segment
			for (var key in this.data){
				if (this.data[key]){
					return false;
				}
			}
			return true;
		},
		active: false,  // returns/set if there is a filter on the currently displayed segments
		showForm: false,
		errorMsg: ""
	};
	
	
	$scope.selectSegments = function(){
		var selectionData = $scope.selection.data;
		$scope.selection.errorMsg = "";
		$scope.loading = true;
		$http.post("/select_segments", selectionData, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			var selectionEmpty = $scope.selection.empty();
			var selectionActive = $scope.selection.active;
			$scope.loading = false;
			if(selectionEmpty){
				if (!selectionActive){
					$scope.selection.errorMsg = "Please provide some criteria";
					return;
				}
			}
			segIds = response.data;
	        if (segIds.length < 1){
	        	$scope.selection.errorMsg = "No segment found with given criteria";
	        	return;
	        }
	        $scope.selection.active = !selectionEmpty;
	        $scope.segIds = segIds;
	        // clear zoom (maybe we have one, it doesn't have to apply to new plots)
	        $scope.getAndClearZooms(); // we simply don't get the zooms, we don't care
	        $scope.setSegment(0);
	    });
	};

	$scope.init = function(){  // update classes and elements
		var data = {order: $scope.orderBy}; //maybe in the future pass some data
		$http.post("/init", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
	        $scope.classes = response.data.classes;
	        $scope.segIds = response.data.segment_ids;
	        $scope.metadata = response.data.metadata;
	        //set selection.data according to metadata:
	        var selectionData = {};
	        $scope.metadata.forEach(function(elm){
	        	selectionData[elm[0]] = undefined;
	        });
	        $scope.selection.data = selectionData;
	        //config plots when dom is rendered (see timeout 0 on google for details):
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
		if ($scope.segIdx < 0){
			if ($scope.segIds.length){
				$scope.segIdx = 0;
			}
		}
		$scope.refreshView();
	}
	
	$scope.setNextSegment = function(){
		var currentIndex = ($scope.segIdx + 1) % ($scope.segIds.length);
		$scope.setSegment(currentIndex);
	};
	
	$scope.setPreviousSegment = function(){
		var currentIndex = $scope.segIdx == 0 ? $scope.segIds.length - 1 : $scope.segIdx - 1;
        $scope.setSegment(currentIndex);
	};
	
	$scope.refreshView = function(){
		$scope.setSegment($scope.segIdx);
	}

	$scope.setSegment = function(index){
		$scope.segIdx = index;
		if (index < 0){
			return;
		}
		$scope.isEditingIndex = false;
		
		var zooms = $scope.getAndClearZooms();
		var param = {segId: $scope.segIds[index], filteredRemResp: $scope.showFiltered, zooms:zooms,
				metadataKeys: $scope.metadata.map(function(elm){return elm[0];})};
		$scope.loading = true;
		$http.post("/get_segment_data", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			$scope.segData = response.data;
			// do not show classes in metadata panel but in a dedicated slot. Thus
			// remove 'classes' and set it to segData.classIds
			$scope.segData.classIds = $scope.segData.metadata.classes || [];
			delete $scope.segData.metadata.classes;  
			// also, set metadata to be a dict of dicts instead of an array
			// (i.e., sorted by 'category': segment, channel etcetera):
			var segMetadata = {};
			for (key in $scope.segData.metadata){
				var elms = key.split(".");
				if (elms.length==1){
					elms = ["segment", elms[0]];
				}
				if (!(elms[0] in segMetadata)){
					segMetadata[elms[0]] ={};
				}
				segMetadata[elms[0]][elms[1]] = $scope.segData.metadata[key];
			}
			$scope.segData.metadata = segMetadata;
			// update plots:
	        $scope.redrawPlots();
	    });
	};
	
//	$scope.getSegmentDisplayMetadata = function(segmentMetadata){
//		var ret = {};
//		if (segmentMetadata){
//			$scope.metadata.forEach(function(element){
//				var key = element[0];
//				var mainKey = "segment"
//				var secondaryKey = key;
//				var idx = key.indexOf(".")
//				if ( idx > -1){
//					mainKey = key.substring(0, idx);
//					secondaryKey = key.substring(idx+1, key.length);
//				}
//				var val = segmentMetadata[key] || "";
//				if (!(mainKey in ret)){
//					ret[mainKey] = {};
//				}
//				ret[mainKey][secondaryKey] = val;
//			});
//		}
//		return ret;
//	};
	
	
	$scope.getAndClearZooms = function(){
		return $scope.plots.map(function(elm){
			zoom = elm.zoom; //2 element array
			//set zoom to zero, otherwise these value are persistent and affect further plots:
			elm.zoom = [null, null];
			return zoom;
		});
	};
	
	$scope.redrawPlots = function(){
		var plotsData = $scope.segData.plotData;
		var plotly = $window.Plotly;
		for (var i=0; i< Math.min($scope.plots.length, plotsData.length); i++){
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
				data.push({
					x0: line[0],
					dx: line[1],
					y: line[2],
					name: line[3],
					type: 'scatter',
		            opacity: 0.95,  // set to zero and uncomment the "use animations" below if you wish,
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