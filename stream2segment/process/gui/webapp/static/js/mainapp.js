function myApp(settings, bottomPlots, rightPlots){
	this.segId = null;  // segment identifier
	this.segIdx = -1;  // current segment index (position)
	this.segmentsCount = 0;
	this.metadata = []; // array of 2-element arrays [key, type] (all elements as string):
						  // example: [('has_data', 'bool'), ('id', 'int'), ...]
	this.classes = []; // Array of dicts. Each dict represents a class and is:
						 // {segments: <int>, id: <int>, label:<string>}

	// selection "window" handling:
	this.selection = {
		showForm: false,
		selExpr: {}  // will be populated when clicking the Select button
	};
	this.closeSelectionForm = function(){this.setWarning('');this.selection.showForm=false;}

	// config "window" handling:
	this.config = {
		showForm: false
	};
	this.closeConfigForm = function(){this.setWarning('');this.config.showForm=false;}
	
	// init the this.plots data:
	this.plots = new Array(bottomPlots.length + rightPlots.length + 1);

	bottomPlots.forEach(function(element, idx){
		var data = {
			visible: idx == 0,
			zoom: [null, null],
			div: document.getElementById('plot-'+element.index),
			xaxis: element.xaxis,
			yaxis: element.yaxis,
			position: element.position
		};
		this.plots[element.index] = data;
	});

	rightPlots.forEach(function(element, idx){
		var data = {
			visible: idx == 0,
			zoom: [null, null],
			div: document.getElementById('plot-'+element.index),
			xaxis: element.xaxis,
			yaxis: element.yaxis,
			position: element.position
		};
		this.plots[element.index] = data;
	});

	this.plots[0] = {
		visible: true,
		zoom: [null, null],
		div: document.getElementById('plot-0'),
		xaxis:{},
		yaxis:{},
		position:''  // position is not currently set for main plot
	};

	// the segment data (classes, plot data, metadata etc...):
	this.segData = {
		classIds: {}, // dict of class ids mapped to a boolean value (selected or not)
		metadata: {}, // dict of keys mapped to dicts. Keys are 'station', 'segment' etc.
					  // and values are dicts of attributes (string) mapped to their segmetn values
		plotData: new Array(this.plots.length), //array of plot data (each element represents a plot)
		snWindows: [] // array of two elements calculated from the server for the current segment:
					  // [signal_window, noise_window] each
					  // window is in turn a 2 element array of integers representing timestamps: [t1, t2]
	};
	
	this.hasPreprocessFunc = $window.__SETTINGS.hasPreprocessFunc;
	this.showPreProcessed = this.hasPreprocessFunc;
	this.showAllComponents = false;

	this.setLoading = function(msg){
		// a non empty `msg` shows up the progress bar and `msg`
		// an empty `msg` hides both. Set this.warnMsg = string and
		// this.loading=true|false to control message and progress bar separately
		// see also 'setWarning' below
		this.loading = !!msg;
		this.warnMsg = msg;
	};

	this.setWarning = function(msg){
		this.setLoading('');
		this.warnMsg = msg;
	};

	this.setLoading("Initializing ...");
	this.init = function(){
		var data = {classes: true, metadata: true};
		// note on data dict above: the server expects also 'metadata' and 'classes' keys which we do provide otherwise
		// they are false by default
		$http.post("/init", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			this.classes = response.data.classes;
			this.metadata = response.data.metadata;
			this.selectSegments();
		});
	};
	
	this.selectSegments = function(selExprObject){
		// sets selExprObject as 'segments_selection' in the config, and refreshes the
		// current view with the first of such selected segments.
		// If the argument is undefined, the current selection on the server config
		// is used
		if (typeof selExprObject === 'object'){
			var data = {segments_selection: selExprObject};
			var selectionEmpty = Object.keys(selExprObject).length == 0;
		}else{
			var data = {segments_selection: null};
			var selectionEmpty = false;
		}
		this.setLoading("Selecting segments (please wait, it might take a while for large databases)");
		$http.post("/set_selection", data, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			this.setLoading("");
			var segmentsCount = response.data.num_segments;
			var errMsg = '';
			if (response.data.error_msg){
				errMsg = response.data.error_msg + ". Check the segment selection (if the problem persists, contact the software maintainers) ";
			}
			if (!errMsg && segmentsCount <= 0){
				if (selectionEmpty){
					errMsg = "No segment found, empty database";
				}else{
					errMsg = "No segment found with the current segment selection";
				}
			}
			if (errMsg){
				this.setWarning(errMsg);
				return;
			}
			this.segmentsCount = segmentsCount;
			this.setSegment(0);
			this.selection.showForm = false;  // close window popup, if any
		});
	};
	
	this.setNextSegment = function(){
		var currentIndex = (this.segIdx + 1) % (this.segmentsCount);
		this.setSegment(currentIndex);
	};
	
	this.setPreviousSegment = function(){
		var currentIndex = this.segIdx == 0 ? this.segmentsCount - 1 : this.segIdx - 1;
		this.setSegment(currentIndex);
	};
	
	this.setSegment = function(index){
		this.segIdx = index;
		this.refreshView(undefined, true);
	};
	
	this.showSelectForm = function(){
		if (this.config.showForm){
			return;
		}
		this.setLoading("Fetching selection");
		$http.post("/get_selection", {}, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			this.setLoading("");
			var errMsg = response.data.error_msg || '';
			if (errMsg){
				this.setWarning(errMsg);
				return false;
			}
			this.selection.selExpr = {};
			var segSelect = response.data.data;
			for (var key of Object.keys(segSelect || {})){
				this.selection.selExpr[key] = segSelect[key];
			}
			this.selection.showForm = true;
		});
	}
	
	this.showConfigForm = function(){
		if (this.config.showForm){
			return;
		}
		this.setLoading("Fetching config");
		$http.post("/get_config", {asstr: true}, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			this.setLoading("");
			var errMsg = response.data.error_msg || '';
			if (errMsg){
				this.setWarning(errMsg);
				return false;
			}
			$window.configEditor.setValue(response.data.data);
		 	$window.configEditor.clearSelection();
		 	this.config.showForm = true;
		});
	}

	this.updateConfig = function(){
		$http.post("/validate_config_str", {data: $window.configEditor.getValue()}, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			this.setLoading("");
			var errMsg = response.data.error_msg || '';
			if (errMsg){
				this.setWarning(errMsg);
				return false;
			}
		 	this.config.showForm = false;
		 	this.refreshView(undefined, false,  response.data.data);
		});
	};

	this.setPlotVisible = function(index){
		var pos = this.plots[index].position;
		this.plots.forEach(function(elm, idx){
			if (elm.position == pos){
				elm.visible = false;
			}
		});
		this.plots[index].visible = true;
		this.refreshView([index]);
	};
	
	this.togglePreProcess = function(){
		this.refreshView();
	};
	
	this.toggleAllComponentView = function(){
		this.refreshView([0]);
	}
	
	this.refreshView = function(indices, refreshMetadata, config){
		/**
		 * Main function that updates the plots and optionally updates the segment metadata
		 * config is a dict of the new config
		 */
		var index = this.segIdx;
		if (index < 0){
			return;
		}
		if (indices === undefined){
			var indices = this.getVisiblePlotIndices();
		}
		var zooms = this.getAndClearZooms();
		var param = {
			seg_index: this.segIdx,
			seg_count: this.segmentsCount,
			pre_processed: this.showPreProcessed,
			zooms:zooms,
			plot_indices: indices,
			all_components: this.showAllComponents
		};

		param.config = config || {};
		if(refreshMetadata){
			param.metadata = true;
			param.classes = true
		}
		this.setLoading("Fetching and calculating segment plots (it might take a while) ...");
		$http.post("/get_segment", param, {headers: {'Content-Type': 'application/json'}}).then(function(response) {
			this.segId = response.data.seg_id;
			response.data.plots.forEach(function(elm, idx){
				this.segData.plotData[indices[idx]] = elm;
			});
			this.segData.snWindows = response.data.sn_windows || [];  // to be safe
			// update metadata if needed:
			if (refreshMetadata){
				this._refreshMetadata(response);
			}
			// update plots:
			this.setLoading("");
			this.redrawPlots(indices);
		});
	}
	
	this._refreshMetadata = function(response){
		var metadata = response.data.metadata || [];
		// create segMetadata and set it to this.segData.metadata
		// the variable has to be parsed in order to order keys and values according to
		// our choice:
		var segMetadata = {
			segment: {},
			event: {},
			channel: {},
			station: {}
		};
		// note that metadata is an array of 2-element arrays: [[key, value], ...]
		metadata.forEach(function(elm, index){
			var key = elm[0];
			var val = elm[1];
			var elms = key.split(".");
			if (elms.length == 1){
				elms = ["segment", elms[0]];
			}
			if (!(elms[0] in segMetadata)){
				segMetadata[elms[0]] = {};
			}
			segMetadata[elms[0]][elms[1]] = val;
		});
		var classIds = {};
		this.classes.forEach(function(elm, index){
			var classId = elm.id;
			classIds[classId] = response.data.classes.indexOf(classId) > -1;
		});
		this.segData.mainInfo = {
			seedId: segMetadata['segment']['data_seed_id'] || "unknown",
			arrivalTime: segMetadata['segment']['arrival_time'] || "unknown",
			eventMag: segMetadata['event']['magnitude'] || " unknown",
			eventDistanceKm: parseInt(segMetadata['segment']['event_distance_km']) || "unknown"
		};
		this.segData.metadata = segMetadata;
		this.segData.classIds = classIds;  // dict created above
	};

	/** functions for getting data for the plot query above **/
	// this function is used for getting the zooms and clearing them:
	this.getAndClearZooms = function(){
		return this.plots.map(function(elm){
			zoom = [this.convertAxisValue(elm.zoom[0], elm.div, 'xaxis'),
					this.convertAxisValue(elm.zoom[1], elm.div, 'xaxis')];
			//set zoom to zero, otherwise these value are persistent and affect further plots:
			elm.zoom = [null, null];
			return zoom;
		});
	};
	
	this.convertAxisValue = function(value, div, axis){
		//given an xaxis value of a plotly div 'div',
		//parses value returning 10**value if div layout type is log
		//in any other case, returns value
		// value: a value as returned by a zoom event listener. Plotly returns the log of the value
		//if axis type is log, or a STRING denoting the date-time if the type is 'date'
		// div: the plotly div
		// axis: either 'xaxis' or 'yaxis'
		if (!value || !div.layout || !div.layout[axis]){
			return value;
		}
		var type = div.layout[axis].type;
		if(type=='log'){
			return Math.pow(10, value);
		}
		return value;
	}

	// this function returns the currently visible plot indices
	this.getVisiblePlotIndices = function(){
		var plotIndices = [];
		// set which indices should we show:
		this.plots.forEach(function(element, index){
			if (element.visible){
				plotIndices.push(index);
				
			}
		});
		return plotIndices;
	};

	/** redraws the plots currently downloaded (data is in this.segData.plotData) **/
	this.redrawPlots = function(indices){
		var plotsData = this.segData.plotData;
		var plotly = $window.Plotly;
		this.setLoading("Drawing plots...");
		var plotStuff = [];  //a list of arrays, each array is (div, data, layout)
		for (var i_=0; i_< indices.length; i_++){
			var i = indices[i_];
			var div = this.plots[i].div;
			var plotData = plotsData[i];
			var data = [];
			var warnings = "";
			if (typeof plotData === 'string'){
				warnings = plotData.replaceAll("\n", "<br>");
			}else{
				data = Array.from(plotData);
			}
			var title = ""
			var layout = getPlotLayout(title, warnings, {xaxis: this.plots[i].xaxis, yaxis: this.plots[i].yaxis});
			plotStuff.push({div: div, data: data, layout:layout, index:i});
		}
		
		// delay execution of redraw. This was due to make all DOM rendered with the correct
		// sizes, which is needed to lay out plots. This is not anymore necessary, but
		// we leave this functionality for safety:
		$timeout(function(){
			this.setLoading("");
			plotStuff.forEach(function(elm){
				var uninit = !(elm.div.data);
				if (uninit){
					var config = {
						displaylogo: false,
						showLink: false,
						modeBarButtonsToRemove: ['sendDataToCloud']
					};
					plotly.plot(elm.div, elm.data, elm.layout, config);
					this.initPlotEvents(elm.index);
				}else{
					elm.div.data = elm.data;
					elm.div.layout = elm.layout;
					plotly.redraw(elm.div);
				}
			});
		}, 100);
	};

	// inits plots events on the given plot index:
	this.initPlotEvents = function(index){
		this.plots[index].div.on('plotly_relayout', function(eventdata){
			// check that this function is called from zoom
			// (it is called from any relayout command also)
			var isZoom = 'xaxis.range[0]' in eventdata && 'xaxis.range[1]' in eventdata;
			if(!isZoom){
				return;
			}
			var zoom = [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']];
			this.plots[index].zoom = [zoom[0], zoom[1]];  // copy (for safety)
			// this.refreshView([index]);
		});
		
		this.plots[index].div.on('plotly_doubleclick', function(eventdata){
			// this.refreshView([index]); // zooms are reset after use, so this redraw normal bounds
		});
	}
	
	this.toggleSegmentClassLabel = function(classId){
		var value = this.segData.classIds[classId];
		var param = {class_id: classId, segment_id: this.segId, value:value};
		$http.post("/set_class_id", param, {headers: {'Content-Type': 'application/json'}}).then(
			function(response) {
				this.classes.forEach(function(elm, index){
					if (elm.id == classId){
						elm.segments += (value ? 1 : -1);  //update count
					}
				});
			},
			function(response) {
				this.segData.classIds[classId] = !value; // restore old value
				// called asynchronously if an error occurs
				// or server returns response with an error status.
				$window.alert('Server error setting the class');
			}
		);
	};
	
	//visibility of some panels
	this.divPanels = {};
	this.isDivVisible = function(key){
		if(!(key in this.divPanels)){
			this.divPanels[key] = true;
		}
		return this.divPanels[key];
	}
	this.toggleDivVisibility = function(key){
		var status = !this.isDivVisible(key); //adds the key if not present
		this.divPanels[key] = status;
		return status
	}
	
	this.setSnWindows = function(){
		this.refreshView([0, 1]);// simply update plots, the changed flags will be set therein:
			 // refresh current segment and spectra only
	};
}

// init our app:
// S2S.init();

/** define here the plotly default layout for all plots, xaxis and yaxis stuff mught be overridden
/* by the user
 */
function getPlotLayout(title, warningMessage, ...layoutOverrides){
	var annotations = [];
	if (title){
		annotations.push({  // https://plot.ly/javascript/reference/#layout-annotations
			xref: 'paper',
			yref: 'paper',
			x: 0,
			xanchor: 'left',
			y: 1.01, //.98,
			yanchor: 'bottom',
			text: title,
			showarrow: false,
			font: {
				color: '#000000'
			},
		});
	}
	if(warningMessage){
		annotations.push({
			xref: 'paper',
			yref: 'paper',
			x: 0.5,  // 0.01,
			xanchor: 'center',
			y: 0.5, //.98,
			yanchor: 'middle',
			text: warningMessage,
			showarrow: false,
			bordercolor: '#ffffff', // '#c7c7c7',
			bgcolor: '#C0392B',
			font: {
				color: '#FFFFFF'
			},
		});
	}
	var _fs = parseFloat(window.getComputedStyle(document.body).getPropertyValue('font-size'));
	var PLOTLY_DEFAULT_LAYOUT = { //https://plot.ly/javascript/axes/
		margin:{'l':55, 't':36, 'b':45, 'r':15},
		pad: 0,
		autosize: true,
		paper_bgcolor: 'rgba(0,0,0,0)',
		font: {
			family: 'Montserrat',
			size: isNaN(_fs) ? 15 : _fs
		},
		xaxis: {
			autorange: true,
			tickangle: 0,
			linecolor: '#aaa',  // should be consistent with div.metadata (see mainapp.css)
			linewidth: 1,
			mirror: true
		},
		yaxis: {
			autorange: true,
			linecolor: '#aaa',  // should be consistent with div.metadata (see mainapp.css)
			linewidth: 1,
			mirror: true
			//fixedrange: true
		},
		annotations: annotations,
		legend: {xanchor:'right', font:{size:10}, x:0.99}
	};
	return mergeDeep(PLOTLY_DEFAULT_LAYOUT, ...layoutOverrides);
}
/* code below copied from https://stackoverflow.com/questions/27936772/how-to-deep-merge-instead-of-shallow-merge*/

/**
 * Simple object check.
 * @param item
 * @returns {boolean}
 */
function isObject(item) {
  return (item && typeof item === 'object' && !Array.isArray(item));
}

/**
 * Deep merge target with all sources. All arguments must be objects
 * @param target
 * @param ...sources
 */
function mergeDeep(target, ...sources) {
	if (!sources.length){
		return target;
	}
	const source = sources.shift();
	for (const key in source) {
		var value = source[key];
		if (isObject(value) && isObject(target[key])) {
			mergeDeep(target[key], value);
		} else {
			target[key] = value;
		}
	}
 	return mergeDeep(target, ...sources);
}