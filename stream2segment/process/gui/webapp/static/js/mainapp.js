function setInfoMessage(msg){
	var elm = document.getElementById('message-dialog');
	elm.style.color = 'inherit';
	elm.querySelector('.loader').style.display='';
	elm.querySelector('.btn-close').style.display='none';
	elm.querySelector('.message').innerHTML = msg || "";
	setDivVisible(elm, !!msg);
}

function setErrorMessage(msg){
	var elm = document.getElementById('message-dialog');
	elm.style.color = 'red';
	elm.querySelector('.loader').style.display='none';
	elm.querySelector('.btn-close').style.display='';
	elm.querySelector('.message').innerHTML = msg || "";
	setDivVisible(elm, !!msg);
}

function isDivVisible(div){
	if (typeof div === 'string') {div = document.getElementById(div); }
	return !div.classList.contains('d-none');
}

function setDivVisible(div, value){
	if (typeof div === 'string') {div = document.getElementById(div); }
	if (value){
		div.classList.remove('d-none');
	}else{
		div.classList.add('d-none');
	}
}

axios.interceptors.response.use((response) => {
	setInfoMessage("");
	return response;
}, (error) => {
	setErrorMessage("[!] " + ((error.message || '').trim() || 'Unknown error'));
	throw error
	// return Promise.reject(error.message);
});

function setSegmentsSelection(segmentsSelection){
	setInfoMessage("Selecting segments ... (it might take a while for large databases)");
	return axios.post("/set_selection", segmentsSelection, {headers: {'Content-Type': 'application/json'}}).then(response => {
		return response;
	});
}

function get_segment_data(segmentIndex, segmentsCount, plots, tracesArePreprocessed, mainPlotShowsAllComponents,
 						  metadataElements, classElements, config){
	/**
	* Main function that updates the plots and optionally updates the segment metadata
	* plots: Array of 3 elements denoting the plot to be redrawn. the 3 elements are:
	* [Python function name (string), destination <div> id (string), plotlty layout (Object)]
	* plots can ba also an Array of Arrays above (redraw more than one plot), or undefined
	* (redraw all visible plots)
	* config: Object or null. If Object, it is the new config to be passed to plots
	*/

	var funcName2ID = {};
	var funcName2Layout = {};
	for (var [fName, divId, layout] of plots){
		funcName2ID[fName] = divId;
		funcName2Layout[fName] = layout;
	}
	var zooms = null;  // NOT USED (here just in case)
	var params = {
		seg_index: segmentIndex,
		seg_count: segmentsCount,
		pre_processed: tracesArePreprocessed,
		zooms: zooms,
		plot_names: Object.keys(funcName2ID),
		all_components: mainPlotShowsAllComponents
	};

	params.config = config || {};
	if(metadataElements || classElements){
		params.metadata = true;
		params.classes = true
	}
	setInfoMessage("Fetching and computing data (it might take a while) ...");
	return axios.post("/get_segment_data", params, {headers: {'Content-Type': 'application/json'}}).then(response => {
		for (var name of Object.keys(response.data.plots)){
			redrawPlot(funcName2ID[name], response.data.plots[name], funcName2Layout[name]);
		}
		// update metadata if needed:
		if (metadataElements){
			for (var [attName, attVal] of response.data.metadata){
				metadataElements[attName].innerHTML = attVal;
			}
		}
		// update classes if needed:
		if (classElements){
			for (var classId of response.data.classes){
				classElements[clsssId].checked=true;
			}
		}
		return response;
	});
}

function redrawPlot(divId, plotlyData, plotlyLayout){
	var div = document.getElementById(divId);
	var initialized = !!div.layout;
	var _fs = parseFloat(window.getComputedStyle(document.body).getPropertyValue('font-size'));
	var _ff = window.getComputedStyle(document.body).getPropertyValue('font-family');
	var layout = {  // set default layout (and merge later with plotlyLayout, if given)
		margin:{'l': 10, 't':10, 'b':10, 'r':10},
		pad: 0,
		autosize: true,
		paper_bgcolor: 'rgba(0,0,0,0)',
		font: {
			family: _ff || "sans-serif",
			size: isNaN(_fs) ? 15 : _fs
		},
		xaxis: {
			autorange: true,
			automargin: true,
			tickangle: 0,
			linecolor: '#aaa',
			linewidth: 1,
			mirror: true
		},
		yaxis: {
			autorange: true,
			automargin: true,
			linecolor: '#aaa',
			linewidth: 1,
			mirror: true
			//fixedrange: true
		},
		annotations: [],
		legend: {xanchor:'right', font:{size:10}, x:0.99}
	};
	// deep merge plotlyLayout into layout
	var objs = [[plotlyLayout, layout]];  // [src, dest]
	while (objs.length){
		var [src, dest] = objs.shift(); // remove 1st element
		Object.keys(src).forEach(key => {
			if ((typeof src[key] === 'object') && (typeof dest[key] === 'object')){
				objs.push([src[key], dest[key]]);
			}else{
				dest[key] = src[key];
			}
		})
	}
	// if data is a string, put it as message:
	if (typeof data === 'string'){
		layout.annotations || (layout.annotations = []);
		layout.annotations.push({
			xref: 'paper',
			yref: 'paper',
			x: 0.5,  // 0.01,
			xanchor: 'center',
			y: 0.5, //.98,
			yanchor: 'middle',
			text: data,
			showarrow: false,
			bordercolor: '#ffffff', // '#c7c7c7',
			bgcolor: '#C0392B',
			font: {
				color: '#FFFFFF'
			}
		});
		data = [];
	}
	// plot (use plotly react if the plot is already set cause it's faster than newPlot):
	if (!initialized){
		var config = {
			displaylogo: false,
			showLink: false,
			modeBarButtonsToRemove: ['sendDataToCloud']
		};
		Plotly.newPlot(div, plotlyData, layout, config);
	}else{
		Plotly.react(div, plotlyData, layout);
	}
}

function setConfig(aceEditor){
	// query config and show form only upon successful response:
	return axios.post("/get_config", {as_str: true}, {headers: {'Content-Type': 'application/json'}}).then(response => {
		aceEditor.setValue(response.data);
		aceEditor.clearSelection();
		return response;
	});
}
//function setClassLabel(classId, value){
//	var segId = parseInt((document.querySelector(`[data-segment-attr="id"]`) || {}).innerHTML);
//	if (isNaN(segId)){
//		setErrorMessage('The segment ID could not be inferred');
//	}
//	var params = {class_id: classId, segment_id: segId, value: value};
//	axios.post("/set_class_id", params, {headers: {'Content-Type': 'application/json'}});
//}