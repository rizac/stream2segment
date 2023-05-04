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
	var msg = 'Internal Server Error';
	var response = error.response;
	if(response.data && response.data.message){
		msg = response.data.message.replaceAll("\n", "<br>");
		if (response.data.traceback){
			msg += "<div class='small'>Traceback: " + response.data.traceback + '</div>'
		}
	}
	setErrorMessage(msg);
	return Promise.reject(error.message);
});

function setSegmentsSelection(inputElements){
	setInfoMessage("Selecting segments ... (it might take a while for large databases)");
	var segmentsSelection = {};
	for(var att of Object.keys(inputElements)){
		var val = inputElements[att].value;
		if (val && val.trim()){
			segmentsSelection[att] = val;
		}
	}
	return axios.post("/set_selection", segmentsSelection, {headers: {'Content-Type': 'application/json'}}).then(response => {
		return response;
	});
}

function getSegmentsSelection(inputElements){
	// queries the current segments selection and puts the selection expressions into the given input elements
	return axios.post("/get_selection", {}, {headers: {'Content-Type': 'application/json'}}).then(response => {
		for(var attname of Object.keys(inputElements)){
			inputElements[attname].value = response.data[attname] || "";
			inputElements[attname].dispatchEvent(new Event("input")); // notify listeners
		}
		return response;
	});
}

function get_segment_data(segmentIndex, segmentsCount, plots, tracesArePreprocessed, mainPlotShowsAllComponents,
 						  attrElements, classElements){
	/**
	* Main function to update the GUI from a given segment.
	* plots: Array of 3-elements Arrays, where the 3 elements are:
	* 	[Python function name (string), destination <div> id (string), plotlty layout (Object)]
	* tracesArePreprocessed: boolean denoting if the traces should be pre-processed
	* mainPlotShowsAllComponents: boolean denoting if the main trace should plot all 3 components / orientations
	* attrElements: Object of segment attributes (string) mapped to the HTML element whose
	*	innerHTML should be set to the relative segment attr value (each element innerHTML is assumed
	*   to be empty). If null / undefined, segment
	* 	attr are not fetched and nothing is set
	* classElements: Object of DB classes ids (integer) mapped to the input[type=checkbox]
	* 	element whose checked state should be set true or false depending on whether the segment
	* 	has the relative class label assigned or not (each input.checked property is assumed to be false).
	*   If null / undefined, segment classes are not fetched and nothing happens
	* this method returns a Promise with argument an Object of metadata (e.g. 'id', 'event.latiture')
	* mapped to their value. The Object 'class.id' is mapped to an Array of ids. If attrElements and
	* classElements are null, the returned Object is empty
	*/
	var funcName2ID = {};
	var funcName2Layout = {};
	for (var [fName, divId, layout] of plots){
		funcName2ID[fName] = divId;
		funcName2Layout[fName] = layout;
	}
	var params = {
		seg_index: segmentIndex,
		seg_count: segmentsCount,
		pre_processed: tracesArePreprocessed,
		zooms: null,  // not used
		plot_names: Object.keys(funcName2ID),
		all_components: mainPlotShowsAllComponents,
		attributes: !!attrElements,
		classes: !!classElements
	}

	setInfoMessage("Fetching and computing data (it might take a while) ...");
	return axios.post("/get_segment_data", params, {headers: {'Content-Type': 'application/json'}}).then(response => {
		for (var name of Object.keys(response.data.plots)){
			redrawPlot(funcName2ID[name], response.data.plots[name], funcName2Layout[name]);
		}
		var ret = {};
		// update metadata if needed:
		if (attrElements){
			for (var att of response.data.attributes){
				attrElements[att.label].innerHTML = att.value;
				ret[att.label] = att.value;
			}
		}
		ret['class.id'] = [];
		// update classes if needed:
		if (classElements){
			for (var classId of response.data.classes){
				ret['class.id'].push(classId);
				classElements[classId].checked=true;
			}
		}
		return ret;
	});
}

function getPageFontInfo(){
	var style = window.getComputedStyle(document.body);
	var fsize = parseFloat(style.getPropertyValue('font-size'));
	var ffamily = style.getPropertyValue('font-family');
	return {
		'size': isNaN(fsize) ? 15 : fsize,
		'family': ffamily || 'sans-serif'
	}
}

function redrawPlot(divId, plotlyData, plotlyLayout){
	var div = document.getElementById(divId);
	var initialized = !!div.layout;
	var font = getPageFontInfo();
	var _ff = window.getComputedStyle(document.body).getPropertyValue('font-family');
	var layout = {  // set default layout (and merge later with plotlyLayout, if given)
		margin:{'l': 10, 't':10, 'b':10, 'r':10},
		pad: 0,
		autosize: true,
		paper_bgcolor: 'rgba(0,0,0,0)',
		font: font,
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
		legend: {
			xanchor:'right',
			font: {
				size: font.size *.9,
				family: font.family,
			},
			x:0.99
		}
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
				size: font.size *.9,
				family: font.family,
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
