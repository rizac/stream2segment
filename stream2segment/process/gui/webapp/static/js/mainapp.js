// Returns {data, status} (like axios' response) so existing callers using `response.data` keep working.
async function postJSON(url, body){
	var response;
	try {
		response = await fetch(url, {
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify(body)
		});
	} catch (error) {
		setErrorMessage('Network error: ' + escapeHtml(error.message));
		return Promise.reject(error.message);
	}
	var text = await response.text();
	var data;
	try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
	if (!response.ok){
		var msg = 'Internal Server Error';
		if (data && typeof data === 'object' && data.message){
			msg = escapeHtml(data.message).replace(/\n/g, "<br>");
			if (data.traceback){
				msg += "<div class='small'>Traceback: " + escapeHtml(data.traceback) + '</div>';
			}
		}
		setErrorMessage(msg);
		return Promise.reject('Request failed with status code ' + response.status);
	}
	setInfoMessage("");
	return {data: data, status: response.status};
}

function escapeHtml(str){
	return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

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

function setDivVisible(div, value){
    // value: true, false or 'toggle' (=invert visible state)
	if (typeof div === 'string') {div = document.getElementById(div); }
	if (value === 'toggle'){
	    value = !div.classList.contains('d-none');
	}
	if (value){
		div.classList.remove('d-none');
	}else{
		div.classList.add('d-none');
	}
}

function setSegmentsSelection(inputElements){
	setInfoMessage("Selecting segments ... (it might take a while for large databases)");
	var segmentsSelection = {};
	for(var att of Object.keys(inputElements)){
		var val = inputElements[att].value;
		if (val && val.trim()){
			segmentsSelection[att] = val;
		}
	}
	return postJSON("/set_selection", segmentsSelection);
}

function getSegmentsSelection(inputElements){
	// queries the current segments selection and puts the selection expressions into the given input elements
	return postJSON("/get_selection", {}).then(response => {
		return response.data;
	});
}

function getSegmentData(
    segmentIndex,
    plots,
    tracesArePreprocessed,
    mainPlotShowsAllComponents,
    loadMetadata
){
	/**
	* Main function to update the GUI from a given segment.
	* plots: Array of 3-elements Arrays, where the 3 elements are:
	* 	[Python function name (string), destination <div> id (string), plotly layout (Object)]
	* tracesArePreprocessed: boolean denoting if the traces should be pre-processed
	* mainPlotShowsAllComponents: boolean denoting if the main trace should plot all 3 components / orientations
	* loadMetadata: boolean denoting if segment attributes and classes should be fetched and returned
	* this method returns a Promise whose resolved value is the response data Object (with keys
	* 'plotData', 'plotLayout' and, if loadMetadata is true, the segment metadata, e.g. 'id', 'event.latitude')
	*/
	var funcName2ID = {};
	var funcName2Layout = {};
	for (var [fName, divId, layout] of plots){
		funcName2ID[fName] = divId;
		funcName2Layout[fName] = layout;
	}
	var params = {
		seg_index: segmentIndex,
		pre_processed: tracesArePreprocessed,
		zooms: null,  // not used
		plot_names: Object.keys(funcName2ID),
		all_components: mainPlotShowsAllComponents,
		attributes: loadMetadata,
		classes: loadMetadata
	}

	setInfoMessage("Fetching and computing data (it might take a while) ...");
	return postJSON("/get_segment_data", params).then(response => {
		for (var name of Object.keys(response.data.plotData)){
			var data = response.data.plotData[name];
			var layout = Object.assign({}, funcName2Layout[name], response.data.plotLayout[name] || {});
			redrawPlot(funcName2ID[name], data, layout);
		}
		return response.data
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
	plotlyLayout = plotlyLayout || {};
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
			var s = src[key], d = dest[key];
			var isObj = v => v !== null && typeof v === 'object' && !Array.isArray(v);
			if (isObj(s) && isObj(d)){
				objs.push([s, d]);
			}else{
				dest[key] = s;
			}
		})
	}
	// if data is a string, put it as message:
	if (typeof plotlyData === 'string'){
		layout.annotations || (layout.annotations = []);
		layout.annotations.push({
			xref: 'paper',
			yref: 'paper',
			x: 0.5,  // 0.01,
			xanchor: 'center',
			y: 0.5, //.98,
			yanchor: 'middle',
			text: plotlyData.replace(/\n/g, "<br>"),
			showarrow: false,
			bordercolor: '#ffffff', // '#c7c7c7',
			bgcolor: '#C0392B',
			font: {
				size: font.size *.9,
				family: font.family,
				color: '#FFFFFF'
			}
		});
		plotlyData = [];
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

function getConfig(){
	// query config and show form only upon successful response:
	return postJSON("/get_config", {as_str: true}).then(response => {
		return response.data;
	});
}

function setConfig(newConfig){
	return postJSON("/set_config", {data: newConfig});
}


function manageClassLabels(newLabel, newDescription) {
    // Optionally creates and returns all class labels. (create a new one only if newLabel is not empty or missing)
    var data = {}
    if (newLabel){
        data = {
            label: newLabel,
            description: newDescription
        }
    }
    return postJSON('/manage_class_labels', data).then(response => { return response.data});
}


function manageClassLabeling(classId, value, segIndex, segCount){
    var params = {
        seg_index: segIndex,
        seg_count: segCount,
        class_id: classId,
        value: value
    };
    return postJSON("/manage_class_labeling", params).then(response => {
        return response.data;
    });
}