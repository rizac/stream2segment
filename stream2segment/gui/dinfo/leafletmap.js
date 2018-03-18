var map = new L.Map('map');

var legend = L.control({position: 'bottomleft'});
legend.onAdd = function (map) {
    var div = L.DomUtil.create('div', 'info-legend');
    // create colorbar html string:
    var colorbar = [0, .25, .5, .75, 1].map(
    		function(value){
    			var gbLevel = Math.floor(255 - 255*value);
    			var color = "rgb(255, " + gbLevel + ", " + gbLevel +")";
    			return `<span class='cmap-chunk' style='background-color: ${color}'></span>`;
    		}
    	).join('');
    div.innerHTML = `<div style='max-width:8em'><div><span class='station_symbol' style='background-color: red'></span>
	    &nbsp;<span>Stations with downloaded waveform data segments. Color: percentage of well-formed segments:</div>
	    <div class='white-space: nowrap;'>0% ${colorbar} 100%</div></div>`;
    return div;
};

legend.addTo(map);


// create the options menu. This is in a function so that it can be called 
// after the page has rendered AND SHOULD BE CALLED ONLY ONCE
function createOptionsMenu(){
	var options = L.control({position: 'topright'});
	options.onAdd = function (map) {
	    var div = L.DomUtil.create('div', 'info-legend options');
	    var child = document.getElementById('options');
	    child.parentNode.removeChild(child);
	    div.appendChild(child);
	    // disable clicks and mouse wheel, so enable scrolling on the div:
	    // https://github.com/dalbrx/Leaflet.ResizableControl/blob/master/src/Leaflet.ResizableControl.js#L98
	    L.DomEvent.disableClickPropagation(child);  // for safety (is it needed?) 
        L.DomEvent.on(child, 'mousewheel', L.DomEvent.stopPropagation);
        
        // add mouse out over effect (hide/ set visible like layers control)
        var icon = L.DomUtil.create('div', 'options_icon');
        icon.innerHTML = 'Options';
        div.appendChild(icon);
        var visibleRemainder = true;
        var showFunc = function(e){setVisible(e, true)};
        var hideFunc = function(e){setVisible(e, false)};
        function setVisible(e, visible){
        		if (visible != visibleRemainder){
        			child.style.display = visible ? 'block' : 'none';
        			icon.style.display = visible ? 'none' : 'inline-block';
        			visibleRemainder = visible;
	        		if (e){
	        			L.DomEvent.stopPropagation(e);
	        		}
        		}
        };
        L.DomEvent.on(div, 'mouseout', hideFunc);
		L.DomEvent.on(div, 'mouseover', showFunc);
        setVisible(null, false);
	    return div;
	    //return child;
	};
	options.addTo(map);
	
}


// store baseLayer and control (clickable legend with the data-centers) in global vars
// maybe not ideal, but we did not find out a better way. See updateMap for info on this object
var _leafletData = null;

function updateMap(){
	/**
	 * Updates the map with the given data and the given selected labels
	 * If the map is already in-place, avoids re-creating layers
	 * but remove its elements (stations) and adds them again. 
	 */
	var fitBounds = false;
	if(!_leafletData){
		// initialize the map if not already init'ed
		L.esri.basemapLayer("Oceans").addTo(map);
		L.esri.basemapLayer("OceansLabels").addTo(map);
		_leafletData = {'control': L.control.layers().addTo(map),  // the control (legend with datacenters checkboxes)
						'dcLayerGroup': {}  // a dict of datacenters names (or id) strings mapped to the relativel
						//leaflet LayerGroup object
		};
		fitBounds = true;
		createOptionsMenu();
	}

	var dcens = {}; //stores layers to checkbox stations of a single datacenter
	var fitBoundMarkers = [];  // used only for zoom. It's a collection of ALL markers we will create
	var floor = Math.floor; //this actually returns an int (at least, something whose str value is with no dots)s
	// ********************************************************************
	// function processing each station and eventually creating its marker:
	// ********************************************************************
	function processStation(staName, array, datacenters, networks, codes, selectedCodes, downloads, selectedDownloads){
		var staId = array[0];
		var lat = array[1];
		var lon = array[2];
		var dcen = datacenters[array[3]];
		var netName = networks[array[4]];
		var ok = 0;
		var malformed = 0;
		// compute malformed and ok:
		var skipStation = true;
		var STARTDATAINDEX = 5;
		for (var idx = STARTDATAINDEX; idx < array.length; idx +=2){
			var downloadId = array[idx];
			var downloadData = array[idx+1];
			if (!selectedDownloads.has(downloadId)){
				continue;
			}
			skipStation = false;
			
			for (var idj = 0; idj < downloadData.length; idj +=2){
				var codeIndex = downloadData[idj];
				var numSegments = downloadData[idj+1];
				if (selectedCodes.has(codeIndex)){
					ok += numSegments;
				}else{
					malformed += numSegments;
				}
			}
		}
		if (skipStation){
			return;
		}
		var total = ok + malformed;
		if (!(dcen in dcens)){
			dcens[dcen] = {'markers': [], 'total':0, 'ok':0};
		}
		var dc = dcens[dcen];
		dc.total += total;
		dc.ok += ok;
		var gb = floor(0.5 + 255 * (1 - (total == 0 ? 0 : ok/total)));
		//console.log(gb);
		
		var fillColor = "rgb(255, " + gb + "," + gb +")";
		var color = '#999999';
		var circle = L.circleMarker([lat, lon], {
		    color: color,
		    opacity: 1,
		    weight: 1,
		    fillColor: fillColor,
		    fillOpacity: 1,
		    radius: 6
		});
		//bind popup with infos:
		var staPopupContent = `<table class='station-info'>
							   <tr><th colspan="2"> ${staName}.${netName} </th></tr>
							   <tr><td colspan="2">(db id: ${staId})</td></tr>
							   <tr><td colspan="2">Segments:</td></tr>
							   <tr><td>Well-formed </td><td> ${ok} </td></tr>
							   <tr><td>Malformed </td><td> ${malformed} </td></tr>
							   <tr><td>Total </td><td> ${total} </td></tr>
							   </table>`; 
		circle.bindPopup(staPopupContent);
		dc.markers.push(circle);
		//if (fitBounds){
			fitBoundMarkers.push(circle);
		//}
	}
	// loop over all data and create markes, calling the function above:
	var data = GLOBALS.sta_data;
	var dcs = GLOBALS.datacenters;
	var networks = GLOBALS.networks;
	var selCodes = GLOBALS.selcodes;
	var downloads = GLOBALS.downloads;
	var codes = GLOBALS.codes;
	var selDownloads = GLOBALS.seldownloads;
	for (var ii=0; ii < data.length; ii+=2){
		var staName = data[ii];
		var staData = data[ii+1];
		processStation(staName, staData, dcs, networks, codes, selCodes, downloads, selDownloads);
	}

	// Note: if we want a control,
	var layerControl = _leafletData['control'];
	var dcLayerGroup = _leafletData['dcLayerGroup'];
	// clear all existing layers first, removing all their markers:
	for (var dcen in dcLayerGroup){
		dcLayerGroup[dcen].clearLayers(); // clear all markers of the layer group (clearLayers is misleading)
	}
	var doc = document;
	var DCID_PREFIX = "_dcen_";
	for (var dcen in dcens){
		var val = dcens[dcen];
		var title = `${dcen}  | Well-formed segments: ${val.ok} of ${val.total} (${Math.round((100*val.ok)/val.total)} %)`; 
		if (dcen in dcLayerGroup){
			var layerGroup = dcLayerGroup[dcen];
			val.markers.forEach(function(elm){layerGroup.addLayer(elm);});
			// update title (text) for the relative control with the new percentages:
			doc.getElementById(DCID_PREFIX+dcen).innerHTML = title;  // here we should set the new title
		}else{
			layerGroup = L.layerGroup(val.markers).addTo(map);
			//this is a way to retrieve the object associated with the control's datacenter checkbox woithout
			// storing it in any variable: add HTML instead of plain text:
			layerControl.addOverlay(layerGroup, `<span id='${DCID_PREFIX+dcen}'> ${title} </span>`);
			dcLayerGroup[dcen] = layerGroup;
		}
	}
	
	//fitBoundMarkers.forEach(function(elm){elm.bringToFront();});
	
	// fit bounds and set stuff only if initializing:
	if(fitBounds){
		// https://stackoverflow.com/questions/16845614/zoom-to-fit-all-markers-in-mapbox-or-leaflet
		var group = new L.featureGroup(fitBoundMarkers);
		map.fitBounds(group.getBounds());
	}
}