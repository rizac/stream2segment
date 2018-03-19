function createLegend(map){
	var legend = L.control({position: 'bottomleft'});
	legend.onAdd = function (map) {
		var div = L.DomUtil.create('div', 'info-legend');
	    var child = document.getElementById('legend');
	    child.parentNode.removeChild(child);
	    div.appendChild(child);
	    return div;
	};
	legend.addTo(map);
}

// create the options menu. This is in a function so that it can be called 
// after the page has rendered AND SHOULD BE CALLED ONLY ONCE
function createOptionsMenu(map){
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
        var isOptionsDivVisible = true;
        function setVisible(e, visible){
        		if (visible != isOptionsDivVisible){
        			child.style.display = visible ? 'block' : 'none';
        			icon.style.display = visible ? 'none' : 'inline-block';
        			isOptionsDivVisible = visible;
	        		if (e){
	        			L.DomEvent.stopPropagation(e);
	        		}
        		}
        };
        var showFunc = function(e){setVisible(e, true)};
        var hideFunc = function(e){setVisible(e, false)};

        // To use the options button as the leaflet control layer (using click event is more complex,
        // as we should keep the icon div above visible, change its name to 'hide', and place it on the div):
        L.DomEvent.on(div, 'mouseover', showFunc);
        L.DomEvent.on(div, 'mouseout', hideFunc);
		
        // Set the default options div visibility:
        setVisible(null, false);
	    return div;
	    //return child;
	};
	options.addTo(map);
	
}

function updateMap(){
	/**
	 * Updates the map with the given data and the given selected labels
	 * If the map is already in-place, avoids re-creating layers
	 * but remove its elements (stations) and adds them again.
	 * This function is NOT called when we toggle datacenters selection, as in this case
	 * we simply add / remove already computed layers
	 */
	var fitBounds = false;
	var map = GLOBALS.map;
	if(!map){
		GLOBALS.map = map = new L.Map('map');
		// initialize the map if not already init'ed
		L.esri.basemapLayer("Topographic").addTo(map);
		// L.esri.basemapLayer("OceansLabels").addTo(map);
		GLOBALS.dcLayerGroups = {};
		fitBounds = true;
		createLegend(map);
		createOptionsMenu(map);
	}
	// some shared functions
	var htmlElement = document.getElementById.bind(document);  // https://stackoverflow.com/questions/1007340/javascript-function-aliasing-doesnt-seem-to-work
	

	var dcens = {}; //stores datacenter id mapped to markers, selected and total segments
	var fitBoundMarkers = [];  // used only for zoom. It's a collection of ALL markers we will create
	// loop over all data and create markes, calling the function above:
	var data = GLOBALS.sta_data;
	var datacenters = GLOBALS.datacenters;
	var networks = GLOBALS.networks;
	var selCodes = GLOBALS.selcodes;
	var downloads = GLOBALS.downloads;
	var codes = GLOBALS.codes;
	var selDownloads = GLOBALS.seldownloads;
	var markersData = [];
	var minVal = 1.0;
	var maxVal = 0.0;
	for (var ii = 0; ii < data.length; ii+=2){
		var staName = data[ii];
		var staData = data[ii+1];
		var [ok, malformed, total] = processStation(staName, staData, selCodes, selDownloads);
		if (!total){
			continue;
		}
		var staId = staData[0];
		var lat = staData[1];
		var lon = staData[2];
		var dcId = staData[3];
		var netName = networks[staData[4]];
		markersData.push([staName, netName, staId, lat, lon, dcId, ok, malformed, total]);
		var myVal = ok/total;
		if(minVal > myVal){
			minVal = myVal;
		}
		if(maxVal < myVal){
			maxVal = myVal;
		}
	}
	// update legend colorbar:
	// conv returns parseInt if a number is >=1 or 0, otherwise the str representation up to the first two nonzero decimals:
	function conv(n){return '' + ((n==0 || n>=1) ? parseInt(0.5+n):  n.toFixed(1-Math.floor(Math.log(n)/Math.log(10))));}
	htmlElement('minval').innerHTML = conv(100 * minVal) + '%';
	htmlElement('maxval').innerHTML = conv(100 * maxVal) + '%';
	
	// create markers with color:
	markersData.forEach(function(val){
		var [staName, netName, staId, lat, lon, dcId, ok, malformed, total] = val;
		if (!(dcId in dcens)){
			dcens[dcId] = {'markers': [], 'total':0, 'ok':0};
		}
		var dc = dcens[dcId];
		dc.total += total;
		dc.ok += ok;
		
		var circle = createMarker(staName, netName, staId, lat, lon, datacenters[dcId], ok, malformed, total, minVal, maxVal);
		dc.markers.push(circle);
		fitBoundMarkers.push(circle);
	});

	dcLayerGroups = GLOBALS.dcLayerGroups;
	// clear all existing layers (removing all their markers) first:
	for (var dcId in dcLayerGroups){
		dcLayerGroups[dcId].clearLayers(); // clear all markers of the layer group (clearLayers is misleading)
	}
	for (var dcId in dcens){
		var val = dcens[dcId];
		// Now bring to front all markers, from those who have lower values to those
		// who have higher values. https://stackoverflow.com/questions/39202182/leaflet-circle-z-index
		// note that sort modifies the array INPLACE!
		val.markers.sort(function(a, b){return a.options.zIndexOffset-b.options.zIndexOffset});
		if (!(dcId in dcLayerGroups)){
			dcLayerGroups[dcId] = L.layerGroup().addTo(map);
		}
		var layerGroup = dcLayerGroups[dcId];
		// set the zindex based on the maximum value found:
		layerGroup.setZIndex(val.markers[val.markers.length-1].options.zIndexOffset);
		// add markers and call bring to front AFTER it is added:
		val.markers.forEach(function(marker){layerGroup.addLayer(marker);marker.bringToFront();});
		//update stats in the dropdown menu Options:
		htmlElement(`dc${dcId}total`).innerHTML = val.total;
		htmlElement(`dc${dcId}sel`).innerHTML = val.ok;
		htmlElement(`dc${dcId}selperc`).innerHTML = `${Math.round((100*val.ok)/val.total)}%`;

	}

	// fit bounds and set stuff only if initializing:
	if(fitBounds){
		// https://stackoverflow.com/questions/16845614/zoom-to-fit-all-markers-in-mapbox-or-leaflet
		var group = new L.featureGroup(fitBoundMarkers);
		map.fitBounds(group.getBounds());
	}
}

function processStation(staName, staData, selectedCodes, selectedDownloads){
	var ok = 0;
	var malformed = 0;
	// compute malformed and ok:
	var skipStation = true;
	var STARTDATAINDEX = 5;
	for (var i = STARTDATAINDEX; i < staData.length; i +=2){
		var downloadId = staData[i];
		var downloadData = staData[i+1];
		if (!selectedDownloads.has(downloadId)){
			continue;
		}
		skipStation = false;
		for (var j = 0; j < downloadData.length; j +=2){
			var codeIndex = downloadData[j];
			var numSegments = downloadData[j+1];
			if (selectedCodes.has(codeIndex)){
				ok += numSegments;
			}else{
				malformed += numSegments;
			}
		}
	}
	if (skipStation){
		return [0.0, 0.0, 0.0];
	}
	return [ok, malformed, ok+malformed];
}

function createMarker(staName, netName, staId, lat, lon, datacenter, ok, malformed, total, minVal, maxVal){
	//datacenters[dcId]
	var val = ok/total;
	val = 255*((val - minVal) / (maxVal - minVal));  // converts minVal to 0, maxVal to 1
	var markerZIndex = Math.floor(0.5 + 1000 * val)  // converts minVal to 0, maxVal to 1000
	val = 255 * (1 - val);  // invert: maxVal to 0, minVal to 255
	val = Math.floor(0.5 + val); // round to intconverts maxVal to 0, minVal to 255
	
	var circle = L.circleMarker([lat, lon], {
	    color: '#333',
	    opacity: 1,
	    weight: 1,
	    fillColor: `rgb(255, ${val}, ${val})`,
	    fillOpacity: 1,
	    radius: 6,
	    zIndexOffset: markerZIndex  // not that this is NOT used for circles but for markers, we will use this feature afterwards
	});
	//bind popup with infos:
	var staPopupContent = `<table class='station-info'>
						   <tr><th colspan="2"> ${staName}.${netName} </th></tr>
						   <tr><td>database id</td><td>${staId}</td></tr>
						   <tr><td>data-center:</td><td>${datacenter}</td></tr>
						   <tr><td colspan="2">Segments:</td></tr>
						   <tr><td>In selected categories:</td><td> ${ok} </td></tr>
						   <tr><td>Not in selected categories:</td><td> ${malformed} </td></tr>
						   <tr><td>Total:</td><td> ${total} </td></tr>
						   </table>`; 
	circle.bindPopup(staPopupContent);
	
	return circle;
}