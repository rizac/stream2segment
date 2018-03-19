function createLegend(map){
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
		    &nbsp;<span>Stations with downloaded waveform data segments. Color: percentage of segments in selected categories:</div>
		    <div class='white-space: nowrap;'>0% ${colorbar} 100%</div></div>`;
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

	var dcens = {}; //stores datacenter id mapped to markers, selected and total segments
	var fitBoundMarkers = [];  // used only for zoom. It's a collection of ALL markers we will create
	var floor = Math.floor; //this actually returns an int (at least, something whose str value is with no dots)s
	// ********************************************************************
	// function processing each station and eventually creating its marker:
	// ********************************************************************
	function processStation(staName, array, datacenters, networks, codes, selectedCodes, downloads, selectedDownloads){
		var staId = array[0];
		var lat = array[1];
		var lon = array[2];
		var dcId = array[3];
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
		if (!(dcId in dcens)){
			dcens[dcId] = {'markers': [], 'total':0, 'ok':0};
		}
		var dc = dcens[dcId];
		dc.total += total;
		dc.ok += ok;
		var gb = floor(0.5 + 255 * (1 - (total == 0 ? 0 : ok/total)));
		//console.log(gb);
		
		var circle = L.circleMarker([lat, lon], {
		    color: '#333',
		    opacity: 1,
		    weight: 1,
		    fillColor: `rgb(255, ${gb}, ${gb})`,
		    fillOpacity: 1,
		    radius: 6
		});
		//bind popup with infos:
		var staPopupContent = `<table class='station-info'>
							   <tr><th colspan="2"> ${staName}.${netName} </th></tr>
							   <tr><td>database id</td><td>${staId}</td></tr>
							   <tr><td>data-center:</td><td>${datacenters[dcId]}</td></tr>
							   <tr><td colspan="2">Segments:</td></tr>
							   <tr><td>In selected categories:</td><td> ${ok} </td></tr>
							   <tr><td>Not in selected categories:</td><td> ${malformed} </td></tr>
							   <tr><td>Total:</td><td> ${total} </td></tr>
							   </table>`; 
		circle.bindPopup(staPopupContent);
		dc.markers.push(circle);
		//if (fitBounds){
			fitBoundMarkers.push(circle);
		//}
	}
	// loop over all data and create markes, calling the function above:
	var data = GLOBALS.sta_data;
	var datacenters = GLOBALS.datacenters;
	var networks = GLOBALS.networks;
	var selCodes = GLOBALS.selcodes;
	var downloads = GLOBALS.downloads;
	var codes = GLOBALS.codes;
	var selDownloads = GLOBALS.seldownloads;
	for (var ii=0; ii < data.length; ii+=2){
		var staName = data[ii];
		var staData = data[ii+1];
		processStation(staName, staData, datacenters, networks, codes, selCodes, downloads, selDownloads);
	}

	dcLayerGroups = GLOBALS.dcLayerGroups;
	// clear all existing layers (removing all their markers) first:
	for (var dcId in dcLayerGroups){
		dcLayerGroups[dcId].clearLayers(); // clear all markers of the layer group (clearLayers is misleading)
	}
	var htmlElement = document.getElementById.bind(document);  // https://stackoverflow.com/questions/1007340/javascript-function-aliasing-doesnt-seem-to-work
	for (var dcId in dcens){
		var val = dcens[dcId];
		if (dcId in dcLayerGroups){
			var layerGroup = dcLayerGroups[dcId];
			val.markers.forEach(function(elm){layerGroup.addLayer(elm);});
		}else{
			// add layerGroup to map and our object:
			dcLayerGroups[dcId] = L.layerGroup(val.markers).addTo(map);
		}
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