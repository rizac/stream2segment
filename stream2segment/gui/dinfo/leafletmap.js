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
		GLOBALS.allMarkers = [];
		fitBounds = true;
		createLegend(map);
		createOptionsMenu(map);
	}
	// some shared functions
	var htmlElement = document.getElementById.bind(document);  // https://stackoverflow.com/questions/1007340/javascript-function-aliasing-doesnt-seem-to-work
	
	// remove all markers:
	GLOBALS.allMarkers.forEach(function(element){map.removeLayer(element);});

	var dcStats = {}; //stores datacenter id mapped to markers, selected and total segments
	var allMarkers = [];  // It's a collection of ALL markers we will create, used also for zoom if map is uninitialized 
	// loop over all data and create markes, calling the function above:
	var {sta_data, datacenters, seldatacenters, networks, codes, selcodes, downloads, seldownloads} = GLOBALS;
	var minVal = 1.0;
	var maxVal = 0.0;
	for (var ii = 0; ii < sta_data.length; ii+=2){
		var staName = sta_data[ii];
		var staData = sta_data[ii+1];
		var [ok, malformed, total] = processStation(staName, staData, selcodes, seldownloads, seldatacenters);
		if (!total){
			continue;
		}
		var staId = staData[0];
		var lat = staData[1];
		var lon = staData[2];
		// get datacenter id and update dc stats:
		var dcId = staData[3];
		if (!(dcId in dcStats)){
			dcStats[dcId] = {'total':0, 'ok':0};
		}
		var dcStat = dcStats[dcId];
		dcStat.total += total;
		dcStat.ok += ok;
		// create the marker and add it to the map:
		var netName = networks[staData[4]];
		var circle = createMarker(staName, netName, staId, lat, lon, dcId, datacenters[dcId], ok, malformed, total).addTo(map);
		allMarkers.push(circle);
	}
	
	// print stats for datacenters:
	for (var dcId in datacenters){
		var {ok, total} = (dcId in dcStats) ? dcStats[dcId] : {'ok': 0, 'total': 0};
		//update stats in the dropdown menu Options:
		htmlElement(`dc${dcId}total`).innerHTML = total;
		htmlElement(`dc${dcId}sel`).innerHTML = ok;
		htmlElement(`dc${dcId}selperc`).innerHTML = `${total ? Math.round((100*ok)/total) : 0}%`;
	}

	// fit bounds and set stuff only if initializing:
	if(fitBounds){
		// https://stackoverflow.com/questions/16845614/zoom-to-fit-all-markers-in-mapbox-or-leaflet
		var group = new L.featureGroup(allMarkers);
		map.fitBounds(group.getBounds());
	}
	
	// now sort markers and then bring them to front.
	// This has to be done at the real end because if the map is uninitialized a view must be set to
	// call bringToFront below. Initializing map.setView(...) would solve the problem
	// of this code placement BUT it does not fit the zoom the first time the map is created
	function sortMarker(marker1, marker2){
		// first who has at least some segment ok, IF the other has not
		var val = marker1.options.zIndexOffset-marker2.options.zIndexOffset; 
		if (!val){ // if both with no segments ok, or both with at least one segment ok, bigger markers to back:
			val = marker2.options.radius-marker1.options.radius;
		}
		return val;
	}
	allMarkers.sort(sortMarker);
	allMarkers.forEach(function(marker){marker.bringToFront();});
	GLOBALS.allMarkers = allMarkers; // assign to global allMarkers
}

function processStation(staName, staData, selectedCodes, selectedDownloads, selectedDatacenters){
	var ok = 0;
	var malformed = 0;
	dcId = staData[3];
	var skipMarker = [0, 0, 0];
	if (!selectedDatacenters.has(dcId)){
		return skipMarker;
	}
	// compute malformed and ok:
	var skipStation = true;
	var STARTDATAINDEX = 5;
	for (var i = STARTDATAINDEX; i < staData.length; i+=2){
		var downloadId = staData[i];
		var downloadData = staData[i+1];
		if (!selectedDownloads.has(downloadId)){
			continue;
		}
		skipStation = false;
		for (var j = 0; j < downloadData.length; j+=2){
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
		return skipMarker;
	}
	return [ok, malformed, ok+malformed];
}

function createMarker(staName, netName, staId, lat, lon, dcId, datacenter, ok, malformed, total){
	//datacenters[dcId]
	var val = ok/total;
	var greenBlue = 255;
	var [minVal, maxVal] = [0, 1];
	if (val > 0 && maxVal > minVal){  // second && is for safety...
		// now we want to set a shade of red according to val: the higher val, the more the marker is red
		// In a rgb context, this means that the higher val, the lower, the lower the var greenBlue
		// The function below creates the value for greenBlue, taking in consideration that
		// visually we are interested to spot the stations with some 'ok' segments (val >0),
		// and thus we set the maximum of greenBlue to 190, meaning that the color immeditaley after
		// the white (val==0) is a kind of pink one
		greenBlue = 190 + ((-190) * (val - minVal) / (maxVal - minVal));
		greenBlue = parseInt(0.5 + greenBlue); // round to int: converts maxVal to 0, minVal to 255
	}
	// set sizes kind-of logaritmically:
	var minRadius = 3;  // lower than this the circle is not clickable ...
	var sizeRadius = 5; // for the biggest case (>= than 1000 segments)
	if (total < 10){
		sizeRadius = 0;
	}else if (total < 50){
		sizeRadius = 1;
	}else if (total < 100){
		sizeRadius = 2;
	}else if (total < 500){
		sizeRadius = 3;
	}else if (total < 1000){
		sizeRadius = 4;
	}

	var circle = L.circleMarker([lat, lon], {
	    color: '#333',
	    opacity: 1,
	    dcId: dcId,
	    weight: 1,
	    fillColor: `rgb(255, ${greenBlue}, ${greenBlue})`,
	    fillOpacity: 1,
	    radius: sizeRadius + minRadius,
	    zIndexOffset: val > minVal ? 1000 : 0  // not that this is NOT used for circles but for markers, we will use this feature afterwards
	});
	//bind popup with infos:
	var staPopupContent = `<table class='station-info'>
						   <tr><th colspan="2"> ${staName}.${netName} </th></tr>
						   <tr><td colspan="2">${datacenter}</td></tr>
						   <tr><td>database id:</td><td class='right'>${staId}</td></tr>
						   <tr><td>Segments:</td><td class='right'> ${total} </td></tr>
						   <tr><td class='right'>In selected categories:</td><td class='right'> ${ok} </td></tr>
						   <tr><td class='right'>Not in selected categories:</td><td class='right'> ${malformed} </td></tr>
						   </table>`; 
	circle.bindPopup(staPopupContent);
	
	return circle;
}