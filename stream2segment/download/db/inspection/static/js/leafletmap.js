function createLegend(map){
	var legend = L.control({position: 'bottomleft'});
	legend.onAdd = function (map) {
		var div = L.DomUtil.create('div');
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
	var options = L.control({position: 'topright', collapsed: false});
	options.onAdd = function (map) {
		var div = L.DomUtil.create('div');
		div.setAttribute('style', 'margin: 0px !important');
		var child = document.getElementById('options');
		//set height and margins:
		child.style.margin = "10px 0";
		child.style.maxHeight = 'calc(100vh - 20px)';
		child.parentNode.removeChild(child);
		div.appendChild(child);
		// mouse wheel on the div should scroll it, not zoom the underlying map
		// (Info here https://gis.stackexchange.com/a/111888):
		L.DomEvent.on(div, 'mouseenter', () => { map.scrollWheelZoom.disable(); });
		L.DomEvent.on(div, 'mouseleave', () => { map.scrollWheelZoom.enable(); });
		return div;
	};
	options.addTo(map);
}

function createMap(){
	// creates the leaflet map, setting its initial bounds. Call map.getBounds() to return them
	GLOBALS.map = map = new L.Map('map', {
		// scrollWheelZoom: false
	});
	// 2 CartoDB gray scale map (very good with overlays, as in our case)
	// the added base layer added is set selected by default (do not add the others then)
	var cartoLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
		attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
		subdomains: 'abcd',
		maxZoom: 19
	}).addTo(map);
	createLegend(map);
	createOptionsMenu(map);
	// create bounds and fit now, otherwise the map mathods latlng will not work:
	var [minLat, minLon] = [1000, 1000];
	var [maxLat, maxLon] =[0, 0];
	var sta_data = GLOBALS.sta_data;
	var stations = [];
	for (var ii = 0; ii < sta_data.length; ii+=2){
		var staName = sta_data[ii];
		var staData = sta_data[ii+1];
		var lat = staData[1];
		var lon = staData[2];
		if (lat < minLat){
			minLat = lat;
		}
		if (lat > maxLat){
			maxLat = lat;
		}
		if (lon < minLon){
			minLon = lon;
		}
		if (lon > maxLon){
			maxLon = lon;
		}
		stations.push([staName, staData, L.latLng(lat, lon)]);
	}
	GLOBALS.sta_data = stations;
	if (minLat == maxLat){
		minLat -= 0.1;
		maxLat += 0.1;
	}
	if(minLon == maxLon){
		minLon -= 0.1;
		maxLon += 0.1;
	}
	var corner1 = L.latLng(minLat, minLon),
	corner2 = L.latLng(maxLat, maxLon),
	__bounds = L.latLngBounds(corner1, corner2);
	map.fitBounds(__bounds);
	// init listeners (just once):
	map.on("zoomend", function (e) {
		// console.log("ZOOMEND", e);
		updateMap();
	});
	return map;
}

function updateMap(){
	// updates the map in a timeout in case of poor perfs
	var loader = document.getElementById("loadingDiv");
	loader.style.display = "block";
	setTimeout(function(){
		var map = _updateMap();
		map.invalidateSize();
		loader.style.display = "none";
	}, 25);
}

function _updateMap(){
	/**
	 * Updates the map with the given data and the given selected labels
	 * This function is called on zoom to resize the markers, as they are of type Leaflet.ploygon,
	 * which are much more lightweight than svgicons but with sizes relative to the map coordinates,
	 * thus zoom dependent, whereas we want them zoom independent
	*/

	var attr = 'data-seldownload-id';
	var seldownloads = new Set(
		 Array.from(document.querySelectorAll(`input[${attr}]`)).
		 filter(elm => elm.checked).
		 map(elm => parseInt(elm.getAttribute(attr)))
	);

	var attr = 'data-seldatacenter-id';
	var seldatacenters = new Set(
		 Array.from(document.querySelectorAll(`input[${attr}]`)).
		 filter(elm => elm.checked).
		 map(elm => parseInt(elm.getAttribute(attr)))
	);

	var attr = 'data-selcode-id';
	var selcodes = new Set(
		 Array.from(document.querySelectorAll(`input[${attr}]`)).
		 filter(elm => elm.checked).
		 map(elm => parseInt(elm.getAttribute(attr)))
	);

	var map = GLOBALS.map;
	if(!map){
		map = createMap();
	}else{
		map.removeLayer(GLOBALS.mapLayer);
	}
	// sta_sta might have been modified if map has to be initialized, set it here:
	sta_data = GLOBALS.sta_data;
	var mapBounds = map.getBounds();  // not used, left here to avoid re-googling it in case ...
	// alias for document.getElementById:
	var htmlElement = document.getElementById.bind(document);  // https://stackoverflow.com/questions/1007340/javascript-function-aliasing-doesnt-seem-to-work

	var dcStats = {}; //stores datacenter id mapped to markers, selected and total segments
	var minVal = 1.0;
	var maxVal = 0.0;
	// Although we do not use svg icons anymore (which have poor perfs) we should not
	// need visibleMarkers to display only visible markers, probably the Object below
	// was kept anyway to make map lighter:
	var visibleMarkers = {};
	var outb =0;
	var below = 0;
	var stazz = 0;
	// now calculate datacenters stats, ok, malformed for each station, and which station markers should be displayed: 
	for (var i = 0; i < sta_data.length; i++){
		var [staName, staData, latLng] = sta_data[i];
		var [ok, malformed, total] = processStation(staName, staData, selcodes, seldownloads, seldatacenters);
		if (!total){
			continue;
		}
		var [staId, lat, lon, dcId, netIndex] = staData;		
		// get datacenter id and update dc stats:
		if (!(dcId in dcStats)){
			dcStats[dcId] = {'total':0, 'ok':0};
		}
		var dcStat = dcStats[dcId];
		dcStat.total += total;
		dcStat.ok += ok;
		// if exactly centered on another marker (according to the current map resolution, in pixels)
		// show only the one with higher value (=% of segments in selected catagories):
		var key = map.latLngToLayerPoint(latLng);
		key = [key.x, key.y].toString();  // in any case js objects convert keys to string
		var insert = !(key in visibleMarkers);
		if (!insert){
			var [mySize, myVal] = getSizeAndValue(ok, malformed, total);
			var [staName_, netName_, staId_, latLng_, dcId_, datacenter_, ok_, malformed_, total_] = visibleMarkers[key];
			var [otherSize, otherVal] = getSizeAndValue(ok_, malformed_, total_);
			if(myVal > otherVal){
				insert = true;
			}
		}
		if(insert){
			var netName = GLOBALS.networks[netIndex];
			visibleMarkers[key] = [staName, netName, staId, latLng, dcId, GLOBALS.datacenters[dcId], ok, malformed, total];
			stazz += 1;
		}else{
			below +=1;
		}
	}
	
	console.log(`inserted ${stazz}, outofbounds ${outb}, overlapping-hidden ${below}`);
	
	// now display the markers:
	var allMarkers = [];
	for (key in visibleMarkers){
		var [staName, netName, staId, latLng, dcId, datacenter, ok, malformed, total] = visibleMarkers[key];
		marker =  createMarker(staName, netName, staId, latLng, dcId, datacenter, ok, malformed, total, map);
		allMarkers.push(marker);
	}
	// now sort them:
	allMarkers.sort(function(m1, m2){return m1.options.zIndexOffset - m2.options.zIndexOffset;});
	
	// print stats for datacenters:
	for (var dcId in GLOBALS.datacenters){  // iterate over Object keys (datacenter id)
		var {ok, total} = (dcId in dcStats) ? dcStats[dcId] : {'ok': 0, 'total': 0};
		//update stats in the dropdown menu Options:
		htmlElement(`dc${dcId}total`).innerHTML = total.toLocaleString('en-US');
		htmlElement(`dc${dcId}sel`).innerHTML = ok.toLocaleString('en-US');
		htmlElement(`dc${dcId}selperc`).innerHTML = `${total ? Math.round((100*ok)/total) : 0}%`;
	}
	// set the mapLayer so that we will know what to clear the next time we call this method:
	GLOBALS.mapLayer = new L.featureGroup(allMarkers).addTo(map);
	return map;
}

function processStation(staName, staData, selectedCodes, selectedDownloads, selectedDatacenters){
	// returns the array [ok, malformed, total] for a given station
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

function createMarker(staName, netName, staId, latLng, dcId, datacenter, ok, malformed, total, map){
	// creates the marker for a given set of stations attributes
	// Uses  L.Ploygon because it's MUCH, much more lightweight than L.marker with custom svg icon
	// The drawback is that we need to resize the marker on zoom
	// for info in svg marker icon, see // copied and modified from https://groups.google.com/forum/#!topic/leaflet-js/GSisdUm5rEc
	var [size, val] = getSizeAndValue(ok, malformed, total);
	var greenBlue = 255 - val;

	var h = size*1.7320508/2;  // height of a triangular equilateral
	var x = size/2.0;
	var y = 2*h/3.0;

	var pt = map.latLngToLayerPoint(latLng);
	var latlng2 = map.layerPointToLatLng(new L.Point(pt.x+x, pt.y-y));
	
	var lon = latLng.lng;
	var lat = latLng.lat;
	var xx = Math.abs(latlng2.lng - lon);
	var yy = Math.abs(latlng2.lat - lat);
	
	// zIndexOffset is an OFFSET. If made in thousands it basically works as a zIndex
	// (see last post here: https://github.com/Leaflet/Leaflet/issues/5560)
	// it is used only if Markers are supplied. If Polygon (as in this case) they will be used
	// for ordering the markers
	var zIndexOffset = (val > 0 ? 10000 : 0) + 100 * size;
	var latlngs = [[lat-yy/2, lon-xx],[lat+yy, lon], [lat-yy/2, lon+xx]];
	var tri = L.polygon(latlngs, {fillOpacity: 1, color: '#333', fillColor:`rgb(255, ${greenBlue}, ${greenBlue})`,
		weight:1, zIndexOffset: zIndexOffset});
	
	//bind popup with infos:
	var staPopupContent = `<h5> ${staName}.${netName} </h5>
							<div>${datacenter}</div>
							<table class='table'>
							<tr><td>database id:</td><td class='text-end'>${staId}</td></tr>
							<tr><td>Segments:</td><td class='text-end'> ${total.toLocaleString('en-US')} </td></tr>
							<tr><td>In selected categories:</td><td class='text-end'> ${ok.toLocaleString('en-US')} </td></tr>
							<tr><td>Not in selected categories:</td><td class='text-end'> ${malformed.toLocaleString('en-US')} </td></tr>
							</table>`;
	tri.bindPopup(staPopupContent);

	return tri;
}

function getSizeAndValue(ok, malformed, total){
	//returns the array [size, total], where size is an int in [7,15] and total an int in [55, 255]
	// the former is the size of the markers, the latter the color to be substracted to green and blue components.
	// It is from 55 in order to leave white markers with no stations in selected categories, and
	// make pink to red the other cases, so that hopefully they are visually recognizable.
	var val = ok/total;
	var retVal = 0;
	if (val > 0){
		retVal = parseInt(0.5+200*val) + 55;
	}
	// set sizes kind-of logaritmically:
	var minRadius = 7;  // lower than this the circle is not clickable ...
	var sizeRadius = 10; // for the biggest case (>= than 1000 segments)
	if (total < 10){
		sizeRadius = 0;
	}else if (total < 50){
		sizeRadius = 2;
	}else if (total < 100){
		sizeRadius = 4;
	}else if (total < 500){
		sizeRadius = 6;
	}else if (total < 1000){
		sizeRadius = 8;
	}
	return [minRadius+sizeRadius, retVal];
}