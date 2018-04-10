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

function createMap(){
	// creates the leaflet map, setting its initial bounds. Call map.getBounds() to return them
	GLOBALS.map = map = new L.Map('map');
	// initialize the map if not already init'ed
	L.esri.basemapLayer("Topographic").addTo(map);
	// L.esri.basemapLayer("OceansLabels").addTo(map);
	createLegend(map);
	createOptionsMenu(map);
	// create bounds and fit now, otherwise the map mathods latlng will not work:
	var [minLat, minLon] = [1000, 1000];
	var [maxLat, maxLon] =[0, 0];
	var sta_data = GLOBALS.sta_data;
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
	}
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
	map.on("moveend", function (e) {
		// console.log("ZOOMEND", e);
		updateMap();
	});
	return map;
}

function updateMap(){
	var loader = document.getElementById("loadingDiv");
	loader.style.display = "block";
	setTimeout(function(){ _updateMap(); loader.style.display = "none";}, 50);
}

function _updateMap(){
	/**
	 * Updates the map with the given data and the given selected labels
	 * This function is called on pan and zoom to minimize the markers, as we use svgicons which are quite heavy,
	 * we need to load only visible one...
	 */
	var {sta_data, datacenters, seldatacenters, networks, codes, selcodes, downloads, seldownloads} = GLOBALS;

	var map = GLOBALS.map;
	if(!map){
		map = createMap();
	}else{
		map.removeLayer(GLOBALS.mapLayer);
	}
	var bounds = map.getBounds();
	// alias for document.getElementById:
	var htmlElement = document.getElementById.bind(document);  // https://stackoverflow.com/questions/1007340/javascript-function-aliasing-doesnt-seem-to-work

	var dcStats = {}; //stores datacenter id mapped to markers, selected and total segments
	var minVal = 1.0;
	var maxVal = 0.0;
	// we use svg icons for the triangles, they have poor perfs, so remove those that are hidden
	// use the object below for that:
	var visibleMarkers = {};
	// now calculate datacenters stats, ok, malformed for each station, and which station markers should be displayed: 
	for (var ii = 0; ii < sta_data.length; ii+=2){
		var staName = sta_data[ii];
		var staData = sta_data[ii+1];
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
		// create the marker and add it to the map:
		var netName = networks[netIndex];
		//var circle = createMarkerOld(staName, netName, staId, lat, lon, dcId, datacenters[dcId], ok, malformed, total).addTo(map);
		
		// decide if we need to add the marker to the visible markers:
		// if out-of-bounds, do not create marker:
		if(bounds){
			var corner1 = L.latLng(lat-.01, lon-.01),
			corner2 = L.latLng(lat+.01, lon+.01),
			_bounds = L.latLngBounds(corner1, corner2);
			if (!bounds.intersects(_bounds)){
				continue;
			}
		}
		// if exactly centered on another marker (according to the current map resolution, in pixels)
		// show only the one with higher value (=% of segments in selected catagories):
		var key = map.latLngToLayerPoint(new L.LatLng(lat, lon));
		key = [key.x, key.y];
		var insert = !(key in visibleMarkers);
		if (!insert){
			var sizeAndValue = getSizeAndValue(ok, malformed, total);
			var [staName_, netName_, staId_, lat_, lon_, dcId_, datacenter_, ok_, malformed_, total_] = visibleMarkers[key];
			var otherSizeAndValue = getSizeAndValue(ok_, malformed_, total_);
			if(sizeAndValue[1] > otherSizeAndValue[1]){
				insert = true;
			}
		}
		if(insert){
			visibleMarkers[key] = [staName, netName, staId, lat, lon, dcId, datacenters[dcId], ok, malformed, total];
		}
	}
	
	// now display the markers. We delete the layer and re-create the markers, which are only those
	// worth to be displayed. First use an image cache to store the leaflet icons (L.icon):
	var allMarkers = [];
//	var imgCache = GLOBALS.imgCache;
//	if (!imgCache){
//		GLOBALS.imgCache = imgCache = {};  // [size, val] -> svg image in bytes
//	}

	// and now create the markers, using the cache to set the already calculated leaflet icons, if any:
	for (key in visibleMarkers){
		var [staName, netName, staId, lat, lon, dcId, datacenter, ok, malformed, total] = visibleMarkers[key];
		var circle = createMarker(staName, netName, staId, lat, lon, dcId, datacenter, ok, malformed, total, map); //.addTo(map);
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
	// set the mapLayer so that we will know what to clear the next time we call this method:
	GLOBALS.mapLayer = new L.featureGroup(allMarkers).addTo(map);
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

function createMarker(staName, netName, staId, lat, lon, dcId, datacenter, ok, malformed, total, map){
	// creates the marker for a given set of stations attributes
	var [size, val] = getSizeAndValue(ok, malformed, total);
	
	

		var greenBlue = 255 - val;

		// copied and modified from https://groups.google.com/forum/#!topic/leaflet-js/GSisdUm5rEc
		var h = size*1.7320508/2;  // height of a triangular equilateral
		var x = size/2.0;
		var y = 2*h/3.0;

		var pt = map.latLngToLayerPoint(new L.LatLng(lat, lon));
		var latlng = map.layerPointToLatLng(new L.Point(pt.x+x, pt.y-y));
		
		var xx = Math.abs(latlng.lng - lon);
		var yy = Math.abs(latlng.lat - lat);
		
//		// here's the SVG for the marker
//		var icon = `<svg class='svg-tirangle' xmlns='http://www.w3.org/2000/svg' version='1.1' width='${size}' height='${h}'>
//			<polygon points='0,${h} ${x},0 ${size},${h}' style='fill:rgb(255, ${greenBlue}, ${greenBlue});stroke:#333;stroke-width:1' />
//			</svg>`;
//		// here's the trick, base64 encode the URL:
//	    var svgURL = "data:image/svg+xml;base64," + btoa(icon);
//	    // create icon
//	    var mySVGIcon = new L.Icon( {
//	        // html: icon,
//	        // className: 'tri-div',
//	    	iconUrl: svgURL,
//	        iconSize: [size, h],
//	        iconAnchor: [x, y],
//	        popupAnchor: [0, 0]
//	    });
//	    imgCache[[size,val]] = mySVGIcon;
	
    // zIndexOffset is an OFFSET. If made in thousands it basically works as a zIndex
	// (see last post here: https://github.com/Leaflet/Leaflet/issues/5560):
    var zIndexOffset = (val > 0 ? 1000 : 0) + size;
//    var tri =  L.marker( [ lat, lon ], { icon: mySVGIcon,
//	    								 zIndexOffset: zIndexOffset
//	    								 // you can put whatever option <n> here and later access it with marker.options.<n>
//    } );
    var latlngs = [[lat-yy/2, lon-xx],[lat+yy, lon], [lat-yy/2, lon+xx]];
    var tri = L.polygon(latlngs, {fillOpacity: 0.8, color: '#333', fillColor:`rgb(255, ${greenBlue}, ${greenBlue})`,
    	weight:1}).addTo(map);
	
	//bind popup with infos:
	var staPopupContent = `<div class='title'> ${staName}.${netName} </div>
						   <div class='subtitle underline'>${datacenter}</div>
						   <table class='station-info'>
						   <tr><td>database id:</td><td class='right'>${staId}</td></tr>
						   <tr><td>Segments:</td><td class='right'> ${total} </td></tr>
						   <tr><td class='right'>In selected categories:</td><td class='right'> ${ok} </td></tr>
						   <tr><td class='right'>Not in selected categories:</td><td class='right'> ${malformed} </td></tr>
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
	var sizeRadius = 5; // for the biggest case (>= than 1000 segments)
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