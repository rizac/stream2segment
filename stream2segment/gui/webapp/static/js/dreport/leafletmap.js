var map = new L.Map('map');

var legend = L.control({position: 'bottomleft'});
legend.onAdd = function (map) {

    var div = L.DomUtil.create('div', 'info-legend');
    var grades = [0, .1, .20, .30, .40, .50, .60, .70, .80, .90, 1];
    
    div.innerHTML = "SDA: 0%";
    // loop through our density intervals and generate a label with a colored square for each interval
    for (var i = 0; i < grades.length; i++) {
    	var gbLevel = Math.floor(255 - 255*grades[i]);
    	var color = "rgb(255, " + gbLevel + ", " + gbLevel +")";
        div.innerHTML +=
            "<span class='cmap-chunk' style='background-color:" + color + "'></span>";
    }
    div.innerHTML += "100%";
    return div;
};

legend.addTo(map);

// store baseLayer and control (clickable legend with the data-centers) in global vars
// maybe not ideal, but we did not find out a better way. See updateMap for info on this object
var _leafletData = null;

function updateMap(data, selLabels){
	/**
	 * Updates the map with the given data and the given selected labels
	 * If the map is already in-place, avoids re-creating layers
	 * but remove its elements (stations) and adds them again. 
	 * 
	 * :param data: a list of lists, as received from the server. Each element is a station, each
	 * station element is its data (lat, lon, datacenter, number of segments falling in any of the selLabels categories, ...)
	 * :param selLabels: an array of arrays e1, ..., eN: each element ei is an array holding: the label name,
	 * the label selection state (ture, false), and the number of segments falling in this category identified by the label
	 */
	var fitBounds = false;
	if(!_leafletData){
		L.esri.basemapLayer("Oceans").addTo(map);
		L.esri.basemapLayer("OceansLabels").addTo(map).setZIndex(-1);
//		_baseLayers = {"Oceans": _defaultLayer, 
//                	   "Streets": L.esri.basemapLayer("Streets"), 
//                       "Topographic": L.esri.basemapLayer("Topographic"), 
//                       "ShadedRelief": L.esri.basemapLayer("ShadedReliefLabels"), 
//                       "Terrain": L.esri.basemapLayer("TerrainLabels"),
//                       "Imagery": L.esri.basemapLayer("ImageryLabels")};
		
//		_leafletData = {'baseLayers': [L.esri.basemapLayer("Streets").addTo(map), 
//		                               L.esri.basemapLayer("Oceans").addTo(map), 
//		                               L.esri.basemapLayer("Topographic").addTo(map), 
//		                               L.esri.basemapLayer("ShadedRelief").addTo(map), 
//		                               L.esri.basemapLayer("Terrain").addTo(map),
//		                               L.esri.basemapLayer("Oceans").addTo(map)],
//		                               // the base layer (not used for the moment)
		_leafletData = {'control': L.control.layers().addTo(map),  // the control (legend with datacenters checkboxes)
						'dcLayerGroup': {}  // a dict of datacenters names (or id) strings mapped to the relativel
						//leaflet LayerGroup object
		};
		fitBounds = true;
	}
	
	var compute = function(array){
		/**
		 * returns a dict representing array, which is an element of the data argument passed to this function
		 */
		var ret = {'total':0, 'malformed':0};
		array.forEach(function(value, index){
			if (index == 0){
				ret.dcurl = value;
			}else if (index == 1){
				ret.staid = value;
			}else if(index == 2){
				ret.lat = value;
			}else if(index == 3){
				ret.lon = value;
			}else if(index == 4){
				ret.total = value;
			}else if(selLabels[index-5][1]){  // count it as malformed
				ret.malformed += value;
			}
		});
		return ret;
	}

	var dcens = {}; //stores layers to checkbox stations of a single datacenter
	var fitBoundMarkers = [];  // used only for zoom. It's a collection of ALL markers we will create
	var floor = Math.floor; //this actually returns an int (at least, something whose str value is with no dots)s
	data.forEach(function(array, index){
		var dic = compute(array);
		
		var lat = dic.lat;
		var lon = dic.lon;
		var total = dic.total;
		var dcen = dic.dcurl;
		var malformed = dic.malformed;
		var ok = total - malformed;
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

		circle.bindPopup("");
		
		// we bind popup content on mouse click (defer for speed). We delegate the angular scope
		// for that, as we will handle therein the post request to get station info
		circle.on('click', function(){
			getScope().setPopupContent(dic.staid, circle);
			circle.openPopup();
		});
		
		dc.markers.push(circle);
		//if (fitBounds){
			fitBoundMarkers.push(circle);
		//}
	});

	var layerControl = _leafletData['control'];
	var dcLayerGroup = _leafletData['dcLayerGroup'];
	var doc = document;
	var DCID_PREFIX = "_dcen_";
	for (var dcen in dcens){
		var val = dcens[dcen];
		var title = dcen +  "  - Ok: " + val.ok + " of " + val.total + " (" + Math.round((100*val.ok)/val.total)+ "%)"; 
		if (dcen in dcLayerGroup){
			var layerGroup = dcLayerGroup[dcen];
			layerGroup.clearLayers(); // clear all markers of the layer group (clearLayers is misleading)
			val.markers.forEach(function(elm){layerGroup.addLayer(elm);});
			// update title (text) for the relative control with the new percentages:
			doc.getElementById(DCID_PREFIX+dcen).innerHTML = title;  // here we should set the new title
		}else{
			layerGroup = L.layerGroup(val.markers).addTo(map);
			//this is a way to retrieve the object associated with the control's datacenter checkbox woithout
			// storing it in any variable: add HTML instead of plain text:
			layerControl.addOverlay(layerGroup, "<span class='btn btn-default' id='"+DCID_PREFIX+dcen+"'>" +  title + "</span>");
			dcLayerGroup[dcen] = layerGroup;
		}
	}
	
	fitBoundMarkers.forEach(function(elm){elm.bringToFront();});
	
	// fit bounds and set stuff only if initializing:
	if(fitBounds){
		// https://stackoverflow.com/questions/16845614/zoom-to-fit-all-markers-in-mapbox-or-leaflet
		var group = new L.featureGroup(fitBoundMarkers);
		map.fitBounds(group.getBounds());
	}
}