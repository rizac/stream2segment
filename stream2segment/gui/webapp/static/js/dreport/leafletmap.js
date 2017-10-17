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

function updateMap(data, selLabels){
	
	var baseLayer = L.esri.basemapLayer("Oceans").addTo(map);
	var layerControl = L.control.layers().addTo(map);
	// start the map in South-East England
	//map.setView(new L.LatLng(51.3, 0.7),9);
	
	
	var compute = function(array){
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
	var markers = [];  // used only for zoom
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
		// var hasWE = (warns != undefined || errs != undefined) ? true : false;
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
		
		// we bind popup content on mouse click (defer for speed):
		circle.on('click', function(){
			getScope().setPopupContent(dic.staid, circle);
			circle.openPopup();
		});
		
		dc.markers.push(circle);
		markers.push(circle);
	});
	
	// we did not find any better solution than redrawing each layer
	map.eachLayer(function (layer) {
		if (layer == baseLayer){
			return;
		}
		map.removeLayer(layer);
		layerControl.removeLayer(layer);
	});

	for (var dcen in dcens){
		var val = dcens[dcen];
		var lg = L.layerGroup(val.markers);
		var title = dcen +  "  - Ok: " + val.ok + " of " + val.total + " (" + Math.round((100*val.ok)/val.total)+ "%)"; 
		//var title = dcen +  "  - total: " + val.total +", ok: " + val.ok +" (" + Math.round((100*val.ok)/val.total)+ "%)"; 
		layerControl.addOverlay(lg, title);
		lg.addTo(map);
	}
	
	// https://stackoverflow.com/questions/16845614/zoom-to-fit-all-markers-in-mapbox-or-leaflet
	var group = new L.featureGroup(markers);
	map.fitBounds(group.getBounds());
	
	/*
	map.fitBounds([
    [44.11217, 0.3363],
    [59.42283, 20.267]
	]);

	
	function unpack(station, errors){
		// errors is eiother ERRORS or WARNINGS, both dicts keyed with station names
		var element = errors[station];
		// element is a dict (object) with numeric keys mapped to arrays denoting the mseeds
		// concerned. The keys are numeric for performance reasons (loading and creating smaller htmls)
		// the error/warning name is columns[key], the mseed concerned are element[key]
		
		var columns = DATA.columns;
		//unpacks warnings / errors returning [num_errors_warnings, div_content]
		var ret = "<table>";
		var count = 0;
		for (var key in element){
			var mseeds = element[key];
			count += mseeds.length;
			ret += "<tr><td colspan=2>" + columns[key] + "(" + mseeds.length +" mseeds)</td></tr>";
			mseeds.forEach(function(elm, index){
				// each element is an array of:
				// [location, channel, start date, start time, end date, end time, db id]
			    // this is optimized in that end date might be empty (same as start date)
			  	var ch_id = station + "." + elm[0] + "." + elm[1];
			  	var timerange = elm[4] ? elm[2] + " " + elm[3] + " to " + elm[4] + " " + elm[5] : elm[2] + ", " + elm[3] + " to " + elm[5];
			  	var dbId = elm[6];
				ret += "<tr><td style='text-align:right'>"+ (index+1) +"</td><td>" + ch_id + " [" + timerange + "] (db id=" + dbId + ")</td></tr>";
			});
			ret += "";
		}
		if (count == 0){
			ret = "none";
		}else{
			ret +="</table>";
		}
		return [count, ret];
	};
	
	function getPopupContent(stationIndex){
		var d = DATA;
		var station = d.index[stationIndex];
		var vals = d.data[stationIndex];
		var lat = vals[0];
		var lon = vals[1];
		var total = vals[2];
		var dc_ = DATACENTERS;
		var dcen = dc_[vals[3]];

		var w_ = WARNINGS;
		var e_ = ERRORS;
		var warnings_ = unpack(station, w_);
		var errors_ = unpack(station, e_);
		
		var infostr = "<table class='info'>"+
			"<tr><td>Station:</td><td>" + station + "</td></tr>" +
			"<tr class='border-top'><td>lat</td><td>"+ lat + "</td></tr>"+
			"<tr><td>lon</td><td>"+ lon + "</td></tr>" +
			"<tr><td>d.c.</td><td>"+ dcen + "</td></tr>" +
			"<tr class='border-top'><td>miniSeeds</td><td>" + total + "</td></tr>" +
			//"<tr><td>ok</td><td>" + ok + "</td></tr>" +
			"<tr><td>errors</td><td>" + errors_[0] + "</td></tr>" +
			"<tr><td>warnings</td><td>" + warnings_[0] + "</td></tr>" +
			"</table>";
		
		infostr += "<div class='info'><div style='color:#e40000'>errors:</div>" + errors_[1] + "</div>";
		infostr += "<div class='info'><div style='color:#0000e4'>warnings:</div>" + warnings_[1] + "</div>";
		return infostr;
	};

	function drawMap(){
		var domLevels = document.querySelectorAll("input[type=checkbox].warnerr");
		var levels = {};
		for (var i=0; i < domLevels.length; i++){
			levels[domLevels[i].getAttribute('data-type')] = domLevels[i].checked;
		}
		
		var d = DATA;
		var cols = d.columns;
		
		var dc_ = DATACENTERS;
		var dcens = {}; //stores layers to checkbox stations of a single datacenter
		//var we = {}; //stores layers to checkbox stations with given error/warning
		var _MARKERS = window._MARKERS || [];
		var isFirstDraw = window._MARKERS ? false : true;
		var toInt = parseInt;  // allocate once (speeds up?)
		d.index.forEach(function(station, index){
			var vals = d.data[index];
			var lat = vals[0];
			var lon = vals[1];
			var total = vals[2];
			var dcen = dc_[vals[3]];
			var malformed = 0;
			// calculate ok's based on the selected levels:
			for (var i=4; i < DATA.columns.length; i++){
				if (!levels[DATA.columns[i]]){
					continue;
				}
				malformed += vals[i];
			}
			var ok = total - malformed;
			if (! (dcen in dcens)){
				dcens[dcen] = {'markers': [], 'total':0, 'ok':0};
			}
			var dc = dcens[dcen];
			dc.total += total;
			dc.ok += ok;
			var gb = toInt(0.5 + 255 * (1 - (total == 0 ? 0 : ok/total)));
			//console.log(gb);
			
			var fillColor = "rgb(255, " + gb + "," + gb +")";
			// var hasWE = (warns != undefined || errs != undefined) ? true : false;
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
			
			// we bind popup content on mouse click (defer for speed):
			circle.on('click', function(){
				circle.setPopupContent(getPopupContent(index));
				//circle.openPopup();
			});
			
			dc.markers.push(circle);
		});
		
		// we did not find any better solution than redrawing each layer
		map.eachLayer(function (layer) {
			if (layer == baseLayer){
				return;
			}
   		 	map.removeLayer(layer);
   		 	layerControl.removeLayer(layer);
		});

		for (var dcen in dcens){
			var val = dcens[dcen];
			var lg = L.layerGroup(val.markers);
			var title = dcen + "  - total: " + val.total +", ok: " + val.ok +" (" + Math.round((100*val.ok)/val.total)+ "%)"; 
			layerControl.addOverlay(lg, title);
			lg.addTo(map);
		}
	
	};	

	var DATA = {{stations}};
	var WARNINGS = {{warnings}};
	var ERRORS = {{errors}};
	var DATACENTERS = {{datacenters}};
	
	var DATA = {{stations}};  //dict of {dc_id: sta_id: [[seg_id, download_s_code, max_gap_o_ratio, sample_rate!=cha_s_rate]
	var WARNINGS = {{warnings}};
	var ERRORS = {{errors}};
	var DATACENTERS = {{datacenters}};

	drawMap();*/
}