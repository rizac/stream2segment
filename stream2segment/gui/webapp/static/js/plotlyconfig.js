var plots = (function(){
	var tSeriesLayout = { //https://plot.ly/javascript/axes/
			margin:{'l':50, 't':35, 'b':70, 'r':18},
			pad: 0,
//			autosize: false,
//			margin:{'l':60, 't':100, 'b':80, 'r':5},
			xaxis: {
				autorange: true,
				tickangle: 0,
				type: 'date',
				titlefont: {
				      color: '#df2200'
				},
				linecolor: '#ddd',
				linewidth: 1,
				mirror: true
			},
			yaxis: {
				autorange: true,
				linecolor: '#ddd',
			    linewidth: 1,
			    mirror: true
				//fixedrange: true
			},
			annotations: [{
			    xref: 'paper',
			    yref: 'paper',
			    x: 0.01,
			    xanchor: 'left',
			    y: .98,
			    yanchor: 'bottom',
			    text: '',
			    showarrow: false,
			    bordercolor: '#ddd', // '#c7c7c7',
			    borderwidth: 1,
			    borderpad: 5,
			    bgcolor: '#f5f5f5', //  'rgba(31, 119, 180, .1)',  // = '#1f77b4',
			    // opacity: 0.1,
			    font: {
			        // family: 'Courier New, monospace',
			        // size: 16,
			        color: '#000000'
			      },
			  }],
//			plot_bgcolor: '#444',
//			paper_bgcolor: '#eee'
		};
	
	// create a deep copy (assuming we have simple dict-like objects
	// see http://stackoverflow.com/questions/728360/how-do-i-correctly-clone-a-javascript-object
	var fftLayout = JSON.parse(JSON.stringify(tSeriesLayout));
	fftLayout.xaxis.type = 'log';
	fftLayout.yaxis.type = 'log';
	
	var plotTypes = {'freq-series': fftLayout, 'time-series': tSeriesLayout};
	
	var plotNodeList = document.querySelectorAll('[data-plot]');
	
	
	
	
	// plotDivs is a node list, thus map must be returned as this:
	var plots = Array.prototype.map.call(plotNodeList, function (div, index) {
		var idx = parseInt(div.getAttribute('data-plotindex'));
		div.setAttribute("id", "plot" + idx);
		plotType = div.getAttribute('data-plot');
		//COPY OBJECTS!!! 
		layout = JSON.parse(JSON.stringify(plotTypes[plotType]));		
		Plotly.plot(div, [{x0:0, dx:1, y:[0], type:'scatter', 'opacity': 0}], layout, {displaylogo: false, showLink: false});
		
		/*
		 * https://plot.ly/javascript/responsive-fluid-layout/
		 * Try to make auto-resize, but 
		 *  this raises an error for two plots saying that they are not plotly divs (WTF?!!!) */
		
		/* window.addEventListener('resize', function() {
			Plotly.Plots.resize(div);
		});
		*/
		
		return {
			'visible': idx < 4,
			'div': div,
			'zoom': [null, null],
			'type': plotType,
			'index': idx
		};
	   
	} );
	
	// sort by plot-index attribute
	plots.sort(function(a,b){
		var a1= a.div.getAttribute('data-plotindex');
		var b1= b.div.getAttribute('data-plotindex');
		return parseInt(a1) - parseInt(b1);
	});
	
	// this forwards the plotly 'on' method which sets a function(eventdata) on the div
	// to a custom callback which is executed with the plots element, the index and the eventData
	plots.on = function(key, callback){
		plots.forEach(function(elm, index, elms){
			elm.div.on(key, function(eventData){
				callback(elm, index, elms, eventData);
			});
		});
	};
	
	return plots;
	
	
})();
