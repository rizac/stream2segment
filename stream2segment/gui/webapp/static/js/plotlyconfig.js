var plots = (function(){
	var tSeriesLayout = { //https://plot.ly/javascript/axes/
			margin:{'l':50, 't':30, 'b':40, 'r':15},
//			autosize: false,
			xaxis: {
				autorange: true,
				tickangle: 0,
				type: 'date',
				titlefont: {
				      color: '#df2200'
				}
			},
			yaxis: {
				autorange: true,
				//fixedrange: true
			},
			annotations: [{
			    xref: 'paper',
			    yref: 'paper',
			    x: 0,
			    xanchor: 'left',
			    y: 1,
			    yanchor: 'bottom',
			    text: '',
			    showarrow: false,
			    //bordercolor: '#c7c7c7',
			    //borderwidth: 2,
			    borderpad: 5,
			    bgcolor: 'rgba(31, 119, 180, .1)',  // = '#1f77b4',
			    // opacity: 0.1,
			    font: {
			        // family: 'Courier New, monospace',
			        // size: 16,
			        color: '#000000'
			      },
			  }]
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
		plotType = div.getAttribute('data-plot');
		//COPY OBJECTS!!! 
		layout = JSON.parse(JSON.stringify(plotTypes[plotType]));
		Plotly.newPlot(div, [{x0:0, dx:1, y:[0], type:'scatter', 'opacity': 0}], layout);
		var idx = parseInt(div.getAttribute('data-plotindex'));
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
	
	return plots;
	
	
})();
