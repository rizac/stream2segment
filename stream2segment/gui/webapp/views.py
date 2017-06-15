'''
Created on Jun 20, 2016

@author: riccardo
'''
from flask import render_template, request, jsonify, Blueprint, current_app
# from stream2segment.gui.webapp.plots import user_defined_plots, View
from stream2segment.gui.webapp import core
from stream2segment.utils import secure_dburl
from stream2segment.gui.webapp.core import set_classes
# from stream2segment.gui.webapp.plotviews import PlotManager

# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
main_page = Blueprint('main_page', __name__, template_folder='templates')


@main_page.route("/")
def main():
    config = current_app.config['CONFIG.YAML']
    set_classes(config)
    ud_plotnames = current_app.config['PLOTMANAGER'].userdefined_plotnames
    keys = ['spectra', 'segment_select', 'segment_orderby']
    settings = {k: config[k] for k in keys}
    return render_template('index.html', title=secure_dburl(current_app.config["DATABASE"]),
                           settings=settings,
                           userDefinedPlots=ud_plotnames)


@main_page.route("/get_segments", methods=['POST'])
def init():
    data = request.get_json()
    dic = core.get_segments(data.get('segment_select', None),
                            data.get('segment_orderby', None),
                            data.get('metadata', True),
                            data.get('classes', False))
    return jsonify(dic)


@main_page.route("/get_segment", methods=['POST'])
def get_segment_data():
    '''view returning the response for the segment data (and/or metadata)'''
    data = request.get_json()
    seg_id = data['seg_id']  # this must be present
    plot_indices = data.get('plot_indices', [])
    filtered = data.get('filtered', False)
    zooms = data.get('zooms', None)
    all_components = data.get('all_components', False)
    metadata = data.get('metadata', False)
    classes = data.get('classes', False)
#     conf = data.get('config', {})
    plotmanager = current_app.config['PLOTMANAGER']
#     if conf:
#         current_app.config['CONFIG.YAML'].update(conf)  # updates also plotmanager
    # NOTE: seg_id is a unicode string, but the query to the db works as well
    return jsonify(core.get_segment_data(seg_id, plotmanager, plot_indices, all_components,
                                         filtered, zooms, metadata, classes))


# @main_page.route("/get_segment_plots", methods=['POST'])
# def get_segment_plots():
#     data = request.get_json()
#     seg_id = data['segId']
#     rem_resp_filtered = data['filteredRemResp']
#     zooms = data['zooms']
#     all_components = data['allComponents']
#     # NOTE: seg_id is a unicode string, but the query to the db works as well
#     return jsonify(core.get_segment_data(seg_id, all_components, rem_resp_filtered, zooms))


# @main_page.route("/select_segments", methods=['POST'])
# def select_segments():
#     data = request.get_json()
#     # NOTE: seg_id is a unicode string, but the query to the db works as well
#     return jsonify(core.get_segment_ids(data['selection'], data.get('order-by', None),
#                                         data.get('with-data', True)))


@main_page.route("/toggle_class_id", methods=['POST'])
def toggle_class_id():
    json_req = request.get_json()
    return jsonify(core.toggle_class_id(json_req['segment_id'], json_req['class_id']))


# @main_page.route("/config_spectra", methods=['POST'])
# def config_spectra():
#     return jsonify(core.config_spectra(request.get_json()))
