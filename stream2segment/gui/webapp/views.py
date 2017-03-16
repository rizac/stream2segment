'''
Created on Jun 20, 2016

@author: riccardo
'''
from flask import render_template, request, jsonify, Blueprint, current_app
from stream2segment.gui.webapp.plots import user_defined_plots, View
from stream2segment.gui.webapp import core
from stream2segment.utils import secure_dburl

# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
main_page = Blueprint('main_page', __name__, template_folder='templates')


@main_page.route("/")
def main():
    return render_template('index.html', title=secure_dburl(current_app.config["DATABASE"]),
                           settings=View.settings,
                           userDefinedPlots=user_defined_plots())  # app.config['DB_URI'])


@main_page.route("/init", methods=['POST'])
def init():
    data = request.get_json()
    dic = core.get_init_data(data.get('order-by', None), data.get('with-data', True))
    return jsonify(dic)


@main_page.route("/get_segment_data", methods=['POST'])
def get_segment_data():
    data = request.get_json()
    seg_id = data['segId']
    rem_resp_filtered = data['filteredRemResp']
    zooms = data['zooms']
    indices = data['plotIndices']
    metadata_keys = data['metadataKeys']

    # NOTE: seg_id is a unicode string, but the query to the db works as well
    return jsonify(core.get_segment_data(seg_id, rem_resp_filtered, zooms, indices, metadata_keys))


@main_page.route("/get_segment_plots", methods=['POST'])
def get_segment_plots():
    data = request.get_json()
    seg_id = data['segId']
    rem_resp_filtered = data['filteredRemResp']
    zooms = data['zooms']
    # NOTE: seg_id is a unicode string, but the query to the db works as well
    return jsonify(core.get_segment_data(seg_id, rem_resp_filtered, zooms))


@main_page.route("/select_segments", methods=['POST'])
def select_segments():
    data = request.get_json()
    # NOTE: seg_id is a unicode string, but the query to the db works as well
    return jsonify(core.get_segment_ids(data['selection'], data.get('order-by', None),
                                        data.get('with-data', True)))


@main_page.route("/toggle_class_id", methods=['POST'])
def toggle_class_id():
    json_req = request.get_json()
    return jsonify(core.toggle_class_id(json_req['segment_id'], json_req['class_id']))


@main_page.route("/config_spectra", methods=['POST'])
def config_spectra():
    return jsonify(core.config_spectra(request.get_json()))
