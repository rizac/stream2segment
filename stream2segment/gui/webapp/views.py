'''
Created on Jun 20, 2016

@author: riccardo
'''

# from stream2segment.gui.webapp import app
from stream2segment.gui.webapp import core
from flask import render_template, request, jsonify
from stream2segment.gui.webapp.plots import get_num_custom_plots
from flask import Blueprint
# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
from flask import current_app
main_page = Blueprint('main_page', __name__, template_folder='templates')


@main_page.route("/")
def main():
    ncp = get_num_custom_plots()
    return render_template('index.html', title=current_app.config["DATABASE"],
                           customPlots=range(ncp))  # app.config['DB_URI'])


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
