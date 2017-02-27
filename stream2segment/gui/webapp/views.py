'''
Created on Jun 20, 2016

@author: riccardo
'''

from stream2segment.gui.webapp import app
from stream2segment.gui.webapp import core
from flask import render_template, request, jsonify
from stream2segment.gui.webapp.core import get_num_custom_plots


@app.route("/")
def main():
    ncp =get_num_custom_plots()
    return render_template('index.html', title=app.config["DATABASE_URI"],
                           numCustomPlots=ncp,
                           customPlots=range(3, 3+ncp))  # app.config['DB_URI'])


@app.route("/get_elements", methods=['POST'])
def get_elements():
    session = core._get_session(app)
    dic = core.get_ids(session)
    return jsonify(dic)


@app.route("/get_classes", methods=['POST'])
def get_classes():
    session = core._get_session(app)
    return jsonify({'classes': core.get_classes(session)})


@app.route("/get_data", methods=['POST'])
def get_data():
    data = request.get_json()
    seg_id = data['segId']
    rem_resp_filtered = data['filteredRemResp']
    zooms = data['zooms']
    # NOTE: seg_id is a unicode string, but the query to the db works as well
    session = core._get_session(app)
    return jsonify(core.get_data(session, seg_id, rem_resp_filtered, zooms))
# 
# @app.route("/toggle_class_id", methods=['POST'])
# def toggle_class_id():
# #    db_uri = app.config['DATABASE_URI']
#     json_req = request.get_json()
# #     class_ids = [] if json_req is None else json_req.get('class_ids', [])
# #     if class_ids:
# #         class_ids = [int(c) for c in class_ids]
#     session = core._get_session(app)
#     return jsonify(core.toggle_class_id(session, json_req['segment_id'], json_req['class_id']))
