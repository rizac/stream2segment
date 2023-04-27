"""
Views for the web app (processing)

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from collections import defaultdict
import json

from flask import (render_template, request, jsonify, Blueprint)

from stream2segment.process.gui.webapp.mainapp import core

# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
# Note that the template_folder and the static paths in the HTML are relative
# to the path of the module WHERE we register this blueprint
# (stream2segment.gui.main)


main_app = Blueprint('main_app', __name__, template_folder='templates')


@main_app.route("/")
def main():
    ud_plots = core.userdefined_plots
    data = core.get_init_data(metadata=True, classes=True)
    classes = data['classes']
    metadata = data['metadata']
    r_plots = [{**p, 'name': n} for n, p in ud_plots.items() if p['position'] == 'r']
    b_plots = [{**p, 'name': n} for n, p in ud_plots.items() if p['position'] == 'b']
    return render_template('mainapp.html',
                           num_segments=len(core.g_segment_ids),
                           title=core.get_db_url(safe=True),
                           rightPlots=r_plots,
                           bottomPlots=b_plots,
                           metadata=metadata,
                           classes=classes,
                           preprocess_func_on=False,
                           preprocessfunc_doc=core.get_func_doc(-1))


@main_app.route("/get_config", methods=['POST'])
def get_config():
    asstr = (request.get_json() or {}).get('as_str', False)
    return jsonify(core.get_config(asstr))


@main_app.route("/get_selection", methods=['POST'])
def get_selection():
    return jsonify(core.get_select_conditions())


@main_app.route("/validate_config_str", methods=['POST'])
def validate_config_str():
    data = request.get_json()
    return jsonify({'data': core.validate_config_str(data['data'])})


@main_app.route("/set_selection", methods=['POST'])
def set_selection():
    sel_conditions = request.get_json() or None
    # sel condition = None: do not update conditions but use already loaded one
    if sel_conditions:
        # remove space-only and empty strings in expressions:
        sel_conditions = {k: v for k, v in sel_conditions.items() if v and v.strip()}
    num_segments = core.set_select_conditions(sel_conditions)
    return jsonify({'num_segments': num_segments, 'error_msg': ''})


@main_app.route("/get_segment_data", methods=['POST'])
def get_segment_data():
    """Return the response for the segment data (and/or metadata)"""
    data = request.get_json()
    seg_index = data['seg_index']
    seg_count = data['seg_count']
    seg_id = core.get_segment_id(seg_index, seg_count)
    plot_names = data.get('plot_names', {})
    preprocessed = data.get('pre_processed', False)
    zooms = data.get('zooms', None)
    all_components = data.get('all_components', False)
    attributes = data.get('attributes', False)
    classes = data.get('classes', False)
    config = data.get('config', {})
    return jsonify(core.get_segment_data(seg_id,
                                         plot_names, all_components,
                                         preprocessed, zooms,
                                         attributes, classes, config))


@main_app.route("/set_class_id", methods=['POST'])
def set_class_id():
    data = request.get_json()
    seg_index = data['seg_index']
    seg_count = data['seg_count']
    seg_id = core.get_segment_id(seg_index, seg_count)
    core.set_class_id(seg_id, data['class_id'], data['value'])
    # the above raises, otherwise return empty json to signal success:
    return jsonify({})
