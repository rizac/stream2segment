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
    sel_conditions = core.get_select_conditions()
    # build metadata (data['metadata'] but with customizing sorting):
    metadata_dict = defaultdict(list)
    for sel_key, sel_type in data['metadata']:
        related_obj_name = "" if '.' not in sel_key else sel_key.split('.')[0]
        # append each metadata element as list of 4 elements:
        # attribute name, type, selection expression, css style for the relative TR
        metadata_dict[related_obj_name].append([sel_key, sel_type,
                                                sel_conditions.get(sel_key, ""),
                                               ""])
    # sort with custom logic: first Segment attributes, then event's, station's ...
    sorted_keys = ["", "event", "station"]
    sorted_keys += sorted(k for k in metadata_dict if k not in sorted_keys)
    # now build the metadata list. Each list element is a list of 4 elements:
    # attribute name, type, selection expression, css style for the relative TR
    metadata = []
    for key in sorted_keys:
        metadata_chunk = sorted(metadata_dict[key], key=lambda k: k[0])
        # add css style (to the first row only):
        metadata_chunk[0][-1] = 'border-top: 2px solid lightgray'
        metadata.extend(metadata_chunk)

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
                           config_text_json=json.dumps(core.get_config(True)),
                           preprocessfunc_doc=core.get_func_doc(-1))


# @main_app.route("/init", methods=['POST'])
# def init():
#     data = request.get_json()
#     dic = core.get_init_data(data.get('metadata', False),
#                              data.get('classes', False))
#     return jsonify(dic)


# @main_app.route("/get_config", methods=['POST'])
# def get_config():
#     asstr = (request.get_json() or {}).get('asstr', False)
#     try:
#         return jsonify({'error_msg': '', 'data': core.get_config(asstr)})
#     except Exception as exc:  # pylint: disable=broad-except
#         return jsonify({'error_msg': str(exc), 'data': {}})


@main_app.route("/get_selection", methods=['POST'])
def get_selection():
    try:
        return jsonify({'error_msg': '', 'data': core.get_select_conditions()})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'error_msg': str(exc), 'data': {}})


@main_app.route("/validate_config_str", methods=['POST'])
def validate_config_str():
    data = request.get_json()
    try:
        return jsonify({'error_msg': '',
                        'data': core.validate_config_str(data['data'])})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'error_msg': str(exc), 'data': {}})


@main_app.route("/set_selection", methods=['POST'])
def set_selection():
    try:
        data = request.get_json()
        sel_conditions = data.get('segments_selection', None)
        # sel condition = None: do not update conditions but use already loaded one
        num_segments = core.set_select_conditions(sel_conditions)
        return jsonify({'num_segments': num_segments, 'error_msg': ''})

    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'num_segments': 0, 'error_msg': str(exc)})


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
    metadata = data.get('metadata', False)
    classes = data.get('classes', False)
    config = data.get('config', {})
    return jsonify(core.get_segment_data(seg_id,
                                         plot_names, all_components,
                                         preprocessed, zooms,
                                         metadata, classes, config))


@main_app.route("/set_class_id", methods=['POST'])
def set_class_id():
    data = request.get_json()
    seg_index = data['seg_index']
    seg_count = data['seg_count']
    seg_id = core.get_segment_id(seg_index, seg_count)
    core.set_class_id(seg_id, data['class_id'], data['value'])
    # the above raises, otherwise return empty json to signal success:
    return jsonify({})
