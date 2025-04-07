"""
Views for the web app (processing)

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from flask import (render_template, request, jsonify, Blueprint)
from werkzeug.exceptions import HTTPException

from stream2segment.process.gui.webapp.mainapp import core

# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
# Note that the template_folder and the static paths in the HTML are relative
# to the path of the module WHERE we register this blueprint
# (stream2segment.gui.main)


main_app = Blueprint('main_app', __name__, template_folder='templates')


@main_app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    import sys
    exc_i = sys.exc_info()
    return jsonify({
        'message': str(exc_i[1].__class__.__name__) + ": " + str(exc_i[1]),
        'traceback': ''  # do not rpovide it for the moment
    }), 500


@main_app.route("/")
def main():
    ud_plots = core.userdefined_plots
    data = core.get_init_data(metadata=True, classes=True)
    classes = data['classes']
    metadata = data['metadata']
    r_plots = [{**p, 'name': n} for n, p in ud_plots.items() if p['position'] == 'r']
    b_plots = [{**p, 'name': n} for n, p in ud_plots.items() if p['position'] == 'b']
    pp_func = core.get_preprocess_function()
    pp_func_doc = core.get_func_doc(pp_func)
    pp_func_defined = pp_func not in (core._default_preprocessfunc, None)
    return render_template('mainapp.html',
                           num_segments=len(core.g_segment_ids),
                           title=core.get_db_url(safe=True),
                           rightPlots=r_plots,
                           bottomPlots=b_plots,
                           metadata=metadata,
                           classes=classes,
                           preprocess_func_on=pp_func_defined,
                           preprocessfunc_doc=pp_func_doc)


@main_app.route("/get_config", methods=['POST'])
def get_config():
    asstr = (request.get_json() or {}).get('as_str', False)
    return jsonify(core.get_config(asstr))


@main_app.route("/get_selection", methods=['POST'])
def get_selection():
    return jsonify(core.get_select_conditions())


@main_app.route("/set_config", methods=['POST'])
def set_config():
    data = request.get_json()
    new_config = core.validate_config_str(data['data'])
    core.reset_global_vars(new_config, None)
    return jsonify(new_config)


@main_app.route("/set_selection", methods=['POST'])
def set_selection():
    sel_conditions = request.get_json() or None
    # sel condition = None: do not update conditions but use already loaded one
    if sel_conditions:
        # remove space-only and empty strings in expressions:
        sel_conditions = {k: v for k, v in sel_conditions.items() if v and v.strip()}
    num_segments = core.get_segments_count(sel_conditions)
    if num_segments < 1:
        raise ValueError('No segment matching the current selection')
    core.reset_global_vars(None, sel_conditions)
    core.reset_segment_ids_array(num_segments)
    return jsonify(num_segments)


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
    return jsonify(core.get_segment_data(seg_id,
                                         plot_names, all_components,
                                         preprocessed, zooms,
                                         attributes, classes))


@main_app.route("/set_class_id", methods=['POST'])
def set_class_id():
    data = request.get_json()
    seg_index = data['seg_index']
    seg_count = data['seg_count']
    seg_id = core.get_segment_id(seg_index, seg_count)
    return jsonify(core.set_class_id(seg_id, data['class_id'], data['value']))

