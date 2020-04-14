'''
Views for the web app (processing)

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from flask import render_template, request, jsonify, Blueprint, current_app
from io import StringIO

import yaml

from stream2segment.gui.webapp import get_session
from stream2segment.gui.webapp.mainapp import core
from stream2segment.utils import secure_dburl
from stream2segment.process.db import configure_classes


# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
main_app = Blueprint('main_app', __name__, template_folder='templates')


@main_app.route("/")
def main():
    plotmanager = core.create_plot_manager(current_app.config['pyfile'],
                                           current_app.config['configfile'])
    config = plotmanager.config or {}
    configure_classes(get_session(current_app), config.get('class_labels', []))
    ud_plots = plotmanager.userdefined_plots
    settings = {'hasPreprocessFunc': plotmanager.has_preprocessfunc}
    # pop keys not to be shown in the gui config form (either already processed, or not
    # regarding plot settings:
    config.pop('class_labels', None)
    # filterfunc_doc = current_app.config['PLOTMANAGER'].get_filterfunc_doc.replace("\n", "<p>")
    return render_template('mainapp.html', title=secure_dburl(current_app.config["DATABASE"]),
                           settings=settings,
                           rightPlots=[_ for _ in ud_plots if _['position'] == 'r'],
                           bottomPlots=[_ for _ in ud_plots if _['position'] == 'b'],
                           preprocessfunc_doc=plotmanager.get_doc(-1))


@main_app.route("/init", methods=['POST'])
def init():
    data = request.get_json()
    # Note: data.get('segment_orderby', None) is not anymore implemented in the config
    # it will default to None (order by event time desending and by event_distance ascending)
    dic = core.init(get_session(current_app),
                    data.get('segment_orderby', None),
                    data.get('metadata', False),
                    data.get('classes', False))
    return jsonify(dic)


@main_app.route("/get_config", methods=['POST'])
def get_config():
    asstr = (request.get_json() or {}).get('asstr', False)
    try:
        return jsonify({'error_msg': '', 'data': core.get_config(asstr)})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'error_msg': str(exc), 'data': {}})


@main_app.route("/validate_config_str", methods=['POST'])
def validate_config_str():
    data = request.get_json()
    try:
        return jsonify({'error_msg': '', 'data': core.validate_config_str(data['data'])})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'error_msg': str(exc), 'data': {}})


@main_app.route("/set_selection", methods=['POST'])
def set_selection():
    try:
        data = request.get_json()
        writeback = True
        segselect = data.get('segment_select', None)
        # segselect is a dict, or None
        if segselect is None:
            writeback = False
            segselect = core.get_segment_select()

        num_segments = core.get_segments_count(get_session(current_app),
                                               segselect)
        # set the plot manager new condition at the end (if provided): in case of errors
        # (e.g. overflow integer in 'id') we do not change the previous selection condition
        # with a erroneous one
        if writeback:
            core.get_plot_manager().config['segment_select'] = segselect
        return jsonify({'num_segments': num_segments, 'error_msg': ''})

    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'num_segments': 0, 'error_msg': str(exc)})


@main_app.route("/get_segment", methods=['POST'])
def get_segment_data():
    '''view returning the response for the segment data (and/or metadata)'''
    data = request.get_json()
    seg_index = data['seg_index']  # this must be present
    seg_id = core.get_segment_id(get_session(current_app), seg_index)
    plot_indices = data.get('plot_indices', [])
    preprocessed = data.get('pre_processed', False)
    zooms = data.get('zooms', None)
    all_components = data.get('all_components', False)
    metadata = data.get('metadata', False)
    classes = data.get('classes', False)
    config = data.get('config', {})
    return jsonify(core.get_segment_data(get_session(current_app), seg_id,
                                         core.get_plot_manager(),
                                         plot_indices, all_components, preprocessed, zooms,
                                         metadata, classes, config))


@main_app.route("/set_class_id", methods=['POST'])
def set_class_id():
    json_req = request.get_json()
    core.set_class_id(get_session(current_app), json_req['segment_id'], json_req['class_id'],
                      json_req['value'])
    # the above raises, otherwise return empty json to signal success:
    return jsonify({})
