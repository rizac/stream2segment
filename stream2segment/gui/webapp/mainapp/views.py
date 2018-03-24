'''
Views for the web app (processing)

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from flask import render_template, request, jsonify, Blueprint, current_app

from stream2segment.gui.webapp import get_session
from stream2segment.gui.webapp.mainapp import core
from stream2segment.utils import secure_dburl
from stream2segment.process.utils import set_classes
import yaml
import json

# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
main_app = Blueprint('main_app', __name__, template_folder='templates')


@main_app.route("/")
def main():
    config = dict(current_app.config['CONFIG.YAML'])
    set_classes(get_session(current_app), config)
    plotmanager = current_app.config['PLOTMANAGER']
    ud_plots = plotmanager.userdefined_plots
    settings = {'segment_select': config.pop('segment_select', {})}
    # pop keys not to be shown in the gui config form (either already processed, or not
    # regarding plot settings:
    for key in ['class_labels', 'save_inventory']:
        config.pop(key, None)
    # create a flatten dict by joininf nested dict keys with the dot:
    settings['config'] = core.flatten_dict(config)
    preprocessfunc_doc = core.get_doc('preprocessfunc', plotmanager)
    segment_select_doc = core.get_doc('segment_select', plotmanager)
    # filterfunc_doc = current_app.config['PLOTMANAGER'].get_filterfunc_doc.replace("\n", "<p>")
    return render_template('mainapp.html', title=secure_dburl(current_app.config["DATABASE"]),
                           settings=settings,
                           rightPlots=[_ for _ in ud_plots if _['position'] == 'r'],
                           bottomPlots=[_ for _ in ud_plots if _['position'] == 'b'],
                           preprocessfunc_doc=preprocessfunc_doc,
                           segment_select_doc=segment_select_doc)


@main_app.route("/get_segments", methods=['POST'])
def init():
    data = request.get_json()
    # Note: data.get('segment_orderby', None) is not anymore implemented in the config
    # it will default to None (order by event time desending and by event_distance ascending)
    dic = core.get_segments(get_session(current_app), data.get('segment_select', None),
                            data.get('segment_orderby', None),
                            data.get('metadata', False),
                            data.get('classes', False))
    return jsonify(dic)


@main_app.route("/get_segment", methods=['POST'])
def get_segment_data():
    '''view returning the response for the segment data (and/or metadata)'''
    data = request.get_json()
    seg_id = data['seg_id']  # this must be present
    plot_indices = data.get('plot_indices', [])
    preprocessed = data.get('pre_processed', False)
    zooms = data.get('zooms', None)
    all_components = data.get('all_components', False)
    metadata = data.get('metadata', False)
    classes = data.get('classes', False)
    config = data.get('config', {})
    plotmanager = current_app.config['PLOTMANAGER']
#     if conf:
#         current_app.config['CONFIG.YAML'].update(conf)  # updates also plotmanager
    # NOTE: seg_id is a unicode string, but the query to the db works as well
    return jsonify(core.get_segment_data(get_session(current_app), seg_id, plotmanager,
                                         plot_indices, all_components, preprocessed, zooms,
                                         metadata, classes, config))


@main_app.route("/set_class_id", methods=['POST'])
def set_class_id():
    json_req = request.get_json()
    core.set_class_id(get_session(current_app), json_req['segment_id'], json_req['class_id'],
                      json_req['value'])
    # the above raises, otherwise return empty json to signal success:
    return jsonify({})
