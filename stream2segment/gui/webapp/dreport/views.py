'''
Views for the web app  (download report)

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from flask import render_template, request, jsonify, Blueprint, current_app

from stream2segment.gui.webapp import get_session
from stream2segment.gui.webapp.dreport import core
from stream2segment.utils import secure_dburl
from stream2segment.gui.webapp.dreport.core import selectablelabels, get_station_data

# http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories:
main_page = Blueprint('main_page', __name__, template_folder='templates')


@main_page.route("/")
def main():
    settings = {'max_gap_overlap_ratio': [-0.5, 0.5]}
    # filterfunc_doc = current_app.config['PLOTMANAGER'].get_filterfunc_doc.replace("\n", "<p>")
    return render_template('dreport.html', title=secure_dburl(current_app.config["DATABASE"]),
                           settings=settings, labels=selectablelabels())


@main_page.route("/get_data", methods=['POST'])
def init():
    data = request.get_json()
    dic = core.get_data(get_session(current_app))
    return jsonify(dic)


@main_page.route("/get_selectable_labels", methods=['POST'])
def selectable_labels():
    return jsonify(selectablelabels())


@main_page.route("/get_station_data", methods=['POST'])
def getstationdata():
    data = request.get_json()
    return jsonify(get_station_data(get_session(current_app), data['station_id'], data['labels']))
