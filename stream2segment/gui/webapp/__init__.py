from flask import Flask
from stream2segment.gui.webapp.core import classannotator
# from flask import url_for
# from flask import request
# app = Flask(__name__, static_folder=None)  # static_folder=None DOES TELL FLASK NOT TO ADD ROUTE
# FOR STATIC FOLDEERS

app = Flask(__name__)
# app.config.from_object('config')

# this has to come AFTER app ABOVE
# from stream2segment.gui.webapp import views  # nopep8

from stream2segment.s2sio.db import ListReader
from flask import jsonify
from stream2segment.gui.webapp import core
from flask import request

@app.route("/")
def main():
    from flask import render_template
    return render_template('index.html', title=app.config["DATABASE_URI"])  # app.config['DB_URI'])


@app.route("/get_elements", methods=['GET'])
def get_elements():
    db_uri = app.config['DATABASE_URI']
    return jsonify(core.get_ids(db_uri))


@app.route("/get_data", methods=['POST'])
def get_data():
    data = request.get_json()
    seg_id = data['segId']
    # NOTE: seg_id is a unicode string, but the query to the db works as well
    db_uri = app.config['DATABASE_URI']
    return jsonify(core.get_data(db_uri, seg_id))


@app.route("/set_class_id", methods=['POST'])
def set_class_id():
    data = request.get_json()
    class_id = data['classId']
    seg_id = data['segmentId']
    old_class_id = core.set_class(seg_id, class_id)
    # Flask complains if return is missing. FIXME: check better!
    return str(old_class_id)

#     seg_id = int(seg_id.replace("NEG", "-"))  # see core for this workaround ...
#     db_uri = app.config['DATABASE_URI']
#     return jsonify(core.get_data(db_uri, seg_id))
