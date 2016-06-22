from flask import Flask
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

@app.route("/")
def main():
    from flask import render_template
    return render_template('index.html', title="A")  # app.config['DB_URI'])


@app.route("/get_elements", methods=['GET'])
def get_elements():
    db_uri = app.config['DATABASE_URI']
    return jsonify(core.get_ids(db_uri))


@app.route("/get_data/<id>", methods=['GET'])
def get_data(id):
    id = int(id.replace("NEG", "-"))  # see core for this workaround ...
    db_uri = app.config['DATABASE_URI']
    return jsonify(core.get_data(db_uri, id))
