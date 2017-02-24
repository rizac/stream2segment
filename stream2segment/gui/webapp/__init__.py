from flask import Flask
# from stream2segment.gui.webapp.core import classannotator
# from flask import url_for
# from flask import request
# app = Flask(__name__, static_folder=None)  # static_folder=None DOES TELL FLASK NOT TO ADD ROUTE
# FOR STATIC FOLDEERS

app = Flask(__name__)
# app.config.from_object('config')

# this has to come AFTER app ABOVE
from stream2segment.gui.webapp import views  # nopep8
