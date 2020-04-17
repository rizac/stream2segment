'''
Functions for launching the web app

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

import uuid
from webbrowser import open as open_in_browser
import os
import random
import threading

from flask import Flask

# from stream2segment.gui.webapp import create_app
# from stream2segment.gui.webapp.mainapp.plots.core import PlotManager


def create_s2s_show_app(dbpath, pyfile=None, configfile=None):
    """
        Creates a new app for processing. Note that config_py_file is the stream2segment gui
        config, not the config passed to Flask `app.config.from_pyfile`.
    """
    os.environ['S2SSHOW_DATABASE'] = dbpath
    os.environ['S2SSHOW_pyfile'] = pyfile
    os.environ['S2SSHOW_configfile'] = configfile

    from stream2segment.gui import webapp
    # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
    app = Flask(webapp.__name__)

#    app = mainapp.app
#     app.config['DATABASE'] = dbpath
#     app.config['pyfile'] = pyfile
#     app.config['configfile'] = configfile
    # app.config['PLOTMANAGER'] = PlotManager(pymodule, configdict)
    # app.config['CONFIG.YAML'] = configdict
    from stream2segment.gui.webapp.mainapp.views import main_app
    app.register_blueprint(main_app)

    return app


def run_in_browser(app, port=None, debug=False):
    app.config.update(
        ENV='development',  # https://stackoverflow.com/a/53919435,
        # DEBUG = True,
        # although we do not use sessions (which write cookies client side),
        # we set a secret key neverthless:
        # https://www.tutorialspoint.com/flask/flask_sessions.htm
        SECRET_KEY=str(uuid.uuid4())
    )
    if port is None:
        port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    if not debug:
        threading.Timer(1.25, lambda: open_in_browser(url)).start()
    app.run(port=port, debug=debug)
