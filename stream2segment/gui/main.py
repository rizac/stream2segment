'''
Functions for launching the web app

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

import os
import uuid
from webbrowser import open as open_in_browser
import random
import threading

from flask import Flask


def create_s2s_show_app(session, pymodule=None, config=None, segments_selection=None):
    """Create a new app for processing. Note that config_py_file is the
    stream2segment GUI config, not the config passed to Flask
    `app.config.from_pyfile`.
    """
    from stream2segment.gui import webapp
    # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
    app = Flask(webapp.__name__)

    from stream2segment.gui.webapp.mainapp import core
    core.init(app, session, pymodule, config, segments_selection)

    # Note that the templae_folder of the Blueprint and the static paths in
    # the HTML are relative to the path of THIS MODULE, so execute the lines
    # below HERE or good luck changing all static paths in the html:
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
