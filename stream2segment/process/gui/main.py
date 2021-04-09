'''
Functions for launching the web app

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

import uuid
from webbrowser import open as open_in_browser
import random
import threading

from flask import Flask

from stream2segment.io import yaml_load
from stream2segment.io.inputvalidation import validate_param
from stream2segment.process.inspectimport import load_source
# valid_session below is imported from stream2segment.process and not stream2segment.io
# for consistency, but it is currently the same exact function
from stream2segment.process.db import get_session
from stream2segment.process.inputvalidation import _extract_segments_selection


def show_gui(dburl, pyfile, configfile):
    """Show downloaded data plots in a system browser dynamic web page"""
    session, pymodule, config_dict, segments_selection = \
        load_config_for_visualization(dburl, pyfile, configfile)
    run_in_browser(create_s2s_show_app(session, pymodule, config_dict,
                                       segments_selection))
    return 0


def load_config_for_visualization(dburl, pyfile=None, config=None):
    """Check visualization arguments and return a tuple of well formed args.
    Raise :class:`BadParam` if any param is invalid
    """
    # in process and download routines, validation is in a separate inputvalidation.py
    # module. Here for the moment we leave it here
    session = validate_param('dburl', dburl, get_session, scoped=True)
    pymodule = None if not pyfile else validate_param('pyfile', pyfile, load_source)
    config_dict = {} if not config else validate_param('configfile', config, yaml_load)
    seg_sel = _extract_segments_selection(config_dict)

    return session, pymodule, config_dict, seg_sel


def create_s2s_show_app(session, pymodule=None, config=None, segments_selection=None):
    """Create a new app for processing. Note that config_py_file is the
    stream2segment GUI config, not the config passed to Flask
    `app.config.from_pyfile`.
    """
    from stream2segment.process.gui import webapp
    # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
    app = Flask(webapp.__name__)

    from stream2segment.process.gui.webapp.mainapp import core
    core.init(app, session, pymodule, config, segments_selection)

    # Note that the templae_folder of the Blueprint and the static paths in
    # the HTML are relative to the path of THIS MODULE, so execute the lines
    # below HERE or good luck changing all static paths in the html:
    from stream2segment.process.gui.webapp.mainapp.views import main_app
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
