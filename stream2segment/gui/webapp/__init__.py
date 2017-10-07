"""
Web app (gui) entry point

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import os

from flask import Flask, g

from stream2segment.utils import get_session as s2s_get_session, load_source
from stream2segment.gui.webapp.plots.core import PlotManager


def get_session(app):
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'session'):
        g.session = s2s_get_session(app.config['DATABASE'])
    return g.session


def create_app(dbpath, pymodule=None, configdict=None):
    """
        Creates a new app. Note that config_py_file is the stream2segment gui config, not
        the config passed to Flask `app.config.from_pyfile`. For Flask config, please provide
        a valid object in `config_object`
    """
    # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
    app = Flask(__name__)
    app.config['DATABASE'] = dbpath
    app.config['PLOTMANAGER'] = PlotManager(pymodule, configdict)
    app.config['CONFIG.YAML'] = configdict
    app.config['CONFIG.KEYS'] = ['spectra', 'segment_select', 'segment_orderby']

    from stream2segment.gui.webapp.views import main_page
    app.register_blueprint(main_page)

    @app.teardown_appcontext
    def close_db(error):
        """Closes the database again at the end of the request."""
        if hasattr(g, 'session'):
            g.session.close()

    return app
