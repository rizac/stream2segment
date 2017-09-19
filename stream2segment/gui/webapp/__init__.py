"""
Web app (gui) entry point

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from flask import Flask, g
from stream2segment.utils import get_session as s2s_get_session, load_source
from flask.json import JSONEncoder
from obspy.core.utcdatetime import UTCDateTime
import os
from stream2segment.gui.webapp.plotviews import PlotManager
#from stream2segment.gui.webapp.plots import set_filter_config, set_spectra_config

# from stream2segment.gui.webapp.core import classannotator
# from flask import url_for
# from flask import request
# app = Flask(__name__, static_folder=None)  # static_folder=None DOES TELL FLASK NOT TO ADD ROUTE
# FOR STATIC FOLDEERS

# app = Flask(__name__)
# app.config.from_object('config')

# this has to come AFTER app ABOVE
# from stream2segment.gui.webapp import views  # nopep8

# _app = None


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
#     global _app
#     if _app is None:

    # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
    app = Flask(__name__)
    app.config['DATABASE'] = dbpath
    app.config['PLOTMANAGER'] = PlotManager(pymodule, configdict)
    app.config['CONFIG.YAML'] = configdict
    app.config['CONFIG.KEYS'] = ['spectra', 'segment_select', 'segment_orderby']
#     if config_py_file is None:
#         config_py_file = os.path.join(os.path.dirname(__file__), "guiconfig.py")
#     # app.config.from_pyfile(config_py_file)
#     modl = load_source(config_py_file)
#     for k in dir(modl):
#         if k == "filter":
#             set_filter_config(**getattr(modl, k))
#         elif k == "spectra":
#             set_spectra_config(**getattr(modl, k))

#     if config_object is not None:
#         app.config.from_object(config_object)

#     from stream2segment.io.db.models import Base
#    db = SQLAlchemy(app, metadata=Base.metadata)

    from stream2segment.gui.webapp.views import main_page
    app.register_blueprint(main_page)

    @app.teardown_appcontext
    def close_db(error):
        """Closes the database again at the end of the request."""
        if hasattr(g, 'session'):
            g.session.close()

    return app

# class CustomJSONEncoder(JSONEncoder):
#     """Encoder which encodes datetime's and utcdatetime's UTCDateTime's"""
# 
#     def default(self, obj):
#         try:
#             if isinstance(obj, UTCDateTime):
#                 if obj.utcoffset() is not None:
#                     obj = obj - obj.utcoffset()
#                 millis = int(
#                     calendar.timegm(obj.timetuple()) * 1000 +
#                     obj.microsecond / 1000
#                 )
#                 return millis
#             iterable = iter(obj)
#         except TypeError:
#             pass
#         else:
#             return list(iterable)
#         return JSONEncoder.default(self, obj)
