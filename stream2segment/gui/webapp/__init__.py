"""
Web app (gui) entry point

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# import os
# from contextlib import contextmanager
# 
# from flask import Flask, g
# 
# from stream2segment.process.db import get_session as s2s_get_session
# 
# 
# def get_session(app):
#     """Opens a new database connection if there is none yet for the
#     current application context.
#     """
#     if not hasattr(g, 'session'):
#         g.session = s2s_get_session(app.config['DATABASE'])
#     return g.session
# 
# 
# @contextmanager
# def create_app(dbpath):
#     """
#         Function used within a 'with' statement to instantiate stuff on the given app
#         ```
#             with create_app(dburl) as app:
#                 ... do stuff on app...
#             return app
#         ```
#     """
#     # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
#     app = Flask(__name__)
#     app.config['DATABASE'] = dbpath
# 
#     @app.teardown_appcontext
#     def close_db(error):
#         """Closes the database again at the end of the request."""
#         if hasattr(g, 'session'):
#             g.session.close()
# 
#     yield app
