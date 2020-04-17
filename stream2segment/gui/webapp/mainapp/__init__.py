# from flask import Flask, session
# import uuid
# 
# # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
# app = Flask(__name__)
# # although we do not use sessions (which write cookies client side),
# # we set a secret key neverthless:
# # https://www.tutorialspoint.com/flask/flask_sessions.htm
# app.secret_key = str(uuid.uuid4())
# 
# # app.config['DATABASE'] = dbpath
# # app.config['pyfile'] = pyfile
# # app.config['configfile'] = configfile
# # app.config['PLOTMANAGER'] = PlotManager(pymodule, configdict)
# # app.config['CONFIG.YAML'] = configdict
# from stream2segment.gui.webapp.mainapp.views import main_app
# app.register_blueprint(main_app)