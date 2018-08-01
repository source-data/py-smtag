from flask import Flask
from smtag_api.config import Config

app = Flask(__name__)

Config.init_app(app)
app.config.from_object(Config)

from smtag_api import views

