from flask import request
from smtag_api import app

@app.route('/')
def hello_world():
    return 'Hello, World!'
