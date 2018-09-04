
from smtag_api import app
import smtag
from smtag.common.utils import cleanup
from smtag.predict.engine import SmtagEngine
smtag.config.working_directory = app.config['SMTAG_PYTHON_WORKING_DIRECTORY']


class PythonPredictor():
    engine = SmtagEngine()
    def complete(self, text, format, tag):
        return self.engine.smtag(text, tag)
    def entity(self, text, format, tag):
        return self.engine.entity(text, tag)
    def role(self, text, format, tag):
        return "implementation pending"
    def tagger(self, text, format, tag):
        return self.engine.tag(text, tag)
    def panelize(self, text, format, tag):
        return self.engine.panelizer(text)
