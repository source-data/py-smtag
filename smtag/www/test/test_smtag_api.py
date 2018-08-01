import unittest
from smtag_api import predictors

class LuaCliPredictorTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_calls_lua(self):
        smtag_predictor = predictors.LuaCliPredictor()
        result = smtag_predictor.complete("text", "xml", "my-tag")
        # import pdb; pdb.set_trace()
        assert len(result) > 0

