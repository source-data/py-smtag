# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from torch import nn, optim
from smtag.common.utils import tokenize, timer
from smtag.common.converter import TString
from test.smtagunittest import SmtagTestCase
from test.mini_trainer import toy_model
from smtag.predict.engine import SmtagEngine, Combine, Connector
from smtag.common.progress import progress
from smtag.common.config import config
MARKING_CHAR = config.marking_char

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()


class EngineTest(SmtagTestCase):

    @classmethod
    def setUpClass(self): # run only once
        self.models = {}
        self.text_example = "AAA YY, XXX, AA"

        self.x = TString(self.text_example).toTensor()
        self.y1 = torch.Tensor(# A A A   Y Y ,   X X X ,   A A
                              [[[0,0,0,0,1,1,0,0,1,1,1,0,0,0,0]]])
        self.models['entity'] = toy_model(self.x, self.y1, selected_features = ["geneprod"], threshold = 1E-04, epochs=1000)

        self.y2 = torch.Tensor(# A A A   Y Y ,   X X X ,   A A
                              [[[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0]]])

        self.models['only_once'] = toy_model(self.x, self.y2, selected_features = [], overlap_features = ["geneprod", "reporter"], threshold = 1E-04, epochs=1000)

        self.anonymized_text_example = self.text_example.replace("X", MARKING_CHAR)
        self.z = TString(self.anonymized_text_example).toTensor()
        self.y3 = torch.Tensor(# A A A   Y Y ,   X X X ,   A A
                              [[[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]]])
        self.models['context'] = toy_model(self.z, self.y3, selected_features=['intervention'], threshold = 1E-04, epochs=1000)

        self.y4 = torch.Tensor(# A A A   Y Y ,   X X X ,   A A
                              [[[0,0,0,0,0,0,1,0,0,0,0,1,0,0,0]]])
        self.models['panelizer'] = toy_model(self.x, self.y4, selected_features = ["panel_start"], threshold = 1E-05, epochs=1000)

        self.cartridge = {
                'entity': [
                    (self.models['entity'], '')
                ],
                'only_once': [
                    (self.models['only_once'], '')
                ],
                'context': [
                    (self.models['context'], 'geneprod')
                ],
                'panelizer': [
                    (self.models['panelizer'], '')
                ]
            }

        self.engine = SmtagEngine(self.cartridge)
        self.engine.DEBUG = True


    def test_panel(self):
        ml = self.engine.panelizer(self.text_example, sdtag='sd-tag', format='xml')
        print(ml)
        expected = '''<smtag><sd-panel>AAA YY</sd-panel><sd-panel>, XXX</sd-panel><sd-panel>, AA</sd-panel></smtag>'''
        self.assertEqual(expected, ml)

    def test_entity(self):
        ml = self.engine.entity(self.text_example, sdtag='sd-tag', format='xml')
        print(ml)
        expected = '''<smtag>AAA <sd-tag type="geneprod" type_score="99">YY</sd-tag>, <sd-tag type="geneprod" type_score="99">XXX</sd-tag>, AA</smtag>'''
        self.assertEqual(expected, ml)

    @unittest.skip("unstable reporter toy model")
    def test_tag(self):
        ml = self.engine.tag(self.text_example, sdtag='sd-tag', format='xml')
        print(ml)
        expected = '''<smtag>AAA <sd-tag type="geneprod" role="reporter" type_score="99" role_score="99">YY</sd-tag>, <sd-tag type="geneprod" role="intervention" type_score="99" role_score="99">XXX</sd-tag>, AA</smtag>'''
        self.assertEqual(expected, ml)

    @unittest.skip("unstable reporter toy model")
    @timer
    def test_all(self):
        ml = self.engine.smtag(self.text_example, sdtag='sd-tag', format='xml')
        print(ml)
        expected = '''<smtag><sd-panel>AAA <sd-tag type="geneprod" role="reporter">YY</sd-tag></sd-panel><sd-panel>, <sd-tag type="geneprod" role="intervention">XXX</sd-tag></sd-panel><sd-panel>, AA</sd-panel></smtag>'''
        self.assertEqual(expected, ml)




from smtag.predict.engine import SmtagEngine
class SmtagEngineSerializationOutputTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_html(self):
        input_text = "qweqwe"
        expected_html = '<ul><li class="sd-panel">qweqwe</li></ul>'
        engine = SmtagEngine()
        html_output = engine.smtag(input_text, "sd-tag", "html")
        # import pdb; pdb.set_trace()
        self.assertEqual(expected_html, html_output)
    def test_xml(self):
        input_text = "qweqwe"
        expected_xml = '<smtag><sd-panel>qweqwe</sd-panel></smtag>'
        engine = SmtagEngine()
        xml_output = engine.smtag(input_text, "sd-tag", "xml")
        # import pdb; pdb.set_trace()
        self.assertEqual(expected_xml, xml_output)

if __name__ == '__main__':
    unittest.main()
