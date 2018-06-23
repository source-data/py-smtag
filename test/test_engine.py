# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from torch import nn, optim
from smtag.utils import tokenize
from smtag.converter import Converter
from test.smtagunittest import SmtagTestCase
from test.mini_trainer import toy_model
from smtag.engine import SmtagEngine, Combine, Connector
from smtag.progress import progress
from smtag.config import MARKING_CHAR

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()


class EngineTest(SmtagTestCase):
    
    @classmethod
    def setUpClass(self): # run only once
        self.text_example = "AAAAAAA XXX AAA"
        self.x = Converter.t_encode(self.text_example)
        self.y = torch.Tensor(# A A A A A A A   X X X   A A A 
                             [[[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]]])
        self.selected_features = ["geneprod"]
        self.entity_model = toy_model(self.x, self.y)

        self.anonymized_text_example = self.text_example.replace("X", MARKING_CHAR)
        self.z = Converter.t_encode(self.anonymized_text_example)
        self.context_model = toy_model(self.z, self.y, selected_feature=['intervention'])

    def test_model_stability(self): 
        '''
        Testing that test model returns the same result
        '''
        iterations = 100
        for i in range(iterations):
            y_1 = self.entity_model(self.x)
            self.assertTensorEqual(self.y, y_1)

    def test_engine_entity(self):
        # need to change this to intantiate SmtagEngine with test cartridge
        e = SmtagEngine()
        ml = e.entities('stained by anti‐SEPA‐1 antibodies')
        print(ml)
        expected = 'stained by anti‐<sd-tag type="geneprod">SEPA‐1</sd-tag> antibodies'
        self.assertEqual(expected, ml)

    def test_engine_combo(self):
        e = SmtagEngine()
        ml = e.entity_and_context('stained on SEPA‐1-/- mice')
        print(ml)
        expected = 'stained on <sd-tag type="geneprod" role="intervention">SEPA‐1</sd-tag>-/- mice'
        self.assertEqual(expected, ml)
    
    def test_engine_entity_reporter_context(self):
        e = SmtagEngine()
        ml = e.entity_reporter_context('Cells expressing GFP-Atg8 in Atg5-/- mice')
        print(ml)
        expected = 'Cells expressing <sd-tag type="geneprod" role="reporter">GFP</sd-tag>-<sd-tag type="geneprod">Atg8</sd-tag> in <sd-tag type="geneprod" role="intervention">Atg5</sd-tag>-/- mice'
        self.assertEqual(expected, ml)


if __name__ == '__main__':
    unittest.main()
