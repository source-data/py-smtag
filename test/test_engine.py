# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from torch import nn, optim
from smtag.utils import tokenize, timer
from smtag.converter import TString
from test.smtagunittest import SmtagTestCase
from test.mini_trainer import toy_model
from smtag.engine import SmtagEngine, Combine, Connector
from smtag.progress import progress
from smtag.config import MARKING_CHAR

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()


class EngineTest(SmtagTestCase):
    
    @classmethod
    def setUpClass(self): # run only once
        self.models = {}
        self.text_example = "AAA YY, XXX AAA"

        self.x = TString(self.text_example).toTensor()
        self.y1 = torch.Tensor(# A A A   Y Y ,   X X X   A A A 
                              [[[0,0,0,0,1,1,0,0,1,1,1,0,0,0,0]]])
        self.models['entity'] = toy_model(self.x, self.y1, selected_features = ["geneprod"], epochs=1000)

        self.y2 = torch.Tensor(# A A A   Y Y ,   X X X   A A A 
                              [[[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0]]])
        self.models['only_once'] = toy_model(self.x, self.y2, selected_features = ["reporter"], epochs=1000)

        self.anonymized_text_example = self.text_example.replace("X", MARKING_CHAR)
        self.z = TString(self.anonymized_text_example).toTensor()
        self.models['context'] = toy_model(self.z, self.y1, selected_features=['intervention'], epochs=1000)

        self.y4 = torch.Tensor(# A A A   Y Y ,   X X X   A A A 
                              [[[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]])
        self.models['panelizer'] = toy_model(self.x, self.y4, selected_features = ["reporter"], epochs=1000)

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

    def test_model_stability(self): 
        '''
        Testing that test model returns the same result
        '''
        iterations = 100
        for i in range(iterations):
            y_1 = self.models['entity'](self.x)
            self.assertTensorEqual(self.y1, y_1)

    @timer
    def test_engine_entity(self):
        ml = SmtagEngine(self.cartridge).all(self.text_example)
        print(ml)
        expected = ""
        #self.assertEqual(expected, ml)


if __name__ == '__main__':
    unittest.main()
