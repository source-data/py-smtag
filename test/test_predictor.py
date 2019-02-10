# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
from math import ceil
from collections import OrderedDict
import torch
from torch import nn, optim
from smtag.common.utils import tokenize
from test.smtagunittest import SmtagTestCase
from test.mini_trainer import toy_model
from smtag.common.converter import TString
from smtag.predict.decode import Decoder
from smtag.predict.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.predict.predictor import Predictor, ContextualPredictor
from smtag.common.mapper import Catalogue
from smtag.common.viz import Show
from smtag.common.importexport import load_model
from smtag import config

MARKING_CHAR = config.marking_char

class PredictorTest(SmtagTestCase):

    @classmethod
    def setUpClass(self): # run only once
        self.text_example = "AAAAAAA XXX AAA"
        self.x = TString(self.text_example)
        self.y = torch.Tensor(# A A A A A A A   X X X   A A A
                             [[[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
                               [1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
                             ]])
        self.selected_features = ["geneprod"]
        self.entity_model = toy_model(self.x.tensor, self.y)
        self.entity_model.eval()
        self.anonymized_text_example = self.text_example.replace("X", MARKING_CHAR)
        self.z = TString(self.anonymized_text_example)
        self.context_model = toy_model(self.z.tensor, self.y, selected_features=['intervention'])
        self.context_model.eval()

    @unittest.skip("not relevant")
    def test_model_stability(self):
        pass
        '''
        Testing that test model returns the same result for same input. 
        '''
        self.entity_model.eval()
        iterations = 10
        x = self.x.tensor
        y_1 = self.entity_model(x)
        print(0, "\n")
        print(y_1)
        for i in range(iterations):
            y_2 = self.entity_model(x)
            print(i, "\n")
            print(y_1,"\n")
            print(y_1.sub(y_2))
            self.assertTensorEqual(y_1, y_2)

    def test_predictor_padding(self):
        p = Predictor(self.entity_model)
        test_string_200 = "a"*200
        test_string_200_encoded = TString(test_string_200)
        padded_string_200_encoded = p.padding(test_string_200_encoded)
        padding_length = ceil(max(config.min_size - 200, 0)/2) + config.min_padding
        print("config.min_size, config.min_size, padding_length", config.min_size, config.min_size, padding_length)
        expected_padded_string_200_encoded = TString(config.padding_char*padding_length + test_string_200 + config.padding_char*padding_length)
        self.assertTensorEqual(expected_padded_string_200_encoded.tensor, padded_string_200_encoded.tensor)

    def test_entity_predictor_1(self):
        p = Predictor(self.entity_model)
        output = p.forward(TString(self.text_example))
        self.assertEqual(list(self.y.size()), list(output.size()))

    def test_context_predictor_anonymization(self):
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,1.0 ,1.0 ,0.2  ,1.0 ,1.0 ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [1.0 ,1.0 ,0   ,0   ,0.8  ,0   ,0   ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ]
                                  ]])
        group = 'test'
        concepts = [Catalogue.GENEPROD, Catalogue.UNTAGGED]
        semantic_groups = OrderedDict([(group, concepts)])
        d = Decoder(input_string, prediction, semantic_groups)
        d.decode()
        d.fuse_adjacent()
        p = ContextualPredictor(self.context_model)
        anonymized_encoded = p.anonymize(d, 'test', Catalogue.GENEPROD)
        anonymized = str(anonymized_encoded)
        expected = "A "+config.marking_char*len("ge ne") + " or others"
        self.assertEqual(expected, anonymized)

if __name__ == '__main__':
    unittest.main()
