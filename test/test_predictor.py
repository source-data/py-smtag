# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from torch import nn, optim
from smtag.utils import tokenize, assertTensorEqual
from smtag.converter import Converter
from smtag.binarize import Binarized
from smtag.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.builder import SmtagModel
from smtag.predictor import EntityPredictor
from smtag.viz import Show
from smtag.importexport import load_model
from smtag.config import PROD_DIR

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()

class FakeModel(nn.Module):
    def __init__(self):
        super(FakeModel, self).__init__()
        self.conv = nn.Conv1d(32,1,1,1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.conv(x)
        y = self.sigmoid(y)
        return y

class PredictorTest(unittest.TestCase):
    
    @staticmethod
    def assertTensorEqual(x, y):
        return assertTensorEqual(x, y)

    def setUp(self):
        self.text_example = "AAA XXX AAA"
        self.x = Converter.t_encode(self.text_example) #torch.ones(1, 32, len(self.text_example)) #
        self.y = torch.Tensor(#"A A A   X X X   A A A"
                              # 0 0 0 0 1 1 1 0 0 0 0
                             [[[0,0,0,0,1,1,1,0,0,0,0]]])
        self.selected_features = ["geneprod"]
        self.model = SmtagModel(FakeModel(), {'selected_features':self.selected_features})
        loss_fn = nn.SmoothL1Loss() # nn.BCELoss() # 
        optimizer = optim.Adam(self.model.parameters(), lr = 0.01)
        optimizer.zero_grad()
        loss = 1
        i = 0 
        while loss > 1E-02 and i < 10000:
            y_hat = self.model(self.x)
            loss = loss_fn(y_hat, self.y)
            loss.backward()
            optimizer.step()
            i += 1
        #print(i)
        #print(loss)
        #print(y_hat)
        self.model.eval()

    def test_predictor_padding(self):
        p = EntityPredictor(self.model)
        test_string_200 = "a"*200
        padded_string_200 = p.padding(test_string_200)
        expected_padded_string_200 = " "*10 + test_string_200 + " "*10
        self.assertEqual(expected_padded_string_200, padded_string_200)
        test_string_20 = "a"*20
        padded_string_20 = p.padding(test_string_20)
        expected_padded_string_20 = " "*60 + test_string_20 + " "*60
        self.assertEqual(expected_padded_string_20, padded_string_20)
   
    def test_entity_predictor_1(self):
        p = EntityPredictor(self.model)
        output = p.forward(self.text_example)
        self.assertEqual(list(self.y.size()), list(output.size()))

    def test_entity_predictor_2(self):
        p = EntityPredictor(self.model)
        ml = p.markup(self.text_example)
        #print(ml)
        expected_ml = 'AAA <sd-tag type="geneprod">XXX</sd-tag> AAA'
        self.assertEqual(expected_ml, ml[0])

    def test_model_stability(self):
        iterations = 10
        for i in range(iterations):
            y_1 = self.model(self.x)
            self.assertTensorEqual(self.y, y_1)

    def test_entity_predictor_3(self):
        real_model = load_model('geneprod.zip', PROD_DIR)
        real_example = "fluorescent images of 200‐cell‐stage embryos from the indicated strains stained by both anti‐SEPA‐1 and anti‐LGG‐1 antibody"
        #real_example = "This is Atg5 speaking."
        p = EntityPredictor(real_model)
        ml = p.markup(real_example)[0]
        print(ml)
        input = Converter.t_encode(real_example)
        real_model.eval()
        prediction = real_model(input)
        Show.print_pretty_color(prediction, real_example)
        Show.print_pretty(prediction)

if __name__ == '__main__':
    unittest.main()

