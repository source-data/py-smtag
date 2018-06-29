# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from torch import nn, optim
from smtag.utils import tokenize
from test.smtagunittest import SmtagTestCase
from test.mini_trainer import toy_model
from smtag.converter import Converter, TString
from smtag.binarize import Binarized
from smtag.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.predictor import SimplePredictor, ContextualPredictor
from smtag.mapper import Catalogue
from smtag.viz import Show
from smtag.importexport import load_model
from smtag.config import PROD_DIR, MARKING_CHAR


class PredictorTest(SmtagTestCase):

    @classmethod
    def setUpClass(self): # run only once
        self.text_example = "AAAAAAA XXX AAA"
        self.x = TString(self.text_example)
        self.y = torch.Tensor(# A A A A A A A   X X X   A A A 
                             [[[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]]])
        self.selected_features = ["geneprod"]
        self.entity_model = toy_model(self.x.toTensor(), self.y)

        self.anonymized_text_example = self.text_example.replace("X", MARKING_CHAR)
        self.z = TString(self.anonymized_text_example)
        self.context_model = toy_model(self.z.toTensor(), self.y, selected_feature=['intervention'])

    def test_model_stability(self): 
        '''
        Testing that test model returns the same result
        '''
        iterations = 10
        for i in range(iterations):
            y_1 = self.entity_model(self.x.toTensor())
            self.assertTensorEqual(self.y, y_1)
            
    def test_predictor_padding(self):
        p = SimplePredictor(self.entity_model)
        test_string_200 = "a"*200
        test_string_200_encoded = TString(test_string_200)
        padded_string_200_encoded = p.padding(test_string_200_encoded)
        expected_padded_string_200_encoded = TString(" "*10 + test_string_200 + " "*10)
        self.assertTensorEqual(expected_padded_string_200_encoded.t, padded_string_200_encoded.t)
        
        test_string_20 = "a"*20
        test_string_20_encoded = TString(test_string_20)
        padded_string_20_encoded = p.padding(test_string_20_encoded)
        expected_padded_string_20_encoded = TString(" "*60 + test_string_20 + " "*60)
        self.assertTensorEqual(expected_padded_string_20_encoded.t, padded_string_20_encoded.t)
   
    def test_entity_predictor_1(self):
        p = SimplePredictor(self.entity_model)
        output = p.forward(TString(self.text_example))
        self.assertEqual(list(self.y.size()), list(output.size()))

    @unittest.skip('need to be changed')
    def test_entity_predictor_2(self):
        p = SimplePredictor(self.entity_model)
        ml = p.markup(TString(self.text_example))
        #print(ml)
        expected_ml = 'AAAAAAA <sd-tag type="geneprod">XXX</sd-tag> AAA'
        self.assertEqual(expected_ml, ml[0])

    #@unittest.skip('need to be changed')
    def test_entity_predictor_3(self):
        real_model = load_model('geneprod.zip', PROD_DIR)
        #real_example = "fluorescent images of 200‐cell‐stage embryos from the indicated strains stained by both anti‐SEPA‐1 and anti‐LGG‐1 antibody"
        real_example = "stained by anti‐SEPA‐1"
        p = SimplePredictor(real_model)
        binarized = p.pred_binarized(TString(real_example), [Catalogue.GENEPROD])
        ml = Serializer().serialize(binarized)
        print(ml[0])
        input = Converter.t_encode(real_example)
        # compare visually with the direct ouput of the model
        real_model.eval()
        prediction = real_model(input)
        Show.print_pretty_color(prediction, real_example)
        Show.print_pretty(prediction)

    def test_context_predictor_anonymization(self):
        input_string = "the cat with a hat"
        input_string_encoded = TString(input_string)
        marks = torch.Tensor(
            # t h e   c a t   w i t h   a   h a t
            [[0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0]]
        )
        p = ContextualPredictor(self.context_model)
        anonymized_encoded = p.anonymize(input_string_encoded, marks, replacement = TString("$"))
        anonymized = str(anonymized_encoded)
        expected = "the $$$ with a hat"
        self.assertEqual(expected, anonymized)

    def test_context_predictor(self):
        entity_p = SimplePredictor(self.entity_model)
        prediction_1 = entity_p.forward(TString(self.text_example))
        bin_pred = Binarized([self.text_example], prediction_1, [Catalogue.GENEPROD])
        tokenized = tokenize(self.text_example)
        bin_pred.binarize_with_token([tokenized])
        
        context_p = ContextualPredictor(self.context_model)
        prediction_2 = context_p.forward(TString(self.text_example), bin_pred.marks)
        Show.print_pretty_color(prediction_1, self.text_example)
        Show.print_pretty_color(prediction_2, self.anonymized_text_example)
        Show.print_pretty(torch.cat((prediction_1, prediction_2),1))

if __name__ == '__main__':
    unittest.main()
