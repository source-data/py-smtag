# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from torch import nn, optim
from smtag.utils import tokenize, assertTensorEqual
from smtag.converter import Converter
from smtag.binarize import Binarized
from smtag.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.builder import SmtagModel, build
from smtag.predictor import EntityPredictor, SemanticsFromContextPredictor
from smtag.viz import Show
from smtag.importexport import load_model
from smtag.config import PROD_DIR, MARKING_CHAR

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()

def miniTrainer(x, y, selected_feature = ['geneprod'], threshold = 1E-02):
        opt = {}
        opt['namebase'] = 'test_importexport'
        opt['learning_rate'] = 0.01
        opt['epochs'] = 100
        opt['minibatch_size'] = 1
        opt['selected_features'] = selected_feature
        opt['nf_table'] =  [8,8]
        opt['pool_table'] = [2,2]
        opt['kernel_table'] = [2,2]
        opt['dropout'] = 0.1
        opt['nf_input'] = x.size(1)
        opt['nf_output'] =  y.size(1)
        opt = opt
        model = build(opt)

        # we do the training loop here instead of using smtag/train to avoid the need to prepare minibatches
        loss_fn = nn.SmoothL1Loss() # nn.BCELoss() # 
        optimizer = optim.Adam(model.parameters(), lr = opt['learning_rate'])
        optimizer.zero_grad()
        loss = 1
        i = 0
        # We stop as soon as the model has reasonably converged or if we exceed a max number of iterations
        max_iterations = opt['epochs']
        while loss > threshold and i < max_iterations:
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            i += 1
            #print(f"{i}: {loss}")
            #sys.stdout.write(f"{i}: {loss}")
            #sys.stdout.flush()
        print(f"Model preparation done! {i} iterations reached loss={float(loss)}")
        # don't forget to set the state of the model to eval() to avoid Dropout
        model.eval()
        return model

class PredictorTest(unittest.TestCase):
    
    @staticmethod
    def assertTensorEqual(x, y):
        return assertTensorEqual(x, y)

    @classmethod
    def setUpClass(self): # run only once
        self.text_example = "AAAAAAA XXX AAA"
        self.x = Converter.t_encode(self.text_example)
        self.y = torch.Tensor(# A A A A A A A   X X X   A A A 
                             [[[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]]])
        self.selected_features = ["geneprod"]
        self.entity_model = miniTrainer(self.x, self.y)

        self.anonymized_text_example = self.text_example.replace("X", MARKING_CHAR)
        self.z = Converter.t_encode(self.anonymized_text_example)
        self.context_model = miniTrainer(self.z, self.y, selected_feature=['intervention'])

    def test_model_stability(self): 
        '''
        Testing that this test model returns the same result
        '''
        iterations = 10
        for i in range(iterations):
            y_1 = self.entity_model(self.x)
            self.assertTensorEqual(self.y, y_1)
            
    def test_predictor_padding(self):
        p = EntityPredictor(self.entity_model)
        test_string_200 = "a"*200
        test_string_200_encoded = Converter.t_encode(test_string_200)
        padded_string_200_encoded = p.padding(test_string_200_encoded)
        expected_padded_string_200_encoded = Converter.t_encode(" "*10 + test_string_200 + " "*10)
        self.assertTensorEqual(expected_padded_string_200_encoded, padded_string_200_encoded)
        
        test_string_20 = "a"*20
        test_string_20_encoded = Converter.t_encode(test_string_20)
        padded_string_20_encoded = p.padding(test_string_20_encoded)
        expected_padded_string_20_encoded = Converter.t_encode(" "*60 + test_string_20 + " "*60)
        self.assertTensorEqual(expected_padded_string_20_encoded, padded_string_20_encoded)
   
    def test_entity_predictor_1(self):
        p = EntityPredictor(self.entity_model)
        output, _ = p.forward(self.text_example)
        self.assertEqual(list(self.y.size()), list(output.size()))

    def test_entity_predictor_2(self):
        p = EntityPredictor(self.entity_model)
        ml = p.markup(self.text_example)
        #print(ml)
        expected_ml = 'AAAAAAA <sd-tag type="geneprod">XXX</sd-tag> AAA'
        self.assertEqual(expected_ml, ml[0])

    def test_entity_predictor_3(self):
        real_model = load_model('geneprod.zip', PROD_DIR)
        #real_example = "fluorescent images of 200‐cell‐stage embryos from the indicated strains stained by both anti‐SEPA‐1 and anti‐LGG‐1 antibody"
        real_example = "stained by anti‐SEPA‐1"
        p = EntityPredictor(real_model)
        ml = p.markup(real_example)[0]
        print(ml)
        input = Converter.t_encode(real_example)
        # compare visually with the direct ouput of the model
        real_model.eval()
        prediction = real_model(input)
        Show.print_pretty_color(prediction, real_example)
        Show.print_pretty(prediction)

    def test_context_predictor_anonymization(self):
        input_string = "the cat with a hat"
        input_string_encoded = Converter.t_encode(input_string)
        marks = torch.Tensor(
            # t h e   c a t   w i t h   a   h a t
            [[0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0]]
        )
        p = SemanticsFromContextPredictor(self.context_model)
        anonymized_encoded = p.anonymize_(input_string_encoded, marks, replacement = Converter.t_encode("$"))
        anonymized = Converter.t_decode(anonymized_encoded)
        expected = "the $$$ with a hat"
        self.assertEqual(expected, anonymized)

    def test_context_predictor(self):
        entity_p = EntityPredictor(self.entity_model)
        prediction_1, string_encoded = entity_p.forward(self.text_example)
        bin_pred = Binarized([self.text_example], prediction_1, ['geneprod'])
        tokenized = tokenize(self.text_example)
        bin_pred.binarize_with_token([tokenized])
        
        context_p = SemanticsFromContextPredictor(self.context_model)
        prediction_2, _ = context_p.forward(string_encoded, bin_pred)
        Show.print_pretty_color(prediction_1, self.text_example)
        Show.print_pretty_color(prediction_2, self.anonymized_text_example)
        Show.print_pretty(torch.cat((prediction_1, prediction_2),1))

if __name__ == '__main__':
    unittest.main()
