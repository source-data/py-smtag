# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from torch import nn, optim
import sys
import os
from smtag.utils import tokenize, assertTensorEqual
from smtag.converter import Converter
from smtag.builder import build
from smtag.predictor import EntityPredictor
from smtag.importexport import load_model
from smtag.config import MODEL_DIR
from smtag.progress import progress
from smtag.importexport import export_model, load_model

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

class ImportExportTest(unittest.TestCase):
    '''
    A test to see if saving and then reloading a trained model is working and produces the same predictio.
    '''
    
    @staticmethod
    def assertTensorEqual(x, y):
        return assertTensorEqual(x,y)
    
    def setUp(self):
        '''
        Training a model with realistic structure with a single example just to make it converge for testing.
        '''
        self.text_example = "AAA XXX AAA"
        self.x = Converter.t_encode(self.text_example) #torch.ones(1, 32, len(self.text_example)) #
        self.y = torch.Tensor(#"A A A   X X X   A A A"
                              # 0 0 0 0 1 1 1 0 0 0 0
                             [[[0,0,0,0,1,1,1,0,0,0,0]]])
        opt = {}
        opt['namebase'] = 'test_importexport'
        opt['learning_rate'] = 0.01
        opt['epochs'] = 100
        opt['minibatch_size'] = 1
        opt['selected_features'] = ['geneprod']
        opt['nf_table'] =  [8,8]
        opt['pool_table'] = [2,2]
        opt['kernel_table'] = [2,2]
        opt['dropout'] = 0.1
        opt['nf_input'] = self.x.size(1)
        opt['nf_output'] =  self.y.size(1)
        self.opt = opt
        self.model = build(self.opt)

        # we do the training loop here instead of using smtag/train to avoid the need to prepare minibatches
        loss_fn = nn.SmoothL1Loss() # nn.BCELoss() # 
        optimizer = optim.Adam(self.model.parameters(), lr = self.opt['learning_rate'])
        optimizer.zero_grad()
        loss = 1
        i = 0
        # We stop as soon as the model has reasonably converged or if we exceed a max number of iterations
        max_iterations = self.opt['epochs']
        while loss > 1E-02 and i < max_iterations:
            y_hat = self.model(self.x)
            loss = loss_fn(y_hat, self.y)
            loss.backward()
            optimizer.step()
            i += 1
            print(f"{i}: {loss}")
            #sys.stdout.write(f"{i}: {loss}")
            #sys.stdout.flush()
        print(f"Model preparation done! {i} iterations reached loss={float(loss)}")
        # don't forget to set the state of the model to eval() to avoid Dropout
        self.model.eval()
        self.myzip = None


    def test_export_reload(self):
        ml_1 = EntityPredictor(self.model).markup(self.text_example)
        y_1 = self.model(self.x)
        self.myzip = export_model(self.model, custom_name='test_model_importexport')
        reloaded = load_model('test_model_importexport.zip')
        ml_2 = EntityPredictor(reloaded).markup(self.text_example)
        y_2 = reloaded(self.x)
        self.assertTensorEqual(y_1, y_2)
        print(ml_1)
        print(ml_2)
        self.assertEqual(ml_1, ml_2)
    
    def tearDown(self):
        if self.myzip is not None:
            os.remove(os.path.join(MODEL_DIR, self.myzip.filename))



if __name__ == '__main__':
    unittest.main()

