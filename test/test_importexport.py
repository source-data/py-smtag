# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from torch import nn, optim
import sys
import os
from py_smtag.common.utils import tokenize
from test.smtagunittest import SmtagTestCase
from test.mini_trainer import toy_model
from py_smtag.common.converter import Converter, TString
from py_smtag.predict.predictor import SimplePredictor
from py_smtag.common.config import MODEL_DIR
from py_smtag.common.progress import progress
from py_smtag.common.mapper import Catalogue
from py_smtag.common.importexport import export_model, load_model

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

class ImportExportTest(SmtagTestCase):
    '''
    A test to see if saving and then reloading a trained model is working and produces the same prediction.
    '''


    def setUp(self):
        '''
        Training a model with realistic structure with a single example just to make it converge for testing.
        '''
        self.text_example = "AAA XXX AAA"
        self.x = Converter.t_encode(self.text_example) #torch.ones(1, 32, len(self.text_example)) #
        self.y = torch.Tensor(#"A A A   X X X   A A A"
                              # 0 0 0 0 1 1 1 0 0 0 0
                             [[[0,0,0,0,1,1,1,0,0,0,0]]])
        self.model = toy_model(self.x, self.y)
        self.myzip = None


    def test_export_reload(self):
        b_1 = SimplePredictor(self.model).pred_binarized(TString(self.text_example), [Catalogue.GENEPROD])
        y_1 = self.model(self.x)
        self.myzip = export_model(self.model, custom_name='test_model_importexport')
        reloaded = load_model('test_model_importexport.zip')
        b_2 = SimplePredictor(reloaded).pred_binarized(TString(self.text_example), [Catalogue.GENEPROD])
        y_2 = reloaded(self.x)
        self.assertTensorEqual(y_1, y_2)
        self.assertTensorEqual(b_1.marks, b_2.marks)

    def tearDown(self):
        if self.myzip is not None:
            os.remove(os.path.join(MODEL_DIR, self.myzip.filename))



if __name__ == '__main__':
    unittest.main()

