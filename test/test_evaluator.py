# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from smtag.common.utils import timer
from smtag.train.loader import Dataset
from smtag.train.minibatches import Minibatches
from smtag.common.converter import TString
from .smtagunittest import SmtagTestCase
from .mini_trainer import toy_model
from smtag.train.evaluator import Accuracy

class EvaluatorTest(SmtagTestCase):

    # @classmethod
    # def setUpClass(self): # runs only once
    #     self.text_example = "AAA YY, XXX, AA"
    #     self.x = TString(self.text_example).toTensor()
    #     self.y1 = torch.Tensor(# A A A   Y Y ,   X X X ,   A A
    #                           [[[0,0,0,0,1,1,0,0,1,1,1,0,0,0,0]]])
    #     self.model = toy_model(self.x, self.y1, selected_features = ["geneprod"], threshold = 1E-04, epochs=1000)

    #     dataset = Dataset()
    #     dataset.input = self.x
    #     dataset.output = self.y1
    #     dataset.text = [self.text_example]
    #     dataset.nf_input = 32
    #     dataset.nf_output = 1
    #     dataset.L = len(self.text_example)
    #     dataset.N = 1
    #     self.dataset = dataset

    def test_tpfp(self):
        pred = torch.Tensor(
                    [
                        [# example 1: 1102
                        [0,0,1,0], # 0tp 1fp
                        [1,1,0,0], # 1tp 1fp
                        [0,0,0,1]  # 1tp 0fp
                        ],
                        [# example 2: 0121
                        [1,0,0,0], # 1tp 0fp
                        [0,1,0,1], # 1tp 1fp
                        [0,0,1,0]  # 1tp 0fp
                        ]
                    ]) # 2 x 3 x 4

        target = torch.Tensor(
                    [
                     [# example 1: 1012
                      [0,1,0,0], # 1p
                      [1,0,1,0], # 2p
                      [0,0,0,1]  # 1p
                     ],
                     [# example 2: 0122
                      [1,0,0,0], # 1p
                      [0,1,0,0], # 1p
                      [0,0,1,1]  # 2p
                     ]
                    ]) # 2 x 3 x 4
 
        p_expected = torch.Tensor([2, 3, 3])
        tp_expected = torch.Tensor([1, 2, 2])
        fp_expected = torch.Tensor([1, 2, 0])
        p, tp, fp = Accuracy.tpfp(pred, target)
        print("p, tp, fp", p, tp, fp)
        self.assertTensorEqual(p, p_expected)
        self.assertTensorEqual(tp, tp_expected)
        self.assertTensorEqual(fp, fp_expected)

        precision = tp / (tp + fp)
        recall = tp / p
        f1 = 2 * recall * precision / (recall + precision)
        print(precision, recall, f1)

    # def test_accuracy(self):
    #     validation = Minibatches(self.dataset, 1)
    #     a = Accuracy(self.model, validation)
    #     precision, accuracy, f1 = a.run()
    #     print("precision, accuracy, f1", precision, accuracy, f1)

if __name__ == '__main__':
    unittest.main()

