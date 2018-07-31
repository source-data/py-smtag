# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from common.operations import t_replace
from test.smtagunittest import SmtagTestCase
from common.converter import Converter

class OperationsTest(SmtagTestCase):
    

    def test_replace(self):
        x = torch.Tensor(
                          [
                           [ # example 1
                            [0,1,0,1],
                            [1,1,0,0],
                            [0,0,0,1]
                           ],
                           [ # example 2
                            [0,1,0,1],
                            [1,1,0,0],
                            [0,0,0,1]
                           ]
                         ]) # 2 x 3 x 4
        mask = torch.Tensor([
                             [0,1,1,0], # example 1
                             [1,0,1,0]  # example 2
                            ] 
                           ) # 2 x 4
        replacement = torch.Tensor([
                                    [
                                     [1],
                                     [1],
                                     [1]
                                    ]
                                   ]) # 1 x 3 x 1
        replaced = t_replace(x, mask, replacement)
        expected = torch.Tensor(
                          [
                           [
                            [0,1,1,1],
                            [1,1,1,0],
                            [0,1,1,1]
                           ],
                           [
                            [1,1,1,1],
                            [1,1,1,0],
                            [1,0,1,1]
                           ]
                         ]) # 2 x 3 x 4
        self.assertTensorEqual(expected, replaced)


    def test_anonymize(self):
        text = "hallo"
        x = Converter.t_encode(text)
        mask = torch.Tensor([[0,0,1,1,0]])
        character = "&"
        replacement = Converter.t_encode(character)
        anonymized = t_replace(x, mask, replacement)
        expected = "ha&&o"
        results = Converter.t_decode(anonymized)
        self.assertEqual(expected, results)


if __name__ == '__main__':
    unittest.main()

