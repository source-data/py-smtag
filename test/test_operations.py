#x = torch.Tensor([[0,1,0,1],[1,1,0,0],[0,0,0,1]])
    #replacement = torch.Tensor([1,0,1]).resize_(3,1)

# -*- coding: utf-8 -*-
import unittest
import torch
from smtag.operations import replace
from smtag.utils import assertTensorEqual
from smtag.converter import Converter

class ConverterTest(unittest.TestCase):
    
    @staticmethod
    def assertTensorEqual(x, y):
        return assertTensorEqual(x, y)

    def test_replace(self):
        x = torch.Tensor([[0,1,0,1],
                          [1,1,0,0],
                          [0,0,0,1]]) # 3 x 4
        mask = torch.Tensor([0,1,1,0]) # 4
        replacement = torch.Tensor([1,
                                    0,
                                    1]) # 3
        replaced = replace(x, mask, replacement)
        expected = torch.Tensor([[0,1,1,1],
                                 [1,0,0,0],
                                 [0,1,1,1]])
        self.assertTensorEqual(expected, replaced)

    
    def test_anonymize(self):
        text = "hallo"
        x = Converter.t_encode(text)
        x.resize_(32, len(text))
        mask = torch.Tensor([0,0,1,1,0])
        character = "&"
        replacement = Converter.t_encode(character).resize_(32,1)
        anonymized = replace(x, mask, replacement).resize_(1, 32, len(text))
        expected = "ha&&o"
        results = Converter.t_decode(anonymized)
        self.assertEqual(expected, results)




if __name__ == '__main__':
    unittest.main()

