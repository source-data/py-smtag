# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from smtag.utils import tokenize
from test.smtagunittest import SmtagTestCase
from smtag.binarize import Binarized
from smtag.mapper import Factory

class BinarizeTest(SmtagTestCase):

    def test_binarize(self):
        '''
        Testing the binarization without fusion.
        '''
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    e    r    s
                                    [0   ,0   ,0.99,0.99,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ]
                                  ]])
        expected_start = torch.Tensor([[
                                    [ 0. ,0.  ,1.   ,0.  ,0.  ,1.  ,0.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_stop = torch.Tensor([[
                                    [ 0. ,0.  ,0.   ,1.  ,0.  ,0.  ,1.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_marks = torch.Tensor([[
                                    [ 0. ,0.  ,1.   ,1.  ,0.  ,1.  ,1.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        
        b = Binarized([input_string], prediction, Factory.from_list(['geneprod']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        print(b.start)
        print(b.stop)
        print(b.marks)
        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)
    
    def test_fuse_adjascent_1(self):
        '''
        Testing the fusion between two similarly labeled terms separated by a tab.
        '''
        input_string = 'A\tge\tne\tor\tothers'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,0.99,0.99,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ]
                                  ]])
        expected_start = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,0.  ,0.  ,0.  ,0.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_stop = torch.Tensor([[
                                    [0.  ,0.  ,0.   ,0.  ,0.  ,0.  ,1.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_marks = torch.Tensor([[
                                    [0.   ,0.  ,1.   ,1.  ,1.  ,1.  ,1.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])

        b = Binarized([input_string], prediction, Factory.from_list(['geneprod']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent(regex="\t")
        #print(b.start)
        #print(b.stop)
        #print(b.marks)

        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)


    def test_fuse_adjascent_2(self):
        '''
        Testing the fusion of two terms at the end of the string.
        '''
        input_string = 'A ge ne'
        prediction = torch.Tensor([[#A         g    e         n    e 
                                    [0   ,0   ,0.99,0.99,0   ,0.99,0.99]
                                  ]])
        expected_start = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_stop = torch.Tensor([[
                                    [0.  ,0.  ,0.   ,0.  ,0.  ,0.  ,1.  ]
                                     ]])
        expected_marks = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,1.  ,1.  ,1.  ,1.  ]
                                     ]])

        b = Binarized([input_string], prediction, Factory.from_list(['geneprod']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        #print(b.start)
        #print(b.stop)
        #print(b.marks)

        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)


if __name__ == '__main__':
    unittest.main()