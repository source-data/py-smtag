# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from smtag.common.utils import tokenize
from test.smtagunittest import SmtagTestCase
from smtag.predict.binarize import Binarized
from smtag.common.mapper import Catalogue

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
                                    [0.  ,0.  ,1.   ,0.  ,0.  ,1.  ,0.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_stop = torch.Tensor([[
                                    [0.  ,0.  ,0.   ,1.  ,0.  ,0.  ,1.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])
        expected_marks = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,1.  ,0.  ,1.  ,1.  ,0. ,0.  ,0.  ,0.,  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ]
                                     ]])

        b = Binarized([input_string], prediction, [Catalogue.GENEPROD])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        
        print("\n")
        print("".join([str(int(x)) for x in list(b.start.view(b.start.numel()))]))
        print("".join([str(int(x)) for x in list(b.stop.view(b.stop.numel()))]))
        print("".join([str(int(x)) for x in list(b.marks.view(b.marks.numel()))]))
        print(",".join([str(int(x)) for x in list(b.score.view(b.marks.numel()))]))
        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)

    def test_fuse_adjascent_1(self):
        '''
        Testing the fusion between two similarly labeled terms separated by a tab.
        '''
        input_string = 'A\tge\tne\tor\tothers'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,0.99,0.99,0.6 ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ]
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

        b = Binarized([input_string], prediction, [Catalogue.GENEPROD])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        print("\nFuse with spacer")
        print("".join([str(int(x)) for x in list(b.start.view(b.start.numel()))]))
        print("".join([str(int(x)) for x in list(b.stop.view(b.stop.numel()))]))
        print("".join([str(int(x)) for x in list(b.marks.view(b.marks.numel()))]))
        print(",".join([str(int(x)) for x in list(b.score.view(b.marks.numel()))]))

        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)


    def test_fuse_adjascent_2(self):
        '''
        Testing the fusion of two terms at the end of the string.
        '''
        input_string = 'A ge n'
        prediction = torch.Tensor([[#A         g    e         n    
                                    [0   ,0   ,0.99,0.99,0.6 ,0.99]
                                  ]])
        expected_start = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,0.  ,0.  ,0. ]
                                     ]])
        expected_stop = torch.Tensor([[
                                    [0.  ,0.  ,0.   ,0.  ,0.  ,1.  ]
                                     ]])
        expected_marks = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,1.  ,1.  ,1.  ]
                                     ]])

        b = Binarized([input_string], prediction, [Catalogue.GENEPROD])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        
        print("\n fuse at the end")
        print("".join([str(int(x)) for x in list(b.start.view(b.start.numel()))]))
        print("".join([str(int(x)) for x in list(b.stop.view(b.stop.numel()))]))
        print("".join([str(int(x)) for x in list(b.marks.view(b.marks.numel()))]))
        print(",".join([str(int(x)) for x in list(b.score.view(b.marks.numel()))]))

        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)

    def test_fuse_adjascent_3(self):
        '''
        Testing the fusion of two terms separated by nothing at the end of the string.
        '''
        input_string = 'A ge-n'
        prediction = torch.Tensor([[#A         g    e    -   n 
                                    [0   ,0   ,0.99,0.99,0.99,0.99]
                                  ]])
        expected_start = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,0.  ,0. ,0. ]
                                     ]])
        expected_stop = torch.Tensor([[
                                    [0.  ,0.  ,0.   ,0.  ,0.  ,1.  ]
                                     ]])
        expected_marks = torch.Tensor([[
                                    [0.  ,0.  ,1.   ,1.  ,1.  ,1.  ]
                                     ]])

        b = Binarized([input_string], prediction, [Catalogue.GENEPROD])
        token_list = tokenize(input_string)
        print(token_list)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        
        print("\nfuse with no spacer and at the end")
        print("".join([str(int(x)) for x in list(b.start.view(b.start.numel()))]))
        print("".join([str(int(x)) for x in list(b.stop.view(b.stop.numel()))]))
        print("".join([str(int(x)) for x in list(b.marks.view(b.marks.numel()))]))
        print(",".join([str(int(x)) for x in list(b.score.view(b.marks.numel()))]))

        self.assertTensorEqual(expected_start, b.start)
        self.assertTensorEqual(expected_stop, b.stop)
        self.assertTensorEqual(expected_marks, b.marks)


if __name__ == '__main__':
    unittest.main()
