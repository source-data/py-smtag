# -*- coding: utf-8 -*-
import unittest
import torch
from smtag.utils import tokenize
from smtag.binarize import Binarized

class BinarizeTest(unittest.TestCase):

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
        
        b = Binarized([input_string], prediction, ['geneprod'])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        print(b.start)
        print(b.stop)
        print(b.marks)
        
        self.assertEqual([x for x in expected_start.resize_(18)], [x for x in b.start.resize_(18)])
        self.assertEqual([x for x in expected_stop.resize_(18)], [x for x in b.stop.resize_(18)])
        self.assertEqual([x for x in expected_marks.resize_(18)], [x for x in b.marks.resize_(18)])
    
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

        b = Binarized([input_string], prediction, ['geneprod'])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent(regex="\t")
        #print(b.start)
        #print(b.stop)
        #print(b.marks)
        self.assertEqual([x for x in expected_start.resize_(17)], [x for x in b.start.resize_(17)])
        self.assertEqual([x for x in expected_stop.resize_(17)], [x for x in b.stop.resize_(17)])
        self.assertEqual([x for x in expected_marks.resize_(17)], [x for x in b.marks.resize_(17)])

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

        b = Binarized([input_string], prediction, ['geneprod'])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        #print(b.start)
        #print(b.stop)
        #print(b.marks)

        self.assertEqual([x for x in expected_start.resize_(7)], [x for x in b.start.resize_(7)])
        self.assertEqual([x for x in expected_stop.resize_(7)], [x for x in b.stop.resize_(7)])
        self.assertEqual([x for x in expected_marks.resize_(7)], [x for x in b.marks.resize_(7)])



if __name__ == '__main__':
    unittest.main()