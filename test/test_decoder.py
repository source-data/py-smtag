# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from collections import OrderedDict
from smtag.common.utils import tokenize
from test.smtagunittest import SmtagTestCase
from smtag.predict.decode import Decoder
from smtag.common.mapper import Catalogue
from smtag.common.converter import StringList


class DecoderTest(SmtagTestCase):


    def test_decode(self):
        '''
        Testing the decoding without fusion.
        '''
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,1.0 ,1.0 ,0   ,1.0 ,1.0 ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [1.0 ,1.0 ,0   ,0   ,1.0 ,0   ,0   ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ]
                                  ]])

        group = 'test'
        concepts = [Catalogue.GENEPROD, Catalogue.UNTAGGED]
        semantic_groups = OrderedDict([(group, concepts)])
        d = Decoder(StringList([input_string]), prediction, semantic_groups)
        d.decode()
        print([t.text for t in d.token_lists[0]])
        print(d.concepts[0])
        print(d.scores[0])
        print(d.char_level_concepts[0])
        #                                          A                   ge                  ne                  or                  others
        expected_concepts = OrderedDict([('test', [Catalogue.UNTAGGED, Catalogue.GENEPROD, Catalogue.GENEPROD, Catalogue.UNTAGGED, Catalogue.UNTAGGED])])
        self.assertEqual(expected_concepts, d.concepts[0])

    def test_fuse_adjacent_1(self):
        '''
        Testing the fusion between two similarly labeled terms separated by a tab.
        '''
        input_string = 'A\tge\tne\tor\tothers'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,1.0 ,1.0 ,0.2  ,1.0 ,1.0 ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [1.0 ,1.0 ,0   ,0   ,0.8  ,0   ,0   ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ]
                                  ]])
        group = 'test'
        concepts = [Catalogue.GENEPROD, Catalogue.UNTAGGED]
        semantic_groups = OrderedDict([(group, concepts)])
        d = Decoder(StringList([input_string]), prediction, semantic_groups)
        d.decode()
        d.fuse_adjacent()
        print([t.text for t in d.token_lists[0]])
        print(d.concepts[0])
        print(d.scores[0])
        print(d.char_level_concepts[0])
        #                                          A                   gene                or                  others
        expected_concepts = OrderedDict([('test', [Catalogue.UNTAGGED, Catalogue.GENEPROD, Catalogue.UNTAGGED, Catalogue.UNTAGGED])])
        self.assertEqual(expected_concepts, d.concepts[0])


    def test_fuse_adjacent_2(self):
        '''
        Testing the fusion of two terms at the end of the string.
        '''
        input_string = 'A ge n'
        prediction = torch.Tensor([[#A         g    e         n    
                                    [0   ,0   ,0.99,0.99,0.6 ,0.99],
                                    [1.  ,1.  ,0    ,0   ,0  ,0   ]
                                  ]])
        group = 'test'
        concepts = [Catalogue.GENEPROD, Catalogue.UNTAGGED]
        semantic_groups = OrderedDict([(group, concepts)])
        d = Decoder(StringList([input_string]), prediction, semantic_groups)
        d.decode()
        d.fuse_adjacent()
        print([t.text for t in d.token_lists[0]])
        print(d.concepts[0])
        print(d.scores[0])
        print(d.char_level_concepts[0])
        #                                          A                   ge n
        expected_concepts = OrderedDict([('test', [Catalogue.UNTAGGED, Catalogue.GENEPROD])])
        self.assertEqual(expected_concepts, d.concepts[0])

    def test_fuse_adjacent_3(self):
        '''
        Testing the fusion of two terms separated by nothing at the end of the string.
        '''
        input_string = 'A ge-n'
        prediction = torch.Tensor([[#A         g    e    -   n 
                                    [0   ,0   ,0.99,0.99,0.99,0.99],
                                    [1.  ,1.  ,0   ,0   ,0   ,0   ]
                                  ]])
        group = 'test'
        concepts = [Catalogue.GENEPROD, Catalogue.UNTAGGED]
        semantic_groups = OrderedDict([(group, concepts)])
        d = Decoder(StringList([input_string]), prediction, semantic_groups)
        d.decode()
        d.fuse_adjacent()
        print([t.text for t in d.token_lists[0]])
        print(d.concepts[0])
        print(d.scores[0])
        print(d.char_level_concepts[0])
        #                                          A                   ge-n
        expected_concepts = OrderedDict([('test', [Catalogue.UNTAGGED, Catalogue.GENEPROD])])
        self.assertEqual(expected_concepts, d.concepts[0])

    def test_cat(self):
        '''
        Testing the decoding without fusion.
        '''
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,1.0 ,1.0 ,0   ,1.0 ,1.0 ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [1.0 ,1.0 ,0   ,0   ,1.0 ,0   ,0   ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ]
                                  ]])

        group = 'test'
        concepts = [Catalogue.GENEPROD, Catalogue.UNTAGGED]
        semantic_groups = OrderedDict([(group, concepts)])
        d = Decoder(StringList([input_string]), prediction, semantic_groups)
        d.decode()
        d.fuse_adjacent()
        cloned = d.clone()
        d.cat_(cloned)
        print(d.prediction)
        print([t.text for t in d.token_lists[0]])
        print(d.concepts[0])
        print(d.scores[0])
        print(d.char_level_concepts[0])
        self.assertTensorEqual(torch.cat((prediction, prediction),1), d.prediction)

if __name__ == '__main__':
    unittest.main()
