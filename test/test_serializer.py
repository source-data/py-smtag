# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from smtag.common.utils import tokenize
from smtag.predict.binarize import Binarized
from smtag.predict.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.common.utils import timer
from smtag.common.mapper import Catalogue

class SerializerTest(unittest.TestCase):

    def test_element_serializer(self):
        #tag, on_features, inner_text
        tag = 'sd-tag'
        on_features = [Catalogue.INTERVENTION, None, None, Catalogue.PROTEIN]
        inner_text = 'test text'
        expected_xml_string = '<sd-tag role="intervention" type="protein">test text</sd-tag>'
        expected_html_string = '<span class="sd-tag role_intervention type_protein">test text</span>'
        xml_string = XMLElementSerializer.make_element(tag, on_features, inner_text)
        html_string = HTMLElementSerializer.make_element(tag, on_features, inner_text)
        #print(xml_string)
        #print(html_string)
        self.assertEqual(expected_xml_string, xml_string)
        self.assertEqual(expected_html_string, html_string)

    def test_serializer_1(self):
        '''
        Simple test to tag 2 words.
        '''
        input_string = 'A gene or protein.'
        prediction = torch.Tensor([[#A         g    e    n    e         o    r         p    r    o    t    e    i    n    .
                                    [0   ,0   ,0.99,0.99,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0,   0,   0,   0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0   ]
                                  ]])

        #self, text_examples, prediction, output_semantics
        b = Binarized([input_string], prediction, Catalogue.from_list(['gene','small_molecule','tissue','protein']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        serializer = Serializer(tag="sd-tag", format="xml")
        predicted_xml_string = serializer.serialize(b)[0]
        expected_xml_string = '<smtag>A <sd-tag type="gene">gene</sd-tag> or <sd-tag type="protein">protein</sd-tag>.</smtag>'
        #expected_html_string = 'A <span class="sd-tag gene">gene</span> or <span class="sd-tag protein">protein</span>.'
        #print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)

    def test_serializer_2(self):
        '''
        Testing tagging of multiple token ("ge ne" as type="gene")
        and multiple attributes for one terms ("others" as role="intervention" type="protein")
        '''
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,0.99,0.99,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99]
                                  ]])

        #self, text_examples, prediction, output_semantics
        b = Binarized([input_string], prediction, Catalogue.from_list(['geneprod','small_molecule','intervention','protein']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        serializer = Serializer(tag="sd-tag", format="xml")
        predicted_xml_string = serializer.serialize(b)[0]
        expected_xml_string = '<smtag>A <sd-tag type="geneprod">ge ne</sd-tag> or <sd-tag role="intervention" type="protein">others</sd-tag></smtag>'
        #print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)

    @timer
    def test_serializer_3(self):
        '''
        Testing tagging with staggered features and xml escaping.
        '''
        input_string = 'A gene or oth>rs'
        prediction = torch.Tensor([[#A         g    e    n    e         o    r         o    t    h    >    r    s
                                    [0   ,0   ,0.99,0.99,0.99,0.99,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99]
                                  ]])

        #self, text_examples, prediction, output_semantics
        b = Binarized([input_string], prediction, Catalogue.from_list(['geneprod','assayed','intervention','protein']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        for _ in range(100):
            serializer = Serializer(tag="sd-tag", format="xml")
            predicted_xml_string = serializer.serialize(b)[0]
        expected_xml_string = '<smtag>A <sd-tag type="geneprod">gene </sd-tag><sd-tag type="geneprod" role="assayed">or</sd-tag> <sd-tag role="intervention" type="protein">oth&gt;rs</sd-tag></smtag>'
        print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)

if __name__ == '__main__':
    unittest.main()

