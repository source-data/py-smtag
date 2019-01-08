# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from xml.etree.ElementTree import fromstring, tostring
from smtag.common.utils import tokenize
from smtag.predict.binarize import Binarized
from smtag.predict.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.common.utils import timer
from smtag.common.mapper import Catalogue
from smtag.predict.updatexml import updatexml_

class SerializerTest(unittest.TestCase):

    def test_element_serializer(self):
        #tag, on_features, inner_text
        tag = 'sd-tag'
        on_features = [Catalogue.INTERVENTION, None, None, Catalogue.PROTEIN]
        scores = [10,0,0,10]
        inner_text = 'test text'
        expected_xml_string = '<sd-tag role="intervention" type="protein" role_score="10" type_score="10">test text</sd-tag>'
        expected_html_string = '<span class="sd-tag role_intervention type_protein role_score_10 type_score_10">test text</span>'
        xml_string = XMLElementSerializer.make_element(tag, on_features, inner_text, scores)
        html_string = HTMLElementSerializer.make_element(tag, on_features, inner_text, scores)
        print(xml_string)
        print(html_string)
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
        expected_xml_string = '<smtag>A <sd-tag type="gene" type_score="99">gene</sd-tag> or <sd-tag type="protein" type_score="99">protein</sd-tag>.</smtag>'
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
                                    [0   ,0   ,0.99,0.99,0.6 ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
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
        expected_xml_string = '<smtag>A <sd-tag type="geneprod" type_score="99">ge ne</sd-tag> or <sd-tag role="intervention" type="protein" role_score="99" type_score="99">others</sd-tag></smtag>'
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
        expected_xml_string = '<smtag>A <sd-tag type="geneprod" type_score="99">gene</sd-tag> <sd-tag type="geneprod" role="assayed" type_score="99" role_score="99">or</sd-tag> <sd-tag role="intervention" type="protein" role_score="99" type_score="99">oth&gt;rs</sd-tag></smtag>'
        print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)


    def test_serializer_4(self):
        '''
        Testing tagging of ambiguous predictions "others" as both intervention and assayed (with lower score)
        '''
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,0.99,0.99,0.6 ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.98,0.98,0.98,0.98,0.98,0.98]
                                  ]])

        #self, text_examples, prediction, output_semantics
        b = Binarized([input_string], prediction, Catalogue.from_list(['geneprod','small_molecule','intervention','assayed']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        serializer = Serializer(tag="sd-tag", format="xml")
        predicted_xml_string = serializer.serialize(b)[0]
        expected_xml_string = '<smtag>A <sd-tag type="geneprod" type_score="99">ge ne</sd-tag> or <sd-tag type="small_molecule" role="intervention" type_score="99" role_score="99">others</sd-tag></smtag>'
        #print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)

    def test_serializer_updatexml(self):
        '''
        Test the update of a pretagged xml object
        '''
        xml_string = '<sd-panel>A <sd-tag type="geneprod">ge ne</sd-tag> or <sd-tag type="protein">others</sd-tag></sd-panel>'
        xml = fromstring(xml_string)
        expected_xml_string = tostring(fromstring('<sd-panel>A <sd-tag type="geneprod">ge ne</sd-tag> or <sd-tag role="intervention" type="protein" role_score="99">others</sd-tag></sd-panel>'))
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99]
                                  ]])

        #self, text_examples, prediction, output_semantics
        b = Binarized([input_string], prediction, Catalogue.from_list(['intervention']))
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        b.fuse_adjascent()
        updatexml_(xml, b)
        resulting_xml_string = tostring(xml)
        print(resulting_xml_string)
        self.assertEqual(expected_xml_string, resulting_xml_string)

if __name__ == '__main__':
    unittest.main()

