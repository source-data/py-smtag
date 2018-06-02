# -*- coding: utf-8 -*-
import unittest
import torch
from smtag.utils import tokenize
from smtag.binarize import Binarized
from smtag.serializer import XMLElementSerializer, HTMLElementSerializer, Serializer

class SerializerTest(unittest.TestCase):

    def test_element_serializer(self):
        #tag, on_features, inner_text
        tag = 'sd-tag'
        on_features = ['intervention', None, None, 'protein']
        inner_text = 'test text'
        expected_xml_string = '<sd-tag role="intervention" type="protein">test text</sd-tag>'
        expected_html_string = '<span class="sd-tag role_intervention type_protein">test text</span>'
        xml_string = XMLElementSerializer.make_element(tag, on_features, inner_text)
        html_string = HTMLElementSerializer.make_element(tag, on_features, inner_text)
        #print(xml_string)
        #print(html_string)
        self.assertEqual(expected_xml_string, xml_string)
        self.assertEqual(expected_html_string, html_string)

    def test_serializer(self):
        #[{xml:tagged_xml_string, html:tagged_html_string}] SDTags(input_string, entities, attrmap)
        input_string = 'A gene or protein.'
        prediction = torch.Tensor([[
                                   [0   ,0   ,0.99,0.99,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ], 
                                   [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0,   0,   0,   0   ], 
                                   [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ], 
                                   [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0   ,0   ]
                                  ]])
        
        #self, text_examples, prediction, output_semantics
        b = Binarized([input_string], prediction, ['gene','small_molecule','tissue','protein'])
        token_list = tokenize(input_string)
        b.binarize_with_token([token_list])
        serializer = Serializer(tag="sd-tag", format="xml")
        predicted_xml_string = serializer.serialize(b)[0]
        expected_xml_string = 'A <sd-tag type="gene">gene</sd-tag> or <sd-tag type="protein">protein</sd-tag>.'
        #expected_html_string = 'A <span class="sd-tag gene">gene</span> or <span class="sd-tag protein">protein</span>.'
        print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)

if __name__ == '__main__':
    unittest.main()

