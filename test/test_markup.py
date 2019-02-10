# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from collections import OrderedDict
from xml.etree.ElementTree import fromstring, tostring
from smtag.common.utils import tokenize
from smtag.predict.decode import Decoder
from smtag.predict.markup import XMLElementSerializer, HTMLElementSerializer, Serializer
from smtag.common.utils import timer
from smtag.common.mapper import Catalogue
from smtag.predict.updatexml import updatexml_

class SerializerTest(unittest.TestCase):

    def test_element_serializer(self):
        tag = 'sd-tag'
        current_concepts = OrderedDict([('role', Catalogue.INTERVENTION), ('crap', Catalogue.UNTAGGED), ('crap2', Catalogue.UNTAGGED), ('entities', Catalogue.PROTEIN)])
        inner_text = 'test text'
        scores = OrderedDict([('role', 0.1), ('crap', 0.2), ('crap2', 0.01), ('entities', 0.9)])
        expected_xml_string = '<sd-tag role="intervention" type="protein" role_score="10" type_score="90">test text</sd-tag>'
        expected_html_string = '<span class="sd-tag role_intervention type_protein role_score_10 type_score_90">test text</span>'
        xml_string = XMLElementSerializer.make_element(tag, current_concepts, inner_text, scores)
        html_string = HTMLElementSerializer.make_element(tag, current_concepts, inner_text, scores)
        print(xml_string)
        print(html_string)
        self.assertEqual(expected_xml_string, xml_string)
        self.assertEqual(expected_html_string, html_string)


    def test_serializer_1(self):
        '''
        Testing tagging of multiple token ("ge ne" as type="gene") with one semantic group ('entities').
        '''
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,1.  ,1.  ,1. ,1.  ,1.  ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ], # geneprod
                                    [1.  ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ], # small_molecule
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.8 ,0.8 ,0.8 ,0.8 ,0.8 ,0.8 ], # cell
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.2 ,0.2 ,0.1 ,0.1 ,0.1 ,0.1 ], # protein
                                    [0   ,1   ,0   ,0   ,1.  ,0   ,0   ,1   ,1   ,1   ,1   ,0   ,0   ,0.1 ,0.1 ,0.1 ,0.1 ]  # untagged
                                  ]])

        #self, text_examples, prediction, output_semantics
        group = 'entities'
        output_semantics = Catalogue.from_list(['geneprod','small_molecule','cell','protein', 'untagged'])
        print(output_semantics)
        semantic_groups = OrderedDict([(group, output_semantics)])
        d = Decoder(input_string, prediction, semantic_groups)
        d.decode()
        print([t.text for t in d.token_list])
        serializer = Serializer(tag="sd-tag", format="xml")
        predicted_xml_string = serializer.serialize(d)
        expected_xml_string = '<smtag><sd-tag type="small_molecule" type_score="100">A</sd-tag> <sd-tag type="geneprod" type_score="100">ge ne</sd-tag> or <sd-tag type="cell" type_score="80">others</sd-tag></smtag>'
        print(predicted_xml_string)
        self.assertEqual(predicted_xml_string, expected_xml_string)

    # @timer
    # def test_serializer_3(self):
    #     '''
    #     Testing tagging with staggered features and xml escaping.
    #     '''
    #     input_string = 'A gene or oth>rs'
    #     prediction = torch.Tensor([[#A         g    e    n    e         o    r         o    t    h    >    r    s
    #                                 [0   ,0   ,0   ,0.99,0.99,0.99,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
    #                                 [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
    #                                 [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
    #                                 [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99]
    #                               ]])

    #     #self, text_examples, prediction, output_semantics
    #     b = Binarized([input_string], prediction, Catalogue.from_list(['geneprod','assayed','intervention','protein']))
    #     token_list = tokenize(input_string)
    #     b.binarize_with_token([token_list])
    #     b.fuse_adjascent()
    #     for _ in range(100):
    #         serializer = Serializer(tag="sd-tag", format="xml")
    #         predicted_xml_string = serializer.serialize(b)[0]
    #     expected_xml_string = '<smtag>A <sd-tag type="geneprod" type_score="99">gene</sd-tag> <sd-tag type="geneprod" role="assayed" type_score="99" role_score="99">or</sd-tag> <sd-tag role="intervention" type="protein" role_score="99" type_score="99">oth&gt;rs</sd-tag></smtag>'
    #     print(predicted_xml_string)
    #     self.assertEqual(predicted_xml_string, expected_xml_string)


    # def test_serializer_4(self):
    #     '''
    #     Testing tagging of ambiguous predictions "others" as both intervention and assayed (with lower score)
    #     '''
    #     input_string = 'A ge ne or others'
    #     prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
    #                                 [0   ,0   ,0.99,0.99,0.6 ,0.99,0.99,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ],
    #                                 [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
    #                                 [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
    #                                 [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.98,0.98,0.98,0.98,0.98,0.98]
    #                               ]])

    #     #self, text_examples, prediction, output_semantics
    #     b = Binarized([input_string], prediction, Catalogue.from_list(['geneprod','small_molecule','intervention','assayed']))
    #     token_list = tokenize(input_string)
    #     b.binarize_with_token([token_list])
    #     b.fuse_adjascent()
    #     serializer = Serializer(tag="sd-tag", format="xml")
    #     predicted_xml_string = serializer.serialize(b)[0]
    #     expected_xml_string = '<smtag>A <sd-tag type="geneprod" type_score="99">ge ne</sd-tag> or <sd-tag type="small_molecule" role="intervention" type_score="99" role_score="99">others</sd-tag></smtag>'
    #     #print(predicted_xml_string)
    #     self.assertEqual(predicted_xml_string, expected_xml_string)

    def test_serializer_updatexml(self):
        '''
        Test the update of a pretagged xml object
        '''
        xml_string = '<sd-panel>A <sd-tag type="geneprod">ge ne</sd-tag> or <sd-tag type="protein">others</sd-tag></sd-panel>'
        xml = fromstring(xml_string)
        expected_xml_string = tostring(fromstring('<sd-panel>A <sd-tag type="geneprod">ge ne</sd-tag> or <sd-tag role="intervention" type="protein">others</sd-tag></sd-panel>'))
        input_string = 'A ge ne or others'
        prediction = torch.Tensor([[#A         g    e         n    e         o    r         o    t    h    e    r    s
                                    [0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0.99,0.99,0.99,0.99,0.99,0.99],
                                    [1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,0   ,0   ,0   ,0   ,0   ,0   ]
                                  ]])

        group = 'roles'
        output_semantics = Catalogue.from_list(['intervention', 'untagged'])
        semantic_groups = OrderedDict([(group, output_semantics)])
        d = Decoder(input_string, prediction, semantic_groups)
        d.decode()
        print([t.text for t in d.token_list])
        updatexml_(xml, d)
        resulting_xml_string = tostring(xml)
        print(resulting_xml_string)
        self.assertEqual(expected_xml_string, resulting_xml_string)

if __name__ == '__main__':
    unittest.main()

