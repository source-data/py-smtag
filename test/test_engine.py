# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from xml.etree.ElementTree import fromstring
from smtag.common.utils import timer
from smtag.predict.engine import SmtagEngine
from test.smtagunittest import SmtagTestCase
from smtag.predict.cartridges import NO_VIZ
#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()


class EngineTest(SmtagTestCase):

    @classmethod
    def setUpClass(self): # run only once
        self.engine = SmtagEngine(NO_VIZ)
        self.engine.DEBUG = False
        self.text_examples = ["We analyzed brain and muscle from Creb1-/- knockout mice after bafilomycin A treatment."] * 2

    def repeated_run(self, method, examples, n=2):
        ml_old = []
        for i in range(n):
            ml = method(examples, sdtag='sd-tag', format='xml')
            print(ml)
            self.assertIsInstance(ml, list)
            self.assertIsInstance(ml[0], str)
            self.assertEqual(len(ml), len(examples))
            if ml_old:
                self.assertEqual(ml_old, ml)
            ml_old = ml

    @timer
    def test_smtag(self):
        self.repeated_run(self.engine.smtag, self.text_examples)  

    def test_tag(self):
        self.repeated_run(self.engine.tag, self.text_examples)

    def test_entity(self):
        self.repeated_run(self.engine.entity, self.text_examples)

    def test_panelize(self):
        self.repeated_run(self.engine.panelizer, self.text_examples)

    def test_role(self):
        pretagged = '<smtag>We analyzed <sd-tag type="tissue" type_score="47">brain</sd-tag> and <sd-tag type="tissue" type_score="44">muscle</sd-tag> from <sd-tag type="geneprod" type_score="49" role_score="46">Creb1</sd-tag>-/- knockout <sd-tag type="organism" type_score="49">mice</sd-tag> after <sd-tag type="small_molecule" type_score="49">bafilomycin A</sd-tag> treatment.</smtag>'
        ml = self.engine.role([pretagged]*2, sdtag='sd-tag')
        expected_inner_text = "".join([x for x in fromstring(pretagged).itertext()])
        self.assertEqual(len(ml), 2)
        self.assertIsInstance(ml[0], bytes)
        inner_text = "".join([x for x in fromstring(ml[0]).itertext()])
        self.assertEqual(expected_inner_text, inner_text)

if __name__ == '__main__':
    unittest.main()
