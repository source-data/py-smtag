# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import torch
from smtag.common.utils import timer
from smtag.predict.engine import SmtagEngine
from test.smtagunittest import SmtagTestCase

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()


class EngineTest(SmtagTestCase):

    @classmethod
    def setUpClass(self): # run only once
        self.engine = SmtagEngine()
        self.engine.DEBUG = False
        self.text_example = "We analyzed brain and muscle from Creb1-/- knockout mice after bafilomycin A treatment."

    def repeated_run(self, method, example, n=2):
        for i in range(n):
            ml = method(example, sdtag='sd-tag', format='xml')
            # print(ml)
            self.assertIsInstance(ml, str)

    @timer
    def test_smtag(self):
        self.repeated_run(self.engine.smtag, self.text_example)  

    def test_tag(self):
        self.repeated_run(self.engine.tag, self.text_example)

    def test_entity(self):
        self.repeated_run(self.engine.entity, self.text_example)

    def test_panelize(self):
        ml = self.engine.panelizer(self.text_example, format="xml")
        self.assertIsInstance(ml, str)

    def test_role(self):
        pretagged = '<smtag>We analyzed <sd-tag type="tissue" type_score="47">brain</sd-tag> and <sd-tag type="tissue" type_score="44">muscle</sd-tag> from <sd-tag type="geneprod" type_score="49" role_score="46">Creb1</sd-tag>-/- knockout <sd-tag type="organism" type_score="49">mice</sd-tag> after <sd-tag type="small_molecule" type_score="49">bafilomycin A</sd-tag> treatment.</smtag>'
        ml = self.engine.role(pretagged, sdtag='sd-tag')
        self.assertIsInstance(ml, bytes)

if __name__ == '__main__':
    unittest.main()
