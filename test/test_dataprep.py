# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from xml.etree.ElementTree  import XML, parse, tostring, fromstring
from test.smtagunittest import SmtagTestCase
from smtag.predict.binarize import Binarized
from smtag.datagen.convert2th import DataPreparator
from smtag import config

class TestDataGen(SmtagTestCase):

    @classmethod
    def setUpClass(self): 
        self.s= '''
        <Article doi="bidon"><figure-caption><sd-panel><sd-tag type="gene" role="normalized"><i>crs1ß</i><sup>+</sup></sd-tag> and <sd-tag type="gene" role="assayed">CA VIII</sd-tag> in the <sd-tag type="tissue" role="experiment">brain</sd-tag> and <sd-tag type="tissue" role="experiment">stomach</sd-tag> and <sd-tag type="tissue" role="experiment">duodenum</sd-tag>.<graphic href="https://api.sourcedata.io/file.php?panel_id=singleton" /></sd-panel></figure-caption></Article>
        '''
        self.xmls = tostring(fromstring(self.s))
        self.xml = fromstring(self.xmls)
        self.options = {
                    'length': 120,
                    'sampling_mode': 'window',
                    'random_shifting': True,
                    'padding': 20,
                    'verbose': False,
                    'iterations': 1,
                    'namebase': 'test20',
                    'compendium': 'test20',
                    'anonymize': [],
                    'enrich': [],
                    'exclusive': [],
                    'ocr': True
        }

    def test_exclusive(self):
        dataprep = DataPreparator(self.options)
        xml = dataprep.exclusive(self.xml, ['.//sd-tag[@type="gene"]'])
        xml_s = tostring(xml)
        xml_s_expected = tostring(fromstring('''
        <Article doi="bidon"><figure-caption><sd-panel><sd-tag type="gene" role="normalized"><i>crs1ß</i><sup>+</sup></sd-tag> and <sd-tag type="gene" role="assayed">CA VIII</sd-tag> in the <sd-tag>brain</sd-tag> and <sd-tag>stomach</sd-tag> and <sd-tag>duodenum</sd-tag>.<graphic href="https://api.sourcedata.io/file.php?panel_id=singleton" /></sd-panel></figure-caption></Article>
        '''))
        # print(xml_excl_s)
        # print(xml_s_expected)
        self.assertEqual(xml_s, xml_s_expected)

    def test_anonymize(self):
        dataprep = DataPreparator(self.options)
        xml = dataprep.anonymize(self.xml, ['.//sd-tag[@type="gene"]'])
        xml_s = tostring(xml)
        gene1 = config.marking_char * len("crs1ß+")
        gene2 = config.marking_char * len("CA VIII")
        xml_s_expected = tostring(fromstring('''
        <Article doi="bidon"><figure-caption><sd-panel><sd-tag type="gene" role="normalized">'''+gene1+'''</sd-tag> and <sd-tag type="gene" role="assayed">'''+gene2+'''</sd-tag> in the <sd-tag type="tissue" role="experiment">brain</sd-tag> and <sd-tag type="tissue" role="experiment">stomach</sd-tag> and <sd-tag type="tissue" role="experiment">duodenum</sd-tag>.<graphic href="https://api.sourcedata.io/file.php?panel_id=singleton" /></sd-panel></figure-caption></Article>
        '''))
        print(xml_s)
        # print(xml_s_expected)
        self.assertEqual(xml_s, xml_s_expected)
        L_expected = len("".join([s for s in fromstring(self.s).itertext()]))
        L = len("".join([s for s in xml.itertext()]))
        self.assertEqual(L, L_expected)

    def test_enrich(self):
        dataprep = DataPreparator(self.options)
        enrich_gene = dataprep.enrich(self.xml, ['.//sd-tag[@type="gene"]'])
        enrich_cell = dataprep.enrich(self.xml, ['.//sd-tag[@type="cell"]'])
        self.assertTrue(enrich_gene)
        self.assertFalse(enrich_cell)

if __name__ == '__main__':
    unittest.main()