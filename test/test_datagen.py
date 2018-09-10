# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
import os
import re
import torch
from test.smtagunittest import SmtagTestCase
from smtag.common.utils import timer
from smtag.datagen.ann2th import BratImport, AnnPreparator, AnnFeaturizer

class BratDatagenTest(SmtagTestCase):

    @classmethod
    def setUpClass(self): # runs only once because class method
        self.annotated_text ="Emerin deficiency at the nuclear membrane in patients with Emery-Dreifuss muscular dystrophy."
        self.annotations = ["T1\tDISO 0 17\tEmerin deficiency\n", "T2\tDISO 59 92\tEmery-Dreifuss muscular dystrophy\n"]
        self.basename = 'test_ann'
        self.ANN_EXTENSION = AnnPreparator.ANN_EXTENSION
        self.TXT_EXTENSION = AnnPreparator.TXT_EXTENSION
        self.brat_importer = BratImport(self.ANN_EXTENSION , self.TXT_EXTENSION)
        self.annotated_text_cleaned = re.sub("[\t\r\n]", " ", self.annotated_text)
        self.parsed_annotations = [
            {'tag_id': 'T1', 'type': 'DISO', 'start': 0, 'stop': 17, 'term': 'Emerin deficiency'}, 
            {'tag_id': 'T2', 'type': 'DISO', 'start': 59, 'stop': 92, 'term': 'Emery-Dreifuss muscular dystrophy'}
        ]
        self.example = {'text': self.annotated_text, 'annot': self.parsed_annotations, 'provenance': self.basename}
        self.raw_examples = [self.example]
        self.mypath = "/tmp"
        with open(os.path.join(self.mypath, self.basename+"."+self.TXT_EXTENSION), "w") as file:
            file.write(self.annotated_text)
        with open(os.path.join(self.mypath, "bogus.bogus"), "w") as file:
            file.write("THIS FILE SHOULD NOT BE UPLOADED")
        with open(os.path.join(self.mypath, self.basename+"."+self.ANN_EXTENSION), "w") as file:
            file.writelines(self.annotations)
        self.authorized_files = [self.basename]

    def test_brat_import_read_ann_from_file(self):
        annotations, annotated_text = self.brat_importer.read_ann_from_file(self.mypath, self.basename)
        # print(annotations)
        # print(self.annotations)
        self.assertEqual(annotations, self.annotations)
        self.assertEqual(annotated_text, self.annotated_text_cleaned)

    def test_brat_parse_annotations(self):
        parsed_annotations = self.brat_importer.parse_annotations(self.annotations)
        # print(parsed_annotations)
        # print(self.parsed_annotations)
        self.assertEqual(parsed_annotations, self.parsed_annotations)

    def test_brat_example_from_file(self):
        example = self.brat_importer.example_from_file(self.mypath, self.basename)
        # print(example)
        # print(self.example)
        self.assertEqual(example, self.example)

    def test_brat_select_files(self):
        authorized_files = self.brat_importer.select_files(self.mypath)
        # print(authorized_files)
        # print(self.authorized_files)
        self.assertEqual(authorized_files, self.authorized_files)
        self.assertNotIn('bogus', authorized_files)

    def test_brat_from_dir(self):
        raw_examples = self.brat_importer.from_dir(self.mypath)
        # print(raw_examples)
        # print(self.raw_examples)
        self.assertEqual(raw_examples, self.raw_examples)


if __name__ == '__main__':
    unittest.main()

