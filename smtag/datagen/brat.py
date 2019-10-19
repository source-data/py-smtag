# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import argparse
import re
from os import path, listdir
from ..common.progress import progress


class BratImport():

    ANN_EXTENSION = 'ann'
    TXT_EXTENSION = 'txt'
    ALLOWED_FILE_TYPES = set([ANN_EXTENSION, TXT_EXTENSION])

    @staticmethod
    def read_ann_from_file(mypath, basename):
        annotation_filename = basename + "." + BratImport.ANN_EXTENSION
        text_filename = basename + "." + BratImport.TXT_EXTENSION
        with open(path.join(mypath, text_filename), 'r') as f:
            annotated_text = f.read()
        annotated_text_cleaned = re.sub('[\r\n\t]', ' ', annotated_text)
        with open(path.join(mypath, annotation_filename), 'r') as f:
            annotations = f.readlines()
        return annotations, annotated_text_cleaned
    
    @staticmethod
    def parse_annotations(annotations):
        # example of annotation for the text "Emerin deficiency at the nuclear membrane in patients with Emery-Dreifuss muscular dystrophy."
        # T1 DISO 0 17   Emerin deficiency
        # T2 DISO 59 92  Emery-Dreifuss muscular dystrophy
        parsed_annotations = []
        parsing_pattern = re.compile(r'^(?P<tag_id>\w+)\t(?P<type>\w+) (?P<start>\d+) (?P<stop>\d+)\t(?P<term>.*)$')
        for a in annotations:
            r = re.search(parsing_pattern, a)
            d = {
                'tag_id': r.group('tag_id'),
                'type': r.group('type'),
                'start': int(r.group('start')),
                'stop': int(r.group('stop')),
                'term': r.group('term')
            }
            parsed_annotations.append(d)
        return parsed_annotations

    @staticmethod
    def example_from_file(mypath, basename):
        annotations, text = BratImport.read_ann_from_file(mypath, basename)
        parsed_annotations = BratImport.parse_annotations(annotations)
        example = {'text': text, 'annot': parsed_annotations, 'provenance': basename} #in the case of xml, annot and text are the same...
        return example

    @staticmethod
    def select_files(mypath):
        unique_basenames = []
        all_files_in_dir = [f for f in listdir(mypath) if path.isfile(path.join(mypath, f))]
        #reduce list to list of unique filenames
        for f in all_files_in_dir:
            try:
                basename, ext = f.split('.')
            except Exception as e:
                import pdb; pdb.set_trace()
                raise e
            if basename not in unique_basenames and ext in BratImport.ALLOWED_FILE_TYPES:
                unique_basenames.append(basename)
        return unique_basenames

    @staticmethod
    def from_dir(mypath):
        raw_examples = []
        filenames = BratImport.select_files(mypath)
        N = len(filenames)
        for i, b in enumerate(filenames):
            progress(i, N, status='loading examples')
            example = BratImport.example_from_file(mypath, b)
            if example['annot']:
                raw_examples.append(example)
            else:
                print("\n skipping {}: no annotations.".format(b))
        return raw_examples

