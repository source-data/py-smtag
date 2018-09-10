# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import argparse
import re
from os import path, listdir
from .dataprep import DataPreparator
from .featurizer import AnnFeaturizer
from ..common.progress import progress


class BratImport():

    def __init__(self, ann_extension, txt_exension):
        self.ANN_EXTENSION = ann_extension
        self.TXT_EXTENSION = txt_exension
        self.ALLOWED_FILE_TYPES = set([self.ANN_EXTENSION, self.TXT_EXTENSION])


    def read_ann_from_file(self, mypath, basename):
        annotation_filename = basename + "." + self.ANN_EXTENSION
        text_filename = basename + "." + self.TXT_EXTENSION
        with open(path.join(mypath, text_filename), 'r') as f:
            annotated_text = f.read()
        annotated_text_cleaned = re.sub('[\r\n\t]', ' ', annotated_text)
        with open(path.join(mypath, annotation_filename), 'r') as f:
            annotations = f.readlines()
        return annotations, annotated_text_cleaned
    
    def parse_annotations(self, annotations):
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

    def example_from_file(self, mypath, basename):
        annotations, text = self.read_ann_from_file(mypath, basename)
        parsed_annotations = self.parse_annotations(annotations)
        example = {'text': text, 'annot': parsed_annotations, 'provenance': basename} #in the case of xml, annot and text are the same...
        return example

    def select_files(self, mypath):
        unique_basenames = []
        all_files_in_dir = [f for f in listdir(mypath) if path.isfile(path.join(mypath, f))]
        #reduce list to list of unique filenames
        for f in all_files_in_dir:
            basename, ext = f.split('.')
            if basename not in unique_basenames and ext in self.ALLOWED_FILE_TYPES:
                unique_basenames.append(basename)
        return unique_basenames

    def from_dir(self, mypath):
        raw_examples = []
        filenames = self.select_files(mypath)
        N = len(filenames)
        for i, b in enumerate(filenames):
            progress(i, N, status='loading examples')
            raw_examples.append(self.example_from_file(mypath, b))
        return raw_examples


class AnnPreparator(DataPreparator):
    
    ANN_EXTENSION = 'ann'
    TXT_EXTENSION = 'txt'

    def __init__(self, parser):
        super(AnnPreparator, self).__init__(parser) #hopefully parser is mutable otherwise use self.parser
        parser.add_argument('dir', type=str, default='../data/test_brat', help='path to directory to scan')
        self.options = self.set_options(parser.parse_args())
        if self.options['verbose']: print(self.options)
        super(AnnPreparator, self).main()

    @staticmethod
    #implements @abstractmethod
    def set_options(args):
        options = super(AnnPreparator, AnnPreparator).set_options(args)
        options['source'] = args.dir
        return options

    #implements @abstractmethod
    def import_examples(self, mypath):
        raw_examples = BratImport(self.ANN_EXTENSION,self.TXT_EXTENSION).from_dir(mypath)
        return raw_examples

    #implements @abstractmethod
    def build_feature_dataset(self, examples):

        dataset = []
        for ex in examples:
            text = ex['text']
            features = AnnFeaturizer.ann2features(ex)
            provenance = {'id':ex['provenance'],'index': 0}
            dataset.append({'provenance': provenance, 'text': text, 'features': features})
        return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates text and tensor files from brat annotations for smtag training")
    p = AnnPreparator(parser)
