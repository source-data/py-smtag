# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import argparse
import re
from os import path, listdir
from datagen.dataprep import DataPreparator
from datagen.featurizer import AnnFeaturizer
from common.progress import progress

class AnnPreparator(DataPreparator):
    ANN_EXTENSION = 'ann'
    TXT_EXTENSION = 'txt'
    ALLOWED_FILE_TYPES = [ANN_EXTENSION, TXT_EXTENSION]

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
        def read_ann_from_file(mypath, basename):
            annotation_filename = basename + '.' + self.ANN_EXTENSION
            text_filename = basename + '.' + self.TXT_EXTENSION
            example = {}
            annot = []
            text = open(path.join(mypath, text_filename), 'r').read()
            text = re.sub('[\r\n\t]', ' ', text)
            with open(path.join(mypath, annotation_filename), 'r') as f:
                annotation_lines = f.readlines()

            #example:
            #Emerin deficiency at the nuclear membrane in patients with Emery-Dreifuss muscular dystrophy.
            #T1 DISO 0 17   Emerin deficiency
            #T2 DISO 59 92  Emery-Dreifuss muscular dystrophy

            parsing_pattern = re.compile(r'^(\w+)\t(\w+) (\d+) (\d+)\t(.*)$')
            for l in annotation_lines:
                r = re.search(parsing_pattern, l)
                a = {}
                a['tag_id'] = r.group(1)
                a['type'] = r.group(2)
                a['start'] = int(r.group(3))
                a['stop'] = int(r.group(4))
                a['term'] = r.group(5)
                annot.append(a)

            example = {'text': text, 'annot': annot, 'provenance': basename} #in the case of xml, annot and text are the same...
            return example

        filenames = []
        all_files_in_dir = [f for f in listdir(mypath) if path.isfile(path.join(mypath, f))]
        #reduce list ot list of unique filenames
        for f in all_files_in_dir:
            basename, ext = f.split('.')
            if basename not in filenames and ext in self.ALLOWED_FILE_TYPES:
                filenames.append(basename)

        raw_examples = []
        total = len(filenames)
        count = 0
        for b in filenames:
            progress(count, total, status='loading examples'); count += 1
            raw_examples.append(read_ann_from_file(mypath, b))

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
