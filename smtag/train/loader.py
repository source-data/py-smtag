# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from ..common.config import NBITS
import numpy as np
import torch
# import logging
import math
from random import shuffle
from ..common.mapper import Catalogue, concept2index
from ..common.progress import progress
from ..common.viz import Show
from ..common.utils import tokenize
from .. import config

# import logging.config
# logging.config.fileConfig('logging.conf')
# logger = logging.getLogger(__name__)


class Dataset:

    class LineTooLong(Exception):

        def __init__(self, line, max_length):
            self.line = line
            self.max_length = max_length

        def __str__(self):
            return "FATAL: Example line is too long: {} > {}".format(len(self.line), self.max_length)

    def __init__(self, N=0, nf_input=0, nf_output=0, L=0):
        self.N = N
        self.nf_input = nf_input
        self.nf_output = nf_output
        self.L = L
        self.text = [None] * N # this is to allow list assignment by index
        self.tokenized = [] # this will be filled only if add_token_lists explicitly called; slow and time consuming; used only for benchmarking
        self.provenance = [None] * N
        self.input = torch.zeros(self.N, self.nf_input, self.L)
        self.output = torch.zeros(self.N, self.nf_output, self.L)

    def from_files(self, basename):
        features_filename = "{}/{}.pyth".format(config.data_dir, basename)
        text_filename = '{}/{}.txt'.format(config.data_dir, basename)
        textcoded_filename = "{}/{}_textcoded.pyth".format(config.data_dir, basename)
        provenance_filename = '{}/{}.prov'.format(config.data_dir, basename)

        print("Loading {} as features for the dataset.".format(features_filename))
        self.output = torch.load(features_filename).float()
        self.N = self.output.size(0) #number of examples
        self.nf_output = self.output.size(1) #number of features
        self.L = self.output.size(2) #length of text snippet

        print("Loading {} as encoded text for the dataset.".format(textcoded_filename))
        self.textcoded = torch.load(textcoded_filename).float()

        print("Loading {} for the original texts of the dataset.".format(text_filename))
        with open(text_filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) > self.L :
                    raise LineTooLong(line, self.L)
                self.text.append(line)

        print("Loading {} as provenance info for the examples in the dataset.".format(provenance_filename))
        with open(provenance_filename, 'r') as f:
            for line in f:
                self.provenance.append(line)

        print("Dataset dimensions:")
        print("{} text examples of size {}".format(self.N, self.L))
        print("{} input features (in-channels).".format(self.nf_input))
        print("{} output features (out-channels).".format(self.nf_output))

    def add_token_lists(self):
        for t in self.text:
            self.tokenized.append(tokenize(t))


class Loader:

    def __init__(self, opt):
        self.selected_features = Catalogue.from_list(opt['selected_features'])
        self.collapsed_features = Catalogue.from_list(opt['collapsed_features'])
        self.overlap_features = Catalogue.from_list(opt['overlap_features'])
        self.features_as_input = Catalogue.from_list(opt['features_as_input'])
        self.validation_fraction = opt['validation_fraction']  # fraction of the whole dataset to be used for validation during training
        self.nf_input = NBITS

        self.nf_collapsed_feature = 0
        self.nf_overlap_feature = 0
        self.nf_output = len(self.selected_features)
        if self.features_as_input:
            self.nf_input += len(self.features_as_input)
        if self.collapsed_features:
            self.nf_output += 1
            self.index_of_collapsed_feature = self.nf_output - 1
        else:
            self.index_of_collapsed_feature = None
        if self.overlap_features:
            self.nf_output += 1
            self.index_of_overlap_feature = self.nf_output - 1
        else:
            self.index_of_overlap_feature = None

        # debugging
        print("nf.output=", self.nf_output)
        print("nf.input=", self.nf_input)
        print("index_of_collapsed_feature=", self.index_of_collapsed_feature)
        print("index_of_overlap_feature", self.index_of_overlap_feature)
        print("concept2index self.selected_features", [concept2index[f] for f in self.selected_features])
        print("concept2index self.collapsed_features", [concept2index[f] for f in self.collapsed_features])
        print("concept2index self.overlap_features", [concept2index[f] for f in self.overlap_features])


    def prepare_datasets(self, file_basename):
        """
        Processes a raw dataset into a ready-to-go dataset by converting text into tensor and assembling selected combinations of features into output features.
        Args:
            file_basename (str): the basename of the text, provenance and feature files to be uploaded and prepared.

        Returns:
            datasets (array of Dataset): the train/validation Dataset objects or the test Dataset that can be processed further by smtag.
        """
        raw_dataset = Dataset()
        raw_dataset.from_files(file_basename)
        N = raw_dataset.N
        L = raw_dataset.L
        nf = raw_dataset.nf_output # it already includes the last row for 'geneprod'!!
        
        assert N != 0, "zero examples!"

        # generate on the fly a 'virtual' geneprod feature as the union (sum) of protein and gene features
        #THIS HAS TO GO! BELONGS TO DATAPREP!!! Probably in Featurizer
        raw_dataset.output[ : , nf-1,  : ] = raw_dataset.output[ : , concept2index[Catalogue.GENE],  : ] + raw_dataset.output[ : ,  concept2index[Catalogue.PROTEIN], : ]

        print("Creating dataset with selected features {}, and shuffling {} examples.".format(self.selected_features, N))
        shuffled_indices = list(range(N))
        shuffle(shuffled_indices)
        datasets = {}
        if self.validation_fraction == 0:
            print("testset mode; for benchmarking")
            datasets["test"]= {} #--testset mode; for benchmarking
            datasets["test"]["first_example"] = 0
            #datasets["test"]["last_example"] = math.ceil(N*self.fraction)
        else:
            print("normal trainset and validation set mode")
            datasets["train"] = {} #--normal trainset and validation set mode
            datasets["valid"] = {}
            datasets["train"]["first_example"] = 0
            datasets["train"]["last_example"] = math.floor(N*(1-self.validation_fraction))
            datasets["valid"]["first_example"] = datasets["train"]["last_example"]
            datasets["valid"]["last_example"] = N
        # N = 11 examples [0,1,2,3,4,5,6,7,8,9,10], validation_fraction = 0.2
        # 8 training, 2 validation
        # train_first_example = 0
        # train_last_example = 8 = floor(11*(1-0.2)) = floor(8.8)
        # range(0,8) is 0,1,2,3,4,5,6,7
        # example_i = 3 => index = 3  - first_example = 3
        # valid first_example =  train_last_example = 8
        # valid_last_example = N
        # N_examples = N - valid_first_example = 11 - 8 = 3
        # list(range(8,11)) = is 8,9,10
        # example_i = 9 => index = 9 - first_example = 1
        for k in datasets: # k 'test' in testing mode otherwise 'valid' or 'train'
            first_example = datasets[k]['first_example']
            last_example = datasets[k]['last_example']
            N_examples = last_example - first_example
            dataset = Dataset(N_examples, self.nf_input, self.nf_output, L)

            print("Generating {} set with {} examples ({}, {})".format(k, N_examples, first_example, last_example))
            print("input dataset['{}'] tensor created".format(k))
            print("output dataset['{}'] tensor created".format(k))
            for example_i in range(first_example, last_example):
                i = shuffled_indices[example_i]
                index = example_i - first_example
                progress(index, N_examples, status="loading {} examples ({} to {}) into dataset['{}'] (example_i={},index={})".format(N_examples, first_example, last_example, k, example_i, index))
                #INPUT: TEXT SAMPLE ENCODING
                assert len(raw_dataset.text[i]) == L, "FATAL: {} > {} with {}".format(len(raw_dataset.text[i]), L, raw_dataset.text[i])
                dataset.text[index] = raw_dataset.text[i]
                dataset.provenance[index] = raw_dataset.provenance[i]
                dataset.input[index, 0:32 , : ] = raw_dataset.textcoded[i, : , : ]

                for j, f in enumerate(self.features_as_input):
                    # for example: j=0, => 32 + 0 = 32
                    dataset.input[index, 32 + j, : ] = raw_dataset.output[i, concept2index[f], : ]

                #OUTPUT SELECTION AND COMBINATION OF FEATURES
                for f in self.selected_features:
                    j = self.selected_features.index(f)
                    dataset.output[index, j, : ] = raw_dataset.output[i, concept2index[f], : ]

                #dataset.output[index, nf_collapsed_feature,:] is already initialized with zeros
                for f in self.collapsed_features:
                    dataset.output[index, self.index_of_collapsed_feature,  : ] += raw_dataset.output[i, concept2index[f], : ]

                #for overlap features, need initialization with ones
                if self.overlap_features:
                    dataset.output[index, self.index_of_overlap_feature, : ].fill_(1)
                for f in self.overlap_features:
                    dataset.output[index, self.index_of_overlap_feature, : ] *= raw_dataset.output[i, concept2index[f], : ]

            print("\ndone\n")
            datasets[k] = dataset
        return datasets