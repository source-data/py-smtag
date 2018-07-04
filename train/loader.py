# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from smtag.config import NBITS
import numpy as np
import torch
# import logging
import math
from common.converter import Converter
from common.mapper import Catalogue, concept2index
from commmon.progress import progress
from common.viz import Show 

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
        self.text = [None] * N
        self.provenance = [None] * N
        self.input = torch.zeros(self.N, self.nf_input, self.L)
        self.output = torch.zeros(self.N, self.nf_output, self.L)

    def from_files(self, basename):
        features_filename = "data/{}.npy".format(basename)
        text_filename = 'data/{}.txt'.format(basename)
        textcoded_filename = "data/{}_textcoded.npy".format(basename)
        provenance_filename = 'data/{}.prov'.format(basename)

        print("Loading {} as features for the dataset.".format(features_filename))
        np_features = np.load(features_filename) #saved file is 3D; need to change this?
        self.N = np_features.shape[0] #number of examples
        self.nf_output = np_features.shape[1] #number of features
        self.L = np_features.shape[2] #length of text snippet 
        self.output = torch.from_numpy(np_features).float() #saved files are from numpy, need conversion to torch and make sure it is a FloatTensor othwerise problems with default dtype 

        print("Loading {} as encoded text for the dataset.".format(textcoded_filename))
        textcoded = np.load(textcoded_filename)
        self.textcoded = torch.from_numpy(textcoded)

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
        print("raw_dataset.nf_output nf=", nf)
        assert N != 0, "zero examples!"

        # generate on the fly a 'virtual' geneprod feature as the union (sum) of protein and gene features
        #THIS HAS TO GO! BELONGS TO DATAPREP!!! Probably in Featurizer
        raw_dataset.output[ : , nf-1,  : ] = raw_dataset.output[ : , concept2index[Catalogue.GENE],  : ] + raw_dataset.output[ : ,  concept2index[Catalogue.PROTEIN], : ]

        print("Creating dataset with selected features {}, and shuffling {} examples.".format(self.selected_features, N))
        shuffled_indices = torch.randperm(N) #shuffled_indices = range(N); shuffle(shuffled_indices)
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
            datasets["valid"]["first_example"] = math.ceil(N*(1-self.validation_fraction+0.000001)) #--the size of the validation set is kept unchanged and not affected by fraction
            datasets["valid"]["last_example"] = N
        # N = 10 examples [0,1,2,3,4,5,6,7,8,9], validation_fraction = 0.8
        # 8 training, 2 validation
        # train first_example = 0
        # train last_example = 8
        # range(0,8) is 0,1,2,3,4,5,6,7
        # valid first_example = 8
        # valid last_example = 10
        # range(8,10) is 8, 9
        for k in datasets: # k 'test' in testing mode otherwise 'valid' or 'train'
            first_example = datasets[k]['first_example']
            last_example = datasets[k]['last_example']
            N_examples = last_example - first_example + 1
            dataset = Dataset(N_examples, self.nf_input, self.nf_output, L)
            
            print("Generating {} set with {} examples ({}, {})".format(k, N_examples, first_example, last_example))
            print("input dataset['{}'] tensor created".format(k))
            print("output dataset['{}'] tensor created".format(k))
            for example_i in range(first_example, last_example):
                progress(example_i - first_example, N_examples, status="loading {} examples ({} to {}) into dataset['{}']".format(N_examples, first_example, last_example, k))
                i = shuffled_indices[example_i]
                index = example_i - first_example + 1

                #INPUT: TEXT SAMPLE ENCODING
                assert len(raw_dataset.text[i]) == L, "FATAL: {} > {} with {}".format(len(raw_dataset.text[i]), L, raw_dataset.text[i])
                dataset.text[index] = raw_dataset.text[i]
                dataset.provenance[index] = raw_dataset.provenance[i]

                #this is a bit slow! Shift to data prep!
                #dataset.input[index, 0:32 , : ] = TString(raw_dataset.text[i]).toTensor()
                dataset.input[index, 0:32 , : ] = raw_dataset.textcoded[i, : , : ]

                j = 0
                for f in self.features_as_input:
                    # for example: j=0, => 32 + 0 = 32
                    dataset.input[index, 32 + j, : ] = raw_dataset.output[i, concept2index[f], : ]
                    j += 1

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
