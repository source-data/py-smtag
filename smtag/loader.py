from smtag import config
import numpy as np
import torch
import logging
import math
from converter import Converter
import mapper
from viz import Visualization 

# import logging.config
# logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


#Dataset class should have text, input_features, output_features and provenance members
#separate from file loading
#separate from computing selected features and randomization

class Dataset:
    class LineTooLong(Exception):
        def __init__(self, line, max_length):
            self.line = line
            self.max_length = max_length
        def __str__(self):
            return f"FATAL: Example line is too long: {len(self.line)} > {self.max_length}"

    def from_files(self, basename):
        features_filename = f"./data/{basename}.npy"
        text_filename = f'./data/{basename}.txt'
        provenance_filename = f'./data/{basename}.prov'
        logger.info(f"Loading {features_filename} as features for the dataset.")
        np_features = np.load(features_filename) #saved file is 3D; need to change this?
        self.N = np_features.shape[0] #number of examples
        self.nf = np_features.shape[1] #number of features
        self.L = np_features.shape[2] #length of text snippet 
        #convert into 4D for the whole smtag package to be able to use 4D convolution modules
        np_features.resize(self.N, self.nf, 1, self.L)
        self.output = torch.from_numpy(np_features) #saved files are from numpy, need conversion to torch
        logger.info(f"Loading {text_filename} for the original texts of the dataset.")
        with open(text_filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) > self.L :
                    raise LineTooLong(line, L)
                self.text.append(line)
        logger.info(f"Loading {provenance_filename} as provenance info for the examples in the dataset.")
        with open(provenance_filename, 'r') as f:
            for line in f:
                self.provenance.append(line)
        logger.info("Dataset dimensions:")
        logger.info(f"{self.N} text examples of size {self.L}")
        logger.info(f"{self.nf} encoded features.")

        
    def __init__(self, N=0, nf_input=0, nf_output=0, L=0):
        self.N = N
        self.nf_input = nf_input
        self.nf_output = nf_output
        self.L = L
        self.text = [None] * N
        self.provenance = [None] * N
        self.input = torch.zeros(self.N, self.nf_input, 1 , self.L)
        self.output = torch.zeros(self.N, self.nf_output, 1, self.L)

class Loader:
    def __init__(self,
                 selected_features,
                 collapsed_features=[],
                 overlap_features={},
                 features2input=[],
                 noise=0,
                 fraction=1,
                 validation_fraction=0.2,
                 cooperative_training=False):
        self.selected_features = selected_features
        self.collapsed_features = collapsed_features
        self.overlap_features = overlap_features
        self.features2input = features2input
        self.noise = noise
        self.fraction = fraction  # fraction of the training set to actually us for training
        self.validation_fraction = validation_fraction  # fraction of the whole dataset to be used for validation during training
        self.nf_input = config.NBITS
        self.nf_collapsed_feature = 0
        self.nf_overlap_feature = 0
        self.nf_output = len(selected_features)
        if len(features2input) > 0:
            self.nf_input = self.nf_input + len(features2input)
        if len(collapsed_features) > 0:
            self.nf_output = self.nf_output + 1
            self.nf_collapsed_feature = self.nf_output
        if len(overlap_features) > 0:
            self.nf_output = self.nf_output + 1
            self.nf_overlap_feature = self.nf_output

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
        nf = raw_dataset.nf
        assert N != 0, "zero examples"
        
        # generate on the fly a 'virtual' geneprod feature as the union (sum) of protein and gene features
        extended_features = torch.zeros(N, nf+1 ,1, L)
        extended_features[ : , 0:nf, : , : ] = raw_dataset.output
        extended_features[ : , nf, : , : ] = raw_dataset.output[ : , mapper.label2index['gene'], : , : ] + raw_dataset.output[ : ,  mapper.label2index['protein'], : , : ]
        nf += 1
        raw_dataset.output = extended_features
        
        logger.debug(f"Creating dataset with selected features {self.selected_features}, and shuffling {N} examples.")
        shuffled_indices = torch.randperm(N) #shuffled_indices = range(N); shuffle(shuffled_indices)
        datasets = {}
        if self.validation_fraction == 0:
            logger.info("testset mode; for benchmarking")
            datasets["test"]= {} #--testset mode; for benchmarking
            datasets["test"]["first_example"] = 0
            datasets["test"]["last_example"] = math.ceil(N*self.fraction)
        else:
            logger.info("normal trainset and validation set mode")
            datasets["train"] = {} #--normal trainset and validation set mode
            datasets["valid"] = {}
            datasets["train"]["first_example"] = 0
            datasets["train"]["last_example"] = math.floor(self.fraction*N*(1-self.validation_fraction))
            datasets["valid"]["first_example"] = math.ceil(N*(1-self.validation_fraction+0.000001)) #--the size of the validation set is kept unchanged and not affected by fraction
            datasets["valid"]["last_example"] = N
        #N = 10 examples [0,1,2,3,4,5,6,7,8,9], validation_fraction = 0.8
        #8 training, 2 validation
        #train first_example = 0
        #train last_example = 8
        #range(0,8) is 0,1,2,3,4,5,6,7
        #valid first_example = 8
        #valid last_example = 10
        #range(8,10) is 8, 9
        for k in datasets:
            first_example = datasets[k]['first_example']
            last_example = datasets[k]['last_example']
            N_examples = last_example - first_example + 1
            dataset = Dataset(N_examples, self.nf_input, self.nf_output, L)
            
            print(f"Generating {k} set with {N_examples} examples ({first_example}, {last_example})")
            print(f"input dataset['{k}'] tensor created")
            print(f"output dataset['{k}'] tensor created")
            for example_i in range(first_example, last_example):
                
                i = shuffled_indices[example_i]
                index = example_i - first_example + 1
                #TEXT SAMPLES AND INPUT TEXT ENCODING
                assert len(raw_dataset.text[i]) == L, "FATAL: {} > {} with {}".format(len(raw_dataset.text[i]), L, raw_dataset.text[i])
                dataset.text[index] = raw_dataset.text[i]
                dataset.provenance[index] = raw_dataset.provenance[i]
                dataset.input[index, 0:32 , :, : ] = Converter.t_encode(raw_dataset.text[i])
                
                #SELECTION AND COMBINATION OF OUTPUT FEATURES
                for f in self.features2input:
                    #argh we need j as the index
                    j = self.features2input.index(f) # j-1???
                    dataset.input[index, self.nf_input - len(self.features2input) + j, : , : ] = raw_dataset.output[i, mapper.label2index[f], : , : ]

                for f in self.selected_features:
                    j = self.selected_features.index(f)
                    dataset.output[index, j, : , : ] = raw_dataset.output[i, mapper.label2index[f], : , : ]

                #dataset.output[index, nf_collapsed_feature,:,:] is already initialized with zeros
                for f in self.collapsed_features:
                    dataset.output[index, self.nf_collapsed_feature, : , : ] = dataset.output[index, self.nf_collapsed_feature, : , : ] + raw_dataset.output[i, mapper.label2index[f], : , : ]  

                #if overlap features need to be computed, needs initialization with ones otherwise zeros will kill all
                if self.nf_overlap_feature:
                    dataset.output[index, self.nf_overlap_feature, : , : ].fill_(1)
                for f in self.overlap_features:
                    dataset.output[index, self.nf_overlap_feature, : , : ] = dataset.output[index, self.nf_overlap_feature, : , : ] * raw_dataset.output[i, mapper.label2index[f], : , : ]
    
            datasets[k] = dataset
        return datasets

def tester():
    logger.debug("> testing")
    selected_features = ['protein', 'gene']
    features2input = ['small_molecule']
    loader = Loader(selected_features, features2input=features2input)
    assert loader.nf_input == config.NBITS + len(features2input) , "Number of features in input should equal NBITS (defined in config) + the number of features to input"
    assert loader.nf_output == len(selected_features), "Number of output features is equal to the number of selected features"

    d = loader.prepare_datasets("test_train")
    logger.debug("> All OK")
    Visualization.show_example([d['train']])

if __name__ == '__main__':           # Only when run
    tester()                         # Not when imported






"""
{
    "Z" = 64, "fraction" = 1,
    "learningRate" = 0.001,
    "overlap_features" = table: 0x05aca518,
    "viz" = false,
    "features" = table: 0x05ac9bb0,
    "epochs" = 10,
    "features2input" = table: 0x05aca620,
    "pooling_table" = table: 0x05aca170,
    "validation_fraction" = 0.2,
    "kernel_table" = table: 0x05aca910,
    "mode" = "unet",
    "suffix" = "2018-04-09-11-14-45",
    "collapsed_features" = table: 0x05aca3f0,
    "dropout" = 0.1,
    "noise" = 0.1,
    "cuda" = false,
    "nf_table" = table: 0x05aca750
}
"""
