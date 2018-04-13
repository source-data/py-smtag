from smtag import config
import numpy as np
import torch
import logging
import math
# import logging.config
# logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

class Dataset:
    class LineTooLong(Exception):
        def __init__(self, line, max_length):
            self.line = line
            self.max_length = max_length
        def __str__(self):
            return f"FATAL: Example line is too long: {len(self.line)} > {self.max_length}"

    def __init__(self, basename):
        features_filename = f"./data/{basename}.npy"
        text_filename = f'./data/{basename}.txt'
        provenance_filename = f'./data/{basename}.prov'
        self.extended_features = None

        logger.info(f"Loading {features_filename} as features for the dataset.")
        self.features = np.load(features_filename)
        self.N = self.features.shape[0]
        self.nf = self.features.shape[1]
        self.L = self.features.shape[2]
        self.features = self.features.reshape(self.N, self.nf, 1, self.L)
        self.features = torch.from_numpy(self.features)

        logger.info(f"Loading {text_filename} for the original texts of the dataset.")
        self.text = []
        with open(text_filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) > self.L :
                    raise LineTooLong(line, L)
                self.text.append(line)

        logger.info(f"Loading {provenance_filename} as provenance info for the examples in the dataset.")
        self.provenance = []
        with open(provenance_filename, 'r') as f:
            for line in f:
                self.provenance.append(line)

        logger.info("Dataset dimensions:")
        logger.info(f"{self.N} text examples of size {self.L}")
        logger.info(f"{self.nf} encoded features.")


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
        if self.nf_output == 0: # assuming autoencoder mode
            self.nf_output = self.nf_input
        # adjust input size in case N models are trained cooperatively using same cores for first and second pass
        # for each output feature, a specialist model is trained that takes the as supplementary input the output of from the other N-1 models
        # the input size has to be adjusted accordingly and set to zero
        # if cooperative_training:
        #     self.nf_input = self.nf_input + self.nf_output - 1

    def prepare_datasets(self, file_basename):
        raw_dataset = Dataset(file_basename)
        N = len(raw_dataset.text)
        L = len(raw_dataset.text[1])
        nf = raw_dataset.nf
        assert N == raw_dataset.N, "ARM: i'm not sure if this needs to be true, just checking, maybe it is fine."
        assert L == raw_dataset.L, "ARM: i'm not sure if this needs to be true, just checking, maybe it is fine."
        # generate on the fly a 'virtual' geneprod feature as the union (sum) of protein and gene features
        raw_dataset.extended_features = torch.zeros(N, nf+1 ,1, L)
        raw_dataset.extended_features[:,0:nf,:,:] = raw_dataset.features
        raw_dataset.extended_features[:,nf,:,:] = raw_dataset.features[:, config.feature_map['gene'],:,:] + raw_dataset.features[:, config.feature_map['protein'],:,:]
        logger.debug(f"Creating dataset with selected features {self.selected_features}, and shuffling {N} examples.")
        shuffled_indices = torch.randperm(N)

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
            datasets["valid"]["last_example"] = N-1



def tester():
    logger.debug("> testing")
    selected_features = ['potein', 'gene']
    features2input = ['zzz']
    loader = Loader(selected_features, features2input=features2input)
    assert loader.nf_input == config.NBITS + len(features2input) , "Number of features in input should equal NBITS (defined in config) + the number of features to input"
    assert loader.nf_output == len(selected_features), "Number of output features is equal to the number of selected features"

    loader.prepare_datasets("test_train")
    logger.debug("> All OK")


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
