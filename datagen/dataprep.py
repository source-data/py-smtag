# -*- coding: utf-8 -*-
#T. Lemberger, 2018

#from abc import ABC, abstractmethod
import argparse
import torch
import os.path
from string import ascii_letters
from math import floor

from nltk import PunktSentenceTokenizer
from random import choice, randrange, random, shuffle
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_BZIP2, ZIP_STORED

from common.mapper import index2concept, Catalogue
from common.converter import TString
from common.utils import cd, timer
from common.config import DATA_DIR
from common.progress import progress

SPACE_ENCODED = TString(" ", dtype=torch.uint8)

class DataPreparator(object):
    """
    An abstract class to prepare text examples as dataset that can be imported in smtag
    """

    def __init__(self, parser):
        """
        Part of the initialisation is to define a common set of command line options. 
        The call to self.main() is left to implementing classes.
        The call to self.set_options(parser.parse_args()) is left to implementing classes to leave the possibility to add specialized command line options.
        """
        self.raw_examples = []
        self.errors = []
        self.split_examples = {'train':[], 'test':[]}
        self.dataset4th = {}
        self.parser = parser
        self.args = []
        parser.add_argument('-f', '--filenamebase', default='test', help='namebase to save trainset and features set files')
        parser.add_argument('-X', '--iterations', default=5, type=int, help='number of times each example is sampled')
        parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
        parser.add_argument('-L', '--length', default=150, type=int, help='length of the text snippets used as example')
        parser.add_argument('-t', '--testfract', default=0.2, type=float, help='fraction of papers in testset')
        parser.add_argument('-W', '--window', action='store_true', help='switches to the sampling fig legends using a random window instead of parsing full sentences')
        parser.add_argument('-S', '--start', action='store_true', help='switches to mode where fig legends are simply taken from the start of the text and truncated appropriately')
        parser.add_argument('-d', '--disable_shifting', action='store_true', help='disable left random padding which is used by default to shift randomly text')
        #parser.add_argument('-c', '--rand_char_padding', action='store_true', help='padding with random characters instead of white spaces') 
        parser.add_argument('-p', '--padding', default=20, help='minimum padding added to text')
        self.options = {}
        #implementation: self.options = self.set_options(parser.parse_args())
        #implementation: call self.main()? Maybe OTT

    #@abstractmethod
    @staticmethod
    def set_options(args):
        """
        Reading the parsed command line arguments to set options.
        Args:
            args: the parsed arguments
        Returns:
            (dict): options correspondning to each argument parsed from command line
        """
        options = {} 
        options['namebase'] = args.filenamebase
        options['iterations'] = args.iterations
        options['verbose'] = args.verbose
        options['testset_fraction'] = args.testfract
        options['length'] = args.length
        if args.window:
            options['sampling_mode'] = 'window' 
        elif args.start:
            options['sampling_mode'] = 'start'
        else:
            options['sampling_mode'] = 'sentence'
        options['random_shifting'] = not args.disable_shifting
        # options['white_space_padding'] = not args.rand_char_padding
        options['padding'] = args.padding

        return options


    @timer
    def sample(self, dataset):
        """
        Once examples are loaded in the dataset, each example will be sampled multiple (=iterations) times, sliced, shifted and padded.
        """
        mode = self.options['sampling_mode']
        # white_space_padding = self.options['white_space_padding']
        random_shifting = self.options['random_shifting']
        min_padding = self.options['padding']
        N = len(dataset)
        print("N =",N)
        length = self.options['length']
        iterations = self.options['iterations']
        index = 0
        text4th = []
        provenance4th = []
        number_of_features = len(Catalogue.standard_channels) # includes the virtual geneprod feature
        tensor4th = torch.zeros((N * iterations, number_of_features, length+min_padding), dtype = torch.uint8) # dtype = np.bool_
        textcoded4th = torch.zeros((N * iterations, 32, length+min_padding), dtype = torch.uint8) # dtype = np.bool_

        length_statistics = []
        print("generating {}*{} x {} x {} tensor.".format(N, self.options['iterations'], number_of_features, length+min_padding))

        total_count = N * iterations
        progress_counter = 1
        for i in range(N):
            text_i = dataset[i]['text'] 
            textcoded4th_i= TString(text_i, dtype=torch.uint8)
            features_i = dataset[i]['features']
            #compute_features_i['marks']['sd-tag']['geneprod'] here or somethign
            provenance_i = dataset[i]['provenance']
            L = len(text_i)
            length_statistics.append(L)
            sentence_ranges = PunktSentenceTokenizer().span_tokenize(text_i) if mode == 'sentence' else None
            
            for j in range(iterations): # j is index of sampling iteration
                progress(progress_counter, total_count, "sampling")
                if mode == 'sentence':
                    fragment = choice(sentence_ranges) 
                elif mode == 'start':
                    fragment = (0, length)
                else: #random window sampling method
                    fragment = (randrange(max(1,L-length+1)), L) # length is the desired fixed length of the snippet; L is the actual length of the figure legend; if L < length, then the same fragment is chose over and over... not good

                #think more about that; sample with/without replacement? use shuffle and don't multi sample small legends
                #legends with few short sentences will have these sentences overrepresented
                #could shuffle but size of tensor cannot be predicted. Not a big deal? First array then copy in tensor once final size known

                start = fragment[0]
                stop = min(start + length, L)
                sub_text = text_i[start:stop]
                # sub_text4th = TString()
                # sub_text4th.s = sub_text # not so nice, but textcoded4th[start:stop] would need to implement __getitem__(); needs to be tested 
                #sub_text4th = textcoded4th_i[ : , :, start:stop] # 3D
                sub_text4th = textcoded4th_i[start:stop]

                padding = length + min_padding - len(sub_text)
                if random_shifting: 
                    random_shift = int(floor(padding * random()))
                else:
                    random_shift = 0
                right_padding = padding - random_shift

                #if white_space_padding:
                text_ij = ' ' * random_shift + sub_text + ' ' * right_padding 
                #else:
                #    left_padding_chars = ''.join([choice(ascii_letters+' ') for i in range(random_shift)])
                #    right_padding_chars = ''.join([choice(ascii_letters+' ') for i in range(right_padding)])
                #    text_ij = left_padding_chars + sub_text + right_padding_chars

                text4th.append(text_ij)
                #add textcoded4th
                #textcoded4th[index] = TString(text_ij).toTensor(dtype=torch.uint8) # pedestrian but maybe slow way

                #faster ??? but lower level
                #pad textcoded4th_ij with suitable encoded spaces matrices
                left_padding_encoded = SPACE_ENCODED.repeat(random_shift) 
                right_padding_encoded = SPACE_ENCODED.repeat(right_padding)
                textcoded4th_ij = left_padding_encoded + sub_text4th + right_padding_encoded
                textcoded4th[index] = textcoded4th_ij.toTensor() # would be nicer to save directly TString and work only with TString ?

                provenance4th.append(provenance_i)

                #fill tensor of features   
                for kind in features_i: 
                    for element in features_i[kind]:
                        for attribute in features_i[kind][element]:
                            f = [None] * random_shift + features_i[kind][element][attribute][start:stop] + [None] * right_padding
                            for pos in range(length+min_padding): 
                                code = f[pos]
                                if code is not None:
                                    tensor4th[index][code][pos] = 1 # True
                index += 1
                progress_counter += 1

        text_avg = float(sum(length_statistics) / N)
        text_sd = float(torch.Tensor(length_statistics).std())
        text_max = max(length_statistics)
        text_min = min(length_statistics)  
        print("\nlength of the {} examples selected:".format(N))
        print("{} +/- {} (min = {}, max = {})".format(text_avg, text_sd, text_min, text_max))
        if self.options['verbose']:
            self.display(text4th, tensor4th)

        return {'text4th':text4th, 'textcoded4th':textcoded4th, 'provenance4th':provenance4th, 'tensor4th':tensor4th} 


    def split_trainset_testset(self, raw_examples):  
        """
        The list of raw examples is split early on into trainset and testset, to make sure they are kept completely separate.
        """
        test_fraction = self.options['testset_fraction']
        print("number of raw_examples", len(raw_examples))
        #hmmm, what if raw_examples is a dictionary instead of a list as is the case in sdgraph2th        
        N = len(raw_examples)
        N_train = int(floor(N * (1 - test_fraction)))
        if isinstance(raw_examples, list):
            shuffle(raw_examples)
            self.split_examples['train'] = raw_examples[:N_train]
            self.split_examples['test'] = raw_examples[N_train:]
        elif isinstance(raw_examples, dict):
            keys = list(raw_examples.keys())
            shuffle(keys)
            self.split_examples['train'] = {k:raw_examples[k] for k in keys[:N_train]}
            self.split_examples['test'] =  {k:raw_examples[k] for k in keys[N_train:]}

    #only called in implementation at the end of __init__()?
    def main(self):
        self.raw_examples = self.import_examples(self.options['source'])

        self.split_trainset_testset(self.raw_examples)

        for subset in ['train', 'test']:
            dataset = self.build_feature_dataset(self.split_examples[subset]) #should return dataset[i]['text'|'provenance'|'features']
            self.dataset4th[subset] = self.sample(dataset)

        self.save(self.options['namebase'])


    #@abstractmethod
    def import_examples(self, source):
        """
        Abstract method to import raw examples from a source eg. text files or database
        """
        #self.examples, errors = neo2leg.neo2xml(self.options)
        pass 
        
    #@abstractmethod
    def build_feature_dataset(self, dataset):
        """
        Abstract method to extract and map features from the loaded examples.
        """ 
        #features, _, _ = xml2features(figure_xml)
        pass


    def save(self, filenamebase):
        """
        Saving datasets prepared for torch to a text file with text example, a npy file for the extracted features and a provenance file that keeps track of origin of each example.
        """
        
        with cd(DATA_DIR):
            for k in self.dataset4th: # 'train' | 'valid' | 'test'
                archive_path = "{}_{}".format(filenamebase, k)
                with ZipFile("{}.zip".format(archive_path), 'w', ZIP_DEFLATED) as myzip:
                    # write feature tensor
                    tensor_filename = "{}.pyth".format(archive_path)
                    torch.save(self.dataset4th[k]['tensor4th'], tensor_filename)
                    myzip.write(tensor_filename)
                    os.remove(tensor_filename) 
                    
                    # write encoded text tensor
                    textcoded_filename = "{}_textcoded.pyth".format(archive_path)
                    torch.save(self.dataset4th[k]['textcoded4th'], textcoded_filename)
                    myzip.write(textcoded_filename)
                    os.remove(textcoded_filename) 

                    # write text examples into text file
                    text_filename = "{}.txt".format(archive_path)
                    with open(text_filename, 'w') as f:
                        for line in self.dataset4th[k]['text4th']: 
                            f.write("{}\n".format(line))
                    myzip.write(text_filename)
                    os.remove(text_filename) 

                    # write provenenance of each example into text file
                    provenance_filename = "{}.prov".format(archive_path)
                    with open(provenance_filename, 'w') as f:
                        for line in self.dataset4th[k]['provenance4th']: 
                            f.write(", ".join([str(line[k]) for k in ['id','index']]) + "\n")
                    myzip.write(provenance_filename)
                    os.remove(provenance_filename)

                    myzip.close()

                for info in myzip.infolist():
                    print("saved {} (size: {})".format(info.filename, info.file_size))

    def log_errors(self, errors):
        """
        Errors that are detected during feature extraction are kept and logged into a log file.
        """
        for e in errors:
            if errors[e]: 
                print("####################################################")
                print(" Writing {} {} errors to errors_{}.log".format(len(errors[e]), e, e))
                print("####################################################" )
            #write log file anyway, even if zero errors, to remove old copy 
            with open('errors_{}.log'.format(e), 'w') as f:
                for line in errors[e]: 
                    ids, err = line
                    f.write(u"\nerror:\t{}\t{}\n".format('\t'.join(ids), err))
            f.close()

    def display(self, text4th, tensor4th):
        """
        Display text fragments and extracted features to the console.
        """
        N, featsize, L = tensor4th.shape
    
        for i in range(N):
            print
            print("Text:")
            print(text4th[i])
            for j in range(featsize):
                feature = str(index2concept[j])
                track = [int(tensor4th[i, j, k]) for k in range(L)]
                print(''.join([['-','+'][x] for x in track]), feature)

