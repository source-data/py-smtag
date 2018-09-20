# -*- coding: utf-8 -*-
#T. Lemberger, 2018

#from abc import ABC, abstractmethod
import argparse
import torch
import os.path
from string import ascii_letters
from math import floor
from xml.etree.ElementTree  import XML, XMLParser, parse

from nltk import PunktSentenceTokenizer
from random import choice, randrange, random, shuffle
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_BZIP2, ZIP_STORED

from ..common.mapper import Catalogue, index2concept
from ..common.converter import TString
from ..common.utils import cd, timer
from .. import config
from ..common.progress import progress
from .featurizer import XMLEncoder, BratEncoder
from .brat import BratImport

# FIX THIS IN mapper
NUMBER_OF_ENCODED_FEATURES = 22 


class Sampler():

    def __init__(self, encoded_examples, length, sampling_mode, random_shifting, min_padding, verbose):
        self.encoded_examples = encoded_examples
        self.mode = sampling_mode
        self.random_shifting = random_shifting
        self.min_padding = min_padding
        self.verbose = verbose
        self.N = len(encoded_examples)
        self.length = length # desired length of snippet
        self.number_of_features = len(Catalogue.standard_channels) # includes the virtual geneprod feature
        print("{} examples; desired length:{}".format(self.N, self.length))
        

    @staticmethod
    def pick_fragment(text, desired_length, mode='random'):
        """
        Picks a random text fragment of desired length, if possible. Returns the text fragment, the start and the end postitions.
        """
        L = len(text) # it is possible that L < desired_length
        if mode == 'start':
            start = 0
        else: #random window sampling method
            start = randrange(max(1, L-desired_length+1)) # length is the desired fixed length of the snippet; L is the actual length of the figure legend; if L < length, then the same fragment is chose over and over... not good
        end = min(start + desired_length, L)
        fragment = text[start:end]
        return fragment, start, end

    @staticmethod
    def pad_and_shift(text, desired_length, random_shifting, min_padding):
        """
        Adds space padding on the left and the right and shifts randomly unless deactivated.
        Returns the padded text.
        """
        # 20 + (150-129) + 20
        # 20 + 11+129+10 + 20
        # left_padding + text + right_padding
        padding = min_padding + (desired_length-len(text)) + min_padding
        left_padding = padding // 2
        right_padding = padding - left_padding
        if random_shifting:
            shift = randrange(-left_padding, right_padding)
        else:
            shift = 0
        left_padding = left_padding + shift
        right_padding = right_padding - shift
        padded_text = ' ' * left_padding + text + ' ' * right_padding
        return padded_text, left_padding, right_padding

    @staticmethod
    def to_tensor(th, index, features, start, stop, left_padding, right_padding): # should include only terms within the desired sample
        for kind in features:
            for element in features[kind]:
                for attribute in features[kind][element]:
                    f = [None] * left_padding + features[kind][element][attribute][start:stop] + [None] * right_padding
                    for pos, code in enumerate(f):
                        if code is not None:
                            th[index][code][pos] = 1

    @staticmethod
    def show_stats(stats, N):
        text_avg = floor(float(sum(stats) / N))
        text_std = floor(float(torch.Tensor(stats).std()))
        text_max = max(stats)
        text_min = min(stats)
        print("\nlength of the {} examples selected:".format(N))
        print("{} +/- {} (min = {}, max = {})".format(text_avg, text_std, text_min, text_max))

    @staticmethod
    def create_tensors(N, iterations, number_of_features, length, min_padding):
        # implementation specific to longitudinal feature encoding
        textcoded4th = torch.zeros((N * iterations, 32, length+(2*min_padding)), dtype=torch.uint8)
        features4th = torch.zeros((N * iterations, number_of_features, length+(2*min_padding)), dtype = torch.uint8)
        return textcoded4th, features4th

    @staticmethod
    def display(text4th, tensor4th):
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

    @timer
    def run(self, iterations):
        """
        Each example will be sampled multiple (=iterations) times, sliced, shifted and padded.
        """
        text4th = []
        provenance4th = []
        textcoded4th, features4th = self.create_tensors(self.N, iterations, self.number_of_features, self.length, self.min_padding)
        length_stats = []
        total = self.N * iterations
        index = 0
        # looping through the examples
        for i in range(self.N):
            text = self.encoded_examples[i]['text']
            encoded = self.encoded_examples[i]['encoded']
            provenance = self.encoded_examples[i]['provenance']
            L = len(text)
            length_stats.append(L)

            # randomly sampling each example
            for j in range(iterations): # j is index of sampling iteration
                progress(i*iterations+j, total, "sampling")
                fragment, start, stop = Sampler.pick_fragment(text, self.length, self.mode) 
                padded_frag, left_padding, right_padding = Sampler.pad_and_shift(fragment, self.length, self.random_shifting, self.min_padding)
                text4th.append(padded_frag)
                textcoded4th[index] = TString(padded_frag, dtype=torch.uint8).toTensor()
                provenance4th.append(provenance)
                Sampler.to_tensor(features4th, index, encoded, start, stop, left_padding, right_padding)
                index += 1

        Sampler.show_stats(length_stats, self.N)
        if self.verbose:
            Sampler.display(text4th, features4th)

        return {
            'text4th': text4th,
            'textcoded4th': textcoded4th,
            'provenance4th':provenance4th,
            'tensor4th': features4th
        }


class DataPreparator(object):
    """
    An  class to prepare examples as dataset that can be imported in smtag
    """

    def __init__(self, options):
        """
        """
        self.length = options['length'] 
        self.sampling_mode = options['sampling_mode']
        self.random_shifting = options['random_shifting']
        self.min_padding = options['padding'] 
        self.verbose = options['verbose']
        self.iterations = options['iterations']
        self.path_compendium = ''
        self.namebase = options['namebase']
        self.train_or_test_dir = options['train_or_test_dir']
        self.dataset4th = {}

    @staticmethod
    def encode_examples(examples):
        """
        Encodes examples provided as XML Elements.
        """

        encoded_examples = []

        for id in examples:
            text = ''.join([s for s in examples[id].itertext()])
            if text:
                encoded_features, _, _ = XMLEncoder.encode(examples[id])
                example = {
                    'provenance': id, 
                    'text': text,
                    'encoded': encoded_features
                }
                encoded_examples.append(example)
            else:
                print("skipping an example in document with id=", id)
                print(str(examples[id]))
        return encoded_examples


    def import_files(self, path, XPath_to_examples='.//figure-caption'):
        """
        Import xml documents from dir. In each document, extracts examples using XPath.
        """

        with cd(config.data_dir):
            self.path_compendium = path + "_" + self.train_or_test_dir
            path = os.path.join(path, self.train_or_test_dir)
            print("loading from:", path)
            filenames = [f for f in os.listdir(path) if f.split(".")[-1] == 'xml']
            examples = {}
            for filename in filenames:
                xml = parse(os.path.join(path, filename))
                for i, e in enumerate(xml.findall(XPath_to_examples)):
                    id = filename + "-" + str(i) # unique id provided filename is unique (hence limiting to single allowed file extension)
                    examples[id] = e
                print("found {} examples in {}".format(i, filename))
        return examples


    def save(self, filenamebase):
        """
        Saving datasets prepared for torch to a text file with text example, a npy file for the extracted features and a provenance file that keeps track of origin of each example.
        """

        if not filenamebase:
            filenamebase = self.path_compendium

        with cd(config.data4th_dir):
            archive_path = "{}".format(filenamebase)
            with ZipFile("{}.zip".format(archive_path), 'w', ZIP_DEFLATED) as myzip:
                
                # write feature tensor
                tensor_filename = "{}.pyth".format(archive_path)
                torch.save(self.dataset4th['tensor4th'], tensor_filename)
                myzip.write(tensor_filename)
                os.remove(tensor_filename)

                # write encoded text tensor
                textcoded_filename = "{}_textcoded.pyth".format(archive_path)
                torch.save(self.dataset4th['textcoded4th'], textcoded_filename)
                myzip.write(textcoded_filename)
                os.remove(textcoded_filename)

                # write text examples into text file
                text_filename = "{}.txt".format(archive_path)
                with open(text_filename, 'w') as f:
                    for line in self.dataset4th['text4th']:
                        f.write("{}\n".format(line))
                myzip.write(text_filename)
                os.remove(text_filename)

                # write provenenance of each example into text file
                provenance_filename = "{}.prov".format(archive_path)
                with open(provenance_filename, 'w') as f:
                    for line in self.dataset4th['provenance4th']:
                        f.write(line+"\n")
                        #f.write(", ".join([str(line[k]) for k in ['id','index']]) + "\n")
                myzip.write(provenance_filename)
                os.remove(provenance_filename)

                myzip.close()

            for info in myzip.infolist():
                print("saved {} (size: {})".format(info.filename, info.file_size))

    def run_on_dir(self, path):
        examples = self.import_files(path)
        encoded_examples = self.encode_examples(examples) # xml elements, attributes and value are encoded into numbered features
        sampler = Sampler(encoded_examples, self.length, self.sampling_mode, self.random_shifting, self.min_padding, self.verbose)
        self.dataset4th = sampler.run(self.iterations) # examples are sampled and transformed into a tensor ready for deep learning
        self.save(self.namebase) # save the tensors
        # save self.global_index and self.rev_globale_index


class BratDataPreparator(DataPreparator):
    def __init__(self, options):
        super(BratDataPreparator, self).__init__(options)

    def import_files(self, path):
        self.path_compendium = path + "_" + self.train_or_test_dir
        with cd(config.data_dir):
            path = os.path.join(path, self.train_or_test_dir)
            brat_examples = BratImport.from_dir(path)
        return brat_examples

    def encode_examples(self, examples):

        encoded_examples = []

        for ex in examples:
            encoded_features = BratEncoder.encode(ex)
            encoded_examples.append({
                'provenance': ex['provenance'], 
                'text': ex['text'], 
                'encoded': encoded_features
            })
        return encoded_examples



def main():
    parser = argparse.ArgumentParser(description='Reads xml and transform into tensor format.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = []
    parser.add_argument('-c', '--path', default='demo_xml', help='path to the source compendium of xml files')
    parser.add_argument('-f', '--filenamebase', default='', help='namebase to save converted trainset and features set files')
    parser.add_argument('-T', '--testset', action='store_true', help='use the testset instead of the trainset')
    parser.add_argument('-b', '--brat', action='store_true', help='Use brat files instead of xml files')
    parser.add_argument('-X', '--iterations', default=5, type=int, help='number of times each example is sampled')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
    parser.add_argument('-L', '--length', default=150, type=int, help='length of the text snippets used as example')
    parser.add_argument('-W', '--window', action='store_true', help='switches to the sampling fig legends using a random window instead of parsing full sentences')
    parser.add_argument('-S', '--start', action='store_true', help='switches to mode where fig legends are simply taken from the start of the text and truncated appropriately')
    parser.add_argument('-d', '--disable_shifting', action='store_true', help='disable left random padding which is used by default to shift randomly text')
    parser.add_argument('-p', '--padding', default=20, help='minimum padding added to text')
    parser.add_argument('-w', '--working_directory', help='Specify the working directory where to read and write files to')

    args = parser.parse_args()
    
    options = {}
    options['namebase'] = args.filenamebase
    options['iterations'] = args.iterations
    options['verbose'] = args.verbose
    options['length'] = args.length
    options['path'] = args.path
    options['train_or_test_dir'] = 'test' if args.testset else 'train'
    if args.window:
        options['sampling_mode'] = 'window'
    elif args.start:
        options['sampling_mode'] = 'start'
    else:
        options['sampling_mode'] = 'sentence'
    options['random_shifting'] = not args.disable_shifting
    options['padding'] = args.padding
    
    if args.working_directory:
        config.working_directory = args.working_directory
    with cd(config.working_directory):
        if args.brat:
            prep = BratDataPreparator(options)
        else:
            prep = DataPreparator(options)
        prep.run_on_dir(options['path'])

if __name__ == "__main__":
    main()
