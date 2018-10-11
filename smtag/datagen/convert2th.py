# -*- coding: utf-8 -*-
#T. Lemberger, 2018

#from abc import ABC, abstractmethod
import argparse
import torch
import os.path
import shutil
import sys
import re
from string import ascii_letters
from math import floor, ceil
from xml.etree.ElementTree  import XML, parse, tostring

from nltk import PunktSentenceTokenizer
from random import choice, randrange, random, shuffle
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_BZIP2, ZIP_STORED

from ..common.mapper import Catalogue, index2concept, NUMBER_OF_ENCODED_FEATURES
from ..common.converter import TString
from ..common.utils import cd, timer
from ..common.progress import progress
from .encoder import XMLEncoder, BratEncoder
from .brat import BratImport
from .context import OCRContext
from .. import config

class Sampler():
    """
    A class to sample fragment from text examples, padd and shift the fragments and encode the corresponding text and features into Tensor format.
    """

    def __init__(self, path, length, sampling_mode, random_shifting, min_padding, verbose):
        self.path = path
        dirnames = os.listdir(self.path)
        self.dirnames = [d for d in dirnames if os.path.isdir(os.path.join(self.path, d)) and d != '.DS_Store']
        self.mode = sampling_mode
        self.random_shifting = random_shifting
        self.min_padding = min_padding
        self.verbose = verbose
        self.N = len(self.dirnames)
        self.length = length # desired length of snippet
        self.number_of_features = NUMBER_OF_ENCODED_FEATURES # includes the virtual geneprod feature
        self.img_cxt_features = config.img_grid_size ** 2 + 2 # square grid + vertical + horizontal
        print("\n{} examples; desired length:{}\n".format(self.N, self.length))


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
    def slice_and_pad(th, index, source_tensor, start, stop, left_padding, right_padding):
        L = stop - start
        th[index, : , left_padding:left_padding+L] = source_tensor[ : , start:stop]

    @staticmethod
    def show_stats(stats, N):
        text_avg = floor(float(sum(stats) / N))
        text_std = torch.Tensor(stats).std()
        if N > 1:
            text_std = floor(float(text_std))
        else:
            text_std = 0
        text_max = max(stats)
        text_min = min(stats)
        print("\nlength of the {} examples selected:".format(N))
        print("{} +/- {} (min = {}, max = {})".format(text_avg, text_std, text_min, text_max))

    @staticmethod
    def create_tensors(N, iterations, number_of_features, img_cxt_features, length, min_padding):
        """
        Creates and initializes the tensors needed  to encode text and features.
        Allows to abstract away the specificity of the encoding. This implementation is for longitudinal features.
        But some other implementation could need several encoding tensors.
        The text is encoded in a 32 feature tensor using the binary representation of the unicode code of each character (see smtag.common.TString)

        Args:
            N: the number of examples
            iterations: the number of times each example is sampled
            number_of_features: number of features for the encoded feature tensor
            length: the desired length of the encoded fragments
            min_padding: the minimum amount of padding put on each side of the fragment

        Returns:
            (textcoded4th, features) where
                textcoded4th: 3D zero-initialized ByteTensor, N*iterations x 32 x full_length, where full_length=length+(2*min_padding)
                features: 3D zero-initialized ByteTensor, N*iterations x number_of_features x full_length, where full_length=length+(2*min_padding)
        """

        textcoded4th = torch.zeros((N * iterations, 32, length+(2*min_padding)), dtype=torch.uint8)
        features4th = torch.zeros((N * iterations, number_of_features, length+(2*min_padding)), dtype = torch.uint8)
        context4th = torch.zeros((N * iterations, img_cxt_features, length+(2*min_padding)), dtype = torch.uint8)
        return textcoded4th, features4th, context4th

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
                if j < len(index2concept):
                    feature = str(index2concept[j])
                else:
                    feature = 'ctx_' + j - len(index2concept)
                track = [int(tensor4th[i, j, k]) for k in range(L)]
                print(''.join([['-','+'][x] for x in track]), feature)

    @timer
    def run(self, iterations):
        """
        Each example will be sampled multiple (=iterations) times, sliced, shifted and padded.

        Args:
            iterations: number of times each example is sampled.

        Returns:
            {'text4th': text4th,
             'textcoded4th': textcoded4th,
             'provenance4th':provenance4th,
             'tensor4th': features4th,
             'context4th':context4th}
            where:
            text4th: a list with a copy of the padded text of the sample
            textcoded4th: a 3D Tensor (samples x 32 x full length) with the fragment text encoded
            provenance4th: an array with the ids of the example from which the sample was taken
            tensor4th: a 3D Tensor with the encoded features corresponding to the fragment
            context4th: a 3D Tensor with location features of text elements extracted from the illustration
        """
        text4th = []
        provenance4th = []
        textcoded4th, features4th, context4th = self.create_tensors(self.N, iterations, self.number_of_features, self.img_cxt_features, self.length, self.min_padding)
        length_stats = []
        total = self.N * iterations
        index = 0

        # looping through the examples # loop through directories   
        with cd(self.path):
            for i, d in enumerate(self.dirnames):
                encoded_example = EncodedExample()
                encoded_example.load(d)
                text = encoded_example.text
                encoded = encoded_example.features
                provenance = encoded_example.provenance
                img_context = encoded_example.context

                L = len(text)
                length_stats.append(L)

                # randomly sampling each example
                for j in range(iterations): # j is index of sampling iteration
                    progress(i*iterations+j, total, "sampling example {}".format(i+1))
                    # a text fragment is picked randomly from the text example
                    fragment, start, stop = Sampler.pick_fragment(text, self.length, self.mode)
                    # it is randomly shifted and padded to fit the desired length
                    padded_frag, left_padding, right_padding = Sampler.pad_and_shift(fragment, self.length, self.random_shifting, self.min_padding)
                    # the final padded fragment is added to the list of samples
                    text4th.append(padded_frag)
                    # the text is encoded using the 32 bit unicode encoding provided by TString
                    textcoded4th[index] = TString(padded_frag, dtype=torch.uint8).toTensor()
                    # the provenance of the fragment needs to be recorded for later reference and possible debugging
                    provenance4th.append(provenance)
                    # the encoded features of the fragment are added to the feature tensor
                    Sampler.slice_and_pad(features4th, index, encoded, start, stop, left_padding, right_padding)
                    # the encoded imgage context features 
                    Sampler.slice_and_pad(context4th, index, img_context, start, stop, left_padding, right_padding)

                    index += 1

        Sampler.show_stats(length_stats, self.N)
        if self.verbose:
            Sampler.display(text4th, features4th)

        return {
            'text4th': text4th,
            'textcoded4th': textcoded4th,
            'provenance4th':provenance4th,
            'tensor4th': features4th,
            'context4th': context4th
        }

class EncodedExample():
    def __init__(self, provenance='', text='', features=None, context=None):
        self._provenance = provenance
        self._text = text
        self._features = features
        self._context = context

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
            with cd(path):
                torch.save(self.features, 'features.pyth')
                torch.save(self.context, 'context.pyth')
                with open('provenance.txt', 'w') as f: 
                    f.write(self.provenance)
                with open('text.txt', 'w') as f:
                    f.write(self.text)
        else:
            print("EncodedExample detected that {} already exists.".format(path))
    
    def load(self, path):
        with cd(path):
            with open('provenance.txt', 'r') as f:
                self._provenance = f.read()
            with open('text.txt', 'r') as f:
                self._text = f.read()
            self._features = torch.load('features.pyth').byte()
            self._context = torch.load('context.pyth') # this is float()

    @property
    def provenance(self):
        return self._provenance

    @property
    def text(self):
        return self._text

    @property
    def features(self):
        return self._features

    @property
    def context(self):
        return self._context


class DataPreparator(object):
    """
    An  class to prepare examples as dataset that can be imported in smtag.
    """

    def __init__(self, options):
        self.length = options['length']
        self.sampling_mode = options['sampling_mode']
        self.random_shifting = options['random_shifting']
        self.min_padding = options['padding']
        self.verbose = options['verbose']
        self.iterations = options['iterations']
        self.namebase = options['namebase'] # namebase where the converted dataset should be saved
        self.compendium = options['compendium'] # the compendium of source documents to be sampled and converted

    def encode_examples(self, subset, examples):
        """
        Encodes examples provided as XML Elements and writes them to disk.
        """

        with cd(os.path.join(config.data_dir, self.compendium)):
            ocr = OCRContext(subset, G=config.img_grid_size)
            for ex in examples:
                xml = ex['xml']
                graphic = ex['graphic']
                prov = ex['provenance']
                text = ''.join([s for s in xml.itertext()])
                path_to_example = os.path.join(subset, prov)
                if not os.path.exists(path_to_example):
                    if text:
                        encoded_features = XMLEncoder.encode(xml) # convert to tensor already here; 
                        ocr_context = ocr.run(text, graphic) # returns a tensor
                        encoded_example = EncodedExample(prov, text, encoded_features, ocr_context)
                        encoded_example.save(path_to_example)
                    else:
                        print("\nskipping an example in document with id=", prov)
                else:
                    print("\n{} was already encoded".format(prov))


    def import_files(self, subset, XPath_to_examples='.//sd-panel', XPath_to_assets = './/graphic'):
        """
        Import xml documents from dir. In each document, extracts examples using XPath_to_examples.
        """

        with cd(config.data_dir):
            path = os.path.join(self.compendium, subset)
            print("\nloading from:", path)
            filenames = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.xml']
            examples = []
            for i, filename in enumerate(filenames):
                try:
                    with open(os.path.join(path, filename)) as f: 
                        xml = parse(f)
                        print("\ndoi:", xml.getroot().get('doi'))
                    for j, e in enumerate(xml.getroot().findall(XPath_to_examples)):
                        provenance = os.path.splitext(filename)[0] + "_" + str(j)
                        g = e.find(XPath_to_assets)
                        if g is not None:
                            basename = re.search(r'panel_id=(\w+)', g.get('href')).group(1)
                            graphic_filename = basename + '.jpg'
                        else:
                            print('no graphic element found')
                            graphic_filename = ''
                        #if not os.path.exists(os.path.join(config.data4th_dir, graphic_filename):
                        examples.append({
                            'xml': e,
                            'provenance': provenance,
                            'graphic': graphic_filename
                        })
                except Exception as e:
                    print("problem parsing", os.path.join(path, filename))
                    print(e)
                progress(i, len(filenames), "loaded {}".format(filename))
        return examples


    def save(self, dataset4th, subset):
        """
        Saving datasets prepared for torch to a text file with text example, a npy file for the extracted features and a provenance file that keeps track of origin of each example.
        """

        with cd(config.data4th_dir):
            with cd(self.namebase):
                os.mkdir(subset)
                with cd(subset):
                    # write feature tensor
                    torch.save(dataset4th['tensor4th'], 'features.pyth')
                    # write encoded text tensor
                    torch.save(dataset4th['textcoded4th'], 'textcoded.pyth')
                    # write image context features
                    torch.save(dataset4th['context4th'], 'context.pyth')
                    # write text examples into text file
                    with open("text.txt", 'w') as f:
                        for line in dataset4th['text4th']:
                            f.write("{}\n".format(line))
                    # write provenenance of each example into text file
                    with open('provenance.txt', 'w') as f:
                        for line in dataset4th['provenance4th']:
                            f.write(line+"\n")

    def run_on_dir(self, subset):
        examples = self.import_files(subset)
        self.encode_examples(subset, examples) # xml elements, attributes and value are encoded into numbered features
        sampler = Sampler(os.path.join(config.data_dir, self.compendium, subset), self.length, self.sampling_mode, self.random_shifting, self.min_padding, self.verbose)
        dataset4th = sampler.run(self.iterations) # examples are sampled and transformed into a tensor ready for deep learning
        self.save(dataset4th, subset) # save the tensors

    def run_on_compendium(self):
        with cd(config.data_dir):
            subsets = os.listdir(self.compendium)
            subsets = [s for s in subsets if s != '.DS_Store']
        with cd(config.data4th_dir):
            if os.path.isdir(self.namebase):
                shutil.rmtree(self.namebase)
            os.mkdir(self.namebase)
        for train_valid_test in subsets:
            self.run_on_dir(train_valid_test)


class BratDataPreparator(DataPreparator):
    def __init__(self, options):
        super(BratDataPreparator, self).__init__(options)

    def import_files(self, subset):
        with cd(config.data_dir):
            path = os.path.join(self.compendium, subset)
            brat_examples = BratImport.from_dir(path)
        return brat_examples

    def encode_examples(self, subset, examples):
        with cd(config.data_dir):
            path = os.path.join(self.compendium, subset)
            for ex in examples:
                encoded_features = BratEncoder.encode(ex)
                encoded_example = EncodedExample(ex['provenance'], ex['text'], encoded_features)
                encoded_example.save(path)



def main():
    parser = argparse.ArgumentParser(description='Reads xml and transform into tensor format.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = []
    parser.add_argument('-c', '--path', default='demo_xml', help='path to the source compendium of xml files')
    parser.add_argument('-f', '--filenamebase', default='', help='namebase to save converted trainset and features set files')
    parser.add_argument('-b', '--brat', action='store_true', help='Use brat files instead of xml files')
    parser.add_argument('-X', '--iterations', default=5, type=int, help='number of times each example is sampled')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
    parser.add_argument('-L', '--length', default=150, type=int, help='length of the text snippets used as example')
    parser.add_argument('-W', '--window', action='store_true', help='switches to the sampling fig legends using a random window instead of parsing full sentences')
    parser.add_argument('-S', '--start', action='store_true', help='switches to mode where fig legends are simply taken from the start of the text and truncated appropriately')
    parser.add_argument('-d', '--disable_shifting', action='store_true', help='disable left random padding which is used by default to shift randomly text')
    parser.add_argument('-p', '--padding', default=config.min_padding, help='minimum padding added to text')
    parser.add_argument('-w', '--working_directory', help='Specify the working directory where to read and write files to')

    args = parser.parse_args()

    options = {}
    options['namebase'] = args.filenamebase
    options['compendium'] = args.path
    options['iterations'] = args.iterations
    options['verbose'] = args.verbose
    options['length'] = args.length
    options['path'] = args.path
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
        prep.run_on_compendium()

if __name__ == "__main__":
    main()
