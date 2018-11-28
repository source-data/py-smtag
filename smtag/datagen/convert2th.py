# -*- coding: utf-8 -*-
#T. Lemberger, 2018

#from abc import ABC, abstractmethod
import argparse
import torch
import os.path
import shutil
import sys
import re
import copy
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
from .ocr import OCREncoder
from .brat import BratImport
from .context import VisualContext
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
        self.ocr_cxt_features = config.img_grid_size ** 2 + 2 # square grid + vertical + horizontal
        # self.viz_cxt_features = config.viz_cxt_features
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
    def create_tensors(N, iterations, number_of_features, ocr_cxt_features, length, min_padding): # viz_cxt_features, 
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

        textcoded4th   = torch.zeros((N * iterations, 32, length+(2*min_padding)), dtype=torch.uint8)
        features4th    = torch.zeros((N * iterations, number_of_features, length+(2*min_padding)), dtype = torch.uint8)
        ocr_context4th = torch.zeros((N * iterations, ocr_cxt_features, length+(2*min_padding)), dtype = torch.uint8)
        # viz_context4th = torch.zeros((N * iterations, viz_cxt_features), dtype = torch.uint8)
        return textcoded4th, features4th, ocr_context4th #, viz_context4th

    @staticmethod
    def display(text4th, tensor4th, ocr_context4th): #, viz_context4th):
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
            for j in range(ocr_context4th.size(1)):
                feature = 'ocr_' + str(j)
                track = [int(ocr_context4th[i, j, k]) for k in range(L)]
                print(''.join([['-','+'][ceil(x)] for x in track]), feature)
            feature = 'viz_' + str(j)
            track = [int(x) for x in viz_context4th[i]]
            print(''.join([str(x) for x in track]), feature)


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
             'ocrcontext4th': ocrcontext4th,
             'vizcontext4th':context4th}
            where:
            text4th: a list with a copy of the padded text of the sample
            textcoded4th: a 3D Tensor (samples x 32 x full length) with the fragment text encoded
            provenance4th: an array with the ids of the example from which the sample was taken
            tensor4th: a 3D Tensor with the encoded features corresponding to the fragment
            context4th: a 3D Tensor with location features of text elements extracted from the illustration
        """
        text4th = []
        provenance4th = []
        # textcoded4th, features4th, ocr_context4th, viz_context4th = self.create_tensors(self.N, iterations, self.number_of_features, self.ocr_cxt_features, self.viz_cxt_features, self.length, self.min_padding)
        textcoded4th, features4th, ocr_context4th = self.create_tensors(self.N, iterations, self.number_of_features, self.ocr_cxt_features, self.length, self.min_padding)
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
                ocr_context = encoded_example.ocr_context
                # viz_context = encoded_example.viz_context

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
                    # the encoded ocr context features 
                    if ocr_context is not None:
                        Sampler.slice_and_pad(ocr_context4th, index, ocr_context, start, stop, left_padding, right_padding)
                    # the visual context features are independent of the position of the text fragment
                    # if viz_context is not None:
                    #     viz_context4th[index] =  viz_context

                    index += 1

        Sampler.show_stats(length_stats, self.N)
        if self.verbose:
            Sampler.display(text4th, features4th, ocr_context4th) #, viz_context4th)

        return {
            'text4th': text4th,
            'textcoded4th': textcoded4th,
            'provenance4th':provenance4th,
            'tensor4th': features4th,
            'ocrcontext4th': ocr_context4th
            # ,
            # 'vizcontext4th': viz_context4th
        }

class EncodedExample():
    def __init__(self, provenance='', text='', features=None, ocr_context=None): #, viz_context=None):
        self._provenance = provenance
        self._text = text
        self._features = features
        self._ocr_context = ocr_context
        # self._viz_context = viz_context

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path) 
            with cd(path):
                torch.save(self.features, 'features.pyth')
                if self.ocr_context is not None:
                    torch.save(self.ocr_context, 'ocr_context.pyth')
                # if self.viz_context is not None:
                #     torch.save(self.viz_context, 'viz_context.pyth')
                with open('provenance.txt', 'w') as f: 
                    f.write(self.provenance)
                with open('text.txt', 'w') as f:
                    f.write(self.text)
        else:
            print("EncodedExample detected that {} already exists. Will not overwrite.".format(path))

    def load(self, path):
        with cd(path):
            with open('provenance.txt', 'r') as f:
                self._provenance = f.read()
            with open('text.txt', 'r') as f:
                self._text = f.read()
            self._features = torch.load('features.pyth').byte()
            if os.path.exists('ocr_context.pyth'):
                self._ocr_context = torch.load('ocr_context.pyth') # this is float()
            # if os.path.exists('viz_context.pyth'):
            #     self._viz_context = torch.load('viz_context.pyth') # this is float()

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
    def ocr_context(self):
        return self._ocr_context

    @property
    def viz_context(self):
        return self._viz_context


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
        self.anonymization_xpath = options['anonymize']
        self.enrichment_xpath = options['enrich'] 
        self.exclusive_xpath = options['exclusive']
        self.ocr = options['ocr']

    def encode_examples(self, subset, examples, anonymize=None):
        """
        Encodes examples provided as XML Elements and writes them to disk.
        """

        if self.ocr:
            ocr = OCREncoder(config.image_dir, G=config.img_grid_size)
        viz = VisualContext(config.image_dir)
        N = len(examples)
        for i, ex in enumerate(examples):
            xml = ex['xml']
            anonymized_xml = ex['anonymized']
            graphic_filename = ex['graphic']
            prov = ex['provenance']
            original_text = ''.join([s for s in xml.itertext()])
            anonymized_text = ''.join([s for s in anonymized_xml.itertext()])
            path_to_encoded = os.path.join(config.encoded_dir, self.compendium, subset, prov)
            progress(i, N, "{}".format(prov))
            if not os.path.exists(path_to_encoded): #path_to_example):
                if original_text:
                    # ENCODING XML
                    encoded_features = XMLEncoder.encode(anonymized_xml) # convert to tensor already here; 

                    # OCR CONTEXT HAPPENS HERE ! Needs the unaltered un anonymized original text for alignment
                    if self.ocr:
                        ocr_context = ocr.encode(original_text, graphic_filename) # returns a tensor
                    else:
                        ocr_context = None

                    # VISUAL CONTEXT HAPPENS HERE
                    # viz_context = viz.get_context(graphic_filename)
                    encoded_example = EncodedExample(prov, anonymized_text, encoded_features, ocr_context) #, viz_context)
                    encoded_example.save(path_to_encoded)
                else:
                    print("\nskipping an example in document with id=", prov)
            else:
                print("{} has already been encoded".format(prov))

    def enrich(self, xml, select):
        if select:
            keep = False
            for xpath in select:
                xml = copy.deepcopy(xml)
                found = xml.findall(xpath)
                if found:
                    return True
            if not found: 
                print(xpath, "not found")
                print(tostring(xml))
        else:
            keep = True
        return keep

    def exclusive(self, xml, keep_only, element='sd-tag'):
        if keep_only:
            xml = copy.deepcopy(xml)
            for xpath in keep_only:
                selected = xml.findall(xpath)
                for e in selected:
                    e.set('temp_attribute_keep_only_this', '1')
            all = xml.findall(".//"+element)
            for e in all:
                if e.get('temp_attribute_keep_only_this', False):
                    del e.attrib['temp_attribute_keep_only_this']
                else:
                    e.attrib = None
        return xml


    def anonymize(self, xml, anonymizations):
        if anonymizations:
            xml = copy.deepcopy(xml)
            for xpath in anonymizations:
                to_be_anonymized = xml.findall(xpath)
                for e in to_be_anonymized: #  # ".//sd-tag[@type='gene']"
                    # FIX PROBLEM WITH MISSING SPACES
                    innertext = "".join([s for s in e.itertext()])
                    for sub in list(e):
                        e.remove(sub)
                    e.text = config.marking_char * len(innertext)
        return xml

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
                #try:
                    with open(os.path.join(path, filename)) as f: 
                        xml = parse(f)
                        print("\n({}/{}) doi:".format(i, len(filenames)), xml.getroot().get('doi'))
                    for j, e in enumerate(xml.getroot().findall(XPath_to_examples)):
                        if self.enrich(e, self.enrichment_xpath):
                            e = self.exclusive(e, self.exclusive_xpath)
                            anonymized = self.anonymize(e, self.anonymization_xpath)
                            provenance = os.path.splitext(filename)[0] + "_" + str(j)
                            g = e.find(XPath_to_assets)
                            if g is not None:
                                basename = re.search(r'panel_id=(\w+)', g.get('href')).group(1)
                                graphic_filename = basename + '.jpg'
                            else:
                                print('no graphic element found')
                                graphic_filename = ''
                            examples.append({
                                'xml': e,
                                'anonymized': anonymized,
                                'provenance': provenance,
                                'graphic': graphic_filename
                            })
                # except Exception as e:
                #     print("problem parsing", os.path.join(path, filename))
                #     print(e)
                # progress(i, len(filenames), "loaded {}".format(filename))
        return examples


    def save(self, dataset4th, subset):
        """
        Saving datasets prepared for torch to a text file with text example, a npy file for the extracted features and a provenance file that keeps track of origin of each example.
        """

        with cd(config.data4th_dir):
            with cd(self.namebase):
                with cd(subset):
                    # write feature tensor
                    torch.save(dataset4th['tensor4th'], 'features.pyth')
                    # write encoded text tensor
                    torch.save(dataset4th['textcoded4th'], 'textcoded.pyth')
                    # write ocr context features
                    torch.save(dataset4th['ocrcontext4th'], 'ocrcontext.pyth')
                    # write visual context features
                    # torch.save(dataset4th['vizcontext4th'], 'vizcontext.pyth')
                    # write text examples into text file
                    with open("text.txt", 'w') as f:
                        for line in dataset4th['text4th']:
                            f.write("{}\n".format(line))
                    # write provenenance of each example into text file
                    with open('provenance.txt', 'w') as f:
                        for line in dataset4th['provenance4th']:
                            f.write(line+"\n")

    def run_on_dir(self, subset):
        print("\nImporting files from {}".format(subset))
        examples = self.import_files(subset)
        print("\nEncoding {} examples".format(len(examples)))
        self.encode_examples(subset, examples) # xml elements, attributes and value are encoded into numbered features
        print("\nSampling examples from {}".format(subset))
        path_to_encoded_data = os.path.join(config.encoded_dir, self.compendium, subset)
        sampler = Sampler(path_to_encoded_data, self.length, self.sampling_mode, self.random_shifting, self.min_padding, self.verbose)
        dataset4th = sampler.run(self.iterations) # examples are sampled and transformed into a tensor ready for deep learning
        print("\nSaving tensors to {}".format(subset))
        self.save(dataset4th, subset) # save the tensors

    def run_on_compendium(self):
        print("config.data_dir", config.data_dir, os.getcwd())
        with cd(config.data_dir):
            subsets = os.listdir(self.compendium)
            subsets = [s for s in subsets if s != '.DS_Store']

        with cd(config.encoded_dir):
            if not os.path.isdir(self.compendium):
                os.mkdir(self.compendium)
                with cd(self.compendium):
                    for s in subsets:
                        os.mkdir(s)

        with cd(config.data4th_dir):
            # overwrite data4th tensors if already there; resampling
            if os.path.isdir(self.namebase):
                shutil.rmtree(self.namebase)
            os.mkdir(self.namebase)
            with cd(self.namebase):
                for s in subsets:
                    os.mkdir(s)

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
    parser.add_argument('-A', '--anonymize', default='', help='Xpath expressions to select xml that will be anonymized. Example .//sd-tag[@type=\'gene\']')
    parser.add_argument('-e', '--exclusive', default='', help='Xpath expressions to keep only specific tags. Example .//sd-tag[@type=\'gene\']')
    parser.add_argument('-y', '--enrich', default='', help='Xpath expressions to make sure all examples include a given element. Example .//sd-tag[@type=\'gene\']')
    parser.add_argument('--noocr', action='store_true', default=False, help='Set this flag to prevent OCR analysis of images.')


    args = parser.parse_args()

    options = {}
    options['namebase'] = args.filenamebase
    options['compendium'] = args.path
    options['iterations'] = args.iterations
    options['verbose'] = args.verbose
    options['length'] = args.length
    options['path'] = args.path
    options['ocr'] = not args.noocr
    if args.window:
        options['sampling_mode'] = 'window'
    elif args.start:
        options['sampling_mode'] = 'start'
    else:
        options['sampling_mode'] = 'sentence'
    options['random_shifting'] = not args.disable_shifting
    options['padding'] = args.padding
    options['anonymize'] =  [a for a in args.anonymize.split(',') if a] # to make sure list is empty if args is ''
    options['exclusive'] =  [a for a in args.exclusive.split(',') if a]
    options['enrich'] =  [a for a in args.enrich.split(',') if a]
    print(options)
    if args.working_directory:
        config.working_directory = args.working_directory
    # with cd(config.working_directory):
    if args.brat:
        prep = BratDataPreparator(options)
    else:
        prep = DataPreparator(options)
    prep.run_on_compendium()

if __name__ == "__main__":
    main()
