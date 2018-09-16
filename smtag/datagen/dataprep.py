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

from ..common.mapper import xml_map
from ..common.converter import TString
from ..common.utils import cd, timer
from .. import config
from ..common.progress import progress
from .featurizer import XMLEncoder, Index

SPACE_ENCODED = TString(" ", dtype=torch.uint8)

# FIX THIS IN mapper
NUMBER_OF_ENCODED_FEATURES = 22 

class DataPreparator(object):
    """
    An  class to prepare examples as dataset that can be imported in smtag
    """

    def __init__(self, options):
        """
        """
        self.examples = []
        self.errors = []
        self.dataset4th = {}
        self.global_index = Index()
        self.max_local_concepts = 0 # maximum number of concepts indexed locally ie in a single panel
        self.options = options

    @staticmethod
    def to_tensor(encoded, left_limit, right_limit, len_global_index, max_local_concepts): # should include only terms within the desired sample
        """
        Takes a dictionary with encoded entity features and transforms it into tensor format.

        Returns
            t: a dictionary of tensors, each potentially with different dimensions, representing features for a given scope
            In this implementation:
            t['global']: 1D tensor where entities are 1-hot encoded using the code of the global index
            t['local']: 2D tensor where entites are 1-hot encoded using the code of the local index in first axe and the encoded feature (eg the role) 1-hot encoded on the second axe
        """

        
        t = {}
        t['global'] = torch.zeros((len_global_index,NUMBER_OF_ENCODED_FEATURES), dtype=torch.uint8)
        t['local'] = torch.zeros((max_local_concepts, NUMBER_OF_ENCODED_FEATURES), dtype=torch.uint8) # MAX_ENTITIES is the maximal number of entities we can list from one snippet of text
        for scope in ['local', 'global']:
            for key in encoded[scope]:
                concept = encoded[scope][key]
                if concept['rightmost'] > left_limit and concept['leftmost'] < right_limit:
                    for feature_code in concept['features'].codes:
                        t[scope][key, feature_code] = 1
        return t
    
    @timer
    def sample(self, encoded_examples):
        """
        Each example will be sampled multiple (=iterations) times, sliced, shifted and padded.
        """

        def pick_fragment(text, desired_length, mode='random'):
            """
            Picks a random text fragment of desired length, if possible. Returns the text fragment, the start and the end postitions.
            """
            L = len(text) # it is possible that L < desired_length
            if mode == 'start':
                start = 0
            else: #random window sampling method
                start = randrange(max(1, L-length+1)) # length is the desired fixed length of the snippet; L is the actual length of the figure legend; if L < length, then the same fragment is chose over and over... not good
            end = min(start + desired_length, L)
            fragment = text[start:end]
            return fragment, start, end

        def pad_and_shift(text, desired_length, random_shifting=True, min_padding=0):
            """
            Adds space padding on the left and the right and shifts randomly unless deactivated.
            Returns the padded text.
            """
            padding = desired_length + min_padding - len(text)
            if random_shifting:
                shift = randrange(-padding, padding)
            else:
                shift = 0
            left_padding = padding + shift
            right_padding = padding - shift
            padded_text = ' ' * left_padding + text + ' ' * right_padding
            return padded_text

        def show_text_stats(length_statistics, N):
            text_avg = float(sum(length_statistics) / N)
            text_std = float(torch.Tensor(length_statistics).std())
            text_max = max(length_statistics)
            text_min = min(length_statistics)
            print("\nlength of the {} examples selected:".format(N))
            print("{} +/- {} (min = {}, max = {})".format(text_avg, text_std, text_min, text_max))

        mode = self.options['sampling_mode']
        random_shifting = self.options['random_shifting']
        min_padding = self.options['padding']
        N = len(encoded_examples)
        length = self.options['length'] # desired length of snippet
        iterations = self.options['iterations']
        text4th = []
        provenance4th = []
        th = {}
        th['global'] = torch.zeros((N * iterations, len(self.global_index), NUMBER_OF_ENCODED_FEATURES), dtype=torch.uint8)
        th['local'] = torch.zeros((N * iterations, self.max_local_concepts, NUMBER_OF_ENCODED_FEATURES), dtype=torch.uint8)
        textcoded4th = torch.zeros((N * iterations, 32, length+(2*min_padding)), dtype=torch.uint8)
        length_statistics = []
        total = N * iterations
        index = 0

        # looping through the examples
        for i in range(N):
            text = encoded_examples[i]['text']
            encoded = encoded_examples[i]['encoded']
            provenance = encoded_examples[i]['provenance']
            L = len(text)
            length_statistics.append(L)

            # randomly sampling each example
            for j in range(iterations): # j is index of sampling iteration
                progress(i*iterations+j, total, "sampling")
                fragment, start, stop = pick_fragment(text, length, mode) 
                padded_frag = pad_and_shift(fragment, length, random_shifting, min_padding)
                text4th.append(padded_frag)
                textcoded4th[index] = TString(padded_frag, dtype=torch.uint8).toTensor()
                provenance4th.append(provenance)
                t_dict = DataPreparator.to_tensor(encoded, start, stop, len(self.global_index), self.max_local_concepts)
                for scope in t_dict:
                    th[scope][index] = t_dict[scope]
                index += 1

        show_text_stats(length_statistics, N)
        if self.options['verbose']:
            self.display(text4th, th['local'])

        return {
            'text4th': text4th,
            'textcoded4th': textcoded4th,
            'provenance4th':provenance4th,
            'global4th': th['global'],
            'local4th': th['local']
        }


    def encode_examples(self, examples, encoder=XMLEncoder.encode):
        """
        Encodes examples provided as XML Elements.
        """

        encoded_examples = []
        M = 0
        for id in examples:
            local_index = Index() # index for the local scope of this document
            encoded_features, text, L = encoder(examples[id], {'global': self.global_index, 'local':local_index})
            if text:
                example = {
                    'provenance': id, 
                    'text': text,
                    'encoded': encoded_features
                }
                encoded_examples.append(example)
            else:
                print("skipping an example in document with id=", id)
                print(str(examples[id]))
            if len(local_index) > M:
                M = len(local_index)
        self.max_local_concepts = M
        return encoded_examples

    def run_on_dir(self, path):
        self.examples = self.import_from_xml(self.options['path'])
        encoded_examples = self.encode_examples(self.examples) # xml elements, attributes and value are encoded into numbered features
        self.dataset4th = self.sample(encoded_examples) # examples are sampled and transformed into a tensor ready for deep learning
        self.save(self.options['namebase']) # save the tensors
        # save self.global_index and self.rev_globale_index

    def import_from_xml(self, path, XPath_to_examples='.//figure-caption'):
        """
        Import xml documents from dir. In each document, extracts examples using XPath.
        """

        path = os.path.join(config.working_directory, path)
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

        with cd(config.data_dir):
                archive_path = "{}".format(filenamebase)
                with ZipFile("{}.zip".format(archive_path), 'w', ZIP_DEFLATED) as myzip:
                    
                    # write local tensor
                    tensor_filename = "{}_local.pyth".format(archive_path)
                    torch.save(self.dataset4th['local4th'], tensor_filename)
                    myzip.write(tensor_filename)
                    os.remove(tensor_filename)

                    # write global tensor
                    tensor_filename = "{}_global.pyth".format(archive_path)
                    torch.save(self.dataset4th['global4th'], tensor_filename)
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
                            f.write(line)
                            #f.write(", ".join([str(line[k]) for k in ['id','index']]) + "\n")
                    myzip.write(provenance_filename)
                    os.remove(provenance_filename)

                    myzip.close()

                for info in myzip.infolist():
                    print("saved {} (size: {})".format(info.filename, info.file_size))


    def display(self, text4th, th):
        """
        Display text fragments and extracted features to the console.
        """
        
        N, R, C = th.size()
        index2concept = [""] * NUMBER_OF_ENCODED_FEATURES
        for e in xml_map['concepts']:
            for a in xml_map['concepts'][e]:
                for v in xml_map['concepts'][e][a]:
                    code = xml_map['concepts'][e][a][v]
                    if code is not None:
                        index2concept[code] = v
        legend = " ".join([str(i)+"="+label for i, label in enumerate(index2concept)])
        for i in range(N):
            print("\n")
            print("Text:")
            print(text4th[i])
            print(legend)
            print("\n")
            print("".join([str(i%10) for i in range(C)]))
            for row in range(R):
                print("".join([['-','+'][x] for x in th[i, row]]))


def main():
    parser = argparse.ArgumentParser(description='Reads xml and transform into tensor format.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = []
    parser.add_argument('-c', '--path', default='test_dir/trainset', help='path to the source compendium of xml files')
    parser.add_argument('-f', '--filenamebase', default='test', help='namebase to save trainset and features set files')
    parser.add_argument('-X', '--iterations', default=5, type=int, help='number of times each example is sampled')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
    parser.add_argument('-L', '--length', default=150, type=int, help='length of the text snippets used as example')
    parser.add_argument('-W', '--window', action='store_true', help='switches to the sampling fig legends using a random window instead of parsing full sentences')
    parser.add_argument('-S', '--start', action='store_true', help='switches to mode where fig legends are simply taken from the start of the text and truncated appropriately')
    parser.add_argument('-d', '--disable_shifting', action='store_true', help='disable left random padding which is used by default to shift randomly text')
    parser.add_argument('-p', '--padding', default=20, help='minimum padding added to text')
    parser.add_argument('-w', '--working_directory', help='Specify the working directory where to read and write files to')

    args = parser.parse_args()
    if args.working_directory:
        config.working_directory = args.working_directory

    options = {}
    options['namebase'] = args.filenamebase
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

    path = args.path
    prep = DataPreparator(options)
    prep.run_on_dir(path)

if __name__ == "__main__":
    main()
