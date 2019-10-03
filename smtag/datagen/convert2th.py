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
import pickle
import threading
import time
from math import floor, ceil
from xml.etree.ElementTree  import XML, parse, fromstring, tostring, XMLParser, ParseError

from nltk import PunktSentenceTokenizer
from random import choice, randrange, random, shuffle
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_BZIP2, ZIP_STORED

from ..common.mapper import Catalogue, index2concept, concept2index, NUMBER_OF_ENCODED_FEATURES
from ..common.converter import TString
from ..common.utils import cd, timer, tokenize, Token, cleanup, xml_escape, innertext
from ..common.progress import progress
from ..common.embeddings import EMBEDDINGS
from .encoder import XMLEncoder, BratEncoder
from .ocr import OCREncoder
from .brat import BratImport
from .context import VisualContext
from .. import config


class EncodedExample:
    features_filename = 'features.pyth'
    text_filename = 'text.txt'
    textcoded_filename = 'textcoded.pyth'
    provenance_filename = 'provenance.txt'
    ocr_context_filename = 'ocr_context.pyth'
    viz_context_filename = 'viz_context.pyth'

    def __init__(self, provenance, text, features, textcoded=None, ocr_context=None, viz_context=None):
        self.provenance = provenance
        self.text = text
        self.features = features
        self.textcoded = textcoded
        self.ocr_context = ocr_context
        self.viz_context = viz_context

class Sampler():

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
        # min_padding + text + filler + min_padding
        # len(filler) = desired_length - len(text)
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
    def slice_and_pad(desired_length, source_tensor, start, stop, min_padding, left_padding, right_padding):
        L = stop - start # length of the chunk
        channels = source_tensor.size(0)
        t = torch.zeros(channels, desired_length+(2*min_padding), dtype=torch.uint8)
        t[  : , left_padding:left_padding+L] = source_tensor[ : , start:stop]
        return t.unsqueeze(0)

    # @staticmethod
    # def show_stats(stats, skipped_examples, N):
    #     text_avg = floor(float(sum(stats) / N))
    #     text_std = torch.Tensor(stats).std()
    #     if N > 1:
    #         text_std = floor(float(text_std))
    #     else:
    #         text_std = 0
    #     text_max = max(stats)
    #     text_min = min(stats)
    #     print("\nlength of the {} examples selected ({} skipped because too short):".format(N, skipped_examples))
    #     print("{} +/- {} (min = {}, max = {})".format(text_avg, text_std, text_min, text_max))

class Augment():
    """
    A class to sample fragment from text examples, padd and shift the fragments and encode the corresponding text and features into Tensor format.
    """

    def __init__(self, length, sampling_mode, random_shifting, min_padding, verbose):
        self.mode = sampling_mode
        self.random_shifting = random_shifting
        self.min_padding = min_padding
        self.verbose = verbose
        self.length = length # desired length of snippet
        self.number_of_features = NUMBER_OF_ENCODED_FEATURES # includes the virtual geneprod feature
        self.ocr_cxt_features = config.img_grid_size ** 2 + 2 # square grid + vertical + horizontal

    def sample_and_save(self, path_to_encoded, encoded_example: EncodedExample, iterations):
        """
        The example will be sampled multiple (=iterations) times, sliced, shifted and padded.

        Args:
            iterations: number of times each example is sampled.

        Saves to disk:
            text4th: a copy of the padded text of the sample
            textcoded4th: a 3D Tensor (1 x NBITS x full length) with the fragment text encoded
            provenance4th: the id of the example from which the sample was taken
            tensor4th: a 3D Tensor with the encoded features corresponding to the fragment
            ocr_context4th: a 3D Tensor with location features of text elements extracted from the illustration
            viz_context4th: a 3D Tensor perceptual vision features
        """

        def sample(j, encoded_example):
            full_path = path_to_encoded + "_" + str(j)
            if os.path.exists(full_path):
                print(f"skipping {full_path}: already exists!")
            else:
                # a text fragment is picked randomly from the text example
                fragment, start, stop = Sampler.pick_fragment(encoded_example.text, self.length, self.mode)
                # it is randomly shifted and padded to fit the desired length
                padded_frag, left_padding, right_padding = Sampler.pad_and_shift(fragment, self.length, self.random_shifting, self.min_padding)
                # use context-aware embeddings
                if config.embeddings_model:
                    textcoded4th = TString(padded_frag).toTensor()
                    textcoded4th = EMBEDDINGS(textcoded4th)
                else:
                    textcoded4th = TString(padded_frag, dtype=torch.uint8).toTensor()
                    #assert str(TString(textcoded4th)) == padded_frag, f"\n'{str(TString(textcoded4th))}'\n is different from original \n'{padded_frag}'"
                # the encoded features of the fragment are selected and padded
                features4th = Sampler.slice_and_pad(self.length, encoded_example.features, start, stop, self.min_padding, left_padding, right_padding)
                # for conveniance, adding a computed feature to represent fused GENE and PROTEIN featres
                features4th[ : , concept2index[Catalogue.GENEPROD],  : ] = features4th[ : , concept2index[Catalogue.GENE],  : ] + features4th[ : ,  concept2index[Catalogue.PROTEIN], : ]
                # the encoded ocr context features is formatted the same way to stay in register
                ocr_context4th = None
                if encoded_example.ocr_context is not None:
                    ocr_context4th = Sampler.slice_and_pad(self.length, encoded_example.ocr_context, start, stop, self.min_padding, left_padding, right_padding)
                # the visual context features are independent of the position of the text fragment
                viz_context4th = encoded_example.viz_context
                #provenance, text, features, textcoded=None, ocr_context=None, viz_context=None
                processed_example = EncodedExample(encoded_example.provenance, padded_frag, features4th, textcoded4th, ocr_context4th, viz_context4th)
                self.save(full_path, processed_example) # tried to use .clone() does not alleviate threading problems...
                if self.verbose:
                    Augment.display(padded_frag, features4th, ocr_context4th, viz_context4th)

        L = len(encoded_example.text)
        adaptive_iterations = min(2, int(max(1.0, L / self.length))) * iterations
        for j in range(adaptive_iterations): # j is index of sampling iteration
            sample(j, encoded_example)
                


    def save(self, full_path, encoded_example:EncodedExample):
        if os.path.exists(full_path):
            print(f"skipping {full_path}: already exists!")
        else:
            os.mkdir(full_path)
            with open(os.path.join(full_path, encoded_example.provenance_filename),'w') as f:
                f.write(encoded_example.provenance)
            with open(os.path.join(full_path, encoded_example.text_filename), 'w') as f:
                f.write(encoded_example.text)
            torch.save(encoded_example.textcoded, os.path.join(full_path, encoded_example.textcoded_filename))
            torch.save(encoded_example.features, os.path.join(full_path, encoded_example.features_filename))
            if encoded_example.ocr_context is not None:
                torch.save(encoded_example.ocr_context, os.path.join(full_path, encoded_example.ocr_context_filename))
            if encoded_example.viz_context is not None:
                torch.save(encoded_example.viz_context, os.path.join(full_path, encoded_example.viz_context_filename))


    @staticmethod
    def display(text4th, tensor4th, ocr_context4th, viz_context4th):
        """
        Display text fragments and extracted features to the console.
        """
        N, featsize, L = tensor4th.shape

        print
        print("Text:")
        print(text4th)
        for j in range(featsize):
            feature = str(index2concept[j])
            track = [int(tensor4th[0, j, k]) for k in range(L)]
            print(''.join([['-','+'][x] for x in track]), feature)
        if ocr_context4th is not None:
            for j in range(ocr_context4th.size(1)):
                feature = 'ocr_' + str(j)
                track = [int(ocr_context4th[0, j, k]) for k in range(L)]
                print(''.join([['-','+'][ceil(x)] for x in track]), feature)
        if viz_context4th is not None:
            feature = 'viz_' + str(j)
            track = [int(x) for x in viz_context4th]
            print(''.join([str(x) for x in track]), feature)

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
        # self.XPath_to_fig_title = options['Xpath_to_fig_title'] # './/fig/caption/title
        self.XPath_to_examples = options['XPath_to_examples'] # './/fig/caption' or './/sd-panel'
        self.XPath_to_assets = options['XPath_to_assets'] # './/sd-panel/graphic' or './/fig/caption/graphic'
        self.ocr = options['ocr']
        self.viz = options['viz']

    @timer
    def encode_examples(self, subset, examples):
        """
        Encodes examples provided as XML Elements and writes them to disk.
        """

        if self.ocr:
            ocr = OCREncoder(config.image_dir)
        augmenter = Augment(self.length, self.sampling_mode, self.random_shifting, self.min_padding, self.verbose)
        N = len(examples)
        for i, ex in enumerate(examples):
            xml = ex['xml']
            processed_xml = ex['processed']
            graphic_filename = ex['graphic']
            prov = ex['provenance']
            original_text = innertext(xml) # needed when aligning OCR terms to the text
            processed_text = innertext(processed_xml) # alterations can be introduced by filtering or anonymization masking
            path_to_encoded = os.path.join(config.data4th_dir, self.namebase, subset, prov)
            progress(i, N, "{}".format(prov+"              "))
            if original_text:
                # ENCODING XML
                encoded_features = XMLEncoder.encode(processed_xml)
                # OCR and percetpual vision features
                ocr_context = None
                viz_context = None
                if self.viz or self.ocr:
                    if (graphic_filename is None or not os.path.exists(os.path.join(config.image_dir, graphic_filename))):
                        print("\nskipped example prov={}: graphic file not available".format(prov))
                    else:
                        if self.ocr:
                            ocr_context = ocr.encode(original_text, graphic_filename) # returns a tensor
                        if self.viz:
                            viz_context_filename = os.path.basename(graphic_filename)
                            viz_context_filename = os.path.splitext(viz_context_filename)[0] + '.pyth'
                            viz_context = torch.load(os.path.join(config.image_dir, viz_context_filename))
                        encoded_example = EncodedExample(prov, processed_text, encoded_features, None, ocr_context, viz_context)
                        augmenter.sample_and_save(path_to_encoded, encoded_example, self.iterations)
                else:
                    encoded_example = EncodedExample(prov, processed_text, encoded_features, None, ocr_context, viz_context)
                    augmenter.sample_and_save(path_to_encoded, encoded_example, self.iterations)
            else:
                print("\nskipping an example without text in document with id=", prov)

    def enrich(self, xml, select):
        if select:
            keep = False
            for xpath in select:
                xml = copy.deepcopy(xml)
                found = xml.findall(xpath)
                if found:
                    return True
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


    def anonymize(self, xml, xpath_expressions):
        """
        Randomly masks selected elements from xml. The element are selected using XPath expressions.
        """
        def mixed_masking(text, p_masking):
            """
            Replaces text with probability p_masking by a string of same length made of a concatenated special 'marking charcater'.
            """
            p = random()
            if p <= p_masking: # True with probability p_masking
                replacement = config.marking_char * len(text)
            else: # True with probability (1 - p_masking)
                replacement = text
            return replacement

        if xpath_expressions:
            xml = copy.deepcopy(xml)
            for xpath in xpath_expressions: 
                to_be_processed = xml.findall(xpath)
                for e in to_be_processed: # 
                    innertext = innertext(e) # "".join([s for s in e.itertext()])
                    for sub in list(e):
                        e.remove(sub)
                    e.text = mixed_masking(innertext, config.masking_proba)
        return xml

    def import_files(self, subset):
        """
        Import xml documents from dir. In each document, extracts examples using XPath_to_examples.
        """

        with cd(config.data_dir):
            path = os.path.join(self.compendium, subset)
            print("\nloading from:", path)
            filenames = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.xml']
            examples = []
            excluded = []
            for i, filename in enumerate(filenames):
                with open(os.path.join(path, filename), "r") as f:
                    xml = parse(f)
                    print("({}/{}) doi:".format(i+1, len(filenames)), xml.getroot().get('doi'), end='\r')
                for j, e in enumerate(xml.getroot().findall(self.XPath_to_examples)):
                    provenance = os.path.splitext(filename)[0] + "_" + str(j)
                    if not self.enrich(e, self.enrichment_xpath):
                        excluded.append(provenance)
                    else:
                        e = self.exclusive(e, self.exclusive_xpath)
                        processed = self.anonymize(e, self.anonymization_xpath)
                        g = e.find(self.XPath_to_assets) # Note: XPath_to_assets is to find graphic element *within* the example; NOT within the whole xml document!
                        if g is not None:
                            url = g.get('href')
                            id = re.search(r'(panel_id|figure_id)=(\d+)', url)
                            prefix = id.group(1)
                            number = id.group(2)
                            graphic_filename = prefix + "_" + number +".jpg"
                        else:
                            print('\nno graphic element found in the xml')
                            graphic_filename = None
                        examples.append({
                            'xml': e,
                            'processed': processed, #
                            'provenance': provenance,
                            'graphic': graphic_filename
                        })
            print("\nnumber of examples excluded because of enrichment: {}".format(len(excluded)))
        return examples


    def run_on_dir(self, subset):
        print("\nImporting files from {}".format(subset))
        examples = self.import_files(subset)
        print("\nEncoding {} examples".format(len(examples)))
        self.encode_examples(subset, examples) # xml elements, attributes and value are encoded into numbered features

    def run_on_compendium(self):
        print("config.data_dir", config.data_dir, os.getcwd())
        with cd(config.data_dir):
            subsets = os.listdir(self.compendium)
            subsets = [s for s in subsets if s != '.DS_Store']

        with cd(config.data4th_dir):
            if not os.path.isdir(self.namebase):
                os.mkdir(self.namebase)
            with cd(self.namebase):
                for s in subsets:
                    if not os.path.isdir(s):
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
        augmenter = Augment(self.length, self.sampling_mode, self.random_shifting, self.min_padding, self.verbose)
        N = len(examples)
        for i, ex in enumerate(examples): # 'text': text, 'annot': parsed_annotations, 'provenance': basename
            progress(i, N, f"{i+1}")
            encoded_features = BratEncoder.encode(ex)
            path_to_encoded = os.path.join(config.data4th_dir, self.namebase, subset, ex['provenance'])
            encoded_example = EncodedExample(ex['provenance'], ex['text'], encoded_features)
            augmenter.sample_and_save(path_to_encoded, encoded_example, self.iterations)

class DecoyDataPreparator(DataPreparator):
    def __init__(self, options):
        super(DecoyDataPreparator, self).__init__(options)
        self.decoy_tags = options['decoy_tags']

    def random_tag(self, text, p, tagset=['<sd-tag type="gene">', '<sd-tag type="protein">']):
        tokenized = tokenize(text)
        token_list = tokenized['token_list']
        escaped_token_list = []
        for token in token_list:
            new_token = Token(
                text = xml_escape(token.text),
                start = token.start,
                stop = token.stop,
                length = token.length,
                left_spacer = xml_escape(token.left_spacer)
            )
            assert not "&<>\"'" in new_token.text, f"{new_token.text} contains characters that should have been xml escaped!"
            assert not "&<>\"'" in new_token.left_spacer, f"left spacer {new_token.left_spacer} contains characters that should have been xml escaped!"
            escaped_token_list.append(new_token)
        N = len(escaped_token_list)
        n = floor(N * p)
        indices = list(range(N))
        shuffle(indices)
        picked = indices[:n]
        for i in picked:
            old_token = escaped_token_list[i]
            # ideally use nltk pos tagger to only tag nouns?
            open_tag = choice(tagset)
            closing_tag = re.sub(r'<([a-zA-Z\-]+)[ >].*', r'</\1>', open_tag)
            tagged_token_text = f"{open_tag}{old_token.text}{closing_tag}"
            new_token = Token(
                text = tagged_token_text,
                start = old_token.start,
                stop = old_token.stop,
                length = old_token.length,
                left_spacer = old_token.left_spacer
            )
            escaped_token_list[i] = new_token
        randomly_tagged_text = "".join([t.left_spacer + t.text for t in escaped_token_list])
        randomly_tagged_text = f"<article>{randomly_tagged_text}</article>"
        return randomly_tagged_text
         
    
    def import_files(self, subset):
        """
        Import decoy text documents from dir. In each document, randomly tag words.
        """

        with cd(config.data_dir):
            path = os.path.join(self.compendium, subset)
            print("\nloading from:", path)
            filenames = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.txt']
            examples = []
            for i, filename in enumerate(filenames):
                with open(os.path.join(path, filename), "r") as f:
                    text = f.read()
                    text = cleanup(text)
                    print(f"{i+1}/{len(filenames)} {filename}          ", end='\r')
                provenance = os.path.splitext(filename)[0]
                if self.decoy_tags == ['notag']:
                    tagged = f"<article>{xml_escape(text)}</article>"
                    
                else:
                    tagged = self.random_tag(text, p=0.02, tagset = self.decoy_tags)
                try:
                    tagged_xml = fromstring(tagged)
                except ParseError as e:
                    print("="*60)
                    print(tagged)
                    print("="*60)
                    faulty_position = int(re.search(r'column (\d+)', str(e)).group(1)) - 1
                    print(f"faulty character: '{tagged[faulty_position-10:faulty_position]}>>>{tagged[faulty_position]}<<<{tagged[faulty_position+1:faulty_position+10]}'")
                    raise(e)
                processed = self.anonymize(tagged_xml, self.anonymization_xpath)
                examples.append({
                    'xml': tagged_xml,
                    'processed': processed,
                    'provenance': provenance,
                    'graphic': None
                })
        return examples

def main():
    parser = config.create_argument_parser_with_defaults(description='Reads xml and transform into tensor format.')
    args = []
    parser.add_argument('-c', '--path', default='demo_xml', help='path to the source compendium of xml files')
    parser.add_argument('-f', '--filenamebase', default='', help='namebase to save converted trainset and features set files')
    parser.add_argument('-b', '--brat', action='store_true', help='Use brat files instead of xml files')
    parser.add_argument('-X', '--iterations', default=5, type=int, help='number of times each example is sampled')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
    parser.add_argument('-L', '--length', default=150, type=int, help='length of the text snippets used as example')
    parser.add_argument('-S', '--start', action='store_true', help='switches to mode where fig legends are simply taken from the start of the text and truncated appropriately')
    parser.add_argument('-d', '--disable_shifting', action='store_true', help='disable left random padding which is used by default to shift randomly text')
    parser.add_argument('-g', '--padding', default=config.min_padding, help='minimum padding added to text')
    parser.add_argument('-A', '--anonymize', default='', help='Xpath expressions to select xml that will be processed. Example .//sd-tag[@type=\'gene\']')
    parser.add_argument('-e', '--exclusive', default='', help='Xpath expressions to keep only specific tags. Example .//sd-tag[@type=\'gene\']')
    parser.add_argument('-y', '--enrich', default='', help='Xpath expressions to make sure all examples include a given element. Example .//sd-tag[@type=\'gene\']')
    parser.add_argument('-E', '--example', default='.//sd-panel', help='Xpath to extract examples from XML documents')
    parser.add_argument('-G', '--graphic', default='.//graphic', help='Xpath to find link to graphic element in an example.')
    parser.add_argument('--decoy', default='', help='List of tags to use in order to generate a randomly tagged decoy dataset. Use "--decoy notag" to generate a decoy without tags')
    parser.add_argument('--noocr', action='store_true', default=False, help='Set this flag to prevent use of image-based OCR data.')
    parser.add_argument('--noviz', action='store_true', default=False, help='Set this flag to prevent use of image-based visual context data.')

    args = parser.parse_args()

    options = {}
    options['namebase'] = args.filenamebase
    options['compendium'] = args.path
    options['iterations'] = args.iterations
    options['verbose'] = args.verbose
    options['length'] = args.length
    options['path'] = args.path
    options['ocr'] = not args.noocr
    options['viz'] = not args.noviz
    options['XPath_to_examples'] = args.example
    options['XPath_to_assets'] = args.graphic
    if args.start:
        options['sampling_mode'] = 'start'
    else:
        options['sampling_mode'] = 'window'
    options['random_shifting'] = not args.disable_shifting
    options['padding'] = int(args.padding)
    options['anonymize'] =  [a for a in args.anonymize.split(',') if a] # to make sure list is empty if args is ''
    options['exclusive'] =  [a for a in args.exclusive.split(',') if a]
    options['enrich'] =  [a for a in args.enrich.split(',') if a]
    options['decoy_tags'] = [a for a in args.decoy.split(',') if a]
    print(options)
    if args.brat:
        prep = BratDataPreparator(options)
    elif args.decoy:
        options['viz'] = False
        options['ocr'] = False
        prep = DecoyDataPreparator(options)
    else:
        prep = DataPreparator(options)
    prep.run_on_compendium()

if __name__ == "__main__":
    main()
