# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from math import ceil
from collections import namedtuple
from smtag.converter import Converter
from smtag.binarize import Binarized
from smtag.serializer import Serializer
from smtag.utils import tokenize
from smtag.operations import t_replace
from smtag.config import MIN_PADDING,  MIN_SIZE, MARKING_CHAR

MARKING_ENCODED = Converter.t_encode(MARKING_CHAR)
SPACE_ENCODED = Converter.t_encode(" ")

class Predictor: #(nn.Module?)

    def __init__(self, model):
        self.model = model

    def padding(self, input_string_encoded): 
        padding_length = ceil(max(MIN_SIZE - input_string_encoded.size(2), MIN_PADDING)/2)
        pad_encoded = SPACE_ENCODED.repeat(1, 1, padding_length) # results in a 3D 1 x 32 X padding_length tensor representation, with N=1 example
        padded_string_encoded = torch.cat([pad_encoded, input_string_encoded, pad_encoded],2)
        return padded_string_encoded
        
    def combine_with_input_features(self, encoded_input_string, additional_input_features=None):
        #CAUTION: should additional_input_features be cloned before modifying it?
        if additional_input_features is not None:
            nf2 = additional_input_features.size(1)
            padding_length = (encoded_input_string.size(2) - additional_input_features.size(2)) / 2
            pad = torch.zeros(1, nf2, padding_length)
            padded_additional_input_features = torch.cat([pad, additional_input_features, pad], 2) # flanking the encoded string with padding tensor left and right along third dimension
            combined_input = torch.cat([encoded_input_string, padded_additional_input_features], 1) # adding additional inputs under the encoded string along second dimension
        else:
            combined_input = encoded_input_string
        return combined_input
    
    def forward(self, input_string_encoded, additional_input_features = None): 
        padded_string_encoded = self.padding(input_string_encoded)
        L = int(padded_string_encoded.size(2))
        padding_length = int((L - input_string_encoded.size(2)) / 2)
        combined_input = self.combine_with_input_features(padded_string_encoded, additional_input_features)
        
        #PREDICTION
        self.model.eval()
        prediction = self.model(combined_input)
        self.model.train()
    
        #remove safety padding
        prediction = prediction[ : , : , padding_length : L-padding_length]
        return prediction, input_string_encoded # return also encoded_input_string for implementations that rather take string as input_string

class EntityPredictor(Predictor):
    
    def __init__(self, model, tag='sd-tag', format='xml'):
        super(EntityPredictor, self).__init__(model) # the model should carry the semantics with him, transmit it to pred and be used by Tagger.element()
        self.tag = tag
        self.format = format
        
    def forward(self, input_string):
        input_string_encoded = Converter.t_encode(input_string)
        return super(EntityPredictor, self).forward(input_string_encoded)

    def markup(self, input_string):
        prediction, _ = self.forward(input_string) # returns also encoded_input_string
        bin_pred = Binarized([input_string], prediction, self.model.output_semantics) # this is where the transfer of the output semantics from the model to the binarized prediction happen; will be used for serializing
        token_list = tokenize(input_string)
        bin_pred.binarize_with_token([token_list])
        bin_pred.fuse_adjascent(regex=" ")
        tagger = Serializer(self.tag, self.format)
        tagged_ml_string= tagger.serialize(bin_pred) # bin_pred has output_semantics
        return tagged_ml_string

class SemanticsFromContextPredictor(Predictor):
    
    def __init__(self, model, tag='sd-tag', format='xml'):
        super(SemanticsFromContextPredictor, self).__init__(model)
        self.tag = tag
        self.format = format

    def anonymize_(self, encoded_input_string, marks, replacement = MARKING_ENCODED): # underscore makes it clear it is in place ,according to pytorch convention
        return t_replace(encoded_input_string, marks, replacement)

    def forward(self, input_string_encoded, binarized_entities): 
        string_encoded= input_string_encoded.clone()
        mask = binarized_entities.marks[0] # need to select appropriate output channel based on modle output semantics!!
        anonymized = self.anonymize_(string_encoded, mask)
        prediction = super(SemanticsFromContextPredictor, self).forward(anonymized)
        return prediction
