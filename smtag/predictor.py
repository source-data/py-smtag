# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from math import ceil
from collections import namedtuple
from smtag.converter import Converter
from smtag.binarize import Binarized
from smtag.serializer import Serializer
from smtag.utils import tokenize

class Predictor: #(nn.Module?)
    MIN_PADDING = 20 # this should be a common param with dataimport
    MIN_SIZE = 140 # input needs to be of minimal size to survive successive convergent convolutions; ideally, should be calculated analytically
    
    def __init__(self, model):
        self.model = model
        self.combined_input = None
        self.input_string = ""
        self.padding_length = 0

    def padding(self, input_string):
        def generate_pad(N):
             return " " * self.padding_length #other implementation could return random string 
        self.padding_length = ceil(max(self.MIN_SIZE - len(input_string), self.MIN_PADDING)/2)
        pad = generate_pad(self.padding_length) 
        return f"{pad}{input_string}{pad}"
        
    def combine_with_input_features(self, encoded_input_string, additional_input_features=None):
        #CAUTION: should additional_input_features be cloned before modifying it?
        if additional_input_features is not None:
            nf2 = additional_input_features.size(1)
            pad = torch.zeros(1, nf2, self.padding_length)
            padded_additional_input_features = torch.cat([pad, additional_input_features, pad], 2) # flanking the encoded string with padding tensor left and right along third dimension
            self.combined_input = torch.cat([encoded_input_string, padded_additional_input_features], 1) # adding additional inputs under the encoded string along second dimension
        else:
            self.combined_input = encoded_input_string
    
    def forward(self, input_string, additional_input_features = None):
        padded_string = self.padding(input_string)
        self.L = len(padded_string) #includes the padding! 
        encoded_input_string = Converter.t_encode(padded_string) # not optimal for speed: we could pad with a tensor with pre-define code for spaces or relevant padding character
        self.combine_with_input_features(encoded_input_string, additional_input_features)
        
        #PREDICTION
        self.model.eval()
        prediction = self.model(self.combined_input)
        self.model.train()
    
        #remove safety padding
        prediction = prediction[ : , : , self.padding_length : self.L-self.padding_length]
        return prediction
        
class EntityPredictor(Predictor):
    
    def __init__(self, model, tag='sd-tag', format='xml'):
        super(EntityPredictor, self).__init__(model) # the model should carry the semantics with him, transmit it to pred and be used by Tagger.element()
        self.tag = tag
        self.format = format
        
    def forward(self, input_string):
        return super(EntityPredictor, self).forward(input_string)

    def markup(self, input_string):
        prediction = self.forward(input_string)
        bin_pred = Binarized([input_string], prediction, self.model.output_semantics) # this is where the transfer of the output semantics from the model to the binarized prediction happen; will be used for serializing
        token_list = tokenize(input_string) #, positions
        bin_pred.binarize_with_token([token_list])
        bin_pred.fuse_adjascent(regex=" ")
        tagger = Serializer(self.tag, self.format)
        tagged_ml_string= tagger.serialize(bin_pred) # bin_pred has output_semantics
        return tagged_ml_string
