# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
import re
from copy import deepcopy
from smtag.utils import xml_escape


class Binarized:
    '''
    Transforms a prediction in an all-or-none (0 or 1) tensor by thresholding. It can be used to binarize several examples, for example when benchmarking.

    Args:
        text_examples (list): a list of text that were used as input for the prediction.
        prediction (torch.Tensor): a N x nf x L Tensor with the predicted output of the model
        output_semantics (list): a list of the concepts (Feature) that correspond to the meaning of each output feature.

    Members:
        start: a N x nf x L tensor with 1 at the position of the first character of a marked term.
        stop: a N x nf x L tensor with 1 at the position of the last character of a marked term.
        marks: a N x nf x L tensor with 1 at the positions of each character of a marked term.
        score: the average score integrated over the length of the marked term.
        tokenized: a list of lists of token for each text example.

    Methods:
        binarize_with_token(tokenized_examples): takes tokenized example, thresholds the prediction tensor and computes start, stop, marks and scores members.
        fuse_adjascent(regex=" "): when to adjascent terms are marked with the same feature, their marking is 'fused' by updating start (of first term), stop (of last term) and marks (encompassing both terms and spacer).
    '''
    def __init__(self, text_examples, prediction, output_semantics): # will need a concept map to set feature-specific thresholds Object Prediction with inputstring, input concept output concept and output tensor
        self.text_examples = text_examples
        self.prediction = prediction
        self.output_semantics = output_semantics

        self.N = prediction.size(0)
        self.nf = prediction.size(1)
        self.L = prediction.size(2)
        dim = (self.N, self.nf, self.L)
        self.start = torch.zeros(dim) #.byte()
        self.stop = torch.zeros(dim) #.byte()
        self.marks = torch.zeros(dim) #.byte()
        self.score = torch.zeros(dim) #.byte()
        self.tokenized = []

    def binarize_with_token(self, tokenized_examples):
        '''
        Takes pre-tokenized examples and thresholds the prediction tensor to computes start, stop, marks and scores members..
        NOTE: stop marks are placed at the last character of a marked term.
        Args:
            tokenized_examples (list of lists): list of lists of token for each example.
        '''
        
        #PROBLEM: set token index
        self.tokenized = tokenized_examples
        for i in range(self.N):
            token = tokenized_examples[i]['token_list'] # tokenized_examples[i] has also fields 'start_index' and 'stop_index'
            for t in token:
                start = t.start
                stop = t.stop
                token_length = t.length
                for k in range(self.nf):
                    avg_score = float(self.prediction[i, k, start:stop].sum()) / token_length
                    concept = self.output_semantics[k]
                    if avg_score >= concept.threshold: 
                        self.start[i, k, start] = 1
                        self.stop[i, k, stop-1] = 1 # stop mark has to be stop-1 to be the last character and not the next; otherwise we would always need later to test if stop < length of string because of last token 
                        self.score[i, k, start] = round(100*avg_score)
                        self.marks[i, k, start:stop].fill_(1)

    def fuse_adjascent(self, regex=" "):
        '''
        When to adjascent terms are marked with the same feature, their marking is 'fused' by updating start (of first term), stop (of last term) and marks (encompassing both terms and spacer).
        
        Args:
            regext (str default=" "): a regex pattern that is used to identify spacers between token that 
        '''
        test = re.compile(regex)
        for i in range(self.N):
            input_string = self.text_examples[i]
            #if self.tokenized: 
            #    pos_iter = PositionIter(token=self.tokenized[i], mode='stop')
            #else:
            #    pos_iter = PositionIter(input_string)
            #for pos, _ in pos_iter:
            for t in self.tokenized[i]['token_list']:
                stop_mark = t.stop-1
                #s = "this is the black cat"
                #                 ||||| |||
                #     0123456789012345678901
                #                 ^    |       s[start:stop] => s[12:17] == 'black'
                #                       ^  |   s[start:stop] => s[18:21] == 'cat'
                # pos = 17: stop[, ,17] > 0.99 and start[,,18] > 0.99 [,,17]==" "
                if stop_mark < self.L - 2: #
                    for k in range(self.nf):
                        if self.stop[i, k, stop_mark] > 0.99 and self.start[i, k, stop_mark+2] > 0.99 and re.match(test, input_string[stop_mark+1]): 
                            self.stop[i, k, stop_mark] = 0 # remove the stop boundary of first term
                            self.start[i, k, stop_mark+2] = 0 # remove the start boundary of next term
                            self.marks[i, k, stop_mark+1] = 1 # fill the gap by marking the space as part of the tagged term
                            self.score[i, k, stop_mark] = 0 #ideally the average of the score of the fused words but would ned to find start of upstream word 

    def cat_(self, other):
        # self.text_examples stays untouched, assumed to be the same, could be tested
        self.prediction = torch.cat((self.prediction, other.prediction), 1)
        self.output_semantics += other.output_semantics
        # self.N = stays the same
        self.nf = self.prediction.size(1) #(after torch.cat of predictions)
        # self.L untouched
        self.start = torch.cat((self.start, other.start), 1)
        self.stop = torch.cat((self.stop, other.stop), 1)
        self.marks = torch.cat((self.marks, other.marks), 1)
        self.score = torch.cat((self.score, other.score), 1)
        assert(self.nf==self.marks.size(1)) # f"{self.nf}<>{self.marks.size(1)}")
        # self.tokenized untouched

    def erase_(self, other):
        # self.text_examples stays untouched, assumed to be the same, could be tested
        # self.prediction untouched ? or zeros at marks that need to be removed?
        # self.output_semantics untouched
        # self.N = stays the same
        #self.nf untouched
        # self.L untouched
        erasor = (1 - other.marks).float() # 0 where marks are, 1 where no mark; used as element-wise filter to erase the marked entites
        self.start *= erasor
        self.stop *= erasor
        self.marks *= erasor
        self.score *= erasor
        # self.tokenized untouched

    def clone(self):
        other = Binarized(deepcopy(self.text_examples), self.prediction.clone(), deepcopy(self.output_semantics))
        other.N = self.N
        other.nf = self.nf
        other.L = self.L
        other.start = self.start.clone()
        other.stop = self.stop.clone()
        other.marks = self.marks.clone()
        other.score = self.score.clone()
        other.tokenized = deepcopy(self.tokenized)
        return other