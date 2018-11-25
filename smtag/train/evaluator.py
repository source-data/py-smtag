# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import torch
import argparse
import numpy as np
from .minibatches import Minibatches
from .loader import Loader
from ..predict.binarize import Binarized
from ..common.progress import progress
from ..common.config import DEFAULT_THRESHOLD
from ..common.importexport import load_model
from .. import config

DEFAULT_THRESHOLD = config.default_threshold

class Accuracy(object):

    def __init__(self, model, minibatches, tokenize=False):
        self.model = model # is this just a reference? if yes, no need to pass model every time
        self.minibatches = minibatches
        self.nf = self.minibatches.nf_output
        self.tokenize  = tokenize
        self.bin_target_start = []
        if torch.cuda.device_count() > 0: # or torch.cuda.is_available() ?
            self.cuda_on = True
        else:
            self.cuda_on = False
        if self.tokenize:
            for i, m in enumerate(self.minibatches):
                progress(i, self.minibatches.minibatch_number, "tokenizing minibatch {}".format(i))
                m_output = m.output
                if self.cuda_on:
                    m_output = m_output.cuda()
                m.add_token_lists()
                b = Binarized(m.text, m_output, self.model.output_semantics)
                b.binarize_with_token(m.tokenized)
                self.bin_target_start.append(b.start)
        epsilon = 1e-12
        self.p_sum = torch.Tensor(self.nf).fill_(epsilon)
        self.tp_sum = torch.Tensor(self.nf).fill_(epsilon)
        self.fp_sum = torch.Tensor(self.nf).fill_(epsilon)
        thresh = [concept.threshold for concept in self.model.output_semantics]
        self.thresholds = torch.Tensor(thresh).resize_(1, self.nf , 1 )
        if self.cuda_on:
            self.p_sum = self.p_sum.cuda()
            self.tp_sum = self.tp_sum.cuda()
            self.fp_sum = self.fp_sum.cuda()
            self.thresholds = self.thresholds.cuda()
            self.bin_target_start = [b.cuda() for b in self.bin_target_start]

    def run(self):
        for i, m in enumerate(self.minibatches):
            self.model.eval()
            m_input = m.input
            m_output = m.output
            if self.cuda_on:
                m_input = m_input.cuda()
                m_output = m_output.cuda()
            with torch.no_grad():
                prediction = self.model(m_input)
            if self.tokenize: 
                bin_pred = Binarized(m.text, prediction, self.model.output_semantics)
                bin_pred.binarize_with_token(m.tokenized)
                p, tp, fp = self.tpfp(bin_pred.start, self.bin_target_start[i], 0.99)
            else:
                p, tp, fp = self.tpfp(prediction, m_output, self.thresholds) # DEFAULT_THRESHOLD)

            self.p_sum += p
            self.tp_sum += tp
            self.fp_sum += fp

        self.model.train()
        precision = self.tp_sum / (self.tp_sum + self.fp_sum)
        recall = self.tp_sum / self.p_sum
        f1 = 2 * recall * precision / (recall + precision)

        return precision, recall, f1

    @staticmethod
    def tpfp(prediction, target, thresholds):
        """
        Computes accuracy at the level of individual characters.

        Args:
            threshold (0..1): to call a character as tagged ('hit')
            prediction (4D Tensor): the prediction for which the accuracy has to be computed
            target (4D Tensor): the target to which the prediction has to be compared to 

        Returns:
            Three 1D Tensors with the number per feature of positives, true positives and false positives, respectively
        """
        # https://en.wikipedia.org/wiki/Precision_and_recall
        nf = target.size(1)
        p = target.sum(2).sum(0) # sum over the characters on a line and sum of these over all examples of the batch
        predx = (prediction - thresholds).clamp(0).ceil() # threshold prediction to binarize in 0 (no hit) or 1 (hit)
        fp = (predx - target).clamp(0).sum(2).sum(0) # false positives when prediction is 1 but target is 0
        tp = predx.sum(2).sum(0) - fp 
        return p.view(nf), tp.view(nf), fp.view(nf) # output as 1 dim tensor with one metric per output feature

class Benchmark():

    def __init__(self, model_basename, dataset_basename, tokenize):
        self.model_name = model_basename
        self.model = load_model(model_basename)
        self.opt = self.model.opt
        self.tokenize = tokenize
        loader = Loader(self.opt)
        self.testset_name = os.path.join(config.data4th_dir, dataset_basename,'test')
        testset = loader.prepare_datasets(self.testset_name)
        minibatches = Minibatches(testset, self.opt['minibatch_size'])
        benchmark = Accuracy(self.model, minibatches, tokenize=self.tokenize)
        self.precision, self.recall, self.f1 = benchmark.run()

    def display(self):
        print("\n\n\033[31;1m========================================================\033[0m")
        print("\n\033[31;1m Data: {}\033[0m".format(self.testset_name))
        print("\n\033[31;1m Model: {}\033[0m".format(self.model_name))
        print("\n Global stats: \033[1m\n")
        print("\t\033[32;1mprecision\033[0m = {}.2f".format(self.precision.mean()))
        print("\t\033[33;1mrecall\033[0m = {}.2f".format(self.recall.mean()))
        print("\t\033[36;1mf1\033[0m = {}.2f".format(self.f1.mean()))

        for i, feature in enumerate(self.model.output_semantics):
            print("\n Feature: '\033[1m{}\033[0m'\n".format(feature))
            print("\t\033[32;1mprecision\033[0m = {}.2f".format(self.precision[i]))
            print("\t\033[33;1mrecall\033[0m = {}.2f".format(self.recall[i]))
            print("\t\033[36;1mf1\033[0m = {}.2f".format(self.f1[i]))

class ScanThreshold():

    def __init__(self, model_basename, dataset_basename, tokenize):
        self.model = load_model(model_basename)
        self.opt = self.model.opt
        print("opt from model", "; ".join(["{}={}".format(k, str(self.opt[k])) for k in self.opt]))
        self.tokenize = tokenize
        loader = Loader(self.opt) # validation_fraction == 0 ==> dataset['single'] is loaded
        validation = loader.prepare_datasets(os.path.join(config.data4th_dir, dataset_basename, 'valid'))
        self.minibatches = Minibatches(validation, int(self.opt['minibatch_size']))


    def run(self, n=11):
        old_thresholds = [concept.threshold for concept in self.model.output_semantics]
        for threshold in np.linspace(0, 1, n):
            for i, _ in enumerate(self.model.output_semantics):
                self.model.output_semantics[i].threshold = threshold
            evaluator = Accuracy(self.model, self.minibatches, tokenize=self.tokenize) # evaluator needs to be redone because thresholds need to be reassigned
            precision, recall, f1 = evaluator.run()
            self.show(precision, recall, f1)
        for i, _ in enumerate(self.model.output_semantics):
                self.model.output_semantics[i].threshold = old_thresholds[i]

    def show(self, precision, recall, f1):
        precision = "; ".join(["{:.3f}".format(p.item()) for p in precision])
        recall = "; ".join(["{:.3f}".format(p.item()) for p in recall])
        f1 = "; ".join(["{:.3f}".format(p.item()) for p in f1])
        print("\t".join(["{:.3f}".format(c.threshold) for c in self.model.output_semantics]), precision, recall, f1)

def main():
    parser = argparse.ArgumentParser(description='Accuracy evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', default='test_entities_test', help='Basename of the dataset to import (testset)')
    parser.add_argument('-m' , '--model',  default='entities.sddl', help='Basename of the model to benchmark.')
    parser.add_argument('-T' , '--no_token', action='store_true', help='Flag to disable tokenization.')
    parser.add_argument('-S' , '--scan', action='store_true', help='Flag to switch to threshold scaning mode.')

    arguments = parser.parse_args()
    basename = arguments.file
    model_basename = arguments.model
    scan_threshold = arguments.scan
    tokenize = not arguments.no_token
    print("model: {}, testset: {}, tokenization: {}".format(model_basename, basename, tokenize))
    if scan_threshold:
        s = ScanThreshold(model_basename, basename, tokenize=tokenize)
        s.run()
    else:
        b = Benchmark(model_basename, basename, tokenize=tokenize)
        b.display()

if __name__ == '__main__':
    main()