# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
import argparse
import numpy as np
from .minibatches import Minibatches
from .loader import Loader
from ..predict.binarize import Binarized
from ..common.progress import progress
from ..common.config import DEFAULT_THRESHOLD
from ..common.importexport import load_model

class Accuracy(object):

    def __init__(self, model, minibatches, tokenize=False):
        self.model = model # is this just a reference? if yes, no need to pass model every time
        self.minibatches = minibatches
        self.tokenize  = tokenize
        if self.tokenize:
            for i, m in enumerate(self.minibatches):
                progress(i, self.minibatches.minibatch_number, "tokenizing minibatch {}".format(i))
                m.add_token_lists()
        if torch.cuda.device_count() > 0: # or torch.cuda.is_available() ?
            self.cuda_on = True
        else:
            self.cuda_on = False

    def run(self):

        nf = self.minibatches.nf_output
        epsilon = 1e-12
        p_sum = torch.Tensor(nf).fill_(epsilon)
        tp_sum = torch.Tensor(nf).fill_(epsilon)
        fp_sum = torch.Tensor(nf).fill_(epsilon)

        if self.cuda_on:
            p_sum = p_sum.cuda()
            tp_sum = tp_sum.cuda()
            fp_sum = fp_sum.cuda()

        output_semantics = self.model.output_semantics

        for m in self.minibatches:
            self.model.eval()
            prediction = self.model(m.input)
            if self.tokenize: 
                bin_pred = Binarized(m.text, prediction, output_semantics)
                bin_pred.binarize_with_token(m.tokenized)
                bin_target = Binarized(m.text, m.output, output_semantics)
                bin_target.binarize_with_token(m.tokenized)
                p, tp, fp = self.tpfp(bin_pred.start, bin_target.start, 0.99)
            else:
                thresholds = [concept.threshold for concept in output_semantics]
                thresholds = torch.Tensor(thresholds).resize_(1, nf , 1 )
                p, tp, fp = self.tpfp(prediction, m.output, thresholds) # DEFAULT_THRESHOLD)

            p_sum += p
            tp_sum += tp
            fp_sum += fp

        self.model.train()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / p_sum
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

    def __init__(self, model_basename, testset_basename, tokenize):
        self.model = load_model(model_basename)
        self.opt = self.model.opt
        self.opt['validation_fraction'] = 0 # single data set mode, no validation
        self.tokenize = tokenize
        loader = Loader(self.opt)
        dataset = loader.prepare_datasets(testset_basename)
        minibatches = Minibatches(dataset['single'], self.opt['minibatch_size'])
        benchmark = Accuracy(self.model, minibatches, tokenize=self.tokenize)
        self.precision, self.recall, self.f1 = benchmark.run()

    def display(self):
        print("\n Global stats: \033[1m")
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
        self.opt['validation_fraction'] = 0 # single data set mode, no validation
        print("opt from model", self.opt)
        self.tokenize = tokenize
        loader = Loader(self.opt) # validation_fraction == 0 ==> dataset['single'] is loaded
        dataset = loader.prepare_datasets(dataset_basename)
        minibatches = Minibatches(dataset['single'], int(self.opt['minibatch_size']))
        self.evaluator = Accuracy(self.model, minibatches, tokenize=self.tokenize)

    def run(self, n=10):
        old_thresholds = [concept.threshold for concept in self.model.output_semantics]
        for threshold in np.linspace(0, 1, n):
            for i, _ in enumerate(self.model.output_semantics):
                self.model.output_semantics[i].threshold = threshold
            precision, recall, f1 = self.evaluator.run()
            self.show(precision, recall, f1)
        for i, _ in enumerate(self.model.output_semantics[i]):
                self.model.output_semantics[i].threshold = old_thresholds[i]

    def show(self, precision, recall, f1):
        precision = "; ".join(["{:.3f}".format(p.item()) for p in precision])
        recall = "; ".join(["{:.3f}".format(p.item()) for p in recall])
        f1 = "; ".join(["{:.3f}".format(p.item()) for p in f1])
        print("\t".join(["{:.3f}".format(c.threshold) for c in self.model.output_semantics]), precision, recall, f1)

def main():
    parser = argparse.ArgumentParser(description='Accuracy evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', default='test_entities_test', help='Namebase of the dataset to import as testset')
    parser.add_argument('-m' , '--model',  default='entities.sddl', help='Basename of the model to benchmark.')
    parser.add_argument('-T' , '--no_token', action='store_true', help='Flag to disable tokenization.')
    parser.add_argument('-S' , '--scan', action='store_true', help='Flag to switch to threshold scaning mode.')

    arguments = parser.parse_args()
    testset_basename = arguments.file
    model_basename = arguments.model
    scan_threshold = arguments.scan
    tokenize = not arguments.no_token
    print("model: {}, testset: {}, tokenization: {}".format(model_basename, testset_basename, tokenize))
    if scan_threshold:
        s = ScanThreshold(model_basename, testset_basename, tokenize=tokenize)
        s.run()
    else:
        b = Benchmark(model_basename, testset_basename, tokenize=tokenize)
        b.display()

if __name__ == '__main__':
    main()