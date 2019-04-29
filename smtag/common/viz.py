# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import math
import torch
from torch.nn import functional as F
from random import random
from .converter import TString
from ..train.evaluator import Accuracy
from tensorboardX import SummaryWriter
from .. import config

NBITS = config.nbits
MARKING_CHAR = config.marking_char

#for code in {1..256}; do printf "\e[38;5;${code}m"$code"\e[0m";echo; done
#for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
#printf "\e[30;1mTesting color\e[0m"
#for i in range(25,50): print(f"\033[{i};1mTesting color {i}\033[0m")

class Show():
    COLORS = {
        "console": [
            "\033[30;1m", #grey
            "\033[34;1m", #blue first since often assayed
            "\033[31;1m", #red
            "\033[33;1m", #yellow
            "\033[32;1m", #green
            "\033[35;1m", #pink
            "\033[36;1m", #turquoise
            "\033[41;37;1m", #red back
            "\033[42;37;1m", #green back
            "\033[43;37;1m", #yellow back
            "\033[44;37;1m", #blue back
            "\033[45;37;1m" #turquoise back
        ],
        "html": [
            "<span style='color:grey'>",
            "<span style='color:blue'>",
            "<span style='color:red'>",
            "<span style='color:yellow'>",
            "<span style='color:green'>",
            "<span style='color:pink'>",
            "<span style='background-color:red;color:white'>",
            "<span style='background-color:green; color:white'>",
            "<span style='background-color:blue; color:white'>",
            "<span style='background-color:turquoise; color:white'>"
        ],
        "markdown": [""] * 12
    }

    CLOSE_COLOR = {
        "console": "\033[0m",
        "html": "</span>",
        "markdown": ""
    }

    BR = {
        "console": "\n",
        "html": "<br/>",
        "markdown": "\n"
    }

    N_COLORS = len(COLORS["console"])

    def __init__(self, format="markdown"):
        self.format = format
        self.col = Show.COLORS[format]
        self.close =  Show.CLOSE_COLOR[format]
        self.nl =  Show.BR[format]

    def example(self, dataloader, model = None):
        out = ""
        minibatch = next(iter(dataloader))
        N = minibatch.input.size(0) # N examples per minibatch
        #select random j-th example in i-th minibatch
        rand_j = math.floor(N * random())
        input = minibatch.input[[rand_j], : , : ] # rand_j index as list to keep the tensor 4D
        target = minibatch.output[[rand_j], : , : ]
        if minibatch.viz_context.size(0) !=0:
            viz_context = minibatch.viz_context[[rand_j], : ]
        else:
            viz_context = torch.Tensor(0)
            if torch.cuda.is_available():
                viz_context.cuda()

        # original_text =  minibatches[rand_i].text[rand_j]
        provenance = minibatch.provenance[rand_j]
        nf_input = input.size(1)
        if model is not None:
            model.eval()
            prediction = model(input, viz_context)
            model.train()

        text = str(TString(input[[0], 0:config.nbits, : ])) #sometimes input has more than NBITS features if feature2input option was chosen
        if nf_input > config.nbits:
            out += "\nAdditional input features:"+self.nl+self.nl
            out += "    "+self.print_pretty(input[[0], NBITS:nf_input, : ]) + self.nl + self.nl

        out+= "\n__Expected:__" + "({})".format(provenance.strip()) + self.nl + self.nl
        # out += self.print_pretty_color(target, original_text) + self.nl + self.nl# visualize anonymized characters with a symbol
        out += self.print_pretty_color(target, text) + self.nl + self.nl# visualize anonymized characters with a symbol
        out += self.print_pretty(target) + self.nl + self.nl

        if model is not None:
            out += "__Predicted:__" + self.nl + self.nl
            out += self.print_pretty_color(prediction, text) + self.nl + self.nl
            out += self.print_pretty(prediction) + self.nl + self.nl
            # thresh = torch.Tensor([0.5])
            # if torch.cuda.device_count() > 0:
            #     thresh = thresh.cuda()
            # p, tp, fp = Accuracy.tpfp(prediction, target, thresh) # need to put 0.5 as cuda() on GPU
            # precision = tp / (tp + fp)
            # recall = tp / p
            # f1 = 2 * recall * precision / (recall + precision)
            # out += "Accuracy of this example:" + self.nl
            # out += "p={}, tp={}, fp={}, precision={:.2f}, recall={:.2f}, f1={:.2f}".format(float(p), float(tp), float(fp), float(precision), float(recall), float(f1))
            # out += self.nl + self.nl

        out += ""
        return out
    
    SYMBOLS = ["_",".",":","^","|"] # for bins 0 to 0.1, 0.11 to 0.2, 0.21 to 0.3, ..., 0.91 to 1

    def print_pretty(self, features):
        f = torch.sigmoid(features)
        # f = F.softmax(f, 1)
        # f -= f.min()
        # f /= f.max()
        out = ""
        N = len(Show.SYMBOLS) # = 5
        for i in range(features.size(1)):
            track = ""
            for j in range(features.size(2)):
                k = min(N-1, math.floor(N*f[0, i, j])) # 0 -> 0; 0.2 -> 1; 0.4 -> 2; 0.6 -> 3; 0.8 -> 4; 1.0 -> 4
                track += Show.SYMBOLS[k]
            out += "Tagging track {}".format(i) + self.nl + self.nl
            out += "    " + track + self.nl + self.nl
        return out

    def print_pretty_color(self, features, text):
        text = text.replace(config.marking_char, '#')
        colored_track = "    "# markdown fixeed font
        nf = features.size(1)
        codes = features.argmax(1).view(-1)
        for code, c in zip(codes, text):
            colored_track += "{}{}{}".format(self.col[nf - 1 - int(code.item())], c, self.close)
        return colored_track

class Plotter(SummaryWriter):

    def __init__(self):
        super(Plotter, self).__init__()

    def add_scalars(self, series, scalar_dict, epoch):
        super(Plotter, self).add_scalars("data/"+series, scalar_dict, epoch)

    def add_example(self, tag, example, epoch=None):
        super(Plotter, self).add_text(tag, example, epoch) # add niter=epoch? https://github.com/lanpa/tensorboardX/issues/6

    def add_progress(self, tag, loss, f1, labels, epoch):
        render_f1 = "; ".join(["{}={:.2f}".format(str(concept), f1[i]) for i, concept in enumerate(labels)])
        render_loss = "{:.4f}".format(loss)
        text = "epoch {}\n\n    loss={}, f1: {}".format(epoch, render_loss, render_f1)
        super(Plotter, self).add_text(tag, text)

    def close(self):
        super(Plotter, self).close()
