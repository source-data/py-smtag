# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import math
from random import random
from .converter import Converter
from ..train.evaluator import Accuracy
from tensorboardX import SummaryWriter
from .. import config

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

    def __init__(self, format="markdown"):
        self.format = format
        self.col = Show.COLORS[format]
        self.close =  Show.CLOSE_COLOR[format]
        self.nl =  Show.BR[format]

    def example(self, minibatches, model = None):
        out = ""
        M = len(minibatches) # M minibatches
        N = minibatches[0].N # N examples per minibatch
        #select random j-th example in i-th minibatch
        rand_i = math.floor(M * random())
        rand_j = math.floor(N *  random())
        input = minibatches[rand_i].input[[rand_j], : , : ] # rand_j index as list to keep the tensor 4D
        target = minibatches[rand_i].output[[rand_j], : , : ]

        # original_text =  minibatches[rand_i].text[rand_j]
        provenance = minibatches[rand_i].provenance[rand_j]
        nf_input = input.size(1)
        if model is not None:
            model.eval()
            prediction = model(input)
            model.train()

        text = Converter.t_decode(input[[0], 0:config.nbits, : ]) #sometimes input has more than 32 features if feature2input option was chosen
        if nf_input > config.nbits:
            out += "\nAdditional input features:"+self.nl+self.nl
            out += "    "+self.print_pretty(input[[0], 32:nf_input, : ]) + self.nl + self.nl

        out+= "\n__Expected:__" + "({})".format(provenance.strip()) + self.nl + self.nl
        # out += self.print_pretty_color(target, original_text) + self.nl + self.nl# visualize anonymized characters with a symbol
        out += self.print_pretty_color(target, text) + self.nl + self.nl# visualize anonymized characters with a symbol
        out += self.print_pretty(target) + self.nl + self.nl

        if model is not None:
            out += "__Predicted:__" + self.nl + self.nl
            out += self.print_pretty_color(prediction, text) + self.nl + self.nl
            out += self.print_pretty(prediction) + self.nl + self.nl
            p, tp, fp = Accuracy.tpfp(prediction, target, 0.5)
            precision = tp / (tp + fp)
            recall = tp / p
            f1 = 2 * recall * precision / (recall + precision)
            out += "p={}, tp={}, fp={}, precision={}, recall={}, f1={}".format(p.item(), tp.item(), fp.item(), precision.item(), recall.item(), f1.item())
            out += self.nl + self.nl

        out += ""
        return out
    
    SYMBOLS = ["_",".",":","^","|"] # for bins 0 to 0.1, 0.11 to 0.2, 0.21 to 0.3, ..., 0.91 to 1

    def print_pretty(self, features):
        out = ""
        N = len(Show.SYMBOLS) # = 5
        for i in range(features.size(1)):
            track = ""
            for j in range(features.size(2)):
                k = min(N-1, math.floor(N*features[0, i, j])) # 0 -> 0; 0.2 -> 1; 0.4 -> 2; 0.6 -> 3; 0.8 -> 4; 1.0 -> 4
                track += Show.SYMBOLS[k]
            out += "Tagging track {}".format(i) + self.nl + self.nl
            out += "    " + track + self.nl + self.nl
        return out

    def print_pretty_color(self, features, text):
        text = text.replace(config.marking_char, '#')
        nf = features.size(1)
        colored_track = "    "# markdown fixeed font
        pos = 0
        for c in text:
            max  = 1
            max_f = -1
            for f in range(nf): # range(2) is 0, 1 should be blue red
                score = 0
                if not math.isnan(features[0, f, pos]): # can be NaN
                    score = math.floor(features[0, f, pos]*10)
                else:
                    print("NaN value!!!", features[0, f, : ])
                if score > max:
                     max = score
                     max_f = f
            if text:
                colored_track += "{}{}{}".format(self.col[max_f + 1], c, self.close)
            pos = pos + 1
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
