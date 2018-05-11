import math
from random import random
from converter import Converter

#for code in {1..256}; do printf "\e[38;5;${code}m'$code'\e[0m";echo; done
#for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
#printf "\e[30;1mTesting color\e[0m"
#for i in range(25,50): print(f"\033[{i};1mTesting color {i}\033[0m")
COLORS = ["\033[30;1m", #grey
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
          "\033[45;37;1m"] #turquoise back
CLOSE_COLOR = "\033[0m"

class Visualization():

    @staticmethod
    def show_example(minibatches):
        M = len(minibatches)
        N = minibatches[0].N
        #select random j-th example in i-th minibatch
        rand_i = math.floor(M * random())
        rand_j = math.floor(N *  random())
        input = minibatches[rand_i].input[rand_j:rand_j+1, : , : , : ] #crappy syntax for pytorch; otherwise result is 3D
        print("inpud size",input.size())
        target = minibatches[rand_i].output[rand_j:rand_j+1, : , : , : ]
        original_text =  minibatches[rand_i].text[rand_j]
        provenance = minibatches[rand_i].provenance[rand_j]
        nf_input = input.size(1)

        text = Converter.t_decode(input[0:1, 0:31, :, : ]) #sometimes input has more than 32 features if feature2input option was chosen
        if nf_input > 32:
            print("Additional input features:")
            Visualization.print_pretty(input[0:1, 32:nf_input, : , : ])
    
        print("Expected:")
        Visualization.print_pretty_color(target, text)
    

    @staticmethod
    def print_pretty(features):
        symbols = ['_','.',':','^','|'] # for bins 0 to 0.1, 0.11 to 0.2, 0.21 to 0.3, ..., 0.91 to 1 
        N = len(symbols)
        for i in range(features.size(1)):
            track = ""
            for j in range(features.size(3)):
                k = math.ceil(N*features[0, i, 0, j]) - 1
                track += symbols[k]
            print(f"Tagging track {i}")
            print(track)

    @staticmethod
    def print_pretty_color(features, text):
        print(text)
        nf = features.size(1)
        colored_track = ""
        pos = 0
        for c in text:
            max  = 1
            max_f = 0
            for f in range(nf): # range(2) is 0, 1 should be blue red
                score = math.floor(features[0, f, 0, pos]*10)
                if score > max:
                     max = score
                     max_f = f
            if text:
                colored_track += f"{COLORS[(max_f % len(COLORS))+1]}{c}{CLOSE_COLOR}"
            else:
                colored_track += f"{COLORS[(max_f % len(COLORS))+1]}{symbols[max+1]}{CLOSE_COLOR}"
            pos = pos + 1
        print(colored_track)