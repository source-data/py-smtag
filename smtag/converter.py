#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import torch

class Converter(object):

    @staticmethod
    def t_encode(input_string):
        #input_string: string to convert
        #returns: 4D tensor 1x32x1xL (1 example, 32 bit,  1 row of characters, L characters) representing characters as 32 features
        L = len(input_string)
        tensor = torch.zeros(1,32,1,L)
        for i in range(L):
            code = ord(input_string[i])
            bits = list("{0:032b}".format(code))
            b = 0
            while bits: 
                bit = int(bits.pop())
                tensor[0][b][0][i] = bit
                b += 1
                
        return tensor

    @staticmethod
    def t_decode(t):
        #tensor is 4D
        L = t.shape[3]
        str = ""
        for i in range(L):
            code = 0
            for j in range(31):
                bit = int(t[0][j][0][i])
                code += bit*(2**j)
                #print i, j, bit, 2**j, code
            #what if code is malformed utf8? like unichr(57085) or unichr(55349)
            str += unichr(code)
        return str     

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Encode decode string into binary tensors" )
    parser.add_argument('input_string', nargs='?', default= "this is so ☾😎 😎 L ‼️", help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string.decode('utf-8')
    print "the decoded of encoded:", Converter.t_decode(Converter.t_encode(input_string))