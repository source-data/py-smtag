# -*- coding: utf-8 -*-
#T. Lemberger, 2018
#Licence:
#test in test/test_converter.py 

import numpy as np
import argparse
import torch

class Converter():
    """
    Conversion operations between unicode strings and tensors.
    """
    
    @staticmethod
    def t_encode(input_string):
        """
        Static method that encodes an input string into a 4D tensor.
        Args
            input_string (str): string to convert
        Returns
            (torch.Tensor): 3D tensor 1x32x1xL (1 example x 32 bits x L characters) representing characters as 32 features
        """
        
        L = len(input_string)
        t = torch.zeros(1,32,L)
        for i in range(L):
            code = ord(input_string[i])
            bits = torch.Tensor([int(b) for b in "{0:032b}".format(code)][::-1])
            t[0, : , i] = bits
        return t

    @staticmethod
    def t_decode(t):
        """
        Static method that decodes a 3D tensor into a unicode string.
        Args:
            t (torch.Tensor): 3D tensor 1x32xL (1 example x 32 bits x L characters) representing characters as 32 features
        Returns
            (str): resulting string
        """
        #tensor is 3D
        L = t.size(2)
        str = ""
        for i in range(L):
            code = 0
            for j in range(31):
                bit = int(t[0][j][i])
                code += bit*(2**j)
            str += chr(code) #python 2: unichr()
        return str     

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Encode decode string into binary tensors" )
    parser.add_argument('input_string', nargs='?', default= "this is so ☾😎 😎 L ‼️", help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string#.encod('utf-8')
    print("the decoded of encoded:", Converter.t_decode(Converter.t_encode(input_string)))
    
    
