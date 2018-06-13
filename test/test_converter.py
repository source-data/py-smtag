# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from test.smtagunittest import SmtagTestCase
from smtag.converter import Converter, TString

class ConverterTest(SmtagTestCase):
    def setUp(self):
        self.input_string = u"😎😇" 
        code = 0x03B1 #GREEK SMALL LETTER ALPHA Unicode: U+03B1, UTF-8: CE B1
        self.single_character = chr(code) # python 2: unichr(code)
        bits = list("{0:032b}".format(code))
        bits.reverse()
        self.tensor = torch.Tensor([int(b) for b in bits]).resize_(1,32,1,1)

    def test_encode_string_into_tensor(self):
        converted = Converter.t_encode(self.input_string)
        self.assertTensorEqual(self.tensor, converted)

    def test_decode_tensor_into_string(self):
        self.assertEqual(self.single_character, Converter.t_decode(self.tensor))

    def test_lossless_encode_decode(self):
        self.assertEqual(self.input_string, Converter.t_decode(Converter.t_encode(self.input_string)))

    def test_lossless_decode_encode(self):
        self.assertTensorEqual(self.tensor, Converter.t_encode(Converter.t_decode(self.tensor)))

class TStringTest(SmtagTestCase):

    def test_lossless_decode_encode(self):
        text = "hallo"
        s1 = TString(text)
        tensor = s1.t
        s2 = TString(tensor)
        self.assertEqual(text, s2.s)
        self.assertTensorEqual(tensor, s2.t)
        self.assertEqual(tensor.size(), s1.size())


if __name__ == '__main__':
    unittest.main()

