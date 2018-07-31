# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from test.smtagunittest import SmtagTestCase
from common.converter import Converter, TString
from common.utils import timer
from timeit import timeit

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

    def test_concat_1(self):
        a = "the "
        b = "cat"
        ab = a + b
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab, str(t_ab))
        self.assertTensorEqual(TString(ab).toTensor(), t_ab.toTensor())
    
    def test_concat_2(self):
        a = "the "
        b = ""
        ab = a + b
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab, str(t_ab))
        self.assertTensorEqual(TString(ab).toTensor(), t_ab.toTensor())
    
    def test_concat_3(self):
        a = ""
        b = "cat"
        ab = a + b
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab, str(t_ab))
        self.assertTensorEqual(TString(ab).toTensor(), t_ab.toTensor())

    def test_slice(self):
        the_cat = "The cat"
        the_cat_ts = TString(the_cat)
        the_ts = the_cat_ts[0:3]
        the = the_cat[0:3]
        expected = TString(the)
        self.assertEqual(expected.s, the_ts.s)
        self.assertTensorEqual(expected.t, the_ts.t)
    
    @timer
    def test_timing(self):
        for _ in range(100000):
            Converter.t_encode("🐌")

class TStringTest(SmtagTestCase):

    def test_lossless_decode_encode(self):
        text = "hallo"
        s1 = TString(text)
        tensor = s1.toTensor()
        s2 = TString(tensor)
        self.assertTensorEqual(s1.toTensor(), s2.toTensor())
        self.assertEqual(s1.size(), s2.size())
        self.assertEqual(len(s1), len(s1))
        self.assertNotEqual(len(s2), 0)
        self.assertEqual(text, str(s1))

    def test_concat(self):
        hello = TString("hello ")
        world = TString("world")
        hello_world = TString("hello world")
        concatenated = hello + world
        self.assertTensorEqual(hello_world.toTensor(), concatenated.toTensor())
    
    def test_len(self):
        
        x = TString("1234567890")
        l1 = len(x)
        l2 = len("1234567890")
        self.assertEqual(l1, l2)

    def test_repeat(self):
        c = "a"
        c10 = c * 10
        s = TString(c)
        s10 = s.repeat(10)
        t10 = s.toTensor().repeat(1,1,10)
        self.assertTensorEqual(t10, s10.toTensor())


if __name__ == '__main__':
    unittest.main()
