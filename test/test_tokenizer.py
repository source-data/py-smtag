# -*- coding: utf-8 -*-
import unittest
from smtag.utils import tokenize

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            ("This iß great, or what? ♨︎♡⚗! Ah.", ['This', 'iß', 'great', ',', 'or', 'what', '?', '♨︎♡⚗', '!', 'Ah', "."]),
            ("Expression of Atg5-/- after X-123-3 (3mM or 3μM) treatment.", ['Expression', 'of', 'Atg5', '-/-', 'after', 'X-123-3', '(', '3mM', 'or', '3μM', ")", 'treatment', "."]),
            ("Atg5-/- IM-123 VSV‐eGFP IFN‐β Atg7-β+/+ cases", ['Atg5','-/-', 'IM-123', 'VSV', '‐', 'eGFP', 'IFN‐β', 'Atg7-β', '+/+', 'cases']),
            ("Atg5",["Atg5"]),
            ("this is β-actin talking to INF-γ", ['this', 'is', 'β-actin', 'talking', 'to', 'INF-γ'])
        ]

    def test_tokenize(self):
        for example in self.examples:
            token = tokenize(example[0])
            print(token)
            self.assertEqual(example[1], token)

if __name__ == '__main__':
    unittest.main()

