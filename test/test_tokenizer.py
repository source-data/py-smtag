# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import unittest
from smtag.utils import tokenize, timer

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            ("This iß great, or what? ♨︎♡⚗! Ah.", ['This', 'iß', 'great', ',', 'or', 'what', '?', '♨︎♡⚗', '!', 'Ah', "."]),
            ("Expression of Atg5-/- after X-123-3 (3mM or 3μM) treatment.", ['Expression', 'of', 'Atg5', '-/-', 'after', 'X-123-3', '(', '3mM', 'or', '3μM', ")", 'treatment', "."]),
            ("Atg5-/- IM-123 VSV‐eGFP IFN‐β Atg7-β+/+ cases", ['Atg5','-/-', 'IM-123', 'VSV', '‐', 'eGFP', 'IFN‐β', 'Atg7-β', '+/+', 'cases']),
            ("Atg5",["Atg5"]),
            ("this is β-actin talking to INF-γ", ['this', 'is', 'β-actin', 'talking', 'to', 'INF-γ']),
            ("a)A shitty panel", ["a", ")", "A", "shitty", "panel"])
        ]

    def test_tokenize(self):
        for example in self.examples:
            token_list, _, _ = tokenize(example[0])
            token_terms = [t.text for t in token_list]
            #print([f"[{t.start}]{t.text}[{t.stop}]" for t in token_list])
            self.assertEqual(example[1], token_terms)

    def test_detokenize(self):
        for example in self.examples:
            token_list, _, _ = tokenize(example[0])
            humpty_dumpty = ''.join(["{}{}".format(t.left_spacer, t.text) for t in token_list])
            #print(humpty_dumpty)
            self.assertEqual(example[0], humpty_dumpty)

    @timer
    def test_speed(self):
        for i in range(10000):
            tokenize(self.examples[0][0])


if __name__ == '__main__':
    unittest.main()

