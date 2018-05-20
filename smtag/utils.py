# -*- coding: utf-8 -*-
import re
from collections import namedtuple

Token = namedtuple('Token', ['text', 'start', 'end'])

def tokenize(s):
    #patterns derived from python nltk library http://www.nltk.org/_modules/nltk/tokenize/punkt.html#PunktLanguageVars.word_tokenize
    
    re_word_start = r"[^\./\(\"\`{\[:;&\#\*@\)}\]\-–—‐−,]" #added . and / and dashes
    re_non_word_chars = r"(?:[\.?!)\";}\]\*:@\'\({\[,])" # added . and + 
    re_multi_char_punct = r"(?:\-–—‐−{2,}|\.{2,}|(?:\.\s){2,}\.)" # added dashes
    re_genotype = r"(?:[+\-–—‐−]/[+\-–—‐−])"
    re_hyphenated = r"(?<=[^\dαβγδεζηθικλμνξπρστυφχψω])(?:[\-–—‐−][^\dαβγδεζηθικλμνξπρστυφχψω])"
    
    re_word_tokenize_fmt = r'''(
        %(MultiChar)s
        |
        %(Genotype)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                         # White-space
            $|                                          # End-of-string
            %(NonWord)s|%(Genotype)s|%(Hyphenated)s  # Punctuation and Hyphenated
            
        )
        |
        \S
    )'''
    
    tokenizing_re = re.compile(re_word_tokenize_fmt % {
                    'NonWord': re_non_word_chars,
                    'MultiChar': re_multi_char_punct,
                    'WordStart': re_word_start,
                    'Genotype': re_genotype,
                    'Hyphenated': re_hyphenated
                    }, re.VERBOSE)
    #tokenizing_re = re.compile(r'''(?=[^\./\(\"\`{\[:;&\#\*@\)}\]\-–—‐−,])\S+?(?=(?:[\.\+?!)\";}\]\*:@\'\({\[])|\s|$)''', re.VERBOSE)
    matches = tokenizing_re.finditer(s)
    token_list = []
    for m in matches:
        token_list.append(Token(m.group(0), m.start(), m.end()))
    return token_list