import os

import re
import collections
import numpy as np
import pandas  as pd


dict_open=open('Lex_ratio.txt')

dict_read= dict_open.read()


txt_line=dict_read.split('\n')
NWORD = collections.defaultdict(lambda: 1)
for a_line in txt_line:
    line_into =a_line.split(' ')
    xx_into=line_into[1].split('\t')

    NWORD[line_into[0]]=float(xx_into[1])
alphabet = 'abcdefghijklmnopqretuvwxyz'

def edits1(word):
    n=len(word)

    new_word = set([word[0:i] + word[i + 1:] for i in range(n)] +
                   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +
                   [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +
                   [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])

    return new_word
def known_edits2(word):
    return  set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORD)


def known(words):
    return  set(w for w in words if w in NWORD)

def correct(words):
    candidates= known([words]) or known(edits1(words)) or known_edits2(words) or [words]
    # for word in candidates:
    #     print(word,NWORD[word])
    xcc_word=sorted(candidates ,key=lambda  w:NWORD[w], reverse=True)
    print(xcc_word)
    # print(max(candidates,key=lambda  w:NWORD[w]))
    return  xcc_word
    # return max(candidates,key=lambda  w:NWORD[w])





while(1):
    xxword=correct(input())

    xuan=input()
    if int(xuan)<=len(xxword)+1:
        print(xxword[int(xuan)-1])