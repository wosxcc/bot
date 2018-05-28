import re
import collections


def words(text):
    return  re.findall('[a-z]+',text.lower())


def train(features):
    model=collections.defaultdict(lambda :1)
    for f  in features:
        model[f]+=1
    return model

NWORDS= train(words(open('Lex_ratio.txt',).read())) ##, encoding='utf-8'

print(NWORDS)