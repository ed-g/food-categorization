#!/usr/bin/env python3
# vim:ts=4:sw=4:expandtab:
from collections import defaultdict
import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import toolz


# Test how maps work.
def test_map():
    numbers = [x for x in range(26)]
    letters = [chr(65 + x) for x in numbers]
    
    d = {}
    for (l,n) in zip(letters,numbers):
        d[l] = n
    
    print (d)
    
    keys = d.keys()
    vals = d.values()
    
    for k,v in zip(
            [x for x in keys][1:]
           ,[x for x in vals][1:]):
        assert d[k] == v
    
def test_normalize():
    d = {'a': {'i1': 3, 'i2': 5},
         'b': {'i1': 6, 'i2': 2},
         'c': {'i1': 8, 'i2': -2},
         'd': {'i1': 2, 'i2': 11},
         }
    print ("d:", d)
    print ("d.values:", d.values())
    dvec = DictVectorizer()
    ary = dvec.fit_transform( d.values() )
    print ("dvec:", dvec)
    print ("ary:\n", ary)
    print ("d.keys()[:]", list(d.keys())[:])
    print ("d.keys()[3]", list(d.keys())[3])
    print ("ary[2]:\n", ary[2])
    print ("ary.toarray():\n", ary.toarray())
    ary = normalize(ary)
    print ("ary = normalize(ary)\n", ary.toarray())
    ary = normalize(ary, axis=0)
    print ("ary = normalize(ary, axis=0)\n", ary.toarray())
    ary = normalize(ary)
    print ("ary = normalize(ary)\n", ary.toarray())
    ary = normalize(ary, axis=0)
    print ("ary = normalize(ary, axis=0)\n", ary.toarray())
if __name__ == '__main__':
    test_normalize()

