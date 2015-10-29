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


