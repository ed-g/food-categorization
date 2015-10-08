from collections import defaultdict
import json
import pandas as pd

trainfile = open('train1.json')
train = json.loads(trainfile.read())
trainfile.close() 

inglist_by_cui = defaultdict(list)
for id, cuisine in train['cuisine'].iteritems():
    inglist_by_cui[cuisine].extend(train['ingredients'][id])

ingfreq_by_cui = {}
for cuisine, ingredients in inglist_by_cui.iteritems():
    ingfreq_by_cui[cuisine] = defaultdict(int)
    for ingredient in ingredients:
        ingfreq_by_cui[cuisine][ingredient] += 1 

for key in ingfreq_by_cui['brazilian']:
    if ingfreq_by_cui['brazilian'][key] == 1:
        print key
