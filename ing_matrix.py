from collections import defaultdict
import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

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

cuisine_axis = []
ingfreq_list = []
for cuisine, ing_hist in ingfreq_by_cui.iteritems():
     cuisine_axis.append(cuisine)
     ingfreq_list.append(ing_hist)

vec = DictVectorizer()
ing_array = vec.fit_transform(ingfreq_list)
ing_array = normalize(ing_array)

# open test file
testfile = open('train2.json')
test = json.loads(testfile.read())
testfile.close()

#make predictions with test data
pred_ids = []
predictions = []
for id, ing_list in test['ingredients'].iteritems():
    pred_ids.append(id)
    ingfreq = defaultdict(int)
    for ingredient in ing_list:
        ingfreq[ingredient] += 1
    pred_vec = vec.transform(ingfreq).transpose()
    cui_sim = ing_array.dot(pred_vec).todense().tolist()
    prediction = cuisine_axis[cui_sim.index(max(cui_sim))]
    predictions.append(prediction)

true_cuisine = [test['cuisine'][id] for id in pred_ids]
correct_list = [true_cuisine[idx] == predictions[idx] for idx in
                range(len(predictions))]
accuracy = sum(correct_list)/float(len(correct_list))

print accuracy
print predictions[:10]
print true_cuisine[:10]
print correct_list[:10]
