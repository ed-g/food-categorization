#!/usr/bin/env python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from collections import defaultdict
import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import toolz

#TEST_INPUT_FILE = 'program_test_data/train_has_5_cuisines.json' 
#assert total_cuisines(get_training_data(TEST_INPUT_FILE)) == 5

TRAIN_INPUT_FILE = 'data/train1.json' 
TEST_INPUT_FILE  = 'data/train2.json'

def get_training_data(json_file_name):
    trainfile = open(json_file_name)
    train = json.loads(trainfile.read())
    trainfile.close() 
    return train


def group_ingredients_by_cuisine(train):
    # inglist_by_cui == ingredient list, by cuisine
    inglist_by_cui = defaultdict(list)
    for id, cuisine in train['cuisine'].items():
        inglist_by_cui[cuisine].extend(train['ingredients'][id])
    return inglist_by_cui


def count_ingredients_in_cuisine(inglist_by_cui):
    # ingfreq_by_cui == ingredient frequency, by cuisine
    ingfreq_by_cui = {}
    for cuisine, ingredients in inglist_by_cui.items():
        ingfreq_by_cui[cuisine] = defaultdict(int)
        for ingredient in ingredients:
            ingfreq_by_cui[cuisine][ingredient] += 1 
    return ingfreq_by_cui

def normalize_matrix(ing_array):
    ing_array = normalize(ing_array) 
    ing_array = normalize(ing_array, axis=0)
    return ing_array



def main():
    # TODO: divide the ingredient counts by total number of sample recipes for each
    # cuisine, so that we have a [0,1.0] frequency of ingredient use across all
    # recipes for each cuisine.


    # NOTE: Don't modify ingfreq_by_cui below this line, we want to keep a
    # correspondence between its keys and values as we create a copy of values in
    # DictVectorizer, so if the keys or values are changed we could get confused as
    # to which cuisine was being represented by which item in the DictVectorizer.
    # ED 2015-10-29

    train = get_training_data(TRAIN_INPUT_FILE)

    inglist_by_cui = group_ingredients_by_cuisine(train)

    ingfreq_by_cui = count_ingredients_in_cuisine(inglist_by_cui)

    vec = DictVectorizer()
    # ing_array = vec.fit_transform(ingfreq_list)
    ing_array = vec.fit_transform( ingfreq_by_cui.values() )

    ing_array = normalize_matrix(ing_array)

    # open test file
    testfile = open(TEST_INPUT_FILE)
    test = json.loads(testfile.read())
    testfile.close()

    #make predictions with test data
    pred_ids = []
    predictions = []
    for id, ing_list in test['ingredients'].items():
        pred_ids.append(id)
        ingfreq = defaultdict(int)
        for ingredient in ing_list:
            ingfreq[ingredient] += 1
        pred_vec = vec.transform(ingfreq).transpose()
        cui_sim = ing_array.dot(pred_vec).todense().tolist()
        cuisines = list(ingfreq_by_cui.keys())
        predicted_cuisine_index = cui_sim.index(max(cui_sim))
        prediction = cuisines [ predicted_cuisine_index ]
        predictions.append(prediction)
    
    true_cuisine = [test['cuisine'][id] for id in pred_ids]
    correct_list = [true_cuisine[idx] == predictions[idx] for idx in
                    range(len(predictions))]
    accuracy = sum(correct_list)/float(len(correct_list))
    
    print (predictions[:10])
    print (true_cuisine[:10])
    print (test['ingredients'][pred_ids[1]])
    
    example_vec = vec.transform([{'sugar': 1}]).transpose()
    example_sim = ing_array.dot(example_vec).todense().tolist()
    
    
    print (list(ingfreq_by_cui.keys()))
    print (example_sim)
    print (accuracy)
    
    # print confusion_matrix(true_cuisine, predictions)

if __name__ == '__main__':
    main()

