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

def get_data(json_file_name):
    datafile = open(json_file_name)
    data = json.loads(datafile.read())
    datafile.close() 
    return data


def group_ingredients_by_cuisine(training_data):
    # inglist_by_cui == ingredient list, by cuisine
    inglist_by_cui = defaultdict(list)
    for id, cuisine in training_data['cuisine'].items():
        inglist_by_cui[cuisine].extend(training_data['ingredients'][id])
    return inglist_by_cui


def count_ingredients_in_cuisine(inglist_by_cui):
    # ingcounts_by_cui == ingredient frequency, by cuisine
    ingcounts_by_cui = {}
    for cuisine, ingredients in inglist_by_cui.items():
        ingcounts_by_cui[cuisine] = defaultdict(int)
        for ingredient in ingredients:
            ingcounts_by_cui[cuisine][ingredient] += 1 
    return ingcounts_by_cui

def create_matrix_and_vectorizer(ingfreq_by_cui):
    vectorizer = DictVectorizer()
    cui_by_ing_matrix = vectorizer.fit_transform( ingfreq_by_cui.values() )
    return cui_by_ing_matrix, vectorizer


def normalize_matrix(matrix):
    matrix = normalize(matrix) 
    matrix = normalize(matrix, axis=0)
    return matrix

def create_vector(test_ing_list, vectorizer):
    ingfreq = defaultdict(int)
    for ingredient in test_ing_list:
        ingfreq[ingredient] += 1
    recipe_vec = vectorizer.transform(ingfreq).transpose()
    return recipe_vec

def predict_cuisine(recipe_vec, cui_by_ing_matrix, cuisines):
    cui_sim = cui_by_ing_matrix.dot(recipe_vec).todense().tolist()
    predicted_cuisine_index = cui_sim.index(max(cui_sim))
    prediction = cuisines [ predicted_cuisine_index ]
    return prediction

def make_predictions(test_data, vectorizer, cui_by_ing_matrix, ingfreq_by_cui):
    # make predictions with test data
    pred_ids = []
    predictions = []
    cuisines = list(ingfreq_by_cui.keys())
    for id, ing_list in test_data['ingredients'].items():
        pred_ids.append(id)
        pred_vec = create_vector(ing_list, vectorizer)
        prediction = predict_cuisine(pred_vec, cui_by_ing_matrix, cuisines)
        predictions.append(prediction)
    return predictions, pred_ids


def calculate_accuracy(test_data, predictions, pred_ids):
    true_cuisine = [test_data['cuisine'][id] for id in pred_ids]
    print (true_cuisine[:10])
    correct_list = [true_cuisine[idx] == predictions[idx] for idx in
                    range(len(predictions))]
    accuracy = sum(correct_list)/float(len(correct_list))
    return accuracy


def main():
    # TODO: divide the ingredient counts by total number of sample recipes for each
    # cuisine, so that we have a [0,1.0] frequency of ingredient use across all
    # recipes for each cuisine.

    train = get_data(TRAIN_INPUT_FILE)

    inglist_by_cui = group_ingredients_by_cuisine(train)

    ingfreq_by_cui = count_ingredients_in_cuisine(inglist_by_cui)

    # NOTE: Don't modify ingfreq_by_cui below this line, we want to keep a
    # correspondence between its keys and values as we create a copy of values in
    # DictVectorizer, so if the keys or values are changed we could get confused as
    # to which cuisine was being represented by which item in the DictVectorizer.
    # ED 2015-10-29

    ing_array, vec = create_matrix_and_vectorizer(ingfreq_by_cui)

    ing_array = normalize_matrix(ing_array)

    test = get_data(TEST_INPUT_FILE)

    # open test file
    # testfile = open(TEST_INPUT_FILE)
    # test = json.loads(testfile.read())
    # testfile.close()

    predictions, pred_ids = make_predictions(test, vec, ing_array, ingfreq_by_cui)

    accuracy = calculate_accuracy(test, predictions, pred_ids)

    print (predictions[:10])
    print (test['ingredients'][pred_ids[1]])
    
    example_vec = vec.transform([{'sugar': 1}]).transpose()
    example_sim = ing_array.dot(example_vec).todense().tolist()
    
    print (list(ingfreq_by_cui.keys()))
    print (example_sim)
    print (accuracy)
    
    # print confusion_matrix(true_cuisine, predictions)

if __name__ == '__main__':
    main()

