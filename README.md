# food-categorization

This is an attempt at the "What's Cooking?" Kaggle competition.

https://www.kaggle.com/c/whats-cooking

The ing_matrix.py script uses a subset of the training data to create a matrix of the frequency of ingredients within each cuisine. Predictions of cuisine type are made for each recipe in a test data set by transforming the list of ingredients into a vector and calculating a dot product with the matrix from the training data to provide a measure of similarity to each cuisine type.
