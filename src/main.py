""" All prediction models go here """
# import numpy as np
# import matplotlib.pyplot as plt
import sys
import csv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import util

def write_word_counts(data_path, varietal, review_col, varietal_col):
    """
    Writes to a csv file a list of words and their counts. Stop words are not
    included.
    @param data_path: path to cleaned data
    @param string varietal: type of grape
    """
    [reviews, varietals] = util.load(data_path, [review_col, varietal_col])[0]
    indices = range(len(varietals))
    indices.reverse()
    for idx in indices:
        if varietals[idx] != varietal:
            del reviews[idx]
    all_text = ' '.join(reviews)
    word_counts = util.word_count(all_text)
    output_path = "../files/output/Word Counts/word_counts_" + varietal + ".csv"
    with open(output_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for (word, freq) in word_counts.items():
            writer.writerow([freq, word])

def cross_val(model, features, response, folds=5, error_metric='accuracy'):
    """
    cross-validation wrapper
    """
    scores = cross_val_score(model, features, response, cv=folds,
            scoring=error_metric)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def price_category(price):
    """
    Classifies a wine based on the type
    @param int price
    @returns category: Extreme value, value, popular premium, premium, super
        premium, ultra premium, luxury, super luxury, icon
    """
    if price < 4:
        return "Extreme Value"
    if price < 10:
        return "Value"
    if price < 15:
        return "Popular Premium"
    if price < 20:
        return "Premium"
    if price < 30:
        return "Super Premium"
    if price < 50:
        return "Ultra Premium"
    if price < 100:
        return "Luxury"
    if price < 200:
        return "Super Luxury"
    return "Icon"

def main():
    """
    main function
    """
    data_path = "../files/Wine/wine_cleaned_google-final.csv"
    red_path = "../files/Wine/reds.csv"
    white_path = "../files/Wine/whites.csv"
    data_clean = util.load(data_path, [1, 2, 4, 5, 12, 14], sample=1.0)[0]
    red_wines = util.load(red_path, [])[0][0]
    white_wines = util.load(white_path, [])[0][0]
    filter_wines = red_wines + white_wines
    [countries, reviews, ratings, prices, varietals, sentiments] = data_clean

    indices = range(len(varietals))
    indices.reverse()
    for idx in indices:
        if varietals[idx] not in filter_wines or prices[idx] == '' \
                or ratings[idx] == '' or countries[idx] == '':
            del countries[idx]
            del reviews[idx]
            del ratings[idx]
            del prices[idx]
            del varietals[idx]
            del sentiments[idx]

    prices = [float(price) for price in prices]
    categories = [price_category(price) for price in prices]
    ratings = [float(rating) for rating in ratings]
    sentiments = [float(sentiment) for sentiment in sentiments]
    logistic = LogisticRegression(solver='saga', max_iter=100, random_state=42,
            multi_class='multinomial')
    linear = LinearRegression()

    if sys.argv[1] == "nb":
        response = varietals
        metric = 'accuracy'
        if sys.argv[2] == "country":
            response = countries
        elif sys.argv[2] == "rating":
            response = ratings
            metric = 'neg_mean_squared_error'
        elif sys.argv[2] == "price":
            response = categories
        x_train, x_test, y_train, y_test = train_test_split(reviews, response,
                test_size=0.1, random_state=0)
        error = util.error_naive_bayes(zip(x_train, y_train), \
                zip(x_test, y_test), error_metric=metric)
        print error
    elif sys.argv[1] == 'bow':
        model = logistic
        metric = 'accuracy'
        num_grams = 1
        response = varietals
        if sys.argv[2] == "country":
            response = countries
        elif sys.argv[2] == "category":
            response = categories
        elif sys.argv[2] == "rating":
            model = linear
            metric = 'neg_mean_squared_error'
            response = ratings
        elif sys.argv[2] == "price":
            model = linear
            metric = 'neg_mean_squared_error'
            response = prices

        features = util.ngram_features(reviews, num_grams, int(sys.argv[3]))[1]
        cross_val(model, features, response, error_metric=metric)

    # model.fit(features, response)
    # util.predict_values(model, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
