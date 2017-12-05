""" All prediction models go here """
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import util

def cross_val(model, features, response, folds=5):
    """
    cross-validation wrapper
    """
    scores = cross_val_score(model, features, response, cv=folds)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def main():
    """
    main function
    """
    data_path = "../files/Wine/wine_cleaned_google-final.csv"
    data_clean = util.load(data_path, [1, 2, 4, 5, 12, 14], sample=1.0)[0]
    [countries, reviews, ratings, prices, varietals, sentiments] = data_clean
    prices = [float(price) for price in prices]
    ratings = [float(rating) for rating in ratings]
    sentiments = [float(sentiment) for sentiment in sentiments]
    logistic = LogisticRegression(solver='saga', max_iter=100, random_state=42,
            multi_class='multinomial')
    linear = LinearRegression()

    # response = varietals
    # model = logistic
    # num_grams = 1
    # features = util.ngram_features(reviews, num_grams, 200)[1]

    response = rating
    model = linear
    features = zip(sentiment, price)
    cross_val(model, features, response)
    # x_train, x_test, y_train, y_test = train_test_split(features, response,
    #         test_size=0.4, random_state=0)
    # util.predict_values(model, x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
