""" All prediction models go here """
# import numpy as np
# import matplotlib.pyplot as plt
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

def cross_val(model, features, response, folds=5):
    """
    cross-validation wrapper
    """
    scores = cross_val_score(model, features, response, cv=folds,
            scoring='mean_squared_error')
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def main():
    """
    main function
    """
    data_path = "../files/Wine/wine_cleaned_google-final.csv"
    data_clean = util.load(data_path, [1, 2, 4, 5, 12, 14], sample=1.0)[0]
    [countries, reviews, ratings, prices, varietals, sentiments] = data_clean
    # prices = [float(price) for price in prices]
    # ratings = [float(rating) for rating in ratings]
    # sentiments = [float(sentiment) for sentiment in sentiments]
    logistic = LogisticRegression(solver='saga', max_iter=100, random_state=42,
            multi_class='multinomial')
    linear = LinearRegression()

    # Logistic regression parameters
    # response = countries
    # model = logistic
    num_grams = 1
    features = util.ngram_features(reviews, num_grams, 200)[1]

    # Linear regression parameters
    response = ratings
    model = linear
    # features = [[sentiment] for sentiment in sentiments]
    # features = zip(sentiments, prices)
    cross_val(model, features, response)
    # model.fit(features, response)
    # x_train, x_test, y_train, y_test = train_test_split(features, response,
    #         test_size=0.4, random_state=0)
    # util.predict_values(model, x_train, y_train, x_test, y_test)

    # Other crap
    # var_list_unique = list(set(varietals))
    # for var in var_list_unique:
    #     print var
    #     write_word_counts(data_path, var, 2, 12)

if __name__ == "__main__":
    main()
