""" All prediction models go here """
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import util

def varietal_prediction():
    """
    predict wine varietals from sommelier reviews
    """
    clean_path = "../files/Wine/wine_clean_harder.csv"
    data_clean = util.load(clean_path, [2, 12], sample=1.0)[0]
    [reviews, varietals] = data_clean
    num_grams = 1
    features = util.ngram_features(reviews, num_grams, 200)[1]
    # x_train, x_test, y_train, y_test = train_test_split(features, varietals,
    #         test_size=0.4, random_state=0)
    clf = LogisticRegression(solver='saga', max_iter=100, random_state=42,
            multi_class='multinomial')
    # util.predict_values(clf, x_train, y_train, x_test, y_test)
    scores = cross_val_score(clf, features, varietals, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def main():
    """
    main function
    """
    #TODO: will add options for running different models
    varietal_prediction()

if __name__ == "__main__":
    main()
