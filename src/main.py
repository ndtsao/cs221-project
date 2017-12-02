import os
import re
import util
import numpy as np
import matplotlib.pyplot as plt

def main():
    clean_path = "../files/Wine/wine_clean.csv"
    data_clean, headers = util.load(clean_path, [2, 4, 5, 12], sample_prob=0.01)
    [reviews, ratings, prices, varieties] = data_clean
    num_grams = 2
    features = util.ngram_features(reviews, num_grams)


if __name__=="__main__":
    main()
