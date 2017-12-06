""" Useful helper functions """
import re
import csv
# import string
import collections
import random
from nltk.corpus import stopwords
import nltk
# from nltk.corpus import wordnet
# from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    """
    @param dict vec1: a feature vector represented by a mapping from a feature
        (string) to a weight (float).
    @param dict vec2: same as vec1
    @return float: the dot product between vec1 and vec2
    """
    if len(vec1) < len(vec2):
        return dot_product(vec2, vec1)
    else:
        return sum(vec1.get(f, 0) * v for f, v in vec2.items())

def increment(vec1, scale, vec2):
    """
    Implements vec1 += scale * vec2 for sparse vectors.
    @param dict vec1: the feature vector which is mutated.
    @param float scale
    @param dict vec2: a feature vector.
    """
    for feature, val in vec2.items():
        vec1[feature] = vec1.get(feature, 0) + val * scale

def word_count(text):
    """
    Takes a string and outputs a dictionary of word counts
    @param string text: input string of words
    @return dict: the counts of every word in |text|
    """
    result = collections.defaultdict(int)
    words = text.split(' ')
    for word in words:
        word = word.lower()
        word = re.sub('[^a-zA-Z-_*]', '', word)
        result[word] += 1
    return result

def remove_non_words(text):
    """
    Remove words that are not in the (English) dictionary
    @param string x
    @return string x with non-English words removed
    Example: 'The Unitedd States' --> 'The States'
    """
    words = list(text.split())
    wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

    filtered_words = [word for word in words if word in wordlist]
    return ' '.join(filtered_words)

def remove_stop_words(text):
    """
    Remove stop words from a string.
    @param string x
    @return string x with stop words removed
    Example: 'The United States' --> {'United': 1, 'States': 1}
    """
    stop_words = set(stopwords.words('english'))
    words = list(text.split())

    for stop_word in stop_words:
        words = [word for word in words if word != stop_word]

    return ' '.join(words)

def normalize(vec):
    """
    Normalizes a probability vector
    @param dict vec with positive entries
    @return dict normalized_vec where weights add up to 1
    """
    norm_constant = sum(vec.values())
    assert norm_constant > 0

    norm_vector = dict()
    for key, value in vec.items():
        norm_vector[key] = float(value) / norm_constant
    return norm_vector

def word_pair_correl(texts):
    """
    Creates a table with correlation scores between every pair of words
    If v1 and v2 are vectors of word counts for word1 and word2, respectively,
    then the correlation score is defined to be
    dotProduct(v1, v2)^2 / (dotProduct(v1, v1) * dotProduct(v2, v2))
    @param list text: list of strings
    @return dictionary of dictionaries with correlation score
    """
    all_text = ' '.join(texts)
    word_counts = [word_count(text) for text in texts]
    all_word_counts = word_count(all_text)
    result = dict()
    for (idx1, (word1, count1)) in enumerate(all_word_counts.items()):
        for idx2 in range(idx1 + 1, len(all_word_counts.items())):
            word2, count2 = all_word_counts.items()[idx2]
            result[(word1, word2)] = float(sum([counts[word1] * counts[word2] \
                    for counts in word_counts])) / (count1 * count2)
    return result

def ngram_features(corpus, n_grams, n_features):
    """
    @param list corpus: list of text documents
    @return [list, nparray]: first element is list of features, second element
    is feature vector for each document
    """
    tfidf_vectorizer = TfidfVectorizer(smooth_idf=False, \
            ngram_range=(1, n_grams), stop_words='english')
    feature_vectors = tfidf_vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(n_features)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    feature_vectors = lsa.fit_transform(feature_vectors)
    return 0, feature_vectors

def predict_values(model, x_train, y_train, x_test, y_test):
    """
    Fits a model to training data and outputs on a test set
    @param function model
    @params matrices x_train, x_test
    @params vectors y_train, y_test
    @param args: other arguments for the model
    """
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    fit = model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    path = "../files/output/price_prediction.csv"
    with open(path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for (data_y, fit) in zip(y_test, predicted):
            writer.writerow([data_y, fit])

def aggregate(data_path, col, output_path):
    """
    Aggregates a column of a csv file
    @param string data_path: path to input csv file
    @param int col: column number, assumed to be smaller than the number of
        columns in the input file
    @param string output_path: path to write to
    """
    data = load(data_path, [col])[0][0]
    table = collections.defaultdict(int)
    for val in data:
        table[val] += 1
    with open(output_path, 'a') as output_file:
        writer = csv.writer(output_file)
        for (count, val) in table.items():
            writer.writerow([count, val])

def load(csv_path, cols, sample=1.0):
    """
    Load all articles from a given file
    @param string csv_path: path to file
    @param list cols: columns of csv file to read. If empty, then the function
        will read all columns
    @param sample: either a number between 0 and 1 to indicate a probability, or
        a list of rows to read
    @return list: each component is the list of values of a particular column
    """
    with open(csv_path, 'rU') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader, None) # skip the header
        if len(cols) == 0:
            cols = range(len(header))

        # counter = 0
        result = [[] for _ in cols]
        for (row_idx, row) in enumerate(reader):
            # if counter % 100 == 0:
            #     print counter
            # counter += 1
            if isinstance(sample, float):
                if random.random() > sample:
                    continue
            else:
                if row_idx not in sample:
                    continue

            for (col_idx, col) in enumerate(cols):
                val = row[col]
                # if del_special_chars:
                #     val = re.sub('[^a-zA-Z-_* ]', '', val)
                # if rem_non_words:
                #     val = remove_non_words(val)
                result[col_idx].append(val)

    return result, header

def clean_data(input_path, output_path, flags):
    """
    Reads a dataset from a csv file and outputs a clean version of it
    @param string input_path: path to source file
    @param string output_path: path to output file
    @param dict flags: cleaning options
    Options are indexed by the following:
        "Special Chars": list of columns for which the function should remove
            all non-alphanumeric characters
        "Top Wines": list of the wine varietals that we want to keep
        "Name in Review": if True, then remove instances where the name of the
            grape appears in the review (should make the prediction harder)
        "Country": if True, then remove rows without a country
        "Price": if True, then remove rows without a price
    """
    raw_data, header = load(input_path, [], sample=1.0)
    country_col = 1
    desc_col = 2
    price_col = 5
    varietal_col = 12

    with open(output_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        # no headers in raw data
        writer.writerow(header)
        counter = 0
        for row in zip(*raw_data):
            row = list(row)
            if "Top Wines" in flags and row[varietal_col] not in \
                    flags["Top Wines"]:
                continue
            if "Name in Review" in flags and flags["Name in Review"]:
                if row[desc_col].find(row[varietal_col]) >= 0:
                    continue
            if "Country" in flags and flags["Country"]:
                if row[country_col] == '':
                    continue
            if "Price" in flags and flags["Price"]:
                if row[price_col] == '':
                    continue
            for (idx, col) in enumerate(row):
                if "Special Chars" in flags and idx in \
                        flags["Special Chars"]:
                    col = re.sub('[^a-zA-Z0-9-_*.\' ]', '', col)
                    row[idx] = remove_stop_words(col)
            if counter % 100 == 0:
                print counter
            counter += 1
            writer.writerow(row)
