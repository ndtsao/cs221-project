""" Useful helper functions """
import re
import csv
# import string
import collections
import random
# from nltk.corpus import stopwords
# import nltk
# from nltk.corpus import wordnet
# from nltk.tokenize import sent_tokenize, word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
'''
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

# def remove_stop_words(text):
#     """
#     Remove stop words from a string.
#     @param string x
#     @return string x with stop words removed
#     Example: 'The United States' --> {'United': 1, 'States': 1}
#     """
#     stop_words = set(stopwords.words('english'))
#     words = list(text.split())
#
#     for stop_word in stop_words:
#         words = [word for word in words if word != stop_word]
#
#     return ' '.join(words)

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

def ngram_features(corpus, n_grams):
    """
    @param list corpus: list of text documents
    @return [list, nparray]: first element is list of features, second element
    is feature vector for each document
    """
    # vectorizer = CountVectorizer()
    # counts = vectorizer.fit_transform(corpus)
    tfidf_vectorizer = TfidfVectorizer(smooth_idf=False, \
            ngram_range=(1, n_grams))
    feature_vectors = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectorizer.get_feature_names(), feature_vectors.toarray()
'''
def load(csv_path, cols, sample_prob=1.0):
    # , del_special_chars=True, rem_non_words=False,):
    """
    Load all articles from a given file
    @param string csv_path: path to file
    @param list cols: columns of csv file to read. If empty, then the function
    will read all columns
    @param bool del_special_chars: delete special characters from text
    @param bool rem_non_words: remove typos from text
    @return list: each component is the list of values of a particular column
    """
    with open(csv_path, 'rU') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader, None) # skip the header
        if len(cols) == 0:
            cols = range(len(header))

        counter = 0
        result = [[] for _ in cols]
        for row in reader:
            if counter % 100 == 0:
                print counter
                print row[0]
            counter += 1
            if random.random() > sample_prob:
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
    """
    raw_data, header = load(input_path, [], sample_prob=1.0)
    varietal_col = 12

    with open(output_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        # no headers in raw data
        writer.writerow(header)
        for row in zip(*raw_data):
            row = list(row)
            if "Top Wines" in flags and row[varietal_col] not in \
                    flags["Top Wines"]:
                continue
            for (idx, col) in enumerate(row):
                if "Special Chars" in flags and idx in \
                        flags["Special Chars"]:
                    row[idx] = re.sub('[^a-zA-Z0-9-_*.\' \-]', '', col)
            writer.writerow(row)




