""" Useful helper functions """
import re
import csv
# import string
import collections
from nltk.corpus import stopwords
import nltk
# from nltk.corpus import wordnet
# from nltk.tokenize import sent_tokenize, word_tokenize

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

def load_articles(csv_path, cols, del_special_chars=True, rem_non_words=False):
    """
    Load all articles from a given file
    """
    articles = []
    summaries = []
    topics = []
    topic_col, summary_col, article_col = cols

    with open(csv_path, 'rU') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader, None) # skip the header

        counter = 0

        for row in reader:

            if counter % 100 == 0:
                print counter

            # kill all non-unicode articles (foreign language -- may want better
            # way to deal with this in the future)
            try:
                article = row[article_col]
                article = article.decode('utf-8', 'ignore')

                # tokenize the article
                # article = sent_tokenize(article)

                # delete numbers
                if del_special_chars:
                    article = re.sub('[^a-zA-Z-_* ]', '', article)
                    # chars_to_remove = '1234567890'
                    # table = {ord(char): None for char in chars_to_remove}
                    # article = article.translate(table).encode('ascii', 'ignore')
                    # article = [a.translate(table).encode('ascii', 'ignore')
                    #         for a in article]

                if rem_non_words:
                    article = remove_non_words(article)
                    # article = [remove_non_words(a) for a in article]

                # temp
                articles.append(article)
                summaries.append(row[summary_col])
                topics.append(row[topic_col])
                # articles.extend(article)
                # for i in range(len(article)):
                #     summaries.append(row[summary_col])
                #     topics.append(row[topic_col])

            except UnicodeError:
                print "non-unicode article"
            counter += 1

    return (topics, summaries, articles)
