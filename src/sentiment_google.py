#!/usr/bin/python

import argparse
# export GOOGLE_APPLICATION_CREDENTIALS="../CS 221 Wine-a94b0e2acae6.json"
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from utilload import load

import random
import collections
import math
import sys
# from util import *

# Import options
IMPORT_PATH = "../files/Wine/testwinereviews.csv"
IMPORT_COLS = range(0, 13)
# 0 = review number
# 1 = country
# 2 = description/review
# 3 = designation
# 4 = points
# 5 = price
# 6 = province
# 7 = region_1
# 8 = region_2
# 9 = taster_name
# 10 = taster_twitter
# 11 = title
# 12 = variety
# 13 = winery


def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0


def google_analyze(text):
    """Run a sentiment analysis request on input text."""
    client = language.LanguageServiceClient()

    '''
    with open(movie_review_filename, 'r') as review_file:
        # Instantiates a plain text document.
        content = review_file.read()
    '''

    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)

    # Print the results
    print_result(annotations)
    return annotations

reviews = load(IMPORT_PATH, IMPORT_COLS)

for i in range(0, len(reviews[0])):
    description = reviews[2][i]
    print 'Review: {}'.format(description)
    google_annotations = google_analyze(description)

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'movie_review_filename',
        help='The filename of the movie review you\'d like to analyze.')
    args = parser.parse_args()

    analyze(args.movie_review_filename)
'''
