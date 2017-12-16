#!/usr/bin/python

import argparse
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from util_load import load
from util_load import clean_data
from examples import clean_example

import csv
import numpy as np
import matplotlib
matplotlib.use('tkagg')    # For some reason this line is required to make pyplot work
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# Import options
INPUT_PATH = "../files/Wine/wine_clean.csv"
# CLEANED_INPUT = "../files/Wine/cleanedreviews.csv"
INPUT_COLS = range(0, 14)
OUTPUT_PATH = "wine_cleaned_google-final.csv"
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
    '''
    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))
    '''
    # print annotations
    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0


def google_analyze(reviews, output_path, header):
    """Run a sentiment analysis request on imported file."""
    with open(output_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        header.append('score')
        header.append('magnitude')
        # writer.writerow(header)

        for i in range(0, len(reviews[0])):
            description = reviews[2][i]
            # print 'Review: {}'.format(description)


            client = language.LanguageServiceClient()

            '''
            with open(movie_review_filename, 'r') as review_file:
                # Instantiates a plain text document.
                content = review_file.read()
            '''

            document = types.Document(
                content=description,
                type=enums.Document.Type.PLAIN_TEXT)
            annotations = client.analyze_sentiment(document=document)

            row_to_be_written = list()
            for j in INPUT_COLS:
                row_to_be_written.append(reviews[j][i])
            row_to_be_written.append(annotations.document_sentiment.score)
            row_to_be_written.append(annotations.document_sentiment.magnitude)
        
            writer.writerow(row_to_be_written)
            # Print the results
            # print_result(annotations)
            if i % 10 == 0:
                print len(reviews[0]) - i
    return annotations



reviews, header = load(INPUT_PATH, INPUT_COLS)
google_annotations = google_analyze(reviews, OUTPUT_PATH, header)

