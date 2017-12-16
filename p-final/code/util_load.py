""" Useful helper functions """
import re
import csv
import collections
import random

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




