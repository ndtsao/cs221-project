""" Useful helper functions """
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
        next(reader, None) # skip the header
        if len(cols) == 0:
            cols = range(len(reader[0]))

        counter = 0
        result = [[] for _ in cols]
        for row in reader:
            if counter % 100 == 0:
                print counter
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

    return result