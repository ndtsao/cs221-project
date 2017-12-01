"""
Example functions
"""
import util

def clean_example():
    """
    Simple example of how to clean the wine dataset
    """
    # file locations
    csv_path = "../files/Wine/winemag-data-130k-v2.csv"
    output_path = "../files/Wine/wine_clean.csv"
    wine_count = "../files/Wine/varietals.csv"

    # remove 10% of the rarest wines in the list
    count_data = util.load(wine_count, [])[0]
    counts, varietals = count_data
    counts = [int(count) for count in counts]
    remove_pct = 0.1
    remove_num = remove_pct * sum(counts)
    partial_sum = 0
    idx = 0
    while partial_sum < remove_num:
        partial_sum += counts[idx]
        idx += 1
    wines_to_keep = varietals[idx:]

    # create flags for the data cleaning function
    flags = dict()
    flags["Top Wines"] = wines_to_keep
    # can change these column indices
    flags["Special Chars"] = [2, 3, 6, 7, 8, 11, 12]
    util.clean_data(csv_path, output_path, flags)

clean_example()
