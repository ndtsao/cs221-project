import re
import csv
import sys
import random
from os import listdir
# from datetime import date, time, datetime
from string import strip, rstrip
# from pdb import set_trace as t

def parseFile(filename, sampleprob=1.0):
    """
    Parses a text file
    @param string filename: name of file, including directory
    @param list sample: If empty, then parse everything. Otherwise, parse the
    the topics with indices that appear in |sample|
    """
    with open(filename, 'rb') as f:
        # pre-process the file
        lines = f.readlines()
        lines = [strip(line, "*:,. ") for line in lines]
        lines = [l for l in lines if len(l) > 0]
        topic_header = "[["
        section_header = "=="
        # Sections that we don't want to include
        excluded_sections = ["See also", "References", "Further reading",
                "External links"]

        topics = []
        summaries = []
        articles = []

        summary_text = ""
        full_text = ""
        # Flag that determines whether to append to summary text or article
        # text
        summary_flag = True
        # When this flag is set to True, the reader will ignore all lines until
        # the next topic header
        skip_to_next_topic = True
        for line in lines:
            modified_line = re.sub('[^a-zA-Z0-9-_*. ]', '', line)
            if line[:2] == topic_header:
                if random.random() > sampleprob:
                    skip_to_next_topic = True
                    continue
                if len(topics) > 0:
                    articles.append(full_text)
                # Reset parameters
                summary_flag = True
                summary_text = ""
                full_text = ""
                skip_to_next_topic = False
                topics.append(modified_line)
            else:
                if not skip_to_next_topic:
                    if line[:2] == section_header:
                        # The text file is structured so that the summary
                        # appears before any of the section headers. When we
                        # first encounter a section header, switch the summary
                        # flag to false and append what we have already read to
                        # the list of summaries. We will not add section headers
                        # to the article text, since the prevalence of some
                        # section headers may throw off our analysis
                        if summary_flag:
                            summaries.append(summary_text)
                            summary_flag = False
                        if modified_line in excluded_sections:
                            skip_to_next_topic = True
                    else:
                        if summary_flag:
                            summary_text += (' ' + modified_line)
                        else:
                            full_text += (' ' + modified_line)

    return topics, summaries, articles

directory = "../../files/Wikipedia/wiki_dump/"
files = listdir(directory)
output_file = "../../files/output/wikipedia_output.csv"
with open(output_file, 'a') as csv_file:
    csvWriter = csv.writer(csv_file)
    num_topics = 0
    for (idx, text_file) in enumerate(files):
        if idx % 100 == 0: print idx, num_topics
        filename = directory + text_file
        topics, summaries, articles = parseFile(filename, sampleprob=0.01)
        num_topics += len(topics)
        for (topic, summary, article) in zip(topics, summaries, articles):
            csvWriter.writerow([topic, summary, article])
