import re
import csv
import sys
from os import listdir
# from datetime import date, time, datetime
from string import rstrip
# from pdb import set_trace as t

def parseFile(dirname, filename):
    # go through the file
    filename = dirname + filename
    with open(filename, 'rb') as f:
        # pre-process the file
        lines = f.readlines()
        lines = map(rstrip, lines)
        lines = [l for l in lines if len(l) > 0]

        terms_flag = False
        summary_flag = False
        case_text_flag = False
        summary_end_flags = ['Headnotes']
        text_end_flags = ['References', 'End document']
        case_text = ''
        summary_text = ''
        terms = []
        for line in lines:
            line = re.sub('[^a-zA-Z0-9-_*. ]', '', line)
            if re.match("Core Terms", line.strip()):
                terms_flag = True
                continue
            if re.match("Case Summary", line.strip()):
                summary_flag = True
                continue
            if re.match("Syllabus", line.strip()) \
                    or re.match("Opinion", line.strip()):
                case_text_flag = True
                continue

            if terms_flag:
                terms = line.split(',')
                terms = [term.split() for term in terms]
                terms_flag = False
                continue

            if summary_flag:
                summary_text += (' ' + line)
                if line in summary_end_flags:
                    summary_flag = False

            if case_text_flag:
                case_text += (' ' + line)
                if line in text_end_flags:
                    case_text_flag = False

        return [terms, case_text, summary_text]

files = listdir(sys.argv[1])
with open(sys.argv[2], 'a') as csv_file:
    csvWriter = csv.writer(csv_file)
    for casefile in files:
        dirname = "../../files/"
        csvWriter.writerow(parseFile(dirname, casefile))
