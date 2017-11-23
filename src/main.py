import os
import re
import util
import numpy as np
import matplotlib.pyplot as plt

def main():
    # (article text, headline, date) tuples for various newspapers
    cnn, cnn_headlines, cnn_dates = util.loadArticles(
            "../files/Datasets -- Cleaned/CNN.csv", 2, 3, 5)
    all_text = ' '.join(cnn)
    # correlation_scores = util.word_pair_correl(cnn)
    article_word_counts = [util.word_count(article) for article in cnn]
    article_distr = [util.normalize(count) for count in article_word_counts]
    headline_word_counts = [util.word_count(article) for headline in cnn]
    headline_distr = [util.normalize(count) for count in headline_word_counts]
    # headline_key = [util.removeNonWords(headline) for headline in cnn_headlines]

    print len(cnn), len(cnn_headlines), len(cnn_dates)
    baseline_word_count = util.word_count(all_text)
    baseline_distr = util.normalize(baseline_word_count)
    log_diffs = []
    # list containing the average rank of headline words for each article
    average_ranks = []
    # list containing the words ranked by importance for each article
    important_words = []

    for (article_idx, pdist) in enumerate(article_distr):
        # dictionary comparing differences in the log distributions between the
        # individual articles and the baseline distribution
        log_diff = dict()
        for (word, prob) in pdist.items():
            log_diff[word] = np.log(baseline_distr[word]) - np.log(prob)
        items = [(diff, word) for (word, diff) in log_diff.items()]
        items.sort()
        important_words.append(items)
        headline = cnn_headlines[article_idx]
        headline_words = headline.split(' ')
        headline_words = [re.sub('[^a-zA-Z-_* ]', '', word) for word in headline_words]
        ranks = []
        for headline_word in headline_words:
            rank = [idx for (idx, (diff, word)) in enumerate(items) if word == headline_word]
            if len(rank) > 0:
                ranks.append(rank)
        # ranks = [[idx for (idx, (diff, word)) in enumerate(items)
        #     if word == headline_word][0] for headline_word in headline_words]
        average_ranks.append(np.mean(ranks) / len(items) if len(ranks) > 0 else 1)
        # print cnn_headlines[article_idx], article_idx

        # nwords = 20
        # print "Top " + str(nwords) + " words:"
        # for idx in range(min(len(items), nwords)):
        #     print items[idx][1]
        log_diffs.append(log_diff)

    sorted_ranks = [(rank, idx) for (idx, rank) in enumerate(average_ranks)]
    sorted_ranks.sort()
    top_rank_param = 20
    nwords = 20
    top_ranks = range(top_rank_param)
    bottom_ranks = range(len(sorted_ranks) - top_rank_param, len(sorted_ranks))
    for i in top_ranks + bottom_ranks:
        rank, idx = sorted_ranks[i]
        print rank, idx
        print cnn_headlines[idx]
        print "Top " + str(nwords) + " words:"
        important = important_words[idx]
        for word_idx in range(min(nwords, len(important) - 1)):
          print important_words[idx][word_idx][1]

    # plot the average ranks
    average_ranks.sort()
    x = range(len(average_ranks))
    nbins = 20
    hist = np.histogram(average_ranks, bins = nbins)[0]
    bins = [float(x)/nbins for x in range(nbins)]
    plt.plot(bins, hist)
    plt.xlabel("Normalized bins")
    plt.ylabel("Count")
    plt.title("Distribution of average rank of headline words")
    plt.axvline(x=0.5)
    plt.show()

if __name__=="__main__":
    main()
