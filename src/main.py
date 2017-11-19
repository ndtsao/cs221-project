import util
import numpy as np

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

    for (article_idx, pdist) in enumerate(article_distr):
        log_diff = dict()
        for (word, prob) in pdist.items():
            log_diff[word] = np.log(baseline_distr[word]) - np.log(prob)
        items = [(diff, word) for (word, diff) in log_diff.items()]
        items.sort()
        # headline = cnn_headlines[article_idx]
        print cnn_headlines[article_idx], article_idx

        nwords = 20
        print "Top " + str(nwords) + " words:"
        for idx in range(min(len(items), nwords)):
            print items[idx][1]
        log_diffs.append(log_diff)

if __name__=="__main__":
    main()
