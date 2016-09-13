__author__ = 'Adward'
import numpy as np


def read_tweet_from_file(fname):
    tweets = []
    with open(fname, mode='r', encoding='utf-8') as tweet_f:
        for row in tweet_f:
            tweets.append(list(filter(lambda w: len(w) >= 5, row.split(' '))))
    return tweets


def get_freq_term(tweets):
    terms = {}
    for tweet in tweets:
        for word in tweet:
            if word in terms:
                terms[word] += 1
            else:
                terms[word] = 1
    items = sorted(terms.items(), key=lambda w: w[1], reverse=True)
    terms = [d[0] for d in items[:10]]
    return terms


def cons_doc_term_matrix(tweets, terms):
    A = [[0]*len(terms) for i in range(len(tweets))]
    for i in range(len(tweets)):
        for j in range(len(tweets[i])):
            word = tweets[i][j]
            for k in range(9):
                if word == terms[k]:
                    A[i][k] += 1
                    break
    return np.array(A)


def cons_corr_matrix(A):
    C = [[0] * 9 for i in range(9)]
    for i in range(9):
        for j in range(9):
            C[i][j] = np.corrcoef(A[:, i], A[:, j])[0, 1]
    return np.array(C)


def compute_pairs(C, terms):
    pairs = []
    for i in range(9):
        max_index = 0 if i else 1
        for j in range(9):
            if j != i:
                if C[i][j] > C[i][max_index]:
                    max_index = j
        pairs.append((terms[i], terms[max_index]))
    pairs.sort(key=lambda x: x[0])
    return pairs


if __name__ == '__main__':
    tweets = read_tweet_from_file('tweets.txt')
    terms = get_freq_term(tweets)
    if terms[0] != 'iphone':
        raise ValueError('The most frequent word is not "iphone"')
    else:
        terms.pop(0)
    A = cons_doc_term_matrix(tweets, terms)
    C = cons_corr_matrix(A)
    pairs = compute_pairs(C, terms)
    print(pairs)