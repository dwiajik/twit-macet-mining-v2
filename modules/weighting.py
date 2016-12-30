import collections
import math

from . import tokenizer

class TfWeighting:
    def __init__(self, labeled_tweets):
        self.term_frequencies = []
        for (tweet, category) in labeled_tweets:
            features = {}
            for token in tweet:
                features["{}".format(token)] = tweet.count(token)
            self.term_frequencies.append((features, category))

    def get_features(self):
        return self.term_frequencies

    def tf(self, tweet):
        features = {}
        for token in tweet:
            features["{}".format(token)] = tweet.count(token)
        return features

class TfIdfWeighting:
    def __init__(self, labeled_tweets):
        #self.term_frequencies = []
        ## Calculate Term Frequencies
        #for (tweet, category) in labeled_tweets:
        #    features = {}
        #    for token in tweet:
        #        features["{}".format(token)] = tweet.count(token)
        #    self.term_frequencies.append((features, category))

        # Calculate Document Frequencies
        self.df = {}
        for (tweet, category) in labeled_tweets:
            token_count = collections.Counter(tweet) # get unique tokens
            for token in token_count:
                try:
                    self.df[token] += 1
                except:
                    self.df[token] = 1

        # Calculate Invers Document Frequencies
        self.idf = {}
        self.tweet_count = len(labeled_tweets)
        for token, doc_count in self.df.items():
            self.idf[token] = math.log(self.tweet_count / (doc_count + 1), 10) # +1 for smoothing
            
        self.tf_idfs = []
        for (tweet, category) in labeled_tweets:
            tf_idf = {}
            for token in tweet:
                tf_idf["{}".format(token)] = tweet.count(token) * self.idf[token]
            self.tf_idfs.append((tf_idf, category))

    def get_features(self):
        return self.tf_idfs

    def tf_idf(self, tweet):
        features = {}
        for token in tweet:
            try:
                features["{}".format(token)] = tweet.count(token) * self.idf[token]
            except:
                features["{}".format(token)] = tweet.count(token) * math.log(self.tweet_count / 1, 10)
        return features
