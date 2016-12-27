import collections
import math
import random
import re
import time

import nltk
from sklearn.svm import LinearSVC

class Classifier:
    def clean_tweet(self, tweet):
        # remove "RT"
        regex = re.compile('RT\s')
        tweet = regex.sub(' ', tweet)

        tweet = tweet.lower()

        # remove non alphabetic character
        regex = re.compile('\shttp.+\s')
        tweet = regex.sub(' ', tweet)
        regex = re.compile('@[a-zA-Z0-9_]+')
        tweet = regex.sub(' ', tweet)
        regex = re.compile('RT\s')
        tweet = regex.sub(' ', tweet)
        regex = re.compile('[^a-zA-Z0-9]')
        tweet = regex.sub(' ', tweet)

        # replace abbreviations
        replacement_word_list = [line.rstrip('\n').rstrip('\r') for line in open('replacement_word_list.txt')]

        replacement_words = {}
        for replacement_word in replacement_word_list:
            replacement_words[replacement_word.split(',')[0]] = replacement_word.split(',')[1]

        new_string = []
        for word in tweet.split():
            if replacement_words.get(word, None) is not None:
                word = replacement_words[word]
            new_string.append(word)

        tweet = ' '.join(new_string)
        return tweet

    def tweet_features_bow(self, tweet):
        features = {}
        tweet = self.clean_tweet(tweet)

        #for word in open('feature_word_list.txt'):
        #    word = word.rstrip('\n').rstrip('\r')
        #    features["count({})".format(word)] = tweet.count(word)

        for word in tweet.split():
            features["{}".format(word)] = tweet.count(word)

        return features

    def tweet_features_tfidf(self, tweet):
        features = {}
        tweet = self.clean_tweet(tweet)
        for word in tweet.split():
            try:
                features["{}".format(word)] = tweet.count(word) * self.idf[word]
            except:
                features["{}".format(word)] = tweet.count(word) * math.log(self.tweet_count / 1)
        return features

    def bow(self, labeled_tweets):
        tweets_features = []
        for (tweet, category) in labeled_tweets:
            features = {}
            tweet = self.clean_tweet(tweet)
            for word in tweet.split():
                features["{}".format(word)] = tweet.count(word)
            tweets_features.append((features, category))
        return tweets_features

    def tfidf(self, labeled_tweets):
        self.df = {}
        for (tweet, category) in labeled_tweets:
            tweet = self.clean_tweet(tweet)
            word_count = collections.Counter(tweet.split())
            for word in word_count:
                try:
                    self.df[word] += 1
                except:
                    self.df[word] = 1

        self.idf = {}
        self.tweet_count = len(labeled_tweets)
        for word, doc_count in self.df.items():
            self.idf[word] = math.log(self.tweet_count / (doc_count + 1))
            
        tweets_features = []
        for (tweet, category) in labeled_tweets:
            features = {}
            tweet = self.clean_tweet(tweet)
            for word in tweet.split():
                features["{}".format(word)] = tweet.count(word) * self.idf[word]
            tweets_features.append((features, category))
        return tweets_features

    def __init__(self, labeled_tweets, weighting = None):
        self.weighting = weighting or 'bow'
        random.shuffle(labeled_tweets)
        if self.weighting is 'tfidf':
            train_set = self.tfidf(labeled_tweets)
        else:
            train_set = self.bow(labeled_tweets)
        self.data_count = len(train_set)

        start_time = time.time()
        self.svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
        self.training_time = round(time.time() - start_time, 2)

    def classify(self, tweet):
        if self.weighting is 'tfidf':
            return self.svm_classifier.classify(self.tweet_features_tfidf(tweet))
        else:
            return self.svm_classifier.classify(self.tweet_features_bow(tweet))

    def get_training_time(self):
        return self.training_time

    def get_data_count(self):
        return self.data_count
