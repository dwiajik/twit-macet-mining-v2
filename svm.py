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

    def tweet_features(self, tweet):
        features = {}
        tweet = self.clean_tweet(tweet)

        #for word in open('feature_word_list.txt'):
        #    word = word.rstrip('\n').rstrip('\r')
        #    features["count({})".format(word)] = tweet.count(word)

        for word in tweet.split():
            features["{}".format(word)] = tweet.count(word)

        return features

    def __init__(self, labeled_tweets):
        random.shuffle(labeled_tweets)
        train_set = [(self.tweet_features(tweet), category) for (tweet, category) in labeled_tweets]
        self.data_count = len(train_set)

        start_time = time.time()
        self.svm_classifier = nltk.classify.SklearnClassifier(LinearSVC(max_iter=10000)).train(train_set)
        self.training_time = round(time.time() - start_time, 2)

    def classify(self, tweet):
        return self.svm_classifier.classify(self.tweet_features(tweet))

    def get_training_time(self):
        return self.training_time

    def get_data_count(self):
        return self.data_count
