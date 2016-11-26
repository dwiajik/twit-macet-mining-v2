import datetime
import json
import os
import random
import re
import sys
import time

import nltk
import nltk.classify
from sklearn.svm import LinearSVC
from nltk.tag import tnt

import mysql.connector
from mysql.connector import Error

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import settings

os.environ['TZ'] = 'Asia/Jakarta'

auth = OAuthHandler(settings.consumer_key, settings.consumer_secret)
auth.set_access_token(settings.access_token, settings.access_secret)


class Classifier:
    def clean_tweet(self, tweet):
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

        for word in open('feature_word_list.txt'):
            word = word.rstrip('\n').rstrip('\r')
            features["count({})".format(word)] = tweet.count(word)

        return features

    def __init__(self):
        labeled_tweets = (
            [(line, 'traffic') for line in open('tweets_corpus/traffic_tweets_combined.txt')] +
            [(line, 'non_traffic') for line in open('tweets_corpus/random_tweets.txt')] +
            [(line, 'non_traffic') for line in open('tweets_corpus/non_traffic_tweets.txt')]
        )
        random.shuffle(labeled_tweets)
        train_set = [(self.tweet_features(tweet), category) for (tweet, category) in labeled_tweets]
        print('Using', len(train_set), 'training data.')

        start_time = time.time()
        self.svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
        svm_time = round(time.time() - start_time, 2)
        print('SVM Classifier training time:', svm_time, 'seconds')

    def svm_classify(self, tweet):
        return self.svm_classifier.classify(self.tweet_features(tweet))


class Location:
    def clean_tweet(self, tweet):
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

    def __init__(self):
        f = open('tagged_name_list.txt', 'r')

        train_data = []
        for line in f:
            train_data.append([nltk.tag.str2tuple(t) for t in line.split()])

        self.tnt_pos_tagger = tnt.TnT()
        self.tnt_pos_tagger.train(train_data)

        grammar = r"""
          LOC: {(<PRFX><PRFX>*<B-LOC><I-LOC>*)|(<B-LOC><I-LOC>*)}
        """
        self.cp = nltk.RegexpParser(grammar)

    def find_locations(self, tweet):
        tweet = self.clean_tweet(tweet)

        tagged_chunked_tweet = self.cp.parse(self.tnt_pos_tagger.tag(nltk.word_tokenize(tweet)))

        result = ''
        for subtree in tagged_chunked_tweet.subtrees():
            if subtree.label() == 'LOC': 
                for leave in subtree.leaves():
                    result += leave[0] + ' '
                result = result[:-1]
                result += ','

        result = result[:-1]
        return result

class TwitterStreamer(StreamListener):
    def __init__(self):
        super(TwitterStreamer, self).__init__()
        self.classifier = Classifier()
        self.location = Location()
        print('\nTweets:')
        with open(os.path.dirname(__file__) + 'classified_tweets.txt', 'a') as f:
            f.write('\nTweets:')

    def on_data(self, data):
        try:
            tweet = json.loads(data)['text'].replace('\n', ' ')

            svm_result = str(self.classifier.svm_classify(tweet))

            if sys.argv[1] == "dev":
                print('| ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      '\t| ' + svm_result,
                      '\t| ' + tweet)
                with open(os.path.dirname(__file__) + 'classified_tweets.txt', 'a') as f:
                    f.write('\n| ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                            '\t| ' + svm_result +
                            '\t| ' + tweet)
                with open(os.path.dirname(__file__) + 'classified_tweets.csv', 'a') as f:
                    f.write('"' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                            '","' + svm_result +
                            '","' + tweet + '"\n')

            if svm_result == "traffic":
                ts = datetime.datetime.strftime(datetime.datetime.strptime(json.loads(data)['created_at'], 
                    '%a %b %d %H:%M:%S +0000 %Y') + datetime.timedelta(hours=7), '%Y-%m-%d %H:%M:%S')

                con = mysql.connector.connect(host=settings.mysql_host, database=settings.mysql_db, user=settings.mysql_user, password=settings.mysql_password)
                cur = con.cursor()
                add_tweet = (
                "INSERT INTO tweets(datetime, twitter_user_id, text, category, locations) VALUES(%s, %s, %s, %s, %s)")
                tweet_data = (
                    ts,
                    json.loads(data)['user']['id_str'],
                    tweet,
                    svm_result,
                    str(self.location.find_locations(tweet))
                )
                cur.execute(add_tweet, tweet_data)
                con.commit()

                cur.close()

        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


twitter_stream = Stream(auth, TwitterStreamer())
# keywords = [line.rstrip('\n') for line in open(os.path.dirname(__file__) + 'name_list.txt')]
users = ['250022672', '187397386', '1118238337', '4675666764', '128175561', '537556372', '106780531', '62327666',
         '454564576', '223476605', '201720189']
keywords = ['Yogyakarta', 'Jogjakarta', 'Jogja', 'Yogya', 'Adisutjipto', 'Adi Sutjipto', 'lalinjogja', 'RTMC_Jogja',
            'ATCS_DIY', 'jogjaupdate', 'jogja24jam', 'infojogja', 'yogyakartacity', 'jogjamedia', 'tribunjogja', 'unisifmyk', 
            'UGM', 'UII', 'UNY', 'UMY', 'lalinyk']
twitter_stream.filter(track=keywords, follow=users)
