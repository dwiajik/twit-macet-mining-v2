import datetime
import json
import os
import sys

import mysql.connector
from mysql.connector import Error

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

from svm import Classifier
from location import Location
import settings

os.environ['TZ'] = 'Asia/Jakarta'

auth = OAuthHandler(settings.consumer_key, settings.consumer_secret)
auth.set_access_token(settings.access_token, settings.access_secret)

labeled_tweets = (
    [(line, 'traffic') for line in open('tweets_corpus/traffic_tweets_combined.txt')] +
    [(line, 'non_traffic') for line in open('tweets_corpus/random_tweets.txt')] +
    [(line, 'non_traffic') for line in open('tweets_corpus/non_traffic_tweets.txt')]
)

class TwitterStreamer(StreamListener):
    def __init__(self):
        super(TwitterStreamer, self).__init__()
        self.classifier = Classifier(labeled_tweets)
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
