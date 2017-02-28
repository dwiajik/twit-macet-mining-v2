import collections
import math

class TFIDF:
    def calculate(self, tweet):
        features = {}
        for word in tweet.split():
            try:
                features["{}".format(word)] = tweet.count(word) * self.idf[word]
            except:
                features["{}".format(word)] = tweet.count(word) * math.log(self.tweet_count / 1)
        return features

    def __init__(self, labeled_tweets):
        self.df = {}
        for (time, tweet, category, cleaned) in labeled_tweets:
            word_count = collections.Counter(cleaned.split())
            for word in word_count:
                try:
                    self.df[word] += 1
                except:
                    self.df[word] = 1

        self.idf = {}
        self.tweet_count = len(labeled_tweets)
        for word, doc_count in self.df.items():
            self.idf[word] = math.log(self.tweet_count / (doc_count + 1))
            
        # tweets_features = []
        # for (time, tweet, category, cleaned) in labeled_tweets:
        #     features = {}
        #     for word in cleaned.split():
        #         features["{}".format(word)] = cleaned.count(word) * self.idf[word]
        #     tweets_features.append((features, category))
        # return tweets_features