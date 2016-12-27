from . import tokenizer

class TfWeighting:
    def __init__(self, labeled_tweets):
        self.tweets_features = []
        for (tweet, category) in labeled_tweets:
            features = {}
            for token in tweet:
                features["{}".format(token)] = tweet.count(token)
            self.tweets_features.append((features, category))

    def get_features(self):
        return self.tweets_features

class TfIdfWeighting:
    def __init__(self, labeled_tweets):
        pass
