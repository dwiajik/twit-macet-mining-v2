import time

import nltk
from sklearn.svm import LinearSVC

class SvmClassifier:
    def __init__(self, training_set):
        start_time = time.time()
        self.svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(training_set)
        self.training_time = round(time.time() - start_time, 2)

    def classify(self, tweet):
        return self.svm_classifier.classify(tweet)

    def get_training_time(self):
        return self.training_time
