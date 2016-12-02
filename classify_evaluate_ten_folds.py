import random
import re
import nltk
import nltk.classify
from nltk.metrics import scores
from sklearn.svm import LinearSVC

import collections
import json
import os
import sys
import time
import datetime

from svm import Classifier

def f_measure(precision, recall):
    return 2*((precision * recall) / (precision + recall))

traffic_tweets = [(line, 'traffic') for line in open('tweets_corpus/traffic_tweets_combined.txt')]
non_traffic_tweets = [(line, 'non_traffic') for line in open('tweets_corpus/random_tweets.txt')] + \
    [(line, 'non_traffic') for line in open('tweets_corpus/non_traffic_tweets.txt')]
random.shuffle(traffic_tweets)
random.shuffle(non_traffic_tweets)

if sys.argv[1] == "balance":
    traffic_tweets = traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]
    non_traffic_tweets = non_traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]

labeled_tweets = (traffic_tweets + non_traffic_tweets)
random.shuffle(labeled_tweets)

svm_times = []
svm_true_positives = []
svm_true_negatives = []
svm_false_positives = []
svm_false_negatives = []
svm_accuracies = []
svm_precisions = []
svm_recalls = []
svm_f_measures = []

print('Start analysis with total:', len(labeled_tweets), 'data')
print('Traffic tweets:', len(traffic_tweets),'data')
print('Non traffic tweets:', len(non_traffic_tweets),'data')

fold = 10

for i in range(fold):
    train_set = labeled_tweets[0:i*int(len(labeled_tweets)/fold)] + labeled_tweets[(i+1)*int(len(labeled_tweets)/fold):len(labeled_tweets)]
    test_set = labeled_tweets[i*int(len(labeled_tweets)/fold):(i+1)*int(len(labeled_tweets)/fold)]

    print('\nIteration', (i+1))
    print('Training data:', len(train_set), 'data')
    print('Test data:', len(test_set), 'data')

    # SVM
    svm_classifier = Classifier(train_set)
     
    svm_true_positive = 0
    svm_true_negative = 0
    svm_false_positive = 0
    svm_false_negative = 0
    for i, (feature, label) in enumerate(test_set):
        observed = svm_classifier.classify(feature)
        if label == 'traffic' and observed == 'traffic':
            svm_true_positive += 1
        if label == 'non_traffic' and observed == 'non_traffic':
            svm_true_negative += 1
        if label == 'traffic' and observed == 'non_traffic':
            svm_false_positive += 1
        if label == 'non_traffic' and observed == 'traffic':
            svm_false_negative += 1

    svm_time = svm_classifier.get_training_time()
    svm_accuracy = (svm_true_positive + svm_true_negative) / (svm_true_positive + svm_true_negative + svm_false_positive + svm_false_negative)
    svm_precision = svm_true_positive / (svm_true_positive + svm_false_positive)
    svm_recall = svm_true_positive / (svm_true_positive + svm_false_negative)
    svm_f_measure = f_measure(svm_precision, svm_recall)

    svm_times.append(svm_time)
    svm_true_positives.append(svm_true_positive)
    svm_true_negatives.append(svm_true_negative)
    svm_false_positives.append(svm_false_positive)
    svm_false_negatives.append(svm_false_negative)
    svm_accuracies.append(svm_accuracy)
    svm_precisions.append(svm_precision)
    svm_recalls.append(svm_recall)
    svm_f_measures.append(svm_f_measure)

    print('SVM Classifier:')
    print('\t', 'Training time:', svm_time)    
    print('\t', 'True positive:', svm_true_positive)
    print('\t', 'True negative:', svm_true_negative)
    print('\t', 'False positive:', svm_false_positive)
    print('\t', 'False negative:', svm_false_negative)
    print('\t', 'Accuracy:', svm_accuracy)
    print('\t', 'Precision:', svm_precision)
    print('\t', 'Recall:', svm_recall)
    print('\t', 'F-Measure:', svm_f_measure)

print('\nSummary SVM Classifier:')
print('\tAverage training time:', sum(svm_times) / len(svm_times))
print('\tAverage true positive:', sum(svm_true_positives) / len(svm_true_positives))
print('\tAverage true negative:', sum(svm_true_negatives) / len(svm_true_negatives))
print('\tAverage false positives:', sum(svm_false_positives) / len(svm_false_positives))
print('\tAverage false negatives:', sum(svm_false_negatives) / len(svm_false_negatives))
print('\tAverage accuracy:', sum(svm_accuracies) / len(svm_accuracies))
print('\tAverage precision:', sum(svm_precisions) / len(svm_precisions))
print('\tAverage recall:', sum(svm_recalls) / len(svm_recalls))
print('\tAverage F-Measure:', sum(svm_f_measures) / len(svm_f_measures))
