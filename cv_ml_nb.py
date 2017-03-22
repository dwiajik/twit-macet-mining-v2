import csv
import os
import random
import time

import nltk
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

fold = 10
n = 1000

cv_accuracies = []
cv_times = []

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/4095-pair-dataset-different.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    next(dataset)
    different_tweets = [({
        'cosine': float(line[5]),
        'dice': float(line[6]),
        'jaccard': float(line[7]),
        # 'manhattan': float(line[8]),
        'overlap': float(line[9]),
        # 'cosine_tfidf': float(line[10]),
        # 'dice_tfidf': float(line[11]),
        # 'manhattan_tfidf': float(line[12]),
        'lcs': float(line[13]),
        }, line[4]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/4095-pair-dataset-similar.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    next(dataset)
    similar_tweets = [({
        'cosine': float(line[5]),
        'dice': float(line[6]),
        'jaccard': float(line[7]),
        # 'manhattan': float(line[8]),
        'overlap': float(line[9]),
        # 'cosine_tfidf': float(line[10]),
        # 'dice_tfidf': float(line[11]),
        # 'manhattan_tfidf': float(line[12]),
        'lcs': float(line[13]),
        }, line[4]) for line in dataset]

for i in range(n):
    random.shuffle(different_tweets)
    random.shuffle(similar_tweets)

    tweets = different_tweets[:600] + similar_tweets[:600]

    random.shuffle(tweets)

    print()
    print('Dataset of {} tweet-pairs'.format(len(tweets)))

    accuracies = []
    times = []

    for i in range(fold):
        train_set = [features for features in tweets[0 : i * int(len(tweets) / fold)]] + \
            [features for features in tweets[(i + 1) * int(len(tweets) / fold) : len(tweets)]]
        test_set = [features for features in tweets[i * int(len(tweets) / fold) : (i + 1) * int(len(tweets) / fold)]]

        # print('\nIteration', (i + 1))
        # print('Training data:', len(train_set), 'data')
        # print('Test data:', len(test_set), 'data')

        start_time = time.time()
        # classifier = nltk.classify.SklearnClassifier(BernoulliNB()).train(train_set)
        classifier = nltk.classify.SklearnClassifier(DecisionTreeClassifier()).train(train_set)
        # classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
        time_elapsed = round(time.time() - start_time, 2)
        accuracy = nltk.classify.accuracy(classifier, test_set)
        accuracies.append(accuracy)
        times.append(time_elapsed)

        # print('Accuracy: {}'.format(accuracy))
        # print('Time elapsed: {}'.format(time_elapsed))

    average_accuracy = sum(accuracies) / len(accuracies)
    average_time = sum(times) / len(times)
    cv_accuracies.append(average_accuracy)
    cv_times.append(average_time)

    print('Average accuracy: {}'.format(average_accuracy))
    print('Average time elapsed: {}'.format(average_time))

print()
print('Summary of {} iterations'.format(n))
print('Average accuracy: {}'.format(sum(cv_accuracies) / len(cv_accuracies)))
print('Average time elapsed: {}'.format(sum(cv_times) / len(cv_times)))