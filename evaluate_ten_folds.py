import argparse
import datetime
import os
import random
import sys

from modules import cleaner, tokenizer
from modules.weighting import TfWeighting, TfIdfWeighting
from modules.classifier import SvmClassifier

def f_measure(precision, recall):
    return 2*((precision * recall) / (precision + recall))

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
parser.add_argument('-w', '--weight', dest='weighting', default='tf', choices=['tf', 'tfidf'], help='Weighting scheme: term frequency or term frequency-inverse document frequency, default "tf"')
parser.add_argument('-d', '--data', default='all', choices=['all', 'under', 'over'], help='Use all data, undersampled data, or oversampled data, default "all"')
parser.add_argument('-o', '--output', default='evaluation.csv', help='File name for output CSV, e.g. evaluation.csv')
args = parser.parse_args()

print('Ngrams: {}'.format(args.ngrams))
print('Weighting scheme: {}'.format(args.weighting))
print('Data imbalance handle: {}\n'.format(args.data))

traffic_tweets = [(line, 'traffic') for line in open('tweets_corpus/traffic_tweets_combined.txt')]
non_traffic_tweets = [(line, 'non_traffic') for line in open('tweets_corpus/random_tweets.txt')] + \
    [(line, 'non_traffic') for line in open('tweets_corpus/non_traffic_tweets.txt')]
random.shuffle(traffic_tweets)
random.shuffle(non_traffic_tweets)

if args.data == 'all':
    pass
elif args.data == 'under':
    traffic_tweets = traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]
    non_traffic_tweets = non_traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]
elif args.data == 'over':
    mul = int(max([len(traffic_tweets), len(non_traffic_tweets)]) / min([len(traffic_tweets), len(non_traffic_tweets)]))
    if len(traffic_tweets) > len(non_traffic_tweets):
        non_traffic_tweets = non_traffic_tweets * (mul + 1)
        non_traffic_tweets = non_traffic_tweets[:len(traffic_tweets)]
    else:
        traffic_tweets = traffic_tweets * (mul + 1)
        traffic_tweets = traffic_tweets[:len(non_traffic_tweets)]

labeled_tweets = (traffic_tweets + non_traffic_tweets)
random.shuffle(labeled_tweets)

print('Start analysis with total:', len(labeled_tweets), 'data')
print('Traffic tweets:', len(traffic_tweets),'data')
print('Non traffic tweets:', len(non_traffic_tweets),'data')

cleaned_tweets = cleaner.clean_tweets(labeled_tweets)
print(cleaned_tweets[:3])

tokenized_tweets = tokenizer.tokenize_tweets(cleaned_tweets, args.ngrams)
print(len(tokenized_tweets))
print(tokenized_tweets[:3])

if args.weighting == 'tf':
    weighter = TfWeighting(tokenized_tweets)
elif args.weighting == 'tfidf':
    weighter = TfIdfWeighting(tokenized_tweets)

dataset = weighter.get_features()
print(dataset[:3])
# svm_model = SvmClassifier(dataset)

with open(os.path.dirname(__file__) + args.output, 'a') as f:
    f.write('"{}"\r\n'.format(sys.argv))
    f.write('"{}"\r\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    f.write('"Start analysis with total: {} data"\r\n'.format(len(labeled_tweets)))
    f.write('"Traffic tweets: {} data"\r\n'.format(len(traffic_tweets)))
    f.write('"Non traffic tweets: {} data"\r\n\r\n'.format(len(non_traffic_tweets)))

data_format = '"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\r\n'
with open(os.path.dirname(__file__) + args.output, 'a') as f:
    f.write(data_format.format(
        'Iteration',
        'Training time',
        'True positive',
        'True negative',
        'False positive',
        'False negative',
        'Accuracy',
        'Precision',
        'Recall',
        'F-measure'
    ))

svm_times = []
svm_true_positives = []
svm_true_negatives = []
svm_false_positives = []
svm_false_negatives = []
svm_accuracies = []
svm_precisions = []
svm_recalls = []
svm_f_measures = []

fold = 10

for i in range(fold):
    train_set = dataset[0:i*int(len(dataset)/fold)] + dataset[(i+1)*int(len(dataset)/fold):len(dataset)]
    test_set = dataset[i*int(len(dataset)/fold):(i+1)*int(len(dataset)/fold)]

    print('\nIteration', (i+1))
    print('Training data:', len(train_set), 'data')
    print('Test data:', len(test_set), 'data')

    # SVM
    svm_classifier = SvmClassifier(train_set)
     
    svm_true_positive = 0
    svm_true_negative = 0
    svm_false_positive = 0
    svm_false_negative = 0
    for index, (feature, label) in enumerate(test_set):
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

    with open(os.path.dirname(__file__) + args.output, 'a') as f:
        f.write(data_format.format(
            i + 1,
            svm_time,
            svm_true_positive,
            svm_true_negative,
            svm_false_positive,
            svm_false_negative,
            svm_accuracy,
            svm_precision,
            svm_recall,
            svm_f_measure
        ))

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


with open(os.path.dirname(__file__) + args.output, 'a') as f:
    f.write((data_format + '\r\n\r\n').format(
        'Total',
        sum(svm_times) / len(svm_times),
        sum(svm_true_positives) / len(svm_true_positives),
        sum(svm_true_negatives) / len(svm_true_negatives),
        sum(svm_false_positives) / len(svm_false_positives),
        sum(svm_false_negatives) / len(svm_false_negatives),
        sum(svm_accuracies) / len(svm_accuracies),
        sum(svm_precisions) / len(svm_precisions),
        sum(svm_recalls) / len(svm_recalls),
        sum(svm_f_measures) / len(svm_f_measures)
    ))

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
