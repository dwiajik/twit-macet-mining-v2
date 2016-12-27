import argparse
import random

from modules import cleaner, tokenizer
from modules.weighting import TfWeighting, TfIdfWeighting
from modules.classifier import SvmClassifier

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
parser.add_argument('-w', '--weight', dest='weighting', default='tf', choices=['tf', 'tfidf'], help='Weighting scheme: term frequency or term frequency-inverse document frequency, default "tf"')
parser.add_argument('-d', '--data', default='all', choices=['all', 'under', 'over'], help='Use all data, undersampled data, or oversampled data, default "all"')
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

cleaned_tweets = cleaner.clean_tweets(labeled_tweets)
print(len(cleaned_tweets))
print(cleaned_tweets[:3])

tokenized_tweets = tokenizer.tokenize_tweets(cleaned_tweets, args.ngrams)
print(len(tokenized_tweets))
print(tokenized_tweets[:3])

if args.weighting == 'tf':
    weighter = TfWeighting(tokenized_tweets)
    print(weighter.get_features()[:3])
elif args.weighting == 'tfidf':
    pass
