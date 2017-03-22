import argparse
import csv
import os
import random

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

fold = 10
n = 1000
calculations = [
    'cosine', 
    'dice', 
    'jaccard', 
    # 'manhattan', 
    'overlap', 
    # 'cosine_tfidf', 
    # 'dice_tfidf', 
    # 'manhattan_tfidf', 
    'lcs'
]

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

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    accuracies = {
        'cosine': [],
        'dice': [],
        'jaccard': [],
        # 'manhattan': [],
        'overlap': [],
        # 'cosine_tfidf': [],
        # 'dice_tfidf': [],
        # 'manhattan_tfidf': [],
        'lcs': [],
    }

    for calculation in calculations:
        for i in range(n):
            random.shuffle(different_tweets)
            random.shuffle(similar_tweets)

            tweets = different_tweets + similar_tweets

            random.shuffle(tweets)

            print()
            print('Dataset of {} tweet-pairs'.format(len(tweets)))

            results = {}

            true = 0
            false = 0
            for (similarity, category) in tweets:
                if similarity[calculation] >= threshold:
                    if category == 'similar':
                        true += 1
                    else:
                        false += 1
                else:
                    if category == 'different':
                        true += 1
                    else:
                        false += 1
            accuracies[calculation].append(true / (true + false))

    for calculation in calculations:
        accuracy = sum(accuracies[calculation]) / len(accuracies[calculation])
        cv_accuracies.append((threshold, calculation, accuracy))

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for (t, c, acc) in cv_accuracies:
        csv_writer.writerow([t, c, acc])
        print('{}\t{}\t{}'.format(t, c, acc))
            