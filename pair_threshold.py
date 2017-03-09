import argparse
import csv
import os
import random

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

fold = 10
calculations = ['cosine', 'dice', 'jaccard', 'manhattan', 'overlap', 'cosine_tfidf', 'dice_tfidf', 'manhattan_tfidf', 'lcs']

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/4095-pair-dataset-different.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    next(dataset)
    different_tweets = [({
        'cosine': float(line[5]),
        'dice': float(line[6]),
        'jaccard': float(line[7]),
        'manhattan': float(line[8]),
        'overlap': float(line[9]),
        'cosine_tfidf': float(line[10]),
        'dice_tfidf': float(line[11]),
        'manhattan_tfidf': float(line[12]),
        'lcs': float(line[13]),
        }, line[4]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/4095-pair-dataset-similar.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    next(dataset)
    similar_tweets = [({
        'cosine': float(line[5]),
        'dice': float(line[6]),
        'jaccard': float(line[7]),
        'manhattan': float(line[8]),
        'overlap': float(line[9]),
        'cosine_tfidf': float(line[10]),
        'dice_tfidf': float(line[11]),
        'manhattan_tfidf': float(line[12]),
        'lcs': float(line[13]),
        }, line[4]) for line in dataset]

random.shuffle(different_tweets)

tweets = different_tweets[:590] + similar_tweets * 5

random.shuffle(tweets)

print('Dataset of {} tweet-pairs'.format(len(tweets)))

results = {}

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    result = {
        'cosine': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'dice': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'jaccard': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'manhattan': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'overlap': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'cosine_tfidf': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'dice_tfidf': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'manhattan_tfidf': { 'true': 0, 'false': 0, 'accuracy': 0 },
        'lcs': { 'true': 0, 'false': 0, 'accuracy': 0 },
    }

    for (similarity, category) in tweets:
        for calculation in calculations:
            if similarity[calculation] >= threshold:
                if category == 'similar':
                    result[calculation]['true'] += 1
                else:
                    result[calculation]['false'] += 1
            else:
                if category == 'different':
                    result[calculation]['true'] += 1
                else:
                    result[calculation]['false'] += 1

    for calculation in calculations:
        result[calculation]['accuracy'] = result[calculation]['true'] / (result[calculation]['true'] + result[calculation]['false'])

    results[threshold] = result


with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for t, res in results.items():
        for c, r in res.items():
            csv_writer.writerow([t, c, r['accuracy']])
            print('{}\t{}\t{}'.format(t, c, r['accuracy']))
            