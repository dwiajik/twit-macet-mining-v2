import argparse
import csv
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy
import os
import time as tm

from modules import cleaner, tokenizer
from modules.similarity import *

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

calculations  = {
    'jaccard': Jaccard(),
    'cosine': Cosine(),
    'weighted_jaccard': WeightedJaccard(),
    'extended_jaccard': ExtendedJaccard(),
    'dice': Dice(),
    'manhattan': Manhattan(),
    'euclidean': Euclidean(),
    'overlap': Overlap(),
    'pearson': Pearson()
}

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

def calculate(hours):
    results = []
    for name, calculation in calculations.items():
        for ngrams in range(1, 7): # 1-6
            for threshold in numpy.arange(0.1, 1.1, 0.1): # 0.1-1.0
                start_time = tm.time()

                cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
                tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

                distincts = []
                tp, tn, fp, fn = 0, 0, 0, 0

                for (time, tweet, category, tokens) in tokenized:
                    if len(distincts) == 0:
                        distincts.append((time, tweet, tokens))
                    else:
                        is_distinct = True
                        for (distinct_time, distinct_tweet, distinct_tokens) in distincts:
                            dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                            distinct_dt = datetime.strptime(distinct_time, '%Y-%m-%d %H:%M:%S')
                            time_diff = dt - distinct_dt

                            if time_diff > timedelta(hours=hours):
                                distincts.remove((distinct_time, distinct_tweet, distinct_tokens))
                                continue

                            index = calculation.index(tokens, distinct_tokens)
                            if index >= threshold:
                                is_distinct = False
                                break

                        if is_distinct:
                            distincts.append((time, tweet, tokens))

                            if category == 'new':
                                tp += 1
                            else:
                                fp += 1
                        else:
                            if category == 'new':
                                fn += 1
                            else:
                                tn += 1

                time_elapsed = tm.time() - start_time
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                fscore = 2 * (precision * recall) / (precision + recall)

                print()
                print('Limit hours: {}'.format(hours))
                print('Calculation: {}'.format(name))
                print('Ngrams: {}'.format(ngrams))
                print('Threshold: {}'.format(threshold))
                print('True positive: {}'.format(tp))
                print('True negative: {}'.format(tn))
                print('False positive: {}'.format(fp))
                print('False negative: {}'.format(fn))
                print('Accuracy: {}'.format(accuracy))
                print('Precison: {}'.format(precision))
                print('Recall: {}'.format(recall))
                print('F-score: {}'.format(fscore))
                print('Time elapsed: {}'.format(time_elapsed))

                results.append([name, hours, ngrams, threshold, tp, tn, fp, fn, accuracy, precision, recall, fscore, time_elapsed])
    return results

p = Pool(8)
pool_results = p.map(calculate, range(6, 49, 6))

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(['calculation', 'limit hours', 'ngrams', 'threshold', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f-score', 'time elapsed'])
    for pool_result in pool_results:
        for result in pool_result:
            csv_writer.writerow(result)
        
