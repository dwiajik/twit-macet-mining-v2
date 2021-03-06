import argparse
import csv
from datetime import datetime, timedelta
from multiprocessing import Pool
import os
import time as tm

from modules import cleaner, tokenizer
from modules.similarity_tfidf import *
from modules.tfidf import TFIDF

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

calculations  = [
    ('jaccard', Jaccard()),
    ('cosine', Cosine()),
    # ('weighted_jaccard', WeightedJaccard()),
    # ('extended_jaccard', ExtendedJaccard()),
    ('dice', Dice()),
    ('manhattan', Manhattan()),
    # ('euclidean', Euclidean()),
    ('overlap', Overlap()),
    # ('pearson', Pearson()),
    # ('combination', Combination())
]

hours = 12

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

cleaned_tweets = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
tfidf = TFIDF(cleaned_tweets)

def calculate(calculation):
    results = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: # 0.1-1.0
        start_time = tm.time()

        distincts = []
        tp, tn, fp, fn = 0, 0, 0, 0

        for (time, tweet, category, cleaned) in cleaned_tweets:
            if len(distincts) == 0:
                distincts.append((time, tweet, cleaned))
            else:
                is_distinct = True
                for (distinct_time, distinct_tweet, distinct_cleaned) in distincts:
                    dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                    distinct_dt = datetime.strptime(distinct_time, '%Y-%m-%d %H:%M:%S')
                    time_diff = dt - distinct_dt

                    if time_diff > timedelta(hours=hours):
                        distincts.remove((distinct_time, distinct_tweet, distinct_cleaned))
                        continue

                    index = calculation[1].index(cleaned, distinct_cleaned, tfidf)
                    if index >= threshold:
                        is_distinct = False
                        break

                if is_distinct:
                    distincts.append((time, tweet, cleaned))

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
        print('Calculation: {}'.format(calculation[0]))
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

        results.append([calculation[0], hours, threshold, tp, tn, fp, fn, accuracy, precision, recall, fscore, time_elapsed])
    return results

p = Pool(5)
pool_results = p.map(calculate, calculations)

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(['calculation', 'limit hours', 'threshold', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f-score', 'time elapsed'])
    for pool_result in pool_results:
        for result in pool_result:
            csv_writer.writerow(result)
        
