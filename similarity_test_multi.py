import argparse
import csv
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy
import os
import time as tm

from modules import cleaner, tokenizer
from modules.jaccard import Jaccard
from modules.cosine import Cosine

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
#parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
#parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Threshold index, default: 0.6')
parser.add_argument('-a', '--algo', type=str, default='jaccard', help='Algorithm: jaccard, cosine')
args = parser.parse_args()

if args.algo == 'jaccard':
    algo = Jaccard()
elif args.algo == 'cosine':
    algo = Cosine()
else:
    raise Exception('Algo not defined')

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

def calculate(hours):
    results = []
    for ngrams in range(1, 7): # 1-6
        for threshold in numpy.arange(0.1, 1.1, 0.1): # 0.1-1.0
            start_time = tm.time()

            cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
            tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

            distincts = []
            # progress = 0
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

                        index = algo.index(tokens, distinct_tokens)
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

                # progress += 1
                # print('\r{}/{}'.format(progress, len(tokenized)), end='')

            time_elapsed = tm.time() - start_time
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fscore = 2 * (precision * recall) / (precision + recall)

            print()
            print('Limit hours: {}'.format(hours))
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

            results.append([hours, ngrams, threshold, tp, tn, fp, fn, accuracy, precision, recall, fscore, time_elapsed])
    return results

p = Pool(4)
pool_results = p.map(calculate, range(12, 49, 12))

with open(os.path.join(os.path.dirname(__file__), args.output), 'w', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(['limit hours', 'ngrams', 'threshold', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f-score', 'time elapsed'])
    for pool_result in pool_results:
        for result in pool_result:
            csv_writer.writerow(result)
        
