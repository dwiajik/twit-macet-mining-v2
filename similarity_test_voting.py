import argparse
import csv
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy
import os

from modules import cleaner, tokenizer
from modules.similarity import *

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

calculations  = [
    {
        'name': 'cosine',
        'calculation': Cosine(),
        'ngrams': 5,
        'threshold': 0.6,
    },
    {
        'name': 'dice',
        'calculation': Dice(),
        'ngrams': 7,
        'threshold': 0.4,
    },
    {
        'name': 'jaccard',
        'calculation': Jaccard(),
        'ngrams': 7,
        'threshold': 0.3,
    },
    {
        'name': 'manhattan',
        'calculation': Manhattan(),
        'ngrams': 7,
        'threshold': 0.2,
    },
    {
        'name': 'overlap',
        'calculation': Overlap(),
        'ngrams': 4,
        'threshold': 0.7,
    }
]

hours = 12

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

# results = []

cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
# tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

tp, tn, fp, fn = 0, 0, 0, 0
progress = 0

for (time, tweet, category, cleaned_tweets) in cleaned:
    is_retweet = False
    for (time2, tweet2, category2, cleaned_tweets2) in cleaned:
        dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        dt2 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')

        if category2 == 'new' and dt > dt2:
            time_diff = dt - dt2
            if time_diff <= timedelta(hours=hours):
                cal_res = []
                for cal_obj in calculations:
                    tweet_tokens = tokenizer.ngrams_tokenizer(cleaned_tweets, cal_obj['ngrams'])
                    tweet2_tokens = tokenizer.ngrams_tokenizer(cleaned_tweets2, cal_obj['ngrams'])
                    index = cal_obj['calculation'].index(tweet_tokens, tweet2_tokens)
                    cal_res.append(index >= cal_obj['threshold'])

                if cal_res.count(True) > len(calculations) / 2:
                    is_retweet = True

        if is_retweet:
            break

    if is_retweet:
        distincts.append((time, tweet, tokens))

        if category == 'retweet':
            tn += 1
        else:
            fn += 1
    else:
        if category == 'retweet':
            fp += 1
        else:
            tp += 1
                # results.append([time, tweet, time2, tweet2] + cal_res)

    progress += 1
    print('\r{}/{}'.format(progress, len(cleaned)), end='')

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = 2 * (precision * recall) / (precision + recall)

print()
print('Limit hours: {}'.format(hours))
print('True positive: {}'.format(tp))
print('True negative: {}'.format(tn))
print('False positive: {}'.format(fp))
print('False negative: {}'.format(fn))
print('Accuracy: {}'.format(accuracy))
print('Precison: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F-score: {}'.format(fscore))

# with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
#     csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
#     # csv_writer.writerow([calculation, 'limit hours', ngrams, 'threshold', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f-score', 'time elapsed'])
#     for result in results:
#         csv_writer.writerow(result)
