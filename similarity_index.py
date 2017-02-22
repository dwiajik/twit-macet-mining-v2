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
    },
    {
        'name': 'dice',
        'calculation': Dice(),
        'ngrams': 7,
    },
    {
        'name': 'jaccard',
        'calculation': Jaccard(),
        'ngrams': 7,
    },
    {
        'name': 'manhattan',
        'calculation': Manhattan(),
        'ngrams': 7,
    },
    {
        'name': 'overlap',
        'calculation': Overlap(),
        'ngrams': 4,
    }
]

hours = 48

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

results = []

cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
# tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

progress = 0
written = 0

for (time, tweet, category, cleaned_tweets) in cleaned:
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
                    cal_res.append(index)
                results.append([time, tweet, time2, tweet2] + cal_res)
                written += 1

    progress += 1
    print('\r{}/{}, written: {}'.format(progress, len(cleaned), written), end='')

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    # csv_writer.writerow([calculation, 'limit hours', ngrams, 'threshold', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f-score', 'time elapsed'])
    for result in results:
        csv_writer.writerow(result)
