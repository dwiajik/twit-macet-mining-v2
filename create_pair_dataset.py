import argparse
import csv
from datetime import datetime, timedelta
import os

from modules import cleaner, tokenizer
from modules.similarity import *

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

calculation = Cosine()
ngrams = 5
threshold = 0.6

progress = 0

results = []

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/pair_dataset_4699.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

for (time, tweet, category, tokens) in tokenized:
    progress += 1
    print('\r{}/{}'.format(progress, len(cleaned)), end='')

    result = []
    highest_score = 0
    index = 0
    highest_index = -1

    for (time2, tweet2, category2, tokens2) in tokenized:
        dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        dt2 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')

        if category2 == 'new' and dt > dt2:
            # time_diff = dt - dt2
            score = calculation.index(tokens, tokens2)
            result.append([time, tweet, time2, tweet2, score])
            if score >= threshold and score > highest_score:
                highest_score = score
                highest_index = index
            index += 1

    if highest_index >= 0:
        result[highest_index].append('similar')

    results.append(result)

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        for r in result:
            csv_writer.writerow(r)
