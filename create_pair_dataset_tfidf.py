import argparse
import csv
from datetime import datetime, timedelta
import os

from modules import cleaner, tokenizer
from modules.similarity_tfidf import *
from modules.tfidf import TFIDF

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

calculation = Manhattan()
ngrams = 1
threshold = 0.55

progress = 0

results = []

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/tweet-2016-07-06-clean.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

cleaned_tweets = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]

tfidf = TFIDF(cleaned_tweets)

# tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]
# tokenized = tokenized[:10]

for (time, tweet, category, cleaned) in cleaned_tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(cleaned_tweets)), end='')

    result = []

    for (time2, tweet2, category2, cleaned2) in cleaned_tweets:
        dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        dt2 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')

        if dt > dt2:
            # time_diff = dt - dt2
            # score = calculation.index(tokens, tokens2)
            score = calculation.index(cleaned, cleaned2, tfidf)

            if score >= threshold:
                result.append([time, tweet, time2, tweet2, score, 'similar'])
            else:
                result.append([time, tweet, time2, tweet2, score, 'different'])


    results.append(result)

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        for r in result:
            csv_writer.writerow(r)
