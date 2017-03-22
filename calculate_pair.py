import argparse
import csv
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import os

from modules import cleaner, tokenizer
from modules import similarity#, similarity_tfidf, tfidf

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

sm = SequenceMatcher(lambda x: x == " ")

ngrams = 1

progress = 0

results = []

with open(os.path.join(os.path.dirname(__file__), 'aaaaa.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2], line[3]) for line in dataset]

cleaned_tweets = [(time, tweet, cleaner.clean(tweet), time2, tweet2, cleaner.clean(tweet2)) for (time, tweet, time2, tweet2) in tweets]
tokenized_tweets = [(time, tweet, cleaned, tokenizer.ngrams_tokenizer(cleaned, ngrams), time2, tweet2, cleaned2, tokenizer.ngrams_tokenizer(cleaned2, ngrams)) for (time, tweet, cleaned, time2, tweet2, cleaned2) in cleaned_tweets]

# tfidf_input = 
# tfidf_obj = tfidf.TFIDF(cleaned_tweets)

for (time, tweet, cleaned, tokens, time2, tweet2, cleaned2, tokens2) in tokenized_tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(tokenized_tweets)), end='')

    sm.set_seqs(cleaned, cleaned2)

    result = [
        time,
        tweet,
        time2,
        tweet2,
        similarity.Cosine().index(tokens, tokens2),
        similarity.Dice().index(tokens, tokens2),
        similarity.Jaccard().index(tokens, tokens2),
        similarity.Manhattan().index(tokens, tokens2),
        similarity.Overlap().index(tokens, tokens2),
        '-',# similarity_tfidf.Cosine().index(cleaned, cleaned2, tfidf_obj),
        '-',# similarity_tfidf.Dice().index(cleaned, cleaned2, tfidf_obj),
        '-',# similarity_tfidf.Manhattan().index(cleaned, cleaned2, tfidf_obj),
        sm.ratio(),
    ]

    results.append(result)

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        csv_writer.writerow(result)
