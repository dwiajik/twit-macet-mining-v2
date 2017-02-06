import argparse
from datetime import datetime, timedelta
import csv
import os

from modules import cleaner, tokenizer
from modules.jaccard import Jaccard
from modules.cosine import Cosine

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Threshold index, default: 0.6')
parser.add_argument('-a', '--algo', type=str, default='jaccard', help='Algorithm: jaccard, cosine')
args = parser.parse_args()

if args.algo == 'jaccard':
    algo = Jaccard()
elif args.algo == 'cosine':
    algo = Cosine()
else:
    raise Exception('Algo not defined')

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15043.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
tokenized = [(time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, args.ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

distincts = []
progress = 0
tp, tn, fp, fn = 0, 0, 0, 0

with open(os.path.join(os.path.dirname(__file__), args.output), 'w', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for (time, tweet, category, tokens) in tokenized:
        if len(distincts) == 0:
            distincts.append((time, tweet, tokens))
            csv_writer.writerow([
                'new',
                category,
                time,
                tweet,
                '[{}]'.format(','.join(tokens))
            ])
        else:
            is_distinct = True
            for (distinct_time, distinct_tweet, distinct_tokens) in distincts:
                dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                distinct_dt = datetime.strptime(distinct_time, '%Y-%m-%d %H:%M:%S')
                time_diff = dt - distinct_dt

                if time_diff > timedelta(hours=12):
                    distincts.remove((distinct_time, distinct_tweet, distinct_tokens))
                    continue

                # This part is too sophisticated
                index = algo.index(tokens, distinct_tokens)
                if index >= args.threshold:
                    is_distinct = False
                    break

            if is_distinct:
                distincts.append((time, tweet, tokens))
                csv_writer.writerow([
                    'new' if is_distinct else 'retweet',
                    category,
                    time,
                    tweet,
                    '[{}]'.format(','.join(tokens))
                ])

                if category == 'new':
                    tp += 1
                else:
                    fp += 1
            else:
                csv_writer.writerow([
                    'new' if is_distinct else 'retweet',
                    category,
                    time,
                    tweet,
                    '[{}]'.format(','.join(tokens)),
                    distinct_time,
                    distinct_tweet,
                    '[{}]'.format(','.join(distinct_tokens)),
                    index
                ])

                if category == 'new':
                    fn += 1
                else:
                    tn += 1

        progress += 1
        print('\r{}/{}'.format(progress, len(tokenized)), end='')

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print()
print('True positive: {}'.format(tp))
print('True negative: {}'.format(tn))
print('False positive: {}'.format(fp))
print('False negative: {}'.format(fn))
print('Accuracy: {}'.format(accuracy))
print('Precison: {}'.format(precision))
print('Recall: {}'.format(recall))
