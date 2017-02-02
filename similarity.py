import argparse
from datetime import datetime, timedelta
import csv
import os

from modules import cleaner, tokenizer
# from modules.jaccard import Jaccard
from modules.cosine import Cosine

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
# parser.add_argument('-j', '--jaccard', type=float, default=0.6, help='Jaccard index, default: 0.6')
parser.add_argument('-c', '--cosine', type=float, default=0.6, help='Cosine index, default: 0.6')
args = parser.parse_args()

# jaccard = Jaccard(args.jaccard)
cosine = Cosine(args.cosine)

with open('tweets_corpus/similarity-dataset15075.csv', newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1]) for line in dataset]

cleaned = [(time, tweet, cleaner.clean(tweet)) for (time, tweet) in tweets]
tokenized = [(time, tweet, tokenizer.ngrams_tokenizer(cleaned_tweets, args.ngrams)) for (time, tweet, cleaned_tweets) in cleaned]

distincts = []
progress = 0
with open(os.path.dirname(__file__) + args.output, 'w', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for (time, tweet, tokens) in tokenized:
        progress += 1
        print('\r{}/{}'.format(progress, len(tokenized)), end='')
        if len(distincts) == 0:
            distincts.append((time, tweet, tokens))
            csv_writer.writerow([time, tweet, '[{}]'.format(','.join(tokens))])
        else:
            is_distinct = True
            for (distinct_time, distinct_tweet, distinct_tokens) in distincts:
                dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S');
                distinct_dt = datetime.strptime(distinct_time, '%Y-%m-%d %H:%M:%S');
                time_diff = dt - distinct_dt

                if time_diff > timedelta(hours=12):
                    distincts.remove((distinct_time, distinct_tweet, distinct_tokens))
                    continue

                # if jaccard.is_similar(tokens, distinct_tokens):
                #     index = jaccard.index(tokens, distinct_tokens)
                #     is_distinct = False
                #     break

                if cosine.is_similar(tokens, distinct_tokens):
                    index = cosine.index(tokens, distinct_tokens)
                    is_distinct = False
                    break

            if is_distinct:
                distincts.append((time, tweet, tokens))
                csv_writer.writerow([time, tweet, '[{}]'.format(','.join(tokens))])
            else:
                csv_writer.writerow([
                    time,
                    tweet,
                    '[{}]'.format(','.join(tokens)),
                    distinct_time,
                    distinct_tweet,
                    '[{}]'.format(','.join(distinct_tokens)),
                    index])
