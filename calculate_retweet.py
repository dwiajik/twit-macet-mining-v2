import csv
import os

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

retweet_count = 0
formal_retweet_count = 0
for time, tweet, category in tweets:
    if category == 'retweet':
        retweet_count += 1
        if tweet[:4] == 'RT @':
            formal_retweet_count += 1

print('Formal retweet count: {}/{}'.format(formal_retweet_count, retweet_count))