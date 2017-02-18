import csv
import os

from modules import cleaner

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

lengths = {}
cleaned_tweets = [cleaner.clean(tweet) for (time, tweet, category) in tweets]
for tweet in cleaned_tweets:
    unigrams = tweet.split()
    l = len(unigrams)
    if l >= 15:
        print(tweet)
    try:
        lengths[l] += 1
    except:
        lengths[l] = 1

total = 0
sums = 0
for length, num in lengths.items():
    sums += length * num
    total += num

avg = sums / total

print(lengths)
print('Total tweet: {}'.format(total))
print('Average: {}'.format(avg))