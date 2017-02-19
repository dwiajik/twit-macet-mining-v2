import csv
from datetime import datetime, timedelta
import os
import time as tm

from modules import cleaner, tokenizer, time as t
from modules.location import Location
from modules.similarity import *

# constant values
hours = 12
calculation = Overlap()
threshold = 0.7
ngrams = 4
l = Location()

start_time = tm.time()
progress = 0

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/similarity_dataset_15028.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

cleaned = [(time, tweet, category, cleaner.clean(tweet)) for (time, tweet, category) in tweets]
tokenized = [(
    time, tweet, category, tokenizer.ngrams_tokenizer(cleaned_tweets, ngrams)) for (time, tweet, category, cleaned_tweets) in cleaned]

distincts = []
tp, tn, fp, fn = 0, 0, 0, 0

for (time, tweet, category, tokens) in tokenized:
    progress += 1
    print('\r{}/{}'.format(progress, len(tokenized)), end='')

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

            index = calculation.index(tokens, distinct_tokens)
            if index >= threshold or (t.is_text_similar(tweet, distinct_tweet) and l.is_first_loc_similar(tweet, distinct_tweet)):
                is_distinct = False
                break
                
        if is_distinct:
            distincts.append((time, tweet, tokens))

            if category == 'new':
                tp += 1
            else:
                # print(tweet)
                fp += 1
        else:
            if category == 'new':
                # print(tweet)
                fn += 1
            else:
                tn += 1

time_elapsed = tm.time() - start_time
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = 2 * (precision * recall) / (precision + recall)

print()
print('Limit hours: {}'.format(hours))
# print('Calculation: {}'.format(name))
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