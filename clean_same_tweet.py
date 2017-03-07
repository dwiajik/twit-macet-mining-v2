import argparse
import csv
import os

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-i', '--input', default='tweets_corpus/tweet-2016-07-06.csv', help='File name for input CSV, e.g. input.csv')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

progress = 0

results = []

with open(os.path.join(os.path.dirname(__file__), args.input), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    tweets = [(line[0], line[1], line[2]) for line in dataset]

for (time, tweet, category) in tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(tweets)), end='')

    new = True

    for [time2, tweet2, category2] in results:
        if tweet == tweet2:
            new = False
            break

    if new:
        results.append([time, tweet, category])

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        csv_writer.writerow(result)
