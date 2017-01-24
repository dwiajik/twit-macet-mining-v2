import argparse
import os

from modules import cleaner, tokenizer
from jaccard import Jaccard

jaccard = Jaccard(0.6)

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

tweets = [line for line in open('tweets_corpus/similarity-dataset15076.txt')]
cleaned = [(tweet.strip('\n'), cleaner.clean(tweet)) for tweet in tweets]
tokenized = [(tweet, tokenizer.ngrams_tokenizer(cleaned_tweets, args.ngrams)) for (tweet, cleaned_tweets) in cleaned]

distincts = []
# duplicates = []
for (tweet, tokens) in tokenized:
	if len(distincts) == 0:
		distincts.append((tweet, tokens))
		with open(os.path.dirname(__file__) + args.output, 'a') as f:
			f.write('"{}","{}"\n'.format(tweet, '[{}]'.format(','.join(tokens))))
	else:
		print(len(distincts))
		is_distinct = True
		for (distinct_tweet, distinct_tokens) in distincts:
			if jaccard.is_similar(tokens, distinct_tokens):
				index = jaccard.index(tokens, distinct_tokens)
				is_distinct = False
				break

		if is_distinct:
			distincts.append((tweet, tokens))
			with open(os.path.dirname(__file__) + args.output, 'a') as f:
				f.write('"{}","{}"\n'.format(tweet, '[{}]'.format(','.join(tokens))))
		else:
			# duplicates.append((tokens, distinct, index))
			with open(os.path.dirname(__file__) + args.output, 'a') as f:
				f.write('"{}","{}","{}","{}","{}"\n'.format(
					tweet,
					'[{}]'.format(','.join(tokens)),
					distinct_tweet,
					'[{}]'.format(','.join(distinct_tokens)),
					index))
