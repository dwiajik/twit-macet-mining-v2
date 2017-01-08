import argparse
import os

from modules import cleaner, tokenizer
from jaccard import Jaccard

jaccard = Jaccard(0.6)

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-n', '--ngrams', type=int, default=1, help='How many n used in n-grams scheme, default "1"')
args = parser.parse_args()

tweets = [line for line in open('tweets_corpus/similarity_test.txt')]
cleaned = [cleaner.clean(tweet) for tweet in tweets]
tokens = [tokenizer.ngrams_tokenizer(tweet, args.ngrams) for tweet in cleaned]

distincts = []
duplicates = []
for tweet in tokens:
	if len(distincts) == 0:
		distincts.append(tweet)
	else:
		print(len(distincts))
		is_distinct = True
		for distinct in distincts:
			if jaccard.is_similar(tweet, distinct):
				index = jaccard.index(tweet, distinct)
				is_distinct = False
				break

		if is_distinct:
			distincts.append(tweet)
			with open(os.path.dirname(__file__) + 'distincts.txt', 'a') as f:
				f.write('[{}]\r\n'.format(','.join(tweet)))
		else:
			duplicates.append((tweet, distinct, index))
			with open(os.path.dirname(__file__) + 'duplicates.csv', 'a') as f:
				f.write('"{}","{}","{}"\r\n'.format('[{}]'.format(','.join(tweet)), '[{}]'.format(','.join(distinct)), jaccard.index(tweet, distinct)))

#for distinct in distincts:
#	with open(os.path.dirname(__file__) + 'distincts.txt', 'a') as f:
#		f.write(distinct)

#for duplicate in duplicates:
#	with open(os.path.dirname(__file__) + 'duplicates.csv', 'a') as f:
#		f.write('"{}","{}","{}"'.format(duplicate[0], duplicate[1], duplicate[2]))