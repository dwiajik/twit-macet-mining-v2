import re

import nltk
from nltk.tag import tnt

class Location:
    def clean_tweet(self, tweet):
        # remove "RT"
        regex = re.compile('RT\s')
        tweet = regex.sub(' ', tweet)

        tweet = tweet.lower()

        # remove non alphabetic character
        regex = re.compile('\shttp.+\s')
        tweet = regex.sub(' ', tweet)
        regex = re.compile('@[a-zA-Z0-9_]+')
        tweet = regex.sub(' ', tweet)
        regex = re.compile('RT\s')
        tweet = regex.sub(' ', tweet)
        regex = re.compile('[^a-zA-Z0-9]')
        tweet = regex.sub(' ', tweet)

        # replace abbreviations
        replacement_word_list = [line.rstrip('\n').rstrip('\r') for line in open('replacement_word_list.txt')]

        replacement_words = {}
        for replacement_word in replacement_word_list:
            replacement_words[replacement_word.split(',')[0]] = replacement_word.split(',')[1]

        new_string = []
        for word in tweet.split():
            if replacement_words.get(word, None) is not None:
                word = replacement_words[word]
            new_string.append(word)

        tweet = ' '.join(new_string)
        return tweet

    def __init__(self):
        f = open('tagged_name_list.txt', 'r')

        train_data = []
        for line in f:
            train_data.append([nltk.tag.str2tuple(t) for t in line.split()])

        self.tnt_pos_tagger = tnt.TnT()
        self.tnt_pos_tagger.train(train_data)

        grammar = r"""
          LOC: {(<PRFX><PRFX>*<B-LOC><I-LOC>*)|(<B-LOC><I-LOC>*)}
        """
        self.cp = nltk.RegexpParser(grammar)

    def find_locations(self, tweet):
        tweet = self.clean_tweet(tweet)

        tagged_chunked_tweet = self.cp.parse(self.tnt_pos_tagger.tag(nltk.word_tokenize(tweet)))

        result = ''
        for subtree in tagged_chunked_tweet.subtrees():
            if subtree.label() == 'LOC': 
                for leave in subtree.leaves():
                    result += leave[0] + ' '
                result = result[:-1]
                result += ','

        result = result[:-1]
        return result
