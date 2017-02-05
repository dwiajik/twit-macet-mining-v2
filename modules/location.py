import os
import re

import nltk
from nltk.tag import tnt

from modules import cleaner

class Location:
    def __init__(self):
        train_data = []

        with open(os.path.join(os.path.dirname(__file__), 'tagged_locations.txt'), 'r') as f:
            for line in f:
                train_data.append([nltk.tag.str2tuple(t) for t in line.split()])

        self.tnt_pos_tagger = tnt.TnT()
        self.tnt_pos_tagger.train(train_data)

        grammar = r"""
          LOC: {(<PRFX><PRFX>*<B-LOC><I-LOC>*)|(<B-LOC><I-LOC>*)}
        """
        self.cp = nltk.RegexpParser(grammar)

    def get_locations(self, tweet):
        tweet = cleaner.clean(tweet)
        tagged_chunked_tweet = self.cp.parse(self.tnt_pos_tagger.tag(nltk.word_tokenize(tweet)))

        locations = []
        for subtree in tagged_chunked_tweet.subtrees():
            if subtree.label() == 'LOC':
                location = []
                for leave in subtree.leaves():
                    location.append(leave[0])
                locations.append(' '.join(location))

        return locations

    def is_first_loc_similar(self, text1, text2):
        try:
            loc1 = self.get_locations(text1)[0]
        except IndexError as e:
            loc1 = ''

        try:
            loc2 = self.get_locations(text2)[0]
        except IndexError as e:
            loc2 = ''

        return loc1 == loc2