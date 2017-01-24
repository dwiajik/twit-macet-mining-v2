import re

def clean(tweet):
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

def clean_tweets(labeled_tweets):
    return [(clean(tweet), category) for (tweet, category) in labeled_tweets]
