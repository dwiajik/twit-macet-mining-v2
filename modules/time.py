import re

def extract_time(text):
    time_regex = re.compile('[0-9]+(\.|:)[0-9]+')
    try:
        return time_regex.search(text).group()
    except AttributeError as e:
        return ''

# buggy
def split_time(text):
    time = extract_time(text)
    separator_regex = re.compile('\.|:')
    separator = separator_regex.search(time).group()
    splitted_time = time.split(separator)
    return int(splitted_time[0]), int(splitted_time[1])

# buggy
def extract_hour(text):
    return split_time(text)[0]

# buggy
def extract_minutes(text):
    return split_time(text)[1]

def is_text_similar(text1, text2):
    return extract_time(text1) == extract_time(text2)

# buggy
def is_time_similar(text1, text2):
    return split_time(text1) == split_time(text2)