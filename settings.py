from os import environ

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

consumer_key = environ.get('CONSUMER_KEY')
consumer_secret = environ.get('CONSUMER_SECRET')
access_token = environ.get('ACCESS_TOKEN')
access_secret = environ.get('ACCESS_SECRET')
