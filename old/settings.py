from os import environ

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

consumer_key = environ.get('CONSUMER_KEY')
consumer_secret = environ.get('CONSUMER_SECRET')
access_token = environ.get('ACCESS_TOKEN')
access_secret = environ.get('ACCESS_SECRET')

mysql_host = environ.get('MYSQL_HOST')
mysql_db = environ.get('MYSQL_DATABASE')
mysql_user = environ.get('MYSQL_USER')
mysql_password = environ.get('MYSQL_PASSWORD')
