from math import *
from decimal import Decimal

def vector(a, b, tfidf):
    tokens, a, b = list(set(a.split() + b.split())), tfidf.calculate(a), tfidf.calculate(b)

    for token in tokens:
        try:
            a[token]
        except:
            a[token] = 0
        try:
            b[token]
        except:
            b[token] = 0
    return tokens, a, b

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

class Jaccard:
    def unique(self, a):
        return list(set(a))

    def intersect(self, a, b):
        return list(set(a) & set(b))

    def union(self, a, b):
        return list(set(a) | set(b))

    def index(self, a = [], b = [], tfidf = None):
        a, b = a.split(), b.split()
        try:
            return len(self.intersect(a, b)) / len(self.union(a, b))
        except ZeroDivisionError as e:
            return 1

class Cosine:
    def index(self, a, b, tfidf = None):
        tokens, vec_a, vec_b = vector(a, b, tfidf)
        # sum of product
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        # square root of sum of a square
        sqrt_soas = sqrt(sum([value * value for attr, value in vec_a.items()]))
        # square root of sum of b square
        sqrt_sobs = sqrt(sum([value * value for attr, value in vec_b.items()]))
        try:
            return sop / (sqrt_soas * sqrt_sobs)
        except ZeroDivisionError as e:
            return 1

class Dice:
    def index(self, a, b, tfidf = None):
        tokens, vec_a, vec_b = vector(a, b, tfidf)
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        dot_a = sum([value * value for attr, value in vec_a.items()])
        dot_b = sum([value * value for attr, value in vec_b.items()])
        try:
            return (2 * sop) / (dot_a + dot_b)
        except ZeroDivisionError as e:
            return 1

class Manhattan:
    def index(self, a, b, tfidf = None):
        tokens, vec_a, vec_b = vector(a, b, tfidf)
        sum_of_subtract = sum(list(map(lambda token: abs(vec_a[token] - vec_b[token]), tokens)))
        try:
            return 1 - (sum_of_subtract / len(tokens))
        except ZeroDivisionError as e:
            return 1

class Overlap:
    def intersect(self, a, b):
        return list(set(a) & set(b))

    def index(self, a = [], b = [], tfidf = None):
        a, b = a.split(), b.split()
        try:
            return len(self.intersect(a, b)) / min(len(a), len(b))
        except ZeroDivisionError as e:
            return 1
