import math

class Cosine:
    def __init__(self, threshold = 0.6):
        self.threshold = threshold

    def vector(self, a, b):
        vec_a, vec_b = {}, {}
        for token in set(a + b):
            vec_a[token], vec_b[token] = 0, 0
        for token in a:
            vec_a[token] += 1
        for token in b:
            vec_b[token] += 1
        return list(set(a + b)), vec_a, vec_b

    def index(self, a, b):
        tokens, vec_a, vec_b = self.vector(a, b)
        # sum of product
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        # square root of sum of a square
        sqrt_soas = math.sqrt(sum([value * value for attr, value in vec_a.items()]))
        # square root of sum of b square
        sqrt_sobs = math.sqrt(sum([value * value for attr, value in vec_b.items()]))
        try:
            return sop / (sqrt_soas * sqrt_sobs)
        except ZeroDivisionError as e:
            return 0

    def is_similar(self, a = [], b = []):
        return self.index(a, b) >= self.threshold
