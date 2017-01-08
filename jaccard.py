class Jaccard:
    def __init__(self, threshold = 0.6):
        self.threshold = threshold

    def unique(self, a):
        return list(set(a))

    def intersect(self, a, b):
        return list(set(a) & set(b))

    def union(self, a, b):
        return list(set(a) | set(b))

    def is_similar(self, a = [], b = []):
        try:
            jaccard_index = len(self.intersect(a, b)) / len(self.union(a, b))
        except ZeroDivisionError as e:
            return True
        return jaccard_index > self.threshold

    def index(self, a = [], b = []):
        try:
            return len(self.intersect(a, b)) / len(self.union(a, b))
        except ZeroDivisionError as e:
            return 1
