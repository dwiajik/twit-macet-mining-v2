class Jaccard:
    def unique(self, a):
        return list(set(a))

    def intersect(self, a, b):
        return list(set(a) & set(b))

    def union(self, a, b):
        return list(set(a) | set(b))

    def index(self, a = [], b = []):
        try:
            return len(self.intersect(a, b)) / len(self.union(a, b))
        except ZeroDivisionError as e:
            return 1
