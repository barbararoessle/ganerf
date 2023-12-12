class MeanTracker(object):
    def __init__(self):
        self.reset()

    def add(self, input, weight=1.0):
        for key, l in input.items():
            if not key in self.mean_dict:
                self.mean_dict[key] = 0
            self.mean_dict[key] = (
                self.mean_dict[key] * self.total_weight + (l if isinstance(l, float) else l.item()) * weight
            ) / (self.total_weight + weight)
        self.total_weight += weight

    def has(self, key):
        return key in self.mean_dict

    def get(self, key):
        return self.mean_dict[key]

    def as_dict(self):
        return self.mean_dict

    def reset(self):
        self.mean_dict = dict()
        self.total_weight = 0