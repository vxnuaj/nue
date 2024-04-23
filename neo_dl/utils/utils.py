import numpy as np

class OneHot:
    def __init__(self, num_classes, labels, dtype = np.float32):
        self.labels = labels
        self.num_classes = num_classes
        self.dtype = dtype

    def encode(self):
        one_hot_y = np.zeros((self.num_classes, self.labels.size))
        one_hot_y[self.labels, np.arange(self.labels.size)] = 1
        return one_hot_y.astype(self.dtype)

