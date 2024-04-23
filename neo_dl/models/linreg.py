import numpy as np

class LinearRegression:
    def __init__(self, input, labels, sample_size, feature_size, alpha, epochs):
        self.input = input
        self.labels = labels
        self.sample_size = sample_size
        self.feature_size = feature_size
        self.alpha = alpha
        self.epochs = epochs
            
    def init_params(self):
        w = np.random.randn(1, self.feature_size)
        b = np.random.randn(1, 1)
        return w, b
    
    def forward(self, w, b):
        pred = np.dot(w, self.input) + b
        return pred

    def mse(self, pred):
        self.labels = np.clip(self.labels, -1e10, 1e10)
        pred = np.clip(pred, -1e10, 1e10)
        l = np.sum((self.labels - pred) ** 2) / self.sample_size
        return l

    def backward(self, pred):
        self.labels = np.clip(self.labels, -1e10, 1e10)
        pred = np.clip(pred, -1e10, 1e10)
        dw = np.dot((self.labels - pred), self.input.T) * 2 / self.sample_size
        db = (np.sum(self.labels - pred )) * 2 / self.sample_size
        return dw, db
    
    def update(self, w, b, dw, db):
        w = w - self.alpha * dw
        b = b - self.alpha * db
        return w, b
    
    def gradient_descent(self, w, b):
        for epoch in range(self.epochs):
            pred = self.forward(w, b)
            l = self.mse(pred)
            dw, db = self.backward(pred)
            w, b = self.update(w, b, dw, db)

            print(f"Epoch: {epoch}")
            print(f"Loss: {l} \n")
        return w, b
    
    def model(self):
        w, b = self.init_params()
        w, b = self.gradient_descent(w, b)
        return w, b