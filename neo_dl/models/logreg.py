import numpy as np

class LogisticRegression:
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
    
    def sigmoid(self, z):
        pred = 1 / (1 + np.exp(-z))
        return pred

    def forward(self, w, b):
        z = np.dot(w, self.input) + b
        pred = self.sigmoid(z)
        return pred
    
    def log_loss(self, pred):
        eps = 1e-10
        l = - (1 / self.sample_size) * np.sum(self.labels * np.log(pred + eps) + (1 - self.labels) * np.log(1 - pred + eps))
        return l
    
    def backward(self, pred):
        dw = np.dot((pred - self.labels), self.input.T) / self.sample_size
        db = np.sum((pred - self.labels)) / self.sample_size
        return dw, db
    
    def update(self, w, b, dw, db):
        w = w - self.alpha * dw
        b = b - self.alpha * db
        return w, b
    
    def gradient_descent(self, w, b):
        for epoch in range(self.epochs):
            pred = self.forward(w, b)
            l = self.log_loss(pred)
            dw, db = self.backward(pred)
            w, b = self.update(w, b, dw, db)
            
            print(f"Epoch: {epoch}")
            print(f"Loss: {l}")

        return w, b
    
    def model(self):
        w, b = self.init_params()
        w, b = self.gradient_descent(w, b)
        return w, b        

