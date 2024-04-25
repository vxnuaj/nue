import numpy as np

class LinearRegression:
    def __init__(self, input, labels, num_features, alpha, epochs):
        self.input = input
        self.labels = labels
        self.num_features = num_features
        self.alpha = alpha
        self.epochs = epochs
        self.params = []
        self.gradients = []
        self.pred = None

    def init_params(self):
        w = np.random.randn(1, self.num_features)
        b = np.random.randn(1, 1)
        self.params = w, b
        return self.params
    
    def forward(self):
        w, b = self.params
        self.pred = np.dot(w, self.input) + b
        return self.pred
    
    def mse(self):
        l = np.sum((self.labels - self.pred) ** 2) / self.labels.size
        return l
    
    def backward(self):
        dw = - np.dot((self.labels - self.pred), self.input.T) * (2/self.labels.size)
        db = 2 * np.sum(self.labels - self.pred, axis = 0, keepdims = True ) / self.labels.size
        self.gradients = dw, db
        return self.gradients

    def update(self):
        dw, db = self.gradients
        w, b = self.params

        w = w - self.alpha * dw
        b = b - self.alpha * db
        
        self.params = w, b
        return self.params
    
    def gradient_descent(self):
        w, b = self.params
        for epoch in range(self.epochs):
            self.pred = self.forward()
            l = self.mse()
            self.gradients = self.backward()
            self.params = self.update()

            print(f"Epoch: {epoch}")
            print(f"Loss: {l}")
        
        return self.params
    
    def model(self):
        self.params = self.init_params()
        self.params = self.gradient_descent()
        return self.params
