import numpy as np

class LogisticRegression:
    def __init__(self, input, labels, num_features, alpha, epochs):
        self.input = input
        self.labels = labels
        self.num_features = num_features
        self.alpha = alpha
        self.epochs = epochs

        self.outputs = []
        self.params = []
        self.gradients = []
        self.pred = None
        self.l = None

    def init_params(self):
        w = np.random.rand(1, self.num_features)
        b = np.random.rand(1, 1)
        self.params = w, b
        return self.params

    def sigmoid(self, z):
        self.pred = 1 / (1 + np.exp(-z))
        return self.pred

    def forward(self):
        w, b = self.params
        z = np.dot(w, self.input) + b
        self.pred = self.sigmoid(z)
        return self.pred
    
    def log_loss(self):
        eps = 1e-10
        self.l = - np.mean(self.labels * np.log(self.pred + eps) + (1 - self.labels) * np.log(1 - self.pred + eps))
        return self.l

    def backward(self):
        dw = np.dot((self.pred - self.labels), self.input.T)
        db = np.mean(self.pred - self.labels, axis = 0, keepdims = True)
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
            self.l = self.log_loss()
            self.gradients = self.backward()
            self.params = self.update()

            print(f"Epoch: {epoch}")
            print(f"Loss: {self.l}")

        return self.params
    
    def model(self):
        self.params = self.init_params()
        self.params = self.gradient_descent()
        return self.params