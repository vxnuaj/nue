import numpy as np
import pandas as pd
import warnings

class LinearRegression:
    def __init__(self):
        return
    
    def init_params(self, feature_size):
        w = np.random.randn(1, feature_size)
        b = np.random.randn(1, 1)
        return w, b
    
    def forward(self, input, w, b):
        if w.shape[1] != input.shape[0]:
            raise ValueError("Incorrect dims! Try transposing the input!")
        pred = np.dot(w, input) + b
        return pred

    def mse(self, labels, pred, sample_size):
        labels = np.clip(labels, -1e10, 1e10)
        pred = np.clip(pred, -1e10, 1e10)
        l = np.sum((labels - pred) ** 2) / sample_size
        return l

    def backward(self, input, labels, sample_size, pred):
        labels = np.clip(labels, -1e10, 1e10)
        pred = np.clip(pred, -1e10, 1e10)
        dw = np.dot((labels - pred), input.T) * 2 / sample_size
        db = (np.sum(labels - pred )) * 2 / sample_size
        return dw, db
    
    def update(self, w, b, dw, db, alpha):
        w = w - alpha * dw
        b = b - alpha * db
        return w, b
    
    def gradient_descent(self, input, labels, sample_size, alpha, epochs, w, b):
        for epoch in range(epochs):
            pred = self.forward(input, w, b)
            l = self.mse(labels, pred, sample_size)
            dw, db = self.backward(input, labels, sample_size, pred)
            w, b = self.update(w, b, dw, db, alpha)

            print(f"Epoch: {epoch}")
            print(f"Loss: {l} \n")
        return w, b
    
    def model(self, input, labels, sample_size, feature_size, alpha, epochs):
        w, b = self.init_params(feature_size)
        w, b = self.gradient_descent(input, labels, sample_size, alpha, epochs, w, b)
        return w, b
    



    # gradient descent