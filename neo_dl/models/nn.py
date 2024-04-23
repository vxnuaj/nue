import numpy as np

class NN:
    def __init__(self):
        return
    
    def layer(self, neurons, input_size):
        w = np.random.randn(neurons, input_size)
        b = np.random.randn(neurons, 1)
        layer = w, b
        return layer

    def ReLU(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))
    
    def forward(self, input, layer):
        w, b = layer
        z = np.dot(w, input) + b
        return z
    
    def cat_cross_entropy(self, pred, y):
        eps = 1e-10
        l = - np.mean(y * np.log(pred + eps))
        return l
        


'''
    
    
    def backward(self, layers):
        dw = []
        db = []

    '''
            
