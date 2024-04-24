import numpy as np

class NN:
    def __init__(self, input, labels, num_features, num_classes, hidden_size, alpha, epochs):
        self.hidden_size = hidden_size
        self.input = input
        self.labels = labels
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.epochs = epochs

        self.outputs = []
        self.params = []
        self.gradients = []
        self.one_hot_y = None
        self.pred = None
        self.l = None
    
    def init_params(self):
        w1 = np.random.rand(self.hidden_size, self.num_features) * np.sqrt(1 / self.num_features)
        b1 = np.zeros((self.hidden_size, 1))
        w2 = np.random.rand(self.num_classes, self.hidden_size) * np.sqrt( 1/ self.hidden_size)
        b2 = np.zeros((self.num_classes, 1))
        self.params = w1, b1, w2, b2
        return self.params
    
    def forward(self):
        w1, b1, w2, b2 = self.params
        z1 = np.dot(w1, self.input)
        a1 = self.ReLU(z1)
        z2 = np.dot(w2, a1)
        self.pred = self.softmax(z2)
        self.outputs = z1, a1, z2
        return self.pred
    
    def ReLU(self, z):
        return np.maximum(z, 0)
    
    def softmax(self, z):
        eps = 1e-10
        return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)
    
    def ReLU_deriv(self, z):
        return z > 0
    
    def one_hot(self):
        self.one_hot_y = np.zeros((self.num_classes, self.labels.size))
        self.one_hot_y[self.labels, np.arange(self.labels.size)] = 1
        return self.one_hot_y
    
    def cat_cross_entropy(self):
        self.l = - np.sum(self.one_hot_y * np.log(self.pred)) / self.one_hot_y.shape[1]
        return self.l
    
    def backward(self):
        z1, a1, _ = self.outputs
        _, _, w2, _ = self.params

        dz2 = self.pred - self.one_hot_y
        dw2 = np.dot(dz2, a1.T) / self.labels.size
        db2 = np.sum(dz2, axis=1, keepdims = True) / self.labels.size
        dz1 = np.dot(w2.T, dz2) * self.ReLU_deriv(z1)
        dw1 = np.dot(dz1, self.input.T) / self.labels.size
        db1 = np.sum(dz1, axis = 1, keepdims = True) / self.labels.size

        self.gradients =  dw1, db1, dw2, db2
        return self.gradients
    
    def update(self):
        w1, b1, w2, b2 = self.params
        dw1, db1, dw2, db2 = self.gradients
        
        w2 = w2 - self.alpha * dw2
        b2 = b2 - self.alpha * db2
        w1 = w1 - self.alpha * dw1
        b1 = b1 - self.alpha * db1
        self.params = w1, b1, w2, b2
        return self.params
    
    def gradient_descent(self):    
        self.one_hot_y = self.one_hot()

        for epoch in range(self.epochs):
            self.pred = self.forward()
            self.l = self.cat_cross_entropy()
            self.gradients = self.backward()
            self.params = self.update()

            print(f"Epoch: {epoch}")
            print(f"Loss: {self.l}")

        return self.params
    
    def model(self):
        self.params = self.init_params()
        self.params = self.gradient_descent()
        return self.params
