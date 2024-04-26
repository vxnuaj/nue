import numpy as np

class NN:
    """

    :param input: The input data matrix of shape (m, n), where m is the number of samples and n is the number of features.
    :type input: numpy.ndarray
    :param labels: The target labels of shape (m, 1).
    :type labels: numpy.ndarray
    :param num_features: The number of features in the input data.
    :type num_features: int
    :param num_classes: The number of classes in the classification task.
    :type num_classes: int
    :param hidden_size: The number of units in the hidden layer.
    :type hidden_size: int
    :param alpha: The learning rate for gradient descent.
    :type alpha: float
    :param epochs: The number of epochs for training.
    :type epochs: int
    """
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
        self.pred_label = None
    
    def init_params(self):
        """
        Initialize the parameters (weights and biases) for the neural network.

        :return: Tuple containing the weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        w1 = np.random.rand(self.hidden_size, self.num_features) * np.sqrt(1 / self.num_features)
        b1 = np.zeros((self.hidden_size, 1))
        w2 = np.random.rand(self.num_classes, self.hidden_size) * np.sqrt( 1/ self.hidden_size)
        b2 = np.zeros((self.num_classes, 1))
        self.params = w1, b1, w2, b2
        return self.params
    
    def forward(self):
        """
        Perform a forward pass through the neural network.

        :return: The predicted probabilities for each class
        :rtype: numpy.ndarray
        """
        w1, b1, w2, b2 = self.params
        z1 = np.dot(w1, self.input)
        a1 = self.ReLU(z1)
        z2 = np.dot(w2, a1)
        self.pred = self.softmax(z2)
        self.outputs = z1, a1, z2
        return self.pred
    
    def ReLU(self, z):
        """
        Apply the Rectified Linear Unit (ReLU) activation function element-wise to the input.

        :param z: The input to the ReLU function.
        :type z: numpy.ndarray
        :return: The output of the ReLU function.
        :rtype: numpy.ndarray
        """
        return np.maximum(z, 0)
    
    def softmax(self, z):
        """
        Apply the softmax activation function to the input.

        :param z: The input to the softmax function.
        :type z: numpy.ndarray
        :return: The output of the softmax function.
        :rtype: numpy.ndarray
        """
        eps = 1e-10
        return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)
    
    def ReLU_deriv(self, z):
        """
        Compute the derivative of the ReLU function.

        :param z: The input to the ReLU function.
        :type z: numpy.ndarray
        :return: The derivative of the ReLU function.
        :rtype: numpy.ndarray
        """
        return z > 0
    
    def one_hot(self):
        """
        Convert the target labels into one-hot encoded format.

        :return: The one-hot encoded labels.
        :rtype: numpy.ndarray
        """
        self.one_hot_y = np.zeros((self.num_classes, self.labels.size))
        self.one_hot_y[self.labels, np.arange(self.labels.size)] = 1
        return self.one_hot_y
    
    def cat_cross_entropy(self):
        """
        Calculate the categorical cross-entropy loss between the predicted and actual labels.

        :return: The categorical cross-entropy loss.
        :rtype: float
        """
        self.l = - np.sum(self.one_hot_y * np.log(self.pred)) / self.one_hot_y.shape[1]
        return self.l
    
    def accuracy(self):
        """
        Calculate the accuracy of the model.

        :return: The accuracy of the model as a percentage.
        :rtype: float
        """
        self.pred_label = np.argmax(self.pred, axis = 0)
        acc = np.sum(self.pred_label == self.labels) / self.labels.size * 100
        return acc

    def backward(self):
        """
        Perform a backward pass through the neural network to compute gradients.

        :return: Tuple containing the gradients of the weights and biases for the hidden and output layers.
        :rtype: tuple
        """
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
        """
        Update the weights and biases of the neural network using gradient descent.

        :return: Tuple containing the updated weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        w1, b1, w2, b2 = self.params
        dw1, db1, dw2, db2 = self.gradients
        
        w2 = w2 - self.alpha * dw2
        b2 = b2 - self.alpha * db2
        w1 = w1 - self.alpha * dw1
        b1 = b1 - self.alpha * db1
        self.params = w1, b1, w2, b2
        return self.params

    
    def gradient_descent(self):    
        """
        Perform gradient descent to train the neural network.

        :return: Tuple containing the final weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        self.one_hot_y = self.one_hot()

        for epoch in range(self.epochs):
            self.pred = self.forward()
            
            acc = self.accuracy()

            self.l = self.cat_cross_entropy()
            self.gradients = self.backward()
            self.params = self.update()

            print(f"Epoch: {epoch}")
            print(f"Accuracy: {acc}%")
            print(f"Loss: {self.l}\n")

        return self.params
    
    def model(self):
        """
        Run the entire neural network model.

        :return: Tuple containing the final weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        self.params = self.init_params()
        self.params = self.gradient_descent()
        return self.params