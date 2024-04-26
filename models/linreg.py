import numpy as np

class LinearRegression:
    """
    
    :param input: The input data of shape (n, m), where n is the number of features and m is the number of samples
    :type input: numpy.ndarray
    :param labels: The target labels of shape (1, m), where m is the number of samples
    :type labels: numpy.ndarray
    :param num_features: The total number of features in the input data per sample
    :type: int
    :param alpha: The learning rate for gradient descent
    :type alpha: float
    :param epochs: The number of epochs for training
    :type epochs: int
    """

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
        """
        Initialize the parameters (weights and bias) for linear regression

        :return: Tuple containing the weights (w) and bias (b)
        :rtype: tuple
        """
        w = np.random.randn(1, self.num_features)
        b = np.random.randn(1, 1)
        self.params = w, b
        return self.params
    
    def forward(self):
        """
        Perform a forward pass to calculate the predicted values.

        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        w, b = self.params
        self.pred = np.dot(w, self.input) + b
        return self.pred
    
    def mse(self):
        """
        Calculate the mean squared error (MSE) between the predicted and actual values.

        :return: The mean squared error.
        :rtype: float
        """
        l = np.sum((self.labels - self.pred) ** 2) / self.labels.size
        return l
    
    def backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: Tuple containing the gradients of the weights (dw) and bias (db).
        :rtype: tuple
        """
        dw = - np.dot((self.labels - self.pred), self.input.T) * (2/self.labels.size)
        db = 2 * np.sum(self.labels - self.pred, axis = 0, keepdims = True ) / self.labels.size
        self.gradients = dw, db
        return self.gradients

    def update(self):
        """
        Update the weights and bias using gradient descent.

        :return: Tuple containing the updated weights (w) and bias (b).
        :rtype: tuple
        """
        dw, db = self.gradients
        w, b = self.params

        w = w - self.alpha * dw
        b = b - self.alpha * db
        
        self.params = w, b
        return self.params
    
    def gradient_descent(self):
        """
        Perform gradient descent to train the linear regression model.

        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
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
        """
        Run the entire linear regression model.

        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
        self.params = self.init_params()
        self.params = self.gradient_descent()
        return self.params
