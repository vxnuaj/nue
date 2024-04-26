import numpy as np

class LogisticRegression:
    """

    :param input: The input data matrix of shape (n, m), where n is the number of features and m is the number of samples
    :type input: numpy.ndarray
    :param labels: The target labels of shape (1, m), where m is the number of samples.
    :type labels: numpy.ndarray
    :param num_features: The number of features in the input data.
    :type num_features: int
    :param alpha: The learning rate for gradient descent.
    :type alpha: float
    :param epochs: The number of epochs for training.
    :type epochs: int
    """
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
        """
        Initialize the parameters (weights and bias) for the logistic regression model.
        :return: Tuple containing the weights (w) and bias (b).
        :rtype: tuple
        """
        w = np.random.rand(1, self.num_features)
        b = np.random.rand(1, 1)
        self.params = w, b
        return self.params

    def sigmoid(self, z):
        """
        Calculate the sigmoid function for a given input.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The sigmoid of z.
        :rtype: float or numpy.ndarray
        """
        self.pred = 1 / (1 + np.exp(-z))
        return self.pred

    def forward(self):
        """
        Perform a forward pass to calculate the predicted probabilities.

        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        w, b = self.params
        z = np.dot(w, self.input) + b
        self.pred = self.sigmoid(z)
        return self.pred
    
    def log_loss(self):
        """
        Calculate the logistic loss (cross-entropy) between the predicted and actual labels.

        :return: The logistic loss.
        :rtype: float
        """
        eps = 1e-10
        self.l = - np.mean(self.labels * np.log(self.pred + eps) + (1 - self.labels) * np.log(1 - self.pred + eps))
        return self.l

    def backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: Tuple containing the gradients of the weights (dw) and bias (db).
        :rtype: tuple
        """
        dw = np.dot((self.pred - self.labels), self.input.T)
        db = np.mean(self.pred - self.labels, axis = 0, keepdims = True)
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
        Perform gradient descent to train the logistic regression model.

        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
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
        """
        Run the entire logistic regression model.

        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
        self.params = self.init_params()
        self.params = self.gradient_descent()
        return self.params