import numpy as np
from nue.models import LogisticRegression

class _PlattScaling():
    '''
    Utility class for implementing Platt Scaling via Logistic Regression, to output raw probabilities of a classifier 
    Currently applicable to Support Vector Machine and K-Nearest Neighbor classifiers

    Learn more here: https://en.wikipedia.org/wiki/Platt_scaling

    :param seed: Set the random seed for the parameters of the Logistic Regression model
    :type seed: int
    :param verbose_train: Set the verbosity for training the model
    :type verbose_train: bool
    :param verbose_test: Set the verbosity for testing the model
    :type verbose_test: bool
    '''
    
    def __init__(self, seed:int = None, verbose_train:bool = False, verbose_test:bool = False):
        self.seed = seed
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self.model = LogisticRegression(seed = self.seed, verbose_train=self.verbose_train, verbose_test=self.verbose_test)
    
    def platt_train(self, X_model_output:np.ndarray, Y_train:np.ndarray, alpha:float = .0001, epochs:int = 1000, metric_freq:int = None):
        '''
        Train a Logistic Regression model for Platt Scaling
        
        :param X_train: The training features, of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The corresponding training labels to X_train of shape (samples, 1) or (samples, )
        :type Y_train: numpy.ndarray 
        '''
        self.X_train = X_model_output
        self.Y_train = Y_train
        self.alpha = alpha
        self.epochs = epochs
        self.loss, self.train_acc, self.__params = self.model.train(X_train = self.X_train, Y_train = self.Y_train, alpha = self.alpha, epochs = self.epochs, metric_freq = metric_freq)
        self.platt_inf(X_test = self.X_train, Y_test = self.Y_train) 
        return self.probs, self.pred, self.train_loss, self.train_acc
    
    def platt_inf(self, X_inf:np.ndarray, Y_inf:np.ndarray):
        '''
        Inference of the Logistic Regression model
        
        :param X_test: The testing features, of shape (samples, features)
        :type X_test: numpy.ndarray 
        :param Y_test: The corresponding testing labels to X_test of shape (samples, 1) or (samples, )
        :type Y_test: numpy.ndarray 
        '''
        self.X_inf = X_inf
        self.Y_inf = Y_inf
        self.test_loss, self.test_acc, self.pred, self.probs = self.model.test(X_test = self.X_inf, Y_test = self.Y_inf, return_probs = True) 
        return self.probs, self.pred, self.test_loss, self.test_acc