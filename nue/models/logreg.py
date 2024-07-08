from nue.metrics import log_loss, logistic_accuracy
import numpy as np

class LogisticRegression:
    """
    :param seed: Set the random seed for initializing parameters. Based on numpy.random.default_rng() 
    :type seed: int  
    """
    def __init__(self, seed:int = None):
    
        self.X_train = np.empty(0)
        self.Y_train = np.empty(0)
        self.alpha = .0001
        self.epochs = 250
        self.seed = seed 
        
        self.__num_features = None
        self.__params = []
        self.__gradients = []

    def _init_params(self):
        """
        Initialize the parameters (weights and bias) for the logistic regression model.
   
        :return: Tuple containing the weights (w) and bias (b).
        :rtype: tuple
        """
        if self.seed == None: 
            w = np.random.rand(1, self.__num_features)
            b = np.zeros((1, 1))
            self.__params == w, b 
        else:   
            rng = np.random.default_rng(seed = self.seed)
            w = rng.normal(size = (1, self.__num_features)) 
            b = np.zeros((1, 1)) 
            self.__params = w, b
            
        return self.__params

    def sigmoid(self, z, eps = 1e-8):
        """
        Calculate the sigmoid function for a given input.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The sigmoid of z.
        :rtype: float or numpy.ndarray
        """
        self.pred = 1 / (1 + np.exp(-z + eps))
        return self.pred

    def _forward(self):
        """
        Perform a forward pass to calculate the predicted probabilities.

        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        w, b = self.__params
        z = np.dot(w, self.X_train) + b
        self.pred = self.sigmoid(z)
        return self.pred
    
    def _backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: Tuple containing the gradients of the weights (dw) and bias (db).
        :rtype: tuple
        """
        
        dz = self.pred - self.Y_train 
        dw = np.dot(dz, self.X_train.T)
        db = np.sum(dz) / self.Y_train.size
        self.__gradients = dw, db
        return self.__gradients
    
    def _update(self):
        """
        Update the weights and bias using gradient descent.

        :return: Tuple containing the updated weights (w) and bias (b).
        :rtype: tuple
        """
        dw, db = self.__gradients
        w, b = self.__params

        w -= self.alpha * dw
        b -= self.alpha * db

        self.__params = w, b
        return self.__params
    
    def _gradient_descent(self, verbose:bool, metric_freq:int):
        """
        Perform gradient descent to train the logistic regression model.

        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
        print(f"Model training!") 
        
        for epoch in range(self.epochs):
            self.pred = self._forward()

            self.train_loss = log_loss(self.Y_train, self.pred)
            self.train_acc = logistic_accuracy(self.Y_train, self.pred)

            self.__gradients = self._backward()
            self.__params = self._update()


            if verbose == True and metric_freq is not None:
                if epoch % metric_freq == 0: 
                    print(f"Epoch: {epoch}")
                    print(f"Loss: {self.train_loss}")
                    print(f"Accuracy: {self.train_acc}%\n")
                    
                    
        print(f"Model trained!\n") 
        
        if verbose == True:
            print(f"Final Training Loss: {self.train_loss}")
            print(f"Final Training Accuracy: {self.train_acc}%\n")
        
        return self.__params
    
    def train(self, X_train:np.ndarray, Y_train:np.ndarray, alpha:float = .0001, epochs:int = 250, verbose:bool = False, metric_freq:int = None ):
        """
        Train the logistic regression model.

        :param X_train: The input data of shape (features, samples)
        :type X_train: numpy.ndarray
        :param Y_train: The target labels of shape (1, samples)
        :type Y_train: numpy.ndarray
        :param alpha: The learning rate for gradient descent
        :type alpha: float 
        :param epochs: The number of epochs for training
        :type epochs: int

        :param verbose: If True, will print out training progress of the model
        :type verbose: bool
        :param metric_freq: Will not apply if verbose is set to False. 
      
            Will print out epoch and loss at the epoch frequency set by metric_freq 
        
        :type metric_freq: int

        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
       
        self.X_train = X_train
        self.Y_train = Y_train
        self.alpha = alpha 
        self.epochs = epochs

        self.__num_features = X_train.shape[0]     
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")
      
        self.__params = self._init_params()
        self.__params = self._gradient_descent(verbose, metric_freq)
        return self.__params

    def test(self, X_test:np.ndarray, Y_test:np.ndarray, verbose:bool = False ):
        '''
        Test the logistic regression model
        
        :param X_test: The validation features, shape (features, samples).
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (1, samples).
        :type Y_test: numpy.ndarray
        :param verbose: If true, will print out loss and r2 score post-test.
        :type verbose: bool  
        '''        

        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_test must be type numpy.ndarray!")
        if not isinstance(Y_test, np.ndarray):
            raise ValueError("Y_test must be type numpy.ndarray!")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")

        print("Model testing!")
        
        w, b = self.__params
        z = np.dot(w, X_test) + b
        a = self.sigmoid(z) 
        
        self.test_loss = log_loss(Y_test, a)
        self.test_acc = logistic_accuracy(Y_test, a)

        print("Model tested!\n")         

        if verbose:
            print(f"Final test loss: {self.test_loss}")
            print(f"Final test accuracy: {self.test_acc}%\n") 
   
    
    def metrics(self, mode = 'train'):
        
        """
       
        Prints the training and / or testing accuracy and loss for the model, dependent on `mode`
        
        :param mode: The metrics to be printed, defined by 'train', 'test', or 'both'
        :type mode: str
        
        """ 
        
        if mode not in ['train', 'test', 'both']:
            raise ValueError("mode must be type str, of 'both', 'train', or 'test'!") 
        
        if mode in ['both', 'test']: 
            if self.train_loss and not hasattr(self, 'test_loss'):
                raise ValueError("You haven't tested the model yet!")
            elif not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!") 
  
        if mode == 'both':
            print(f"Train loss: {self.train_loss} | Train accuracy: {self.train_acc}%\nTest loss: {self.test_loss} | Test accuracy: {self.test_acc}%")
        elif mode == 'test':
                print(f"Test loss: {self.test_loss} | Test accuracy: {self.test_acc}%") 
      
        elif mode == 'train':
            if not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!")
            print(f"Train loss: {self.train_loss} | Train accuracy: {self.train_acc}%") 
    
    @property
    def X_train(self):
        return self._X_train 
    
    @X_train.setter
    def X_train(self, X_train):
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be type numpy.ndarray!")
        self._X_train = X_train
    
    @property
    def Y_train(self):
        return self._Y_train  
       
    @Y_train.setter
    def Y_train(self, Y_train):
        if not isinstance(Y_train, np.ndarray):
            raise ValueError("Y_train must be type numpy.ndarray!") 
        self._Y_train = Y_train
    
    @property
    def alpha(self):
        return self._alpha 
   
    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, float):
            raise ValueError("alpha must be type float!")
        self._alpha = alpha     
    
    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, epochs):
        if not isinstance(epochs, int):
            raise ValueError("epochs must be type int!")
        self._epochs = epochs

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be type int or set as none!")
        self._seed = seed
        

    def __str__(self):
        if self.train_loss and not hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train Accuracy: {self.train_acc}%"
        elif self.train_loss and hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train Accuracy: {self.train_acc}%\nTest loss: {self.test_loss} | Test Accuracy: {self.test_acc}%"
        