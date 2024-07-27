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
        
        self.seed = seed 
        
        self.__num_features = None
        self.__params = None
        self.__gradients = None
        
    def train(self, X_train:np.ndarray, Y_train:np.ndarray, alpha:float = .0001, epochs:int = 250, verbose:bool = False, metric_freq:int = None ):
        """
        Train the logistic regression model.

        :param X_train: The input data of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The target labels of shape (samples, 1) or (samples, )
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

        :return: list containing the final weights (w) and bias (b).
        :rtype: list
        """
       
        self.X_train = X_train
        self.Y_train = Y_train
        self.alpha = alpha 
        self.epochs = epochs
        self.verbose_train = verbose

        self.__num_features = X_train.shape[1]     
       
        self.__params = self._init_params()
        self.__params = self._gradient_descent(verbose, metric_freq)
        return self.__params 

    def _init_params(self):
        """
        Initialize the parameters (weights and bias) for the logistic regression model, based on the chosen seed in the init method.
   
        :return: List containing the weights (w) and bias (b).
        :rtype: list
        """
        if self.seed == None: 
            w = np.random.rand(1, self.__num_features)
            b = np.zeros((1, 1))
            
            self.__params = [w, b] 
        else:   
            rng = np.random.default_rng(seed = self.seed)
            w = rng.normal(size = (1, self.__num_features)) 
            b = np.zeros((1, 1)) 
            self.__params = [w, b]

        return self.__params

    def sigmoid(self, z, eps = 1e-8):
        """
        Calculate the sigmoid function for a given input.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The sigmoid of z.
        :rtype: float or numpy.ndarray
        """
        self.output = 1 / (1 + np.exp(-z + eps))
        return self.output

    def _forward(self):
        """
        Perform a forward pass to calculate the predicted probabilities.

        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        
        w, b = self.__params
        z = np.dot(w, self.X_train.T) + b
        self.output = self.sigmoid(z)
        return self.output
    
    def _backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: List containing the gradients of the weights (dw) and bias (db).
        :rtype: list
        """
       
        dz = self.output - self.Y_train.T
        dw = np.dot(dz, self.X_train)
        db = np.sum(dz) / self.Y_train.size
        self.__gradients = [dw, db] 
        return self.__gradients
    
    def _update(self):
        """
        Update the weights and bias using gradient descent.

        :return: List containing the updated weights (w) and bias (b).
        :rtype: list
        """
        dw, db = self.__gradients
        w, b = self.__params

        w -= self.alpha * dw
        b -= self.alpha * db

        self.__params = [w, b] 
        return self.__params
    
    def _gradient_descent(self, verbose:bool, metric_freq:int):
        """
        Perform gradient descent to train the logistic regression model.

        :return: List containing the final weights (w) and bias (b).
        :rtype: list
        """
        print(f"Model training!") 
        
        for epoch in range(self.epochs):
            self.output = self._forward()

            self.train_loss = log_loss(self.Y_train.T, self.output)
            self.train_acc = logistic_accuracy(self.Y_train.T, self.output)

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

        self.X_test = X_test
        self.Y_test = Y_test

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")

        print("Model testing!")
       
        w, b = self.__params
        z = np.dot(w, X_test.T) + b
        a = self.sigmoid(z) 
        
        self.test_loss = log_loss(Y_test.T, a)
        self.test_acc = logistic_accuracy(Y_test.T, a)

        print("Model tested!\n")         

        if verbose:
            print(f"Final test loss: {self.test_loss}")
            print(f"Final test accuracy: {self.test_acc}%\n") 
  
        return self.test_loss, self.test_acc
   
    def inference(self, X_inf:np.ndarray, Y_inf:np.ndarray, verbose:bool = False):
        
        self.X_inf = X_inf
        self.Y_inf = Y_inf
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!") 
        
        w, b = self.__params
        z = np.dot(w, X_inf) + b
        a = self.sigmoid(z) 
       
        self.inf_loss = log_loss(Y_inf.T, a) 
        self.inf_acc = logistic_accuracy(Y_inf.T, a)
        
        self.pred = np.round(a, decimals = 0) 
       
        return self.pred 
   
    
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
        assert isinstance(X_train, np.ndarray), "X_train must be type numpy.ndarray!"
        
        self._X_train = X_train
    
    @property
    def Y_train(self):
        return self._Y_train  
       
    @Y_train.setter
    def Y_train(self, Y_train):
        assert isinstance(Y_train, np.ndarray), "Y_train must be of type numpy.ndarray!"
        self._Y_train = Y_train

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        assert isinstance(alpha, (float, int)), "alpha must be of type float or int!"
        self._alpha = alpha

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        assert isinstance(epochs, int), "epochs must be of type int!"
        self._epochs = epochs

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int) or seed is None, "seed must be of type int or None!"
        self._seed = seed

    @property
    def verbose_train(self):
        return self._verbose_train

    @verbose_train.setter
    def verbose_train(self, verbose):
        assert isinstance(verbose, bool), "verbose must be of type bool!"
        self._verbose_train = verbose


    def __str__(self):
        if self.train_loss and not hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train Accuracy: {self.train_acc}%"
        elif self.train_loss and hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train Accuracy: {self.train_acc}%\nTest loss: {self.test_loss} | Test Accuracy: {self.test_acc}%"
        
        