import numpy as np
from nue.metrics import mse, r2_score


class LinearRegression:
   
    """
    
    :param seed: Set the random seed for initializing parameters. Based on numpy.random.default_rng() 
    :type seed: int 
    """
    def __init__(self, modality = 'ols', seed:int = None):
        self.X_train = np.empty(0) 
        self.Y_train = np.empty(0) 
        self.modality = modality 
      
        if self.modality == 'sgd':
            self.seed = seed

            self.__num_features = None
            self.__params = []
            self.__gradients = []
            
        elif modality == 'ols':
            return 
            
    def _init_params(self):
        """
        Initialize the parameters (weights and bias) for linear regression

        :return: Tuple containing the weights (w) and bias (b)
        :rtype: tuple
        """
        
        if self.seed == None:
            w = np.random.randn(1, self.__num_features)
            b = np.zeros((1, 1))
            self.__params = w, b
        else:
            rng = np.random.default_rng(seed = self.seed)
            w = rng.normal(size = (1, self.__num_features))
            b = np.zeros((1,1)) 
            self.__params = w, b 
        return self.__params
   
    def train(self, X_train:np.ndarray, Y_train: np.ndarray, alpha:float = .0001, epochs:int = 50000, verbose:bool=False, metric_freq:int = None): 
        """
        Train the linear regression model via Gradient Descent or Ordinary Least Squares, depending on the set modality.

        :param X_train: The input data of shape (samples, feature).
        :type X_train: numpy.ndarray
        :param Y_train: The target labels of shape (samples, 1)
        :type Y_train: numpy.ndarray
        :param alpha: The learning rate for gradient descent
        :type alpha: float 
        :param epochs: The number of epochs for training
        :type epochs: int

        :param verbose: If True, will print out training progress of the model
        :type verbose: bool
        :param metric_freq: Will not apply if verbose is set to False. If verbose is True and metric_freq is not set, it will default to 1. 
      
            Will print out epoch and loss at the epoch frequency set by metric_freq 
        
        :type metric_freq: int
        
        :return: Tuple containing the final weights (w) and bias (b).
        :rtype: tuple
       
        """

        self.X_train = X_train
        self.Y_train = Y_train
        self.verbose_train = verbose 
       
        if self.verbose_train and metric_freq is None: 
            self.metric_freq = 1
        elif self.verbose_train and metric_freq is not None:
            self.metric_freq = metric_freq  
        else:
            self.metric_freq = metric_freq
        
        if self.modality == 'sgd':  
   
            self.__num_features = X_train.shape[1]
            self.alpha = alpha
            self.epochs = epochs
       
            self.__params = self._init_params()
            self.__params = self._gradient_descent()
       
        elif self.modality == 'ols':
            
            self.__params = self._ols(Y_train, X_train)
        
        return self.__params
    
    def _forward(self):
        """
        Perform a forward pass to calculate the predicted values.

        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        w, b = self.__params
        self._pred = np.dot(w, self.X_train.T) + b

        return self._pred
    
    def _backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: Tuple containing the gradients of the weights (dw) and bias (db).
        :rtype: tuple
        """
        
        dz = -2 * (self.Y_train.T - self._pred)
        dw = np.dot(dz, self.X_train) /  self.Y_train.size
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
        b -=self.alpha * db
        
        self.__params = w, b
        return self.__params
    
    def _gradient_descent(self):
        """
        Perform gradient descent to train the linear regression model.
        
        :param verbose: If True, will print out training progress of the model
        :type verbose: bool
        :param metric_freq: Will not apply if verbose is set to False. 
      
            Will print out epoch and loss at the epoch frequency set by metric_freq 
        
        :type metric_freq: int

        :return: Name mangled tuple containing the final weights (w) and bias (b).
        
        """

        print(f"Model training via Gradient Descent!")
        for epoch in range(self.epochs):
            self._pred = self._forward()
            self.train_loss = mse(self.Y_train, self._pred.T)
            self.__gradients = self._backward()
            self.__params = self._update()

            if self._verbose_train == True and self.metric_freq is not None: 
                if epoch % self.metric_freq == 0:
                    print(f"Epoch: {epoch}") 
                    print(f"Loss: {self.train_loss}\n")
   
    
        self.train_r2 = r2_score(self.Y_train, self._pred.T) 
         
        print(f"Model trained!\n")
     
        self.coef, self.intercept = [i.flatten() for i in self.__params]
       
        if self._verbose_train == True: 
            print(f"\nFinal Training Loss: {self.train_loss}")  
            print(f"Training R2 score: {self.train_r2}") 
            print(f"Coefficients: {self.coef}\nIntercept: {self.intercept}\n")
        return self.__params
    
    def _ols(self, Y_train, X_train):
      
        print(f"Model training via OLS!")
       
        eps = 1e-8
        
        X_mean = np.mean(X_train, axis = 0)
        Y_mean = np.mean(Y_train, axis = 0)
  
  
        X_train_adj = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        beta = np.dot(np.linalg.inv(np.dot(X_train_adj.T, X_train_adj)), np.dot(X_train_adj.T, Y_train))

        b, w = beta[0], beta[1:]
        
        self._pred = np.dot(w.T, X_train.T) + b
       
        self.__params = w, b
        self.coef, self.intercept = [i.flatten() for i in self.__params]
     
        self.train_loss = mse(self.Y_train, self._pred.T)
        self.train_r2 = r2_score(self.Y_train, self._pred.T)
  
        if self._verbose_train:
            print(f"\nFinal Training Loss: {self.train_loss}")
            print(f"Training R2 score: {self.train_r2}")
            print(f"Coefficient: {self.coef}\nIntercept: {self.intercept}\n")
  
        print(f"Model trained!\n")
   
        return w, b
    
    def test(self, X_test:np.ndarray, Y_test:np.ndarray, verbose:bool = False):
        
        '''
        Test the linear regression model.
        
        :param X_test: The validation features, shape (samples, features).
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (samples, 1).
        :type Y_test: numpy.ndarray
        :param verbose: If true, will print out loss and r2 score post-test.
        :type verbose: bool
        ''' 
      
        self.X_test = X_test
        self.Y_test = Y_test
        self.verbose_test = verbose 
       
       
        print("Model testing!") 
       
        w, b = self.__params
     
        if self.modality == 'sgd': 
            self._pred = np.dot(w, self.X_test.T) + b 
        elif self.modality == 'ols':
            self._pred = np.dot(w.T, self.X_test.T) + b

        self.test_loss = mse(self.Y_test, self._pred.T)
        self.test_r2 = r2_score(self.Y_test, self._pred.T)
       
        print("Model tested!\n")
       
        if verbose: 
            print(f"Final Test loss: {self.test_loss}")
            print(f"Test R2 score: {self.test_r2}") 
    
    def metrics(self, mode = 'train'):
       
        if mode not in ['train', 'test', 'both']:
            raise ValueError("mode must be type str, of 'both', 'train', or 'test'!") 
        
        if mode in ['both', 'test']: 
            if self.train_loss and not hasattr(self, 'test_loss'):
                raise ValueError("You haven't tested the model yet!")
            elif not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!") 
            
            if mode == 'both': 
                print(f"Train loss: {self.train_loss} | Train R2 score: {self.train_r2} \nTest loss: {self.test_loss} | Test R2 score: {self.test_r2}\nCoefficients: {self.coef} | Intercept: {self.intercept}") 
            elif mode == 'test':
                print(f"Test loss: {self.test_loss} | Test R2 score: {self.test_r2}\nCoefficients: {self.coef} | Intercept: {self.intercept}")
        elif mode == 'train':
            if not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!")
            print(f"Train Loss: {self.train_loss} | Train R2 score: {self.train_r2}\nCoefficients: {self.coef} | Intercept: {self.intercept}")
             
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
    def verbose_train(self):
        return self._verbose_train
    
    @verbose_train.setter
    def verbose_train(self, verbose_train):
       
        if not isinstance(verbose_train, bool): 
            raise ValueError("verbose must be type bool")
        self._verbose_train = verbose_train 
        
    @property
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
        if modality not in ['sgd', 'ols']:
            raise ValueError("modality must be stochastic gradient descent (sgd) or ordinary least squares (ols)!")
        self._modality = modality
    
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

    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter 
    def X_test(self, X_test):
        if not isinstance(X_test, np.ndarray):
            raise ValueError('X_test must be type numpy.ndarray!')
        self._X_test = X_test 
        
    @property
    def Y_test(self):
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, Y_test):
        if not isinstance(Y_test, np.ndarray):
            raise ValueError('Y_test must be type numpy.ndarray')
        self._Y_test = Y_test 
       
    @property
    def verbose_test(self):
        return self._verbos_test
    
    @verbose_test.setter
    def verbose_test(self, verbose_test):
        if not isinstance(verbose_test, bool):
            raise ValueError('verbose_test must be type bool')
        self._verbose_test = verbose_test 
        
    def __str__(self):
        if self.train_loss and not hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train R2 score: {self.train_r2}\nCoefficients: {self.coef} | Intercept: {self.intercept} " 
        elif self.train_loss and hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train R2 score: {self.train_r2} \nTest loss: {self.test_loss} | Test R2 score: {self.test_r2}\n Coefficients: {self.coef} | Intercept: {self.intercept}" 
        