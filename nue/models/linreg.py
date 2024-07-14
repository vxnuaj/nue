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
            self.alpha = .0001 
            self.epochs = 50000
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
    
    def forward(self):
        """
        Perform a forward pass to calculate the predicted values.

        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        w, b = self.__params
        self.pred = np.dot(w, self.X_train) + b

        return self.pred
    
    def _backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: Tuple containing the gradients of the weights (dw) and bias (db).
        :rtype: tuple
        """
        dz = -2 * (self.Y_train - self.pred)
        dw = np.dot(dz, self.X_train.T) /  self.Y_train.size
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
    
    def _gradient_descent(self, verbose:bool, metric_freq:int):
        """
        Perform gradient descent to train the linear regression model.
        
        :param verbose: If True, will print out training progress of the model
        :type verbose: bool
        :param metric_freq: Will not apply if verbose is set to False. 
      
            Will print out epoch and loss at the epoch frequency set by metric_freq 
        
        :type metric_freq: int

        :return: Name mangled tuple containing the final weights (w) and bias (b).
        :rtype: tuple
        """
        print(f"Model training!")
        for epoch in range(self.epochs):
            self.pred = self.forward()
            self.train_loss = mse(self.Y_train, self.pred)
            self.__gradients = self._backward()
            self.__params = self._update()

            if verbose == True and metric_freq is not None: 
                if epoch % metric_freq == 0:
                    print(f"Epoch: {epoch}") 
                    print(f"Loss: {self.train_loss}\n")
   
    
        self.train_r2 = r2_score(self.Y_train, self.pred) 
         
        print(f"Model trained!\n")
     
        self.coef, self.intercept = self.__params 
       
        if verbose == True: 
            print(f"Final Training Loss: {self.train_loss}")  
            print(f"Training R2 score: {self.train_r2}") 
            print(f"Coefficients: {self.coef}\nIntercept: {self.intercept}\n")
        return self.__params
    
    def train(self, X_train:np.ndarray, Y_train: np.ndarray, alpha:float = .0001, epochs:int = 50000, verbose:bool=False, metric_freq:int = None): 
        """
        Train the linear regression model via Gradient Descent.

           
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
        self.__num_features = X_train.shape[0]

        if not isinstance(verbose, bool): 
            raise ValueError("verbose must be type bool")
        
        self.__params = self._init_params()
        self.__params = self._gradient_descent(verbose, metric_freq)
        return self.__params
    
    def fit(self, X_train:np.ndarray, Y_train:np.ndarray):
        '''
        Fit the model via Ordinary Least Squares 
        '''
        
        eps = 1e-8 
        
        Y_mean = np.mean(Y_train)
        X_mean = np.mean(X_train)

        w = np.sum((X_train - X_mean) * (Y_train -Y_mean), axis = 0) / np.sum(np.square(X_train - X_mean) + eps, axis = 0) / X_train.shape[0]
        b = np.sum(Y_mean - w * X_mean) / X_train.shape[0]
        self.__params = w, b
        return self.__params
  
   
    def test(self, X_test:np.ndarray, Y_test:np.ndarray, verbose:bool = False):
        
        '''
      
        Test the linear regression model.
        
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
        pred = np.dot(w, X_test) + b 
        self.test_loss = mse(Y_test, pred)
        self.test_r2 = r2_score(Y_test, pred)
       
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
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
        if modality not in ['gd', 'ols']:
            raise ValueError("modality must be gradient descent (gd) or ordinary least squares (ols)!")
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

    def __str__(self):
        if self.train_loss and not hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train R2 score: {self.train_r2}\nCoefficients: {self.coef} | Intercept: {self.intercept} " 
        elif self.train_loss and hasattr(self, 'test_loss'):
            return f"Train loss: {self.train_loss} | Train R2 score: {self.train_r2} \nTest loss: {self.test_loss} | Test R2 score: {self.test_r2}\n Coefficients: {self.coef} | Intercept: {self.intercept}" 
        