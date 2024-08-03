from nue.metrics import log_loss, logistic_accuracy
import numpy as np

class LogisticRegression:
    """
    Initialize the Logistic Regression model. 
    
    :param seed: Set the random seed for initializing parameters. Based on numpy.random.default_rng() 
    :type seed: int 
    :param verbose_train: Set the verbosity of the model during training
    :type verbose_train: bool
    :param verbose_test: Set the verbosity of the model during testing
    :type verbose_test: bool 
    """
    def __init__(self, seed:int = None, verbose_train = False, verbose_test = False):
    
        self.seed = seed 
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test 
    
    def train(self, X_train:np.ndarray, Y_train:np.ndarray, alpha:float = .0001, epochs:int = 1000, metric_freq:int = None ):
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
        self.metric_freq = metric_freq
        self.__num_features = X_train.shape[1]     
        self.__params = self._init_params()
        self.train_loss, self.train_acc, self.__params = self._gradient_descent()
      
        return self.train_loss, self.train_acc, self.__params 

    def test(self, X_test:np.ndarray, Y_test:np.ndarray, return_probs = False):
        '''
        Test the logistic regression model
        
        :param X_test: The validation features, shape (features, samples).
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (1, samples).
        :type Y_test: numpy.ndarray
        '''        

        self.X_test = X_test
        self.Y_test = Y_test
        self.return_probs = return_probs

        print("Logistic Regression Testing!")
       
        self.pred, self.probs = self.inference(self.X_test, self.Y_test, self.verbose_test) 
    
    
        w, b = self.__params
        
        z = np.dot(w, self.X_test.T) + b
        a = self.sigmoid(z) 
        self.pred = np.round(a, decimals = 0)
       
        self.test_loss = log_loss(self.Y_test.T, a)
        self.test_acc = logistic_accuracy(self.Y_test.T, a)

        print("Logistic Regression Tested!")         

        if self.verbose_test:
            print(f"Logistic Regression Test Loss: {self.test_loss}")
            print(f"Logistic Regression Test Accuracy: {self.test_acc}%") 
            if self.return_probs:
                print(f"Logistic Regression Probabilities: \n\n{self.probs}\n")
         
        if self.return_probs: 
            return self.test_loss, self.test_acc, self.pred, self.probs.flatten()
        return self.test_loss, self.test_acc, self.pred
   
    
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
        self._output = 1 / (1 + np.exp(-z + eps))
        return self._output

    def _forward(self):
        """
        Perform a forward pass to calculate the predicted probabilities.

        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        
        w, b = self.__params
        z = np.dot(w, self.X_train.T) + b
        self._output = self.sigmoid(z)
        return self._output
    
    def _backward(self):
        """
        Perform a backward pass to calculate the gradients of the weights and bias.

        :return: List containing the gradients of the weights (dw) and bias (db).
        :rtype: list
        """
       
        dz = self._output - self.Y_train.T
        dw = np.dot(dz, self.X_train) / self.Y_train.size
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
    
    def _gradient_descent(self):
        """
        Perform gradient descent to train the logistic regression model.

        :return: List containing the final weights (w) and bias (b).
        :rtype: list
        """
        print(f"Logistic Regression Training!") 
        
        for epoch in range(self.epochs):
            self._output = self._forward()
            self.train_loss = log_loss(self.Y_train.T, self._output)
            self.train_acc = logistic_accuracy(self.Y_train.T, self._output)
            self.__gradients = self._backward()
            self.__params = self._update()
            
            if self.verbose_train == True and self.metric_freq is not None:
                if epoch % self.metric_freq == 0: 
                    print(f"Epoch: {epoch}")
                    print(f"Loss: {self.train_loss}")
                    print(f"Accuracy: {self.train_acc}%\n")
        
        self.weights, self.bias = [i for i in self.__params]            
                    
        print(f"Logistic Regression Finished Training!") 
        
        if self.verbose_train == True:
            print(f"Logistic Regression Training Loss: {self.train_loss}")
            print(f"Logistic Regression Training Accuracy: {self.train_acc}%")
        
        return self.train_loss, self.train_acc, self.__params
    
    def inference(self, X_inf:np.ndarray, Y_inf:np.ndarray, verbose:bool = False):
        
        self.X_inf = X_inf
        self.Y_inf = Y_inf
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!") 
        
        w, b = self.__params
        z = np.dot(w, self.X_inf.T) + b
        self._output = self.sigmoid(z) 
       
        self.inf_loss = log_loss(self.Y_inf.T, self._output) 
        self.inf_acc = logistic_accuracy(self.Y_inf.T, self._output)
        
        self.pred = np.round(self._output, decimals = 0) 
       
        return self.pred, self._output
   
    
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
        if np.any(Y_train == -1):
            Y_train = np.where(Y_train == -1, 0, 1)  
        self._Y_train = Y_train

    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter
    def X_test(self, X_test):
        assert isinstance(X_test, np.ndarray), "X_test must be type numpy.ndarray!"
        self._X_test = X_test
    
    @property
    def Y_test(self):
        return self._Y_test  
       
    @Y_test.setter
    def Y_test(self, Y_test):
        assert isinstance(Y_test, np.ndarray), "Y_train must be of type numpy.ndarray!"
        if np.any(Y_test == -1):
            Y_test = np.where(Y_test == -1, 0, 1)  
        self._Y_test = Y_test

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
        
        