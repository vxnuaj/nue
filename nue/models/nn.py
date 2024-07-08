import numpy as np
from nue.metrics import nn_accuracy, cce
from nue.preprocessing import one_hot


class NN:
    """
    :param seed: Set the random seed for initializing parameters. Based on numpy.random.default_rng() 
    :type seed: int  
    """
    def __init__(self, seed): 
        self.X_train = np.empty(0)
        self.Y_train = np.empty(0) 
        self.alpha = .01
        self.epochs = 250
        self.seed = seed
        
        self.outputs = []
        self.__params = []
        self.__gradients = []
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
        
        if self.seed == None:
            w1 = np.random.rand(self.hidden_size, self.__num_features) * np.sqrt(2 / self.__num_features)
            b1 = np.zeros((self.hidden_size, 1))
            w2 = np.random.rand(self.num_classes, self.hidden_size) * np.sqrt( 2/ self.hidden_size)
            b2 = np.zeros((self.num_classes, 1))
            self.__params = w1, b1, w2, b2
        else:
            rng = np.random.default_rng(seed = self.seed)
            w1 = rng.normal(size = (self.hidden_size, self.__num_features)) * np.sqrt(2 / self.__num_features)
            b1 = np.zeros((self.hidden_size, 1))
            w2 = rng.normal( size = (self.num_classes, self.hidden_size)) * np.sqrt(2 / self.hidden_size)
            b2 = np.zeros((self.num_classes, 1)) 
            self.__params = w1, b1, w2, b2 
        return self.__params
    
    def _forward(self):
        """
        Perform a forward pass through the neural network.

        :return: The predicted probabilities for each class
        :rtype: numpy.ndarray
        """
        w1, b1, w2, b2 = self.__params
        z1 = np.dot(w1, self.X_train) + b1
        a1 = self.leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        self.pred = self.softmax(z2)
        self.outputs = z1, a1, z2, self.pred
        return self.pred
    
    def leaky_relu(self, z):
        """
        Apply the Rectified Linear Unit (ReLU) activation function element-wise to the input.

        :param z: The input to the ReLU function.
        :type z: numpy.ndarray
        :return: The output of the ReLU function.
        :rtype: numpy.ndarray
        """
        return np.where(z > 0, z, z * .01) 
   
    def leaky_relu_deriv(self, z):
        """
        Compute the derivative of the ReLU function.

        :param z: The input to the ReLU function.
        :type z: numpy.ndarray
        :return: The derivative of the ReLU function.
        :rtype: numpy.ndarray
        """
        return np.where(z > 0, 1, .01) 
   
    
    def softmax(self, z):
        """
        Apply the softmax activation function to the input.

        :param z: The input to the softmax function.
        :type z: numpy.ndarray
        :return: The output of the softmax function.
        :rtype: numpy.ndarray
        """
        eps = 1e-6
        return np.exp(z + eps) / np.sum(np.exp(z + eps), axis = 0, keepdims = True)
    
    def _backward(self):
        """
        Perform a backward pass through the neural network to compute gradients.

        :return: Tuple containing the gradients of the weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        z1, a1, _, a2 = self.outputs
        _, _, w2, _ = self.__params

        dz2 = a2 - self.one_hot_y
        dw2 = np.dot(dz2, a1.T) / self.Y_train.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims = True) / self.Y_train.shape[1]
        dz1 = np.dot(w2.T, dz2) * self.leaky_relu_deriv(z1)
        dw1 = np.dot(dz1, self.X_train.T) / self.Y_train.shape[1]
        db1 = np.sum(dz1, axis = 1, keepdims = True) / self.Y_train.shape[1]

        self.__gradients =  dw1, db1, dw2, db2
        return self.__gradients
    
    def _update(self):
        """
        Update the weights and biases of the neural network using gradient descent.

        :return: Tuple containing the updated weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        w1, b1, w2, b2 = self.__params
        dw1, db1, dw2, db2 = self.__gradients
        
        w2 -= self.alpha * dw2
        b2 -= self.alpha * db2
        w1 -= self.alpha * dw1
        b1 -= self.alpha * db1
        self.__params = w1, b1, w2, b2
        return self.__params

    
    def _gradient_descent(self, verbose:bool, metric_freq:int):    
        """
        Perform gradient descent to train the neural network.

        :return: Tuple containing the final weights and biases for the hidden and output layers.
        :rtype: tuple
        """
        self.one_hot_y = one_hot(self.Y_train)

        print(f"Model training!")

        for epoch in range(self.epochs):
            self.pred = self._forward()
            
            self.train_acc = nn_accuracy(self.Y_train, self.pred)
            self.train_loss = cce(self.one_hot_y, self.pred) 

            self.__gradients = self._backward()
            self.__params = self._update()

            if verbose == True and metric_freq is not None:
                if epoch % metric_freq == 0:
                    print(f"Epoch: {epoch}")
                    print(f"Accuracy: {self.train_acc}%")
                    print(f"Loss: {self.train_loss}\n")

        print(f"Model trained!\n")
        
        if verbose == True:
            print(f"Final Training Loss: {self.train_loss}")
            print(f"Final Training Accuracy: {self.train_acc}%\n") 
            
        return self.__params
    
    def train(self, X_train:np.ndarray, Y_train:np.ndarray, hidden_size:int, num_classes:int, alpha:float = .01, epochs:int = 250, verbose:bool = False, metric_freq:int = None, load_model_filepath:str = None, save_model_filepath:str = None):
        """
        Train the neural network

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
        :param save_model: If True, will save model parameters to specified filepath as .npz file
        :type save_model: bool
        :param filepath: Denoting the intended filepath to save the model parameters to.
        :type filepath: str 
        
        :return: Tuple containing the final weights and biases for the hidden and output layers.
        :rtype: tuple
        """
    
        self.X_train = X_train
        self.Y_train = Y_train
        self.hidden_size = hidden_size       
        self.num_classes = num_classes 
        self.alpha = alpha
        self.epochs = epochs
        
        self.__num_features = X_train.shape[0]
       
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")

 
        if load_model_filepath is None:
            print("Initializing new model parameters!")
            self.__params = self.init_params()
        elif isinstance(load_model_filepath, str):
            try:
                print("Found model! Loading parameters!")
                with np.load(load_model_filepath) as parameters:
                    self.__params = [parameters[f"arr_{i}"] for i in range(len(parameters.files))] 
            except FileNotFoundError as e:
                raise ValueError(f"Unable to initialize model: {e} ")
        else:
            raise ValueError("load_model_filepath must be str or None!")
        
                 
        self.__params = self._gradient_descent(verbose, metric_freq) 
        
        if isinstance(save_model_filepath, str):
            *params, = self.__params
            np.savez(save_model_filepath, *params) 
        
        
        return self.__params
  
    def test(self, X_test:np.ndarray, Y_test:np.ndarray, model_filepath:str = None, verbose:bool = False):
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_test must be type numpy.ndarray!")
        if not isinstance(Y_test, np.ndarray):
            raise ValueError("Y_test must be type numpy.ndarray!")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")
        
        print(f"Model testing!") 


        one_hot_y = one_hot(Y_test)

        if model_filepath == None:
            raise ValueError("Can't proceed, you didn't provide a model_filepath!")
        else:
            with np.load(model_filepath) as parameters:
                self.__params = [parameters[f"arr_{i}"] for i in range(len(parameters.files))]
                w1, b1, w2, b2 = self.__params

        z1 = np.dot(w1, X_test) + b1
        a1 = self.leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = self.softmax(z2)

        self.test_loss = cce(one_hot_y, a2)
        self.test_acc = nn_accuracy(Y_test, a2) 

        print(f"Model tested!\n")


        if verbose:
            print(f"Final test loss: {self.test_loss}")
            print(f"Final test accuracy: {self.test_acc}%\n") 
            


    def metrics(self, mode = 'train'):
         
        """
        Prints the training and / or testing accuracy and loss for the model, dependent on `mode`
        
        Will not work if model.train() and model.test() are not run consecutively. As an alternative, run model.test(*args, verbose = True).
        
        :param mode: The metrics to be printed, defined by 'train', 'test', or 'both'
        :type mode: str 
        """
        
        if mode.lower() not in ['train', 'test', 'both']:
            raise ValueError("mode must be type str, of 'both', 'train', or 'test'!") 
        
        if mode.lower in ['both', 'test']: 
            if not hasattr(self, 'test_loss'):
                raise ValueError("You haven't tested the model yet!")
            elif not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!") 
  
        if mode.lower() == 'both':
            print(f"Train loss: {self.train_loss} | Train accuracy: {self.train_acc}%\nTest loss: {self.test_loss} | Test accuracy: {self.test_acc}%\n")
        elif mode.lower() == 'test':
                print(f"Test loss: {self.test_loss} | Test accuracy: {self.test_acc}%\n") 
        
        elif mode.lower() == 'train':
            if not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!")
            print(f"Train loss: {self.train_loss} | Train accuracy: {self.train_acc}%\n") 

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
    def hidden_size(self):
        return self._hidden_size
   
    @hidden_size.setter
    def hidden_size(self, hidden_size):
        self._hidden_size = hidden_size 

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
        