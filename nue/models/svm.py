import numpy as np
from nue.metrics import svm_hinge_loss, svm_accuracy
from nue.models import LogisticRegression
from nue.calibration import _PlattScaling

class SVM():
   
    '''
    Initialize the Support Vector Machine 
    
    :param seed: Set the random seed for initializing parameter, based on numpy.random.default_rng(). 
    :type seed: int
    :param verbose_train: Set the verbosity of the model during training
    :type bool:
    :param verbose_test: Set the verbosity of the model during testing
    :type bool:
    ''' 
    
    def __init__(self, seed:int = None, verbose_test = False, verbose_train = False):
        self.seed = seed
        self.verbose_test = verbose_test 
        self.verbose_train = verbose_train
        
    def train(self, X_train:np.ndarray, Y_train:np.ndarray, modality = 'soft', C = .01, alpha:float = .0001, epochs:int = 250, metric_freq:int = 1 ):
        '''
        Train the SVM
         
        :param X_train: The input data of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The target labels of shape (samples, 1) or (samples, ). Binary, labels must be -1 or 1.
        :type Y_train: numpy.ndarray
        :param seed: Set the random seed for initializing parameters. Based on numpy.random.default_rng() 
        :type seed: int
        :param modality: Set the modality for the SVM. Choose between soft and hard, for soft-margin and hard-margin respectively. 
        :type modality: str 
        :param C: Set the regularization strength when using modality as 'soft'
        :type C: float
        :param alpha: The learning rate for gradient descent
        :type alpha: float 
        :param epochs: The number of epochs for training
        :type epochs: int

        :param metric_freq: Will not apply if verbose is set to False. 
      
            Will print out epoch and loss at the epoch frequency set by metric_freq 
        
        :type metric_freq: int

        :return: list containing the final weights (w) and bias (b).
        :rtype: list
        '''
       
        self.X_train = X_train
        self.Y_train = Y_train
        self.modality = modality 
        self.C = C
        self.alpha = alpha
        self.epochs = epochs 
        self.metric_freq = metric_freq
        
        self.__num_features = X_train.shape[1]
        
        self.__params = self._init_params()
        self.train_loss, self.train_acc, self.__params = self._gradient_descent()

        return self.train_loss, self.train_acc, self.__params

    def test(self, X_test:np.ndarray, Y_test:np.ndarray, return_probs:bool = False, platt_kwargs:dict = {}): # TODO -- add ability to use holdout set for platt scaling.
        '''
        Test the SVM
       
        :param X_test: The validation features, shape (samples, feature).
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (samples, 1) or (samples, ). If binary, labels must be -1 or 1.
        :type Y_test: numpy.ndarray
        :param return_probs: Experimenatal. If True, return probabilities of the SVM via Platt's method. If False, only return loss and predictions. 
        :type return_probs: bool 
        :param platt_kwargs: The key value arguments for the _PlattScaling() class, of __init__, platt_train, and platt_test, in the form of a dictionary. If not supplied, _PlattScaling() class is initialized with default values. Supplying the platt_kwargs dict is extremely recommended. See the _PlattScaling class for more details.
        :type platt_kwargs: dict 
        
        :return self.test_loss: The SVM's testing loss
        :rtype self.test_loss: float
        :return self.test_acc: The SVM's testing accuracy
        :rtype self.tests_acc: float
        :return self.predictions: The predictions of the SVM
        :rtype self.predictions: float or int (not sure lol)
        :return raw_output: Only returns if return_probs is True. Returns the raw outputs of the SVM without applying the 
        :rtype raw_output: numpy.ndarray
        '''        

        self.X_test = X_test
        self.Y_test = Y_test
        self.return_probs = return_probs
        self.platt_kwargs = platt_kwargs
       
        print("SVM Testing!\n")
        w, b = self.__params
        self.raw_output = np.dot(w, X_test.T) + b
        self.test_loss = svm_hinge_loss(self.predictions, self.Y_test.T, self.__params, modality = self.modality, C = self.C)
        self.test_acc, _ = svm_accuracy(self.Y_test.T, self.raw_output)
        self.predictions = np.sign(self.raw_output) 
        
        if self.return_probs:
            try: 
                seed = self.platt_kwargs.get('seed', 1) 
                verbose_train = self.platt_kwargs.get('verbose_train', True) 
                verbose_test = self.platt_kwargs.get('verbose_test', True) 
            except:
                seed = 1
                verbose_train = False
                verbose_test = False 
            
            self.platt_model = _PlattScaling(seed = seed, verbose_train = verbose_train, verbose_test = verbose_test)
            self._train_platt_model()
       
            print(f"SVM Tested!") 
 
            if self.verbose_test:
                print(f"SVM Test Loss: {self.test_loss}")
                print(f"SVM Test Accuracy: {self.test_acc}%")
                print(f"SVM Test Probabilities:\n\n{self.probs.flatten()}") 
        
            return self.test_loss, self.test_acc, self.predictions, self.probs.flatten()

        print(f"SVM Tested!")

        if self.verbose_test:
                print(f"SVM test loss: {self.test_loss}")
                print(f"SVM test accuracy: {self.test_acc}%\n")
 
        return self.test_loss, self.test_acc, self.predictions  

    def inference(self, X_inf:np.ndarray, Y_inf:np.ndarray, return_raw_score = False):
        '''
        Inference of the SVM
        
        :param X_inf: The validation features, shape (samples, feature)
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (samples, 1) or (samples, ). If binary, labels must be -1 or 1.
        :type Y_test: numpy.ndarray
        :param verbose: If true, will print out loss and r2 score post-test.
        :type verbose: bool 
        
        :return self.pred: The predictions of the SVM
        :rtype self.pred: numpy.ndarray
        :return self.output: The raw outputs of the SVM
        :rtype self.output: numpy.ndarray
        ''' 

        self.X_inf = X_inf
        self.Y_inf = Y_inf
      
        w, b = self.__params
        self.output = np.dot(w, X_inf.T) + b
        self.inf_loss = svm_hinge_loss(self.output, self.Y_inf.T, self.__params, modality = 'hard')
        self.inf_acc = svm_accuracy(self.Y_inf.T, self.output)
        self.pred = np.maximum(self.output, 0)        

        if return_raw_score:
            return self.pred, self.output
        return self.pred
    
    def _init_params(self):
        """
        Initialize the parameters (weights and bias) for the Support Vector Machine, based on the chosen seed in the init method.
        
        :return: Tuple contianing the weights (w) and bias (b) 
        :rtype: tuple
        """     
        if self.seed == None:
            w  = np.random.rand(1, self.__num_features)
            b = np.zeros((1,1))
            self.__params = [w, b] 
        else:
            rng = np.random.default_rng(seed = self.seed)
            w = rng.normal(size = (1, self.__num_features))
            b = np.zeros((1, 1))
            self.__params = [w, b] 
        
        return self.__params

    def _forward(self):
        '''
        Compute the forward pass to compute the predicted probabilities
        
        :return: The predicted probabilities, or the functional margin
        :rtype: numpy.ndarray 
        '''
        w, b = self.__params

        self.predictions = np.dot(w, self.X_train.T) + b
        
        return self.predictions

    def _backward(self):
        '''
        Compute the gradients for the parameters (w and b) with a backward pass.

        :return: List containing the gradients of the weights (dw) and bias (db)
        :rtype: list
        ''' 
      
        if self.modality == 'soft':  
            dz = self.C * np.maximum(0, -self.Y_train.T) 
            dw = np.dot(dz, self.X_train) / self.Y_train.size
            db = np.sum(dz) / self.Y_train.size
            self.__gradients = [dw, db] 
        else:
            dz = np.maximum(0, -self.Y_train.T) 
            dw = np.dot(dz, self.X_train) / self.Y_train.size
            db = np.sum(dz) / self.Y_train.size
            self.__gradients = [dw, db] 
        return self.__gradients
    
    def _update(self):
        '''
        Update the weights and bias using gradient descent.
        
        :return: list contianing the update weights (w) and bias (b) 
        :rtype: list 
        '''

        dw, db = self.__gradients
        w, b = self.__params
        
        w -= self.alpha * dw
        b -= self.alpha * db
        
        self.__params = [w, b]

        return self.__params
    
    def support_vector(self):
        '''
        Identify the support vectors of the given SVM based on their functional margin and comute their geometric margin.
        '''

        k = 6
        w, _ = self.__params 

        self._functional_margin = (self.Y_train.T * self.predictions).flatten()
        min_indices = np.argpartition(self._functional_margin, k)[:k]
        self.support_vectors = self.X_train[min_indices, :]
        weight_norm = np.linalg.norm(w, ord = 2, axis = 1)
        if weight_norm == 0:
            raise ValueError("The norm of the weight vector is zero, cannot compute geometric margin.")

        self.geometric_margins = self._functional_margin[min_indices] / weight_norm
        return self.support_vectors, self.geometric_margins
        
    def _gradient_descent(self): 
        '''
        Perform gradient descent to train the logistic regression model
        
        :return: List containing the final weights (w) and bias (b).
        :rtype: List
        '''

        print(f"SVM Training!")
        
        for epoch in range(self.epochs):
            self.predictions = self._forward()
            
            self.train_loss = svm_hinge_loss(self.predictions, self.Y_train.T, self.__params, self.modality, self.C)
            self.train_acc, _ = svm_accuracy(self.Y_train.T, self.predictions)

            self.__gradients = self._backward()
            self.__params = self._update()
            
            if self.verbose_train: 
                if epoch % self.metric_freq == 0:
                    print(f"Epoch: {epoch}")
                    print(f"Loss {self.train_loss}")
                    print(f"Accuracy: {self.train_acc}%\n")
                
        if self.verbose_train:
            print(f"Final Training Loss: {self.train_loss}")
            print(f"Final Training Accuracy: {self.train_acc}%\n")

        self.weights, self.bias = [i for i in self.__params]
        self.support_vectors = self.support_vector()

        print(f"SVM Finished Training!")

        return self.train_loss, self.train_acc, self.__params

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

    def _train_platt_model(self):
      
        self.platt_Y_train = self.platt_kwargs.get('Y_train', self.Y_train)
        self.platt_alpha = self.platt_kwargs.get('alpha', .0001) 
        self.platt_epochs = self.platt_kwargs.get('epochs', 1000) 
        self.platt_metric_freq = self.platt_kwargs.get('metric_freq', None) 
        self.probs, _, self.platt_train_loss, self.platt_train_acc = self.platt_model.platt_train(model_output = self.raw_output, Y_train = self.platt_Y_train, alpha = self.platt_alpha, epochs = self.platt_epochs, metric_freq = self.platt_metric_freq) 
            
    def _test_platt_model(self):
      
        self.platt_X_test = self.platt_kwargs.get('X_test', self.X_test)
        self.platt_Y_test = self.platt_kwargs.get('Y_test', self.Y_test)
        self.probs, _, self.platt_test_loss, self.platt_test_acc = self.platt_model.platt_inf(model_output = self.raw_output.T, Y_inf = self.platt_Y_test)
        
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
       
        if np.any(Y_train == 0):
            Y_train = np.where(Y_train == 0, -1, 1)
         
        self._Y_train = Y_train
   
    @property
    def X_test(self):
        return self._X_test
   
    
    @X_test.setter
    def X_test(self, X_test):
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_train must be type numpy.ndarray!")
        self._X_test = X_test
    
    @property
    def Y_test(self):
        return self._Y_test 
       
    @Y_test.setter
    def Y_test(self, Y_test):
        if not isinstance(Y_test, np.ndarray):
            raise ValueError("Y_train must be type numpy.ndarray!") 

        if np.any(Y_test == 0):
            Y_test = np.where(Y_test == 0, -1, 1)
        
        self._Y_test = Y_test
   
    @property
    def X_inf(self):
        return self._X_inf
    
    @X_inf.setter
    def X_inf(self, X_inf):
        if not isinstance(X_inf, np.ndarray):
            raise ValueError("X_train must be type numpy.ndarray!")
        self._X_inf = X_inf
    
    @property
    def Y_inf(self):
        return self._Y_inf  
       
    @Y_inf.setter
    def Y_inf(self, Y_inf):
        if not isinstance(Y_inf, np.ndarray):
            raise ValueError("Y_train must be type numpy.ndarray!") 
        
        if np.any(Y_inf == 0):
            Y_inf = np.where(Y_inf == 0, -1, 1)
        self._Y_inf = Y_inf
   
    @property
    def alpha(self):
        return self._alpha 
   
    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise ValueError("alpha must be type float or int!")
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
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
        if modality.lower() not in ['soft', 'hard']: 
            raise ValueError("modality must be 'soft' or 'hard'!")
        self._modality = modality
        
    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self, C):
        if self.modality == 'hard':
            self._C = None
        else:
            self._C = C
   
    @property
    def platt_train_verbose(self):
        return self._platt_train_verbose
    
    @platt_train_verbose.setter
    def platt_train_verbose(self, platt_train_verbose):
        assert isinstance(platt_train_verbose, bool), 'platt_train_verbose must be type bool.' 
        self._platt_train_verbose = platt_train_verbose
        
    @property
    def platt_test_verbose(self):
        return self._platt_test_verbose
    
    @platt_test_verbose.setter
    def platt_test_verbose(self, platt_test_verbose):
        assert isinstance(platt_test_verbose, bool), 'platt_test_verbose must be type bool.' 
        self._platt_test_verbose = platt_test_verbose