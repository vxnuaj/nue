import numpy as np
from nue.metrics import knn_accuracy
from nue.calibration import _PlattScaling

class KNN():

    '''
    Initialize the KNN. Currently works only for numerical labels.
    
    :param verbose_test: Set the verbosity during testing
    :type verbose_test: bool
    ''' 


    def __init__(self, verbose_test = False):
        self.verbose_test = verbose_test
       
    def train(self, X_train, Y_train, K = 10, modality = 'brute', distance_metric:float = 2):
      
        '''
        Train the KNN (well not really, but you know...).
        
        :param X_train: The training data for the KNN of shape (samples, features)
        :type X_train: numpy.ndarray 
        
        :param Y_train: The training labels corresponding to X_train
        :type Y_train: numpy.ndarray
        
        :param K: The K amount of neighbors to parse, when predicting classes.
        :type K: int 
       
        :param modality: The modality of the KNN, of either 'brute', 'kd-tree', or 'ball-tree'. Currently, only the 'brute' method is in production and that will be the only which works.
        :type modality: str 
        
        :param distance_metric: The order of the LP norm as a distance metric. Default is 2, as the L2 norm. Can be asssigned as a quasi-norm.
        :type distance_metric: int 
       
        :param feature_dim: The index of the feature dimension to take the median of, when computing the 'kd-tree' algorithm 
        :type feature_dim: int
        '''
        
        print(f"Model Training!")
       
        self.Y_train = Y_train 
        self.X_train = X_train
      
        self.K = K
        self.modality = modality
        self.distance_metric = distance_metric  
        
        print(f"Finished Training!") 
      
    def test(self, X_test, Y_test, testing_size = 10, return_probs = False, platt_kwargs = {}):
        '''
        Inference of the KNN. Assumes that each row of `X_train` represents a sample.
      
        :param X_test: Input data of shape (samples, features)
        :type X_test: numpy.ndarray 
        :param Y_test: Input labels of the data
        :type Y_test: numpy.ndarray
        :param testing_size: The sample size for the test set
        :type testing_size: int 
        :param return_probs: Experimenatal. If True, return probabilities of the SVM via Platt's method. If False, only return loss and predictions. 
        :type return_probs: bool  
        :param platt_kwargs: The key value arguments for the _PlattScaling() class, of __init__, platt_train, and platt_test, in the form of a dictionary. If not supplied, _PlattScaling() class is initialized with default values. Supplying the platt_kwargs dict is extremely recommended. See the _PlattScaling class for more details.
        :type platt_kwargs: dict 
        '''     

        self.X_test = X_test
        self.Y_test = Y_test
        self.return_probs = return_probs
        self.platt_kwargs = platt_kwargs
        self.testing_size = testing_size 
        
        if self.modality.lower() == 'brute':
            self._predict_brute() 

        '''elif self.modality.lower() == '':
            self._predict_kd_tree()'''
            
        print(f"KNN Testing!")
       
        self.test_acc = knn_accuracy(self.Y_test.flatten(), self.predictions) 
           
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
            
            print(f"KNN Finished Testing")

            if self.verbose_test:
                print(f"KNN Test Accurascy: {self.test_acc}")
                print(f"KNN Test Probabilities: {self.probs.flatten()}")

            return self.test_acc, self.predictions, self.probs.flatten()

        print(f"KNN Finished Testing")

        if self.verbose_test:
            print(f"KNN Accuracy: {self.test_acc}%")
       
        return self.test_acc, self.predictions 
      
    def _predict_brute(self):
       
        '''
        Predict classes using the KNN model.
        
        :param testing_size: The sample size for the test set
        :type testing_size: int
        :param K: The nearest kth neighbors.
        :type K: int
        
        '''
        
        if isinstance(self.testing_size, str) and self.testing_size.lower() == 'all':
            self.testing_size = self.X_test.shape[0]
        elif self.testing_size > self.X_test.shape[0]:
            self.testing_size = self.X_test.shape[0]
            
        self.X_test = self.X_test[:self.testing_size, :]
        self.Y_test = self.Y_test[:self.testing_size]
        
        if self.K > self.X_train.shape[0]:
            raise ValueError("K must not be greater than the number of samples in the training set!")
   
        self.predictions = np.empty(self.X_test.shape[0], dtype = self.Y_test.dtype)
        self.raw_probs = []

        class_label_1 = 1

        for index, test_row in enumerate(self.X_test): 
        
            '''
            if self.verbose_test == True:
                print(f"Sample: {index}")'''
            
            self.distances = np.linalg.norm(self.X_train - test_row, ord = self.distance_metric, axis = 1) # Takes the difference in L2 size of X_train and the testing data
            self.nearest_k_distances = np.sort(self.distances)[:self.K].reshape(-1, 1) # Returns the nearest K distances. 
            self.nearest_k_index = np.argsort(self.distances)[:self.K] # Yields the indices of the lowest distances of X_train up to the kth value 
            self.nearest_k_labels = self.Y_train[self.nearest_k_index] # Indexes Y_train, gets the labels of the corresponding values the lowest distance based on their indices as previously assigned
        
            unique_labels, label_freq = np.unique(self.nearest_k_labels, return_counts=True)
    
            prob = label_freq / self.nearest_k_labels.size
            
            if class_label_1 in unique_labels:
                class_1_prob = float(prob[np.where(unique_labels == class_label_1)[0][0]])
            else:
                class_1_prob = 0.0  
  
            self.raw_probs.append(class_1_prob)          
           
            if len(np.unique(self.nearest_k_labels)) == 1:  # All labels are the same
                self.predictions[index] = self.nearest_k_labels[0]
            else:
                # Handle tie-breaking by distance
                labels, counts = np.unique(self.nearest_k_labels, return_counts=True)  # Get unique labels and their counts
                max_count_indices = np.where(counts == np.max(counts))[0]  # Indices of labels with the highest count

                if len(max_count_indices) > 1:  # Tie in the counts
                    tied_labels = labels[max_count_indices]  # Labels that are tied
                    tied_indices = np.isin(self.nearest_k_labels, tied_labels)  # Indices in nearest_k_labels that are tied
                    tied_distances =self.distances[self.nearest_k_index][tied_indices]  # Distances of the tied labels

                    min_distance_index = np.argmin(tied_distances)  # Index of the tied label with the smallest distance
                    self.predictions[index] = self.nearest_k_labels[tied_indices][min_distance_index]  # Select the label with the smallest distance
                else:
                    self.predictions[index] = labels[max_count_indices[0]]  # Select the most frequent label
 
        print('probs', self.raw_probs) # GOT THE PROBS!
  
    def _train_platt_model(self):
        
        self.platt_Y_train = self.platt_kwargs.get('Y_train', self.nearest_k_labels)[:self.testing_size]
        self.platt_alpha = self.platt_kwargs.get('alpha', .01) 
        self.platt_epochs = self.platt_kwargs.get('epochs', 1000) 
        self.platt_metric_freq = self.platt_kwargs.get('metric_freq', None) 
       
        self.probs, _, self.platt_train_loss, self.platt_train_acc = self.platt_model.platt_train(model_output = self.nearest_k_distances.T, Y_train = self.platt_Y_train, alpha = self.platt_alpha, epochs = self.platt_epochs, metric_freq = self.platt_metric_freq) 
            
    def _test_platt_model(self):
        self.platt_X_test = self.platt_kwargs.get('X_test', self.X_test)
        self.platt_Y_test = self.platt_kwargs.get('Y_test', self.Y_test)
        self.probs, _, self.platt_test_loss, self.platt_test_acc = self.platt_model.platt_inf(model_output = self.nearest_k_distances.T, Y_inf = self.platt_Y_test)
        
    def metrics(self):
        '''
        Prints the metrics of the KNN. Can only be called if the function, self.predict has been succesfully run. Alternative is self.predict(args, ... , verbose = True) 
        '''
        
        if not self.predictions: 
            raise ValueError("model has not yet been tested! run self.predict after running self.fit!")
        print(knn_accuracy(self.Y_test, self.predictions))
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, K):
        if not isinstance(K, int):
            raise ValueError("K must be type int!")
        elif K < 1: 
            raise ValueError("K must be 1 or greater!")
        self._K = K
        
    @property
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
      
        if not isinstance(modality, str):
            raise ValueError("modality must be type str!")
        elif modality in ['kd-tree, ball-tree']:
            raise NotImplementedError("the kd-tree and ball-tree algorithsm weren't implemented yet! try the default brute algorithm!")
       
        self._modality = modality
        
    @property
    def distance_metric(self):
        return self._distance_metric
    
    @distance_metric.setter
    def distance_metric(self, distance_metric):
        if not isinstance(distance_metric, (float, int)):
            raise ValueError("the distance_metric must be type float to take the LP norm as a distance metric!")
        elif distance_metric < 0:
            raise ValueError("the distance_metric can't be less than 0!")
        self._distance_metric = distance_metric
        