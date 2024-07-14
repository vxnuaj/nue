import numpy as np

class KNN():

    def __init__(self, K = 10, modality = 'brute', distance_metric:float = 2):

        '''
        Initialize the KNN. Currently works only for numerical labels.
      
        :param K: The K amount of neighbors to parse, when predicting classes.
        :type K: int 
       
        :param modality: The modality of the KNN, of either 'brute', 'kd-tree', or 'ball-tree'. Currently, only the 'brute' method is in production and that will be the only which works.
        :type modality: str 
        
        :param distance_metric: The order of the LP norm as a distance metric. Default is 2, as the L2 norm. Can be asssigned as a quasi-norm.
        :type distance_metric: int 
       
        :param feature_dim: The index of the feature dimension to take the median of, when computing the 'kd-tree' algorithm 
        :type feature_dim: int
        '''
        self.K = K
        self.modality = modality
        self.distance_metric = distance_metric 
        self.X_train = None
        self.dist_X_test = None
       
    def accuracy(self, Y_test, predictions):
        
        '''
        Compute the accuracy of the model
        
        :param Y_test: The labels of the testing data
        :type Y_test: numpy.ndarray
        
        :param predictions: The predictions of the KNN
        :type predictions: numpy.ndarray 
        
        ''' 
        acc = np.sum(Y_test == predictions) / Y_test.size * 100
        return acc
        
    def fit(self, X_train, Y_train):
      
        '''
        Fit the KNN.
        
        :param X_train: The training data for the KNN of shape (samples, features)
        :type X_train: numpy.ndarray 
        
        :param Y_train: The training labels corresponding to X_train
        :type Y_train: numpy.ndarray
        '''
       
        self.Y_train = Y_train 
        self.X_train = X_train
        
        print(f"Finished fitting\n") 
        return 
      
    def _predict_brute(self, testing_size, K, verbose):

        if testing_size == None:
            testing_size = self.X_test.shape[0]
            
        self.X_test = self.X_test[:testing_size, :]
        self.Y_test = self.Y_test[:testing_size]
        
        if K > self.X_train.shape[0]:
            raise ValueError("K must not be greater than the number of samples in the training set!")
   
        self.predictions = np.empty(self.X_test.shape[0], dtype = self.Y_test.dtype)

        for index, test_row in enumerate(self.X_test): 
        
            if verbose == True:
                print(f"Sample: {index}")
            
            distances = np.linalg.norm(self.X_train - test_row, ord = self.distance_metric, axis = 1) # Takes the difference in L2 size of X_train and the testing data
            nearest_k_index = np.argsort(distances)[:K] # Yields the indices of the lowest distances of X_train up to the kth value 
            nearest_k_labels = self.Y_train[nearest_k_index] # Indexes Y_train, gets the labels of the corresponding values the lowest distance based on their indices as previously assigned
           
            if len(np.unique(nearest_k_labels)) == 1:  # All labels are the same
                self.predictions[index] = nearest_k_labels[0]
            else:
                # Handle tie-breaking by distance
                max_label = np.argmax(np.bincount(nearest_k_labels.astype(int))) # Gets the most occuring label in nearest_k_labels
                max_counts = np.bincount(nearest_k_labels.astype(int))[max_label] # Gets how many times the maximum label appeared.

                if np.sum(np.bincount(nearest_k_labels.astype(int)) == max_counts) > 1: # Checks if there are other labels that appear the similar amount to the previous label of max_counts.
                    # Compute distances for the tied labels
                    tied_indices = np.where(np.bincount(nearest_k_labels.astype(int)) == max_counts)[0] # Returns the index of the label that is tied with the other maximum label, baesd on it's frequency.
                    tied_distances = distances[tied_indices] # Returns distance values of the tied datapoints, based on their index.

                    min_distance_index = np.argmin(tied_distances) # Returns the index of the tied datapoint that is lowest distance from the input.
                    self.predictions[index] = nearest_k_labels[tied_indices[min_distance_index]] # Selects the label of the value with the lowest distance and adds it to self.predictions
                else:
                    self.predictions[index] = max_label     
        
     
    def predict(self, X_test, Y_test, testing_size = 10, verbose = False):
        '''
        Inference of the KNN. Assumes that each row of `X_train` represents a sample.
      
        :param X_test: Input data of shape (samples, features)
        :type X_test: numpy.ndarray 
        
        :param Y_test: Input labels of the data
        :type Y_test: numpy.ndarray
        
        :param K: The amount of K-nearest neighbors to consider
        :type K: int 
        
        :param verbose: The verbosity of the output, if you want accuracy metrics printed, set to verbose = True
        :type verbose: bool
        '''     

        self.X_test = X_test
        self.Y_test = Y_test
        
        if self.modality.lower() == 'brute':
            self._predict_brute(testing_size, self.K, verbose) 

        elif self.modality.lower() == '':
            self._predict_kd_tree()
            
        print(f"\nFinished testing\n")

        if verbose and Y_test is not None and Y_test.any():
            print(f"Accuracy: {self.accuracy(self.Y_test, self.predictions)}%")
        
        return self.predictions
   
    def metrics(self):
        '''
        Prints the metrics of the KNN. Can only be called if the function, self.predict has been succesfully run. Alternative is self.predict(args, ... , verbose = True) 
        '''
        
        if not self.predictions: 
            raise ValueError("model has not yet been tested! run self.predict after running self.fit!")
        print(self.accuracy(self.Y_test, self.predictions))
    
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
        
    def __str__(self):
        return f"K: {self.K}\nModality: {self.modality}\nDistance Metric: {self.distance_metric}\n"
        