'''Assume Binary Classification for now.'''
''' FAILED'''


import numpy as np

class Node():
    '''
    Initialize a Node of the Decision Tree
    
    :param feature: The optimal feature index to split upon
    :type feature: int
    :param threshold: The optimal threshold value of a given feature to split upon
    :type threshold: int
    :param left_node: The left split of the current Node
    :type left_node: Node object
    :param right_node: The right split of the current Node
    :type right_node: Node object
    :param value: The predicted value that a node would yield if it was a leaf node
    :type value: None or the predicted label if Node is a leaf node.
    '''
    def __init__(self, feature = None, threshold = None, left_node=None, right_node=None, value = None):

        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.value = value

    def _leaf(self):
        return self.value is not None 
    
    @property
    def feature(self):
        return self._feature
    
    @feature.setter
    def feature(self, feature):
        if not isinstance(feature, (int, float, type(None))):
            raise ValueError('feature must be type int or float or None!')
        self._feature = feature
        
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, threshold):
        if not isinstance(threshold, (int, type(None))):
            raise ValueError('threshold must be type int!')
        self._threshold = threshold

    @property 
    def left_node(self):
        return self._left_node
    
    @left_node.setter
    def left_node(self, left_node):
        if not isinstance(left_node, (Node, type(None))):
            raise ValueError('left_node must be the Node object!')
        self._left_node = left_node
   
    @property 
    def right_node(self):
        return self._right_node
    
    @right_node.setter
    def right_node(self, right_node):
        if not isinstance(right_node, (Node, type(None))):
            raise ValueError('left_node must be the Node object!') 
        self._right_node = right_node 
        
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        self._value = value 

class DecisionTree():
    
    '''
    Initialize a Decision Tree model
    
    :param max_depth: The maximum depth of the Decision Tree
    :type max_depth: int
    :param min_node_samples: The minimum samples allowed per Node of the tree.
    :type min_node_samples: int 
    ''' 
    
    def __init__(self, max_depth, min_node_samples):
        self.max_depth = max_depth
        self.min_node_samples = min_node_samples
   
    def fit(self, X_train, Y_train):
        '''
        Fit the Decision Tree 
        
        :param X: Training samples of shape (samples, features)
        :type X: numpy.ndarray
        
        :param Y: Corresponding labels to the training samples of shape (samples, 1)
        :type Y: numpy.ndarray
        '''
        
        self.X_train = X_train
        self.Y_train = Y_train
        
        self.total_samples = X_train.shape[0] # Total samples in the X
        self.total_features = X_train.shape[1] # Total features in the X
        self.root = self._grow_tree(X_train, Y_train) # Initializing the recursive algorithm
    
    def _grow_tree(self, X, Y, depth = 0):
       
        '''
        Grow the Decision Tree
        
        :param X: A given set of samples
        :type X: numpy.ndarray 
        :param Y: A given set of labels corresponding to X
        :type Y: numpy.ndarray 
        :param depth: The given depth value for the current node. 0 as default, assigned +1 for every recursive iteration.
        :type depth: int 
        
        :return: The root Node, containing sub Nodes as Node.left_right and Node.right_node
        :rtype: Node object
        '''
       
        n_samples , n_features = X.shape 
        n_labels = len(np.unique(Y))
     
        # Stopping the growth of the Tree via stopping criteria of `self.max_depth or self.min_node_samples` or if a pure node has been identified. 
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_node_samples):
            leaf_value = np.argmax(np.bincount(Y.flatten()))
            return Node(value = leaf_value) 
       
        best_feat, best_thresh = self._best_split(X, Y) # Identifying the best features and best threshold value of the feature to split upon
        print(f"Best split: feature = {best_feat}, threshold = {best_thresh}") 
        
        if best_feat is None or best_thresh is None:
            leaf_value = np.argmax(np.bincount(Y.flatten()))
            print(f"No valid split found. Creating leaf node with value: {leaf_value}")
            return Node(value = leaf_value)
         
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh) # Splitting based on the column of X determined by the best feature and the best threshold
        print(f"Left split size: {len(left_idxs)}, Right split size: {len(right_idxs)}") 
      
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = np.argmax(np.bincount(Y.flatten())) if len(Y) > 0 else 0
            print(f"Creating leaf node due to empty split, value: {leaf_value}") 
            return Node(value=leaf_value)  
       
        depth += 1 
       
        left_node = self._grow_tree(X[left_idxs, :], Y[left_idxs], depth) # Recursively growing the right nodes
        right_node = self._grow_tree(X[right_idxs, :], Y[right_idxs], depth) # Recursively growing the left nodes
       
        print(f"Depth: {depth}, Samples: {n_samples}, Unique labels: {n_labels}")    
        
        return Node(feature = best_feat, threshold = best_thresh, left_node = left_node, right_node=right_node) # Returning the root node, with a recursively grown left and right nodes.
    
    def _best_split(self, X, Y):
        '''
        Identify the best split for a node of the Decision Tree
        
        :param X: A given set of samples
        :type X: numpy.ndarray 
        :param Y: A given set of labels corresponding to X. Must be type int.
        :type Y: numpy.ndarray 
        
        :return split_idx: The optimal feature split index
        :rtype split_idx: int
        :return thresh_idxs: The optimal threshold value split for the given feature
        :rtype thresh_idxs: int
        '''
       
        best_gain = -float('inf')
        split_idx, thresh_idx = None, None  
        
        n_features = X.shape[1] # Getting the total number of features

       
        for feat_idx in range(n_features): # Iterating over the total number of features
            
            X_col = X[:, feat_idx] # In the current iteration, get the column of the current feature index.
            thresholds = np.unique(X_col) # Get a column vector of the possible split threshold values
          
            for thresh in thresholds: # Iterating over the possible threshold values

                left_idxs, right_idxs = self._split(X_col, thresh)
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue 
                
                information_gain = self._information_gain(Y, X_col, thresh) 
                print(f"Feature: {feat_idx}, Threshold: {thresh}, Gain:{information_gain}")
                
            if information_gain > best_gain:
                best_gain = information_gain
                split_idx = feat_idx
                thresh_idx = thresh
        print(f"Best split: feature = {split_idx}, threshold = {thresh_idx}, gain={best_gain}")        
        return split_idx, thresh_idx
   
    
    def _split(self, X_col, thresh):
       
        '''
        Identify the indices of a given column of features of X that are above the optimal threshold value 
        
        :param X_col: The given feature column to split upon, previously identified by split_idx
        :type X_col: numpy.ndarray
        :param thresh: The optimal threshold value to split upon
        :type thresh: int
        
        :return left_idxs: The indices of samples that will be split onto a left node
        :rtype left_idxs: int
        :return right_idxs: THe indices of the sampels thast will be split onto a right node
        :rtype right_idxs: int
        '''
        
        left_idxs = np.argwhere(X_col <= thresh).flatten()
        right_idxs = np.argwhere(X_col > thresh).flatten()
        
        return left_idxs, right_idxs
 
  
    def _entropy(self, Y):
 
        '''
        Compute the entropy for a given set of samples
        
        :param Y: Input labels of shape (samples, 1).
        :type Y: numpy.ndarray 
        
        :return: The entropy for a given set of samples 
        :rtype: float 
        '''
  
        eps = 1e-10
   
        freqs = np.bincount(Y.flatten())  
        prob = freqs / Y.size 
        prob = prob[prob > 0] 
        entropy = float(- np.sum(prob * np.log(prob + eps)))
        
        return entropy
    
    def _information_gain(self, Y, X_col, thresh):

        '''
        Compute the information gain for a given split
        
        :param Y: Input labels of shape (samples, 1), of a parent node
        :type Y: numpy.ndarray
        :param left_node: The left split of a node, given by self._split()
        :type left_node: numpy.ndarray
        :param right_node: The right split of a node, given by self._split()
        :type right_node: numpy.ndarray  
        
        :return: The information gain for a given split
        :rtype: float
        '''
       
        left_idxs, right_idxs = self._split(X_col, thresh=thresh)
      
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
       
       
        n = len(Y)
        n_l, n_r = len(left_idxs), len(right_idxs) 
        
        parent_entropy = self._entropy(Y) 
        left_entropy = self._entropy(Y[left_idxs])
        right_entropy = self._entropy(Y[right_idxs])
       
        
        weighted_entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy
        information_gain = float((parent_entropy - weighted_entropy))
        
        return information_gain    
   
    def accuracy(self, pred, Y):
        acc = np.sum(pred == Y) / Y.size
        return acc
    
    def predict(self, X_test, Y_test = None, verbose = False):
       
        '''
        Traverse the Tree for all samples within X to predict the class
        
        :param X_test: The testing samples
        :type X_test: numpy.ndarray
        :param Verbose: The verbosity of the output. If True, returns accuracy metric
        
        :return: An array of predictions
        :rtype: numpy.ndarray
        ''' 
  
        self.verbose = verbose
        self.X_test = X_test
        self.Y_test = Y_test
         
        pred = np.array([self._tree_traverse(x, self.root) for x in X_test])
        
        print(f"Unique predictions: {np.unique(pred)}")
        print(f"Prediction counts: {np.bincount(pred)}")      
       
        if verbose == True:
            print(f"Accuracy: {self.accuracy(pred, Y_test)}")
        
        return pred        
 
    def _tree_traverse(self, x, node):
        print(f"Traversing node: feature={node.feature}, threshold={node.threshold}, value={node.value}")

        if node.value is not None:    
            return node.value
    
        if x[node.feature] <= node.threshold:
            return self._tree_traverse(x, node.left_nodej)
        return self._tree_traverse(x, node.right_node)
   
    @property
    def max_depth(self):
        return self._max_depth
    
    @max_depth.setter
    def max_depth(self, max_depth):
        if not isinstance(max_depth, int):
            raise ValueError('max_depth must be type int!')
        self._max_depth = max_depth
   
    
    @property
    def min_node_samples(self):
        return self._min_node_samples
    
    @min_node_samples.setter
    def min_node_samples(self, min_node_samples):
        if not isinstance(min_node_samples, int):
            raise ValueError('min_node_samples must be type int!')
        self._min_node_samples = min_node_samples
   
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
            raise ValueError('Y_train must be type numpy.ndarray!')
        self._Y_train = Y_train
  
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, verbose):
        if not isinstance(verbose, bool):
            raise ValueError('verbose must be type bool')
        self._verbose = verbose 

    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter
    def X_test(self, X_test):
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_test must be type numpy.ndarray")
        self._X_test = X_test

    @property
    def Y_test(self):
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, Y_test):
        if self.verbose and Y_test is None:
            raise ValueError('If verbose == True, you must provide a Y_test set of labels!') 
        elif not isinstance(Y_test, np.ndarray):
            raise ValueError("Y_test must be type numpy.ndarray")        
        self._Y_test = Y_test    
    
   