import numpy as np

class Node():
    def __init__(self, feature = None, threshold = None, left_node = None, right_node = None, value = None, depth = None):
        '''
        Initialize the Node of a deecision Tree 
       
        :param feature: The feature index of a given node, to optimally split
        :type feature: int
        :param threshold: The threshold index, to optimally split a Node based on the optimal `feature` 
        :type threshold: int
        :param left_node: The left node of the current node in the tree
        :type left_node: Node object
        :param right_node: The right node of the current node in the tree
        :type right_node: Node object 
        :param value: The final output value of a leaf node, only applicable of a given node is a leaf node 
        
        '''
        self.feature = feature  
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.value = value 
        self.depth = depth
         
    def _is_leaf(self):
        return self.value is not None # returns False if self.value is None, otherwise returns True
        
    @property
    def feature(self):
        return self._feature
    
    @feature.setter
    def feature(self, feature): 
        if not isinstance(feature, (int, type(None))):
            raise ValueError('feature must be type int or None!')
        self._feature = feature   
  
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, threshold):
        if not isinstance(threshold, (int, type(None))):
            raise ValueError('threshold must be type int or None!')
        self._threshold = threshold
   
    @property
    def left_node(self):
        return self._left_node
    
    @left_node.setter
    def left_node(self, left_node):
        if not isinstance(left_node, (Node, type(None))):
            raise ValueError("left_node must be a Node object or None!")
        self._left_node = left_node 
        
    @property
    def right_node(self):
        return self._right_node
    
    @right_node.setter
    def right_node(self, right_node):
        if not isinstance(right_node, (Node, type(None))):
            raise ValueError("right_node must be a Node object or None!") 
        self._right_node = right_node
   
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        if not isinstance(value, (float, int, type(None))):
           raise ValueError("value must be type int or float or None!") 
        self._value = value 
        
class DecisionTree():
    def __init__(self, max_depth, min_node_samples = 1):
        self.max_depth = max_depth
        self.min_node_samples = min_node_samples
        self.root = None
       
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.root = self._grow_tree(self.X_train, self.Y_train)
        
    def _grow_tree(self, X, Y, depth = 0):
        
        n_features, n_samples = X.shape 
        n_labels = len(np.unique(Y))
        
        # checking stopping criteria
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_node_samples):
            leaf_value = int(np.argmax(np.bincount(Y.flatten()))) # Gets the most common label if there are multiple classes or the only label if there is only one class in the leaf node
            return Node(value = leaf_value)
           
        best_feat, best_thresh = self._best_split(X, Y)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
      
        depth += 1 
        print(f"Tree Depth: {depth}")

        left_node = self._grow_tree(X[left_idxs, :], Y[left_idxs], depth) 
        right_node = self._grow_tree(X[right_idxs, :], Y[right_idxs], depth)
        
        return Node(feature = int(best_feat), threshold = int(best_thresh), left_node = left_node, right_node = right_node, depth = depth) 
            
    def _best_split(self, X, Y):
     
        n_samples, n_features = X.shape 
        best_gain = - 1 # setting the best_gain to -1, prior to growing the tree 
      
       
        for feat in range(n_features):
            thresholds = X[:, feat]
            X_col = X[:, feat]

            for thresh in thresholds: 
                information_gain = self._information_gain(X_col, Y, thresh)

                if information_gain > best_gain: # setting new information gain based on the best information gain thus far.
                    best_gain = information_gain
                    best_feat = feat
                    best_thresh = thresh 

        return best_feat, best_thresh 
        
         
    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col <= thresh).flatten()   
        right_idxs = np.argwhere(X_col > thresh).flatten() 
        return left_idxs, right_idxs 
       
    def _information_gain(self, X_col, Y, thresh):
       
        parent_entropy = self._entropy(Y)
        
        left_idxs, right_idxs = self._split(X_col=X_col, thresh = thresh)

        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)
      
        left_entropy = self._entropy(Y[left_idxs])
        right_entropy = self._entropy(Y[right_idxs])
       
        weighted_entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy
       
        information_gain = parent_entropy - weighted_entropy
       
        return information_gain 
        
    def _entropy(self, Y):

        eps = 1e-10

        freqs = np.bincount(Y.flatten())
        prob = freqs / Y.size
        entropy = - np.sum(prob * np.log(prob + eps)) 
        
        return entropy 
   
    def _accuracy(self, Y, pred):
        print(pred)
        print(Y)
        acc = np.sum(Y == pred) / Y.size * 100
        return acc
    
    def predict(self, X_test, Y_test = None, verbose = None):
        self.X_test = X_test
        self.Y_test = Y_test
        self.verbose = verbose
        
        pred = np.array([self._traverse(x) for x in X_test])

        if verbose:
            print(f"Accuracy: {self._accuracy(Y_test.flatten(), pred)}%")

        return pred

    def _traverse(self, x):
        node = self.root
        while not node._is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left_node
            elif x[node.feature] > node.threshold:
                node = node.right_node
        return node.value
    
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, verbose):
        if not isinstance(verbose, bool):
            raise ValueError('verbose not a boolean!')
        if self.Y_test is None:
            raise ValueError('Y_test must be assinged for verbose = True to work!')    
        self._verbose = verbose
        
    @property
    def Y_test(self):
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, Y_test):
        if not isinstance(Y_test, np.ndarray):
            raise ValueError('Y_test must be numpy.ndarray!')
        self._Y_test = Y_test