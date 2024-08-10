import numpy as np
from nue.metrics import dt_accuracy, entropy, gini

class DecisionTree():
    
    '''
    Initialize the DecisionTree. 

    :param verbose_train: The verbosity of the model output while training
    :type verbose_train: bool :param verbose_Test: The verbosity of the model output while training
    :type verbose_test: bool 
    ''' 
    
    def __init__(self, verbose_train = False, verbose_test = False):
        
        self.verbose_train = verbose_train 
        self.verbose_test = verbose_test
        self.n_leaf = 0

    def train(self, X_train, Y_train, max_depth = 100, min_sample_split = 2, modality = 'entropy', alpha = None):
       
        '''
        Train the Decision Tree.
        
        :param X_train: The training data for the Decision Tree, of shape (samples, feature)
        :type X_train: numpy.ndarray
        :param Y_train: The labels for the corresponding X_train, of shape (samples, ) or (samples, 1)
        :type Y_train: numpy.ndarray
        :param max_depth: The maximum depth allowed in the decision tree.
        :type max_depth: int
        :param min_sample_split: The least amount of samples allowed for a Node to split
        :type min_sample_split: int
        :param modality: The modality for fitting the tree. Entropy is the default. Currently supports 'entropy' or 'gini'  
        :param alpha: The cost complexity parameter, similar to regularization for lin, log, or nn
        :type alpha: float or int
        '''
       
        print(f"Model Training!") 
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.modality = modality
        self.alpha = alpha
        
        self.root = self._grow_tree(self.X_train, self.Y_train)
    
        self._orig_data_test()
    
        if self.verbose_train:
            self.train_metrics() 
     
        print(f"Finished Training!") 
        
        return self.train_acc, self.train_uncertainty
      
    def test(self, X_test, Y_test = None, return_probs = False):
        
        '''
        Predict a label given a set of testing samples and labels
        
        :param X_test: The set of testing samples of shape (samples, features)
        :type X_test: numpy.ndarray 
        :param Y_test: The set of labels correpsonding to X_test of shape (samples, 1) or (samples, )
        :type Y_test: numpy.ndarray
        :param return_probs: Return the probabilities of a given class or classes at a leaf node.
        :type return_probs: bool 
        
        :return pred: The predictions of the decision tree, for x in X_test
        :rtype pred: numpy.ndarray
        :return probs: The probabilities for the final set of classes at the leaf node. Only applicable if `self.return_probs = True`
        :rtype probs: numpy.ndarray
        ''' 
        
        self.X_test = X_test 
        self.Y_test = Y_test
        self.return_probs = return_probs

        if self.return_probs:
           
            pred_and_prob = [self._traverse(x) for x in self.X_test]
            pred, probs = zip(*pred_and_prob)
            pred = np.array(pred)
            
            self.probs = np.array(probs, dtype=object).flatten()
          
            idxs = np.array([i for i, p in enumerate(probs) if len(p) == 1])
            idxs_labels = pred[idxs]
            
            #print(f'IDXS:\n\n{idxs}')
            #print(f'IDXSLABELS:\n\n{idxs_labels}')
            #print(f"Probs:\n\n{self.probs}")

            for i, idx in enumerate(idxs):
                if idxs_labels[i] == 0:
                    self.probs[idx] = np.insert(self.probs[idx], 1, 0)
                elif idxs_labels[i] == 1:
                    self.probs[idx] = np.insert(self.probs[idx], 0, 0) 
       
            self.probs = np.array([p[1:] for p in self.probs]).flatten()
            
        else:
            pred = np.array([self._traverse(x) for x in X_test])

        self.test_acc = dt_accuracy(Y_test.flatten(), pred) 
       
        if self.verbose_test:
            self.test_metrics()
        
        if self.return_probs:
            return self.test_uncertainty, self.test_acc, pred, np.array(probs).flatten()
        else:
            return self.test_uncertainty, self.test_acc, pred
    
    def _grow_tree(self, X, Y, depth = 0):
      
        '''
        Grow the Decision Tree recursively.
        
        :param X: The training data, used to grow the decision tree, of shape (samples, features)
        :type X: numpy.ndarray
        :param Y: The labels for the corresponding X, of shape (samples,) or (samples, 1)
        :type Y: numpy.ndarray 
        :param depth: The current depth of the given Node in the Decision Tree
        :type: int
        
        :return: The root node, holding the left_node(s) and right_node(s) splits as Node.left_node and Node.right_node
        :return type: Node instance
        '''
 
        n_samples, _ = X.shape
        n_labels = len(np.unique(Y))
       
        # Stopping Criteria 
        if (depth == self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            self.n_leaf += 1
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value, Y = Y) 
         
        best_feat, best_thresh = self._best_split(X, Y)
       
        # If there is no best feature or best threshold index, return the Node. Occurs if the information_gain doesn't exceed -1 for the current node.
        # This then only occurs when we introduce the alpha parameter (aka complexity parameter), as a means of regularization
        # See here - https://www.youtube.com/watch?v=Tg2OGohaUTc
        # The alpha parameter adds a penalty to the weighted entropy, thereby when we compute the information entropy as parent_entropy - weighted_entropy, 
        # some of the values for information entropy are negative. The amount of negative values are determiend by the magnitude of alpha.
        # Then, given that, more weights are pruned with no optimal solution the bigger alpha (complexity parameter), reducing overfitting.
        
        if best_feat is None or best_thresh is None:
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value, Y = Y) 
        
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh) 
     
        depth += 1
        
        if self.verbose_train:
            print(f"Tree Depth: {depth}") 
      
        left_node = self._grow_tree(X[left_idxs, :], Y[left_idxs], depth) 
        right_node = self._grow_tree(X[right_idxs, :], Y[right_idxs], depth) 
     
        return Node(left_node = left_node, right_node = right_node, Y = Y, feature = best_feat, threshold = best_thresh) 
        
    def _best_split(self, X, Y):
       
        '''
        Identify the best splits of the node, feature and threshold split, based on the information gain or gini index. 
       
        :param X: The  training data, used to grow the decision tree, of shape (samples, features)
        :type X: numpy.ndarray
        :param Y: The labels for the corresponding X, of shape (samples,) or (samples, 1)
        :type Y: numpy.ndarray 
        
        :return best_feat: The best feature index of X, as a splitting feature index
        :rtype best_feat: int
        :return best_thresh: The best threshold to split the samples X, with the correpsonding feature index.
        :rtype best_thresh: int or float
        '''
        
        n_samples, n_features = X.shape 
        best_gain = -1 
        best_feat = None
        best_thresh = None
         
        for feat in range(n_features):
            X_col = X[:, feat]
            thresholds = np.unique(X[:, feat])

            for thresh in thresholds:
                self.information_gain = self._information_gain(X_col, Y, thresh)

                if self.information_gain > best_gain:
                    best_gain = self.information_gain
                    best_feat = feat
                    best_thresh = thresh
       
        return best_feat, best_thresh

    def _split(self, X_col, thresh):
        
        '''
        Split the optimal feature column of X (X_col) given the threshold (`thresh`)
        
        :param X_col: The feature column of X to split using `thresh`
        :type X_col: numpy.ndarray
        :param thresh: The given threshold value to split X_col
        :type thresh: float or int
        
        :return left_idxs: The sample indices corresponding to samples that will be split to the left_node
        :type left_idxs: int
        :return right_idxs: The sample indices corresponding to samples that will be split to the right_node
        :type right_idxs: int
        ''' 
        
        left_idxs = np.argwhere(X_col < thresh).flatten() # what happens if we don't flatten either?
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs
                
    def _information_gain(self, X_col, Y, thresh):
        
        '''
        Compute the information gain for a given split.
        
        :param X_col: The feature column of X.
        :type X_col: numpy.ndarray 
        :param Y: The set of labels corresponding to samples of X
        :type Y: numpy.ndarray
        :param thresh: The given threshold to split `X_col`
        
        :return: The information gain of the given split
        :rtype information_gain: float or int
        '''         
        left_idxs, right_idxs = self._split(X_col, thresh)  
        
        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)  
    
        if self.modality == 'entropy': 
           
            parent_entropy = entropy(Y) 
            left_entropy, right_entropy = entropy(Y[left_idxs]), entropy(Y[right_idxs]) 
            
            if self.alpha: 
                weighted_entropy = ((n_l / n) * left_entropy + (n_r / n) * right_entropy) + self.alpha * np.abs(self.n_leaf)
            else:
                weighted_entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy
 
            information_gain = parent_entropy - weighted_entropy 
         
        elif self.modality == 'gini':
            parent_gini = gini(Y)
            left_gini, right_gini = gini(Y[left_idxs]), gini(Y[right_idxs]) 
            weighted_gini = (n_l / n) * left_gini + (n_r / n) * right_gini
            information_gain = parent_gini - weighted_gini     
             
        return information_gain
        
    def _most_common_label(self, Y):
        
        '''
        Compute the most common label in a given node, through the given labels (Y) within that node.
         
        :param Y: The labels for a given node
        :type Y: numpy.ndarray 
        
        :return: The most common label in Y
        :rtype: float or int
        ''' 
        unique_labels, counts = np.unique(Y.flatten(), return_counts = True)
        most_common_index = np.argmax(counts)
        return unique_labels[most_common_index]

    def _orig_data_test(self):
        # should also be able to display the uncertainty for the original datset for the leaf node.
        pred = np.array([self._orig_data_traverse(x) for x in self.X_train])
        self.train_acc = dt_accuracy(self.Y_train.flatten(), pred)
        return pred

    def _orig_data_traverse(self, x):
        
        node = self.root
   
        while not node._is_leaf(): 
        
            if x[node.feature] < node.threshold:
                node = node.left_node
            elif x[node.feature] >= node.threshold:
                node = node.right_node
     
        if self.modality == 'entropy':
            self.train_uncertainty = entropy(node.Y)
        elif self.modality == 'gini':
            self.train_uncertainty = gini(node.Y) 
             
        return node.value


    def _traverse(self, x):
        
        '''
        Traverse the Decision Tree
        
        :param x: A single sample (column) of X_test
        :type x: numpy.ndarray
        
        :return: The predicted value, based on a leaf node of the Decision Tree, for a given sample x
        :rtype x: float or int
        '''
        
        node = self.root
   
        while not node._is_leaf(): 
        
            if x[node.feature] < node.threshold:
                node = node.left_node
            elif x[node.feature] >= node.threshold:
                node = node.right_node
      
        if self.return_probs:
            _, freqs = np.unique(node.Y, return_counts=True) 
            probs = np.array(freqs) / len(node.Y) 
            
        if self.modality == 'entropy':
            self.test_uncertainty = entropy(node.Y)
        elif self.modality == 'gini':
            self.test_uncertainty = gini(node.Y) 
       
        if self.return_probs:
            return node.value, probs  
        
        return node.value
      
    def train_metrics(self):
        print(f"\nTotal Leaf Nodes: {self.n_leaf} ☘︎")
        print(f"Training Accuracy: {self.train_acc}%") 
        print(f"Average Model Uncertainty: {self.train_uncertainty}")
       
    def test_metrics(self):
        print(f"\nTotal Leaf Nodes: {self.n_leaf} ☘︎") 
        print(f"Testing Accuracy: {self.test_acc}%")
        print(f"Average Model Uncertainty: {self.test_uncertainty}")
      
        if hasattr(self, 'probs'):
            print(f"Decision Tree Probabilities:\n\n{self.probs}") 
       
    @property
    def X_train(self):
        return self._X_train
    
    @X_train.setter
    def X_train(self, X_train):
        if not isinstance(X_train, (np.ndarray)):
            raise ValueError('X_train must be type numpy.ndarray.')
        self._X_train = X_train
   
    @property
    def Y_train(self):
        return self._Y_train
    
    @Y_train.setter
    def Y_train(self, Y_train):
        if not isinstance(Y_train, (np.ndarray)):
            raise ValueError('Y_train must be type numpy.ndarray.')   
        self._Y_train = Y_train 
      
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (int, float, type(None))):
            raise ValueError('alpha must be type float.')
        self._alpha = alpha 
       
    @property
    def verbose_train(self):
        return self._verbose_train
    
    @verbose_train.setter
    def verbose_train(self, verbose_train):
        if not isinstance(verbose_train, bool):
            raise ValueError('verbose_train must be type bool.')
        self._verbose_train = verbose_train

    @property
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
        if modality.lower() not in ['entropy', 'gini']:
            raise ValueError("modality must be 'entropy or 'gini'.")
        self._modality = modality
        
    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter
    def X_test(self, X_test):
        if not isinstance(X_test, np.ndarray):
            raise ValueError('X_test must be type numpy.ndarray.')
        self._X_test = X_test
        
    @property
    def Y_test(self):
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, Y_test):
        if not isinstance(Y_test, np.ndarray):
            raise ValueError('Y_test must be type numpy.ndarray.')
        self._Y_test = Y_test 
       
    @property
    def verbose_predict(self):
        return self._verbose_predict
   
    @verbose_predict.setter
    def verbose_predict(self, verbose_predict):
        if not isinstance(verbose_predict, bool):
            raise ValueError('verbose must be type bool')
        self._verbose_predict = verbose_predict 
    
    @property
    def return_probs(self):
        return self._return_probs    
   
    @return_probs.setter
    def return_probs(self, return_probs):
        assert isinstance(return_probs, bool), 'return_probs must be type bool'
        self._return_probs = return_probs
     
        
class Node():
    
    '''
    Initializes a Node of a Decision Tree. Primarily for internal use of the `DecisionTree` class. 
   
    :param value: The value of the node, if the Node is a leaf or pure node.
    :type value: float or int 
    :param left_node: The left node of the given Node instance. Recursively grown via the `DecisionTree._grow_tree()`
    :type left_node: Node 
    :param right_node: The right node of the given Node instance. Recursively grown via the `DecisionTree._grow_tree()`.
    :type right_node: Node
    :param feature: The optimal feature index for which to split the samples within the current Node
    :type feature: int
    :param threshold: The optimal threshold value, within the range of the optimal feature column vector, to split the current set of samples within the current Node, into left or right Nodes.
    :type threshold: float or int
    ''' 
    
    def __init__(self, value = None, Y = None, left_node = None, right_node = None, feature = None, threshold = None):

        self.value = value
        self.Y = Y
        self.left_node = left_node
        self.right_node = right_node
        self.feature = feature
        self.threshold = threshold

    def _is_leaf(self):
        
        '''
        Assess if the current Node is a leaf node or not. 
        
        :return: A boolean value, True if self.value isn't None. Otherwise, if it is None, returns False 
        :rtype: bool
        ''' 
        
        return self.value is not None # Returns True if self.value isn't None. Otherwise, if it is None, returns False.


class RandomTree:
    def __init__(self, verbose_train = False):
        self.verbose_train = verbose_train
        self.root = None # the model isn't trained if None
        self.n_leaf = 0 # number of leaf nodes. at init is 0

    def train(self, X_train, Y_train, min_node_samples = 2, max_depth = 100, max_features = 5, criterion = 'gini', alpha = 0): 
        self.X_train = X_train
        self.Y_train = Y_train
        self.min_node_samples = min_node_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.alpha = alpha
        self.root = self._grow_tree(self.X_train, self.Y_train)
    
    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = np.array([self._traverse(x) for x in X_test])
        self.accuracy = self._accuracy()

        print(f"Accuracy: {self.accuracy}%") 

        return self.preds

    def _grow_tree(self, X, Y, depth = 0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))

        # if a node has no value to assign itself to
        if len(Y) == 0: 
            return Node(value = None) 
        
        # stopping criteria
        if (depth == self.max_depth or n_classes == 1 or n_samples < self.min_node_samples):
            leaf_val = self._most_common_label(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y = Y)

        best_thresh, best_feat = self._best_split(X, Y)
        
        # pruning criteria
        if best_thresh is None or best_feat is None:
            leaf_val = self._most_common_label(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y = Y)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        depth += 1

        if self.verbose_train:
            print(f"Tree Depth: {depth}")

        left_node = self._grow_tree(X[left_idxs], Y[left_idxs], depth = depth)
        right_node = self._grow_tree(X[right_idxs], Y[right_idxs], depth = depth)
        return Node(right_node = right_node, left_node = left_node, threshold = best_thresh, feature = best_feat, Y = Y)

    def _most_common_label(self, Y):
        labels, freqs = np.unique(Y.flatten(), return_counts = True)
        most_common_idx = np.argmax(freqs)
        return labels[most_common_idx]

    def _best_split(self, X, Y):
        n_samples, n_features = X.shape
        best_thresh = None
        best_feat = None
        best_gain = -1
        
        feat_idxs = np.array([np.random.randint(low = 0, high = X.shape[1]) for feat in range(self.max_features)])
        
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for thresh_val in thresholds:
                inf_gain = self._inf_gain(X_col, Y, thresh_val)
                if inf_gain > best_gain:
                    best_gain = inf_gain
                    best_thresh = thresh_val
                    best_feat = feat_idx
        
        return best_thresh, best_feat

    def _inf_gain(self, X_col, Y, thresh):
        
        left_idxs, right_idxs = self._split(X_col, thresh)

        # if a left_idxs or right_idxs has no value to split upon, indicates that the given split with len == 0 is the worst possible split to go down.
        # prevents empty nodes with no values from being considered
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
         
        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)

        if self.criterion == 'gini':
            parent_gini = self._gini(Y)
            left_gini, right_gini = self._gini(Y[left_idxs]), self._gini(Y[right_idxs])
            if self.alpha: 
                weighted_gini = ((n_l / n ) * left_gini + (n_r / n) * right_gini) + (self.alpha * np.abs((self.n_leaf)))
            else:
                weighted_gini = (n_l / n ) * left_gini + (n_r / n) * right_gini
            return parent_gini - weighted_gini
            
        elif self.criterion == 'entropy':
            parent_entropy = self._entropy(Y)
            left_ent, right_ent = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])
            if self.alpha: 
                weighted_ent = ((n_l / n ) * left_ent + (n_r / n) * right_ent) + (self.alpha * np.abs((self.n_leaf)))
            else: 
                weighted_ent = (n_l / n ) * left_ent + (n_r / n) * right_ent
            return parent_entropy - weighted_ent

    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col < thresh).flatten()
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, Y):
        _, freqs = np.unique(Y.flatten(), return_counts = True)
        probs = freqs / Y.size
        gini = 1 - np.sum(np.square(probs))
        return gini

    def _entropy(self, Y):
        _, freqs = np.unique(Y.flatten(), return_counts =True)
        probs = freqs / Y.size
        entropy = - np.sum(probs * np.log(probs))
        return entropy

    def _accuracy(self):
        self.acc = np.sum(self.preds.flatten() == self.Y_test.flatten()) / self.Y_test.size * 100
        return self.acc

    def _traverse(self, x):
        node = self.root
        while not node._is_leaf():
            if x[node.feature] >= node.threshold:
                node = node.right_node
            elif x[node.feature] < node.threshold:
                node = node.left_node
        return node.value

class RandomForest:

    def __init__(self, verbose_train = False):
        self.verbose_train = verbose_train

    def train(self, X_train, Y_train, max_features = 5, n_bootstraps = 10, rtree_dict = None, alpha_range = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.max_features = max_features
        self.n_bootstraps = n_bootstraps
        self.rtree_dict = rtree_dict
        self.alpha_range = alpha_range
        self.models = []

        self._get_dicts()
        
        for i in range(n_bootstraps):
            b_idx = self._bootstrap_idx()
            model = RandomTree(**self._init_dict)
            print(f"Training Tree #{i}")
            model.train(X_train[b_idx],Y_train[b_idx], max_features = self.max_features, **self._train_dict)
            self.models.append(model)

        print(f"\nAll {i} Trees have finished Training.\n")

    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = []

        for model in self.models:
            preds = model.test(self.X_test, self.Y_test)
            self.preds.append(preds)
           
        self._get_preds()        
        self._accuracy()
        
        print(f"\nFinal Forest Accuracy: {self.accuracy}")

    def _get_dicts(self):
        self._init_dict = {k:v for k,v in self.rtree_dict.items() if k in ['verbose_train']}
        self._train_dict = {k:v for k,v in self.rtree_dict.items() if k in ['min_node_samples', 'max_depth', 'criterion']}

    def _bootstrap_idx(self):
        n_samples = self.Y_train.size
        b_idx = np.random.randint(low = 0, high = n_samples, size = n_samples)
        return b_idx

    def _get_preds(self):
        self.preds = np.array(self.preds)
        self.preds = np.apply_along_axis(self._most_common_label, axis = 0, arr = self.preds)

    def _most_common_label(self, preds):
        pred, freqs = np.unique(preds, return_counts = True)
        most_common_idx = np.argmax(freqs)
        return pred[most_common_idx]

    def _accuracy(self):
        self.accuracy = np.sum(self.preds.flatten() == self.Y_test.flatten()) / self.Y_test.size * 100
        

    @property
    def max_features(self):
        return self._max_features

    @max_features.setter
    def max_features(self, max_features):
        assert 0 < max_features < self.X_train.shape[1], "max_features can't be or exceed the total number of features in X_train."
        self._max_features = max_features

