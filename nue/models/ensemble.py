import numpy as np
import pprint
import time
from nue.models import DecisionTree, RandomTree
from nue.metrics import accuracy
from termcolor import colored

# ----- Bagged Trees (bagging) ----- 

class BaggedTrees:
    def __init__(self, verbose_test = False):
        self.verbose_test = verbose_test
        self._preds = []
        
    def train(self, X_train, Y_train, n_bootstrap, dtree_dict, alpha_range:tuple = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_bootstrap = n_bootstrap
        self.dtree_dict = dtree_dict
        self.alpha_range = alpha_range
        self.models = []

        for i in range(n_bootstrap):
            print(f"\nTraining Tree {i}")
            X_bootstrap, Y_bootstrap = self._bootstrap_samples(self.X_train, self.Y_train)
            if self.alpha_range:
                model_init = {k:v for k,v in dtree_dict.items() if k in ['verbose_train', 'verbose_test']}
                model_train = {k:v for k,v in dtree_dict.items() if k in ['modality', 'max_depth', 'min_sample_split']}
                alpha = np.random.uniform(low = alpha_range[0], high = alpha_range[1])
                model = DecisionTree(**model_init)
                model.train(X_bootstrap, Y_bootstrap, alpha = alpha, **model_train)
            else:
                model_init = {k:v for k,v in dtree_dict.items() if k in ['verbose_train', 'verbose_test']}
                model_train = {k:v for k,v in dtree_dict.items() if k in ['max_depth', 'min_sample_split', 'alpha', 'modality']} # instead of drawing alpha, could use a random alpha value for different models in the ensemble, drawn randomly
                model = DecisionTree(**model_init)
                model.train(X_bootstrap, Y_bootstrap, **model_train)

            self.models.append(model)
            
    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
      
        self._preds = self._get_model_pred()
        self.accuracy = self._accuracy(Y_test, self._preds)

        if self.verbose_test:
            print(f"\nFinal BaggedTrees Test Accuracy: {self.accuracy}%")

        return self._preds, self.accuracy

    def _bootstrap_samples(self, X, Y):
        bootstrap_idx = np.random.randint(low = 0, high = Y.size, size = (X.shape[0]))
        X_bootstrap = X[bootstrap_idx]
        Y_bootstrap = Y[bootstrap_idx]
        return X_bootstrap, Y_bootstrap
           
    def _get_model_pred(self):

        all_preds = [ ]

        for model in self.models:
            _, _, pred = model.test(self.X_test, self.Y_test)
            all_preds.append(pred)
        all_preds = np.array(all_preds)

        pred = np.apply_along_axis(self._most_common, axis = 0, arr = all_preds)
        self._preds.append(pred)
        return np.array(self._preds)

    def _most_common(self, all_preds):
        labels, freqs = np.unique(all_preds, return_counts = True)
        most_common_idx = np.argmax(freqs)
        return labels[most_common_idx]

    def _accuracy(self, Y, preds):
        acc = np.sum(Y.flatten() == preds.flatten()) / Y.size * 100
        return acc

# ----- Random Forest ------

class RandomForest:

    def __init__(self, verbose_test = False, n_extremely_randomized_feats = None):
        self.verbose_test = verbose_test
        self.n_extremely_randomized_feats = n_extremely_randomized_feats

    def train(self, X_train, Y_train, max_features = 5, n_bootstraps = 10, rtree_dict = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.max_features = max_features
        self.n_bootstraps = n_bootstraps
        self.rtree_dict = rtree_dict
        self.models = []

        self._get_dicts()
        
        for i in range(n_bootstraps):
            b_idx = self._bootstrap_idx()
            model = RandomTree(n_extremely_randomized_feats = self.n_extremely_randomized_feats, **self._init_dict)
            print(f"Training Tree #{i}")
            model.train(X_train[b_idx],Y_train[b_idx], max_features = self.max_features, **self._train_dict)
            print(model.n_extremely_randomized_feats)
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
       
        if self.verbose_test:
            print(f"\nFinal Forest Accuracy: {self.accuracy}")

        return self.preds, self.accuracy 

    def _get_dicts(self):
        self._init_dict = {k:v for k,v in self.rtree_dict.items() if k in ['verbose_train']}
        self._train_dict = {k:v for k,v in self.rtree_dict.items() if k in ['min_node_samples', 'max_depth', 'criterion', 'alpha']}

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

# ----- majority classifier -----

class MajorityClassifier:
    
    '''
    Initalize a set of Ensemble models.
    
    :param models: The set of models used to run the Voting Classifier. Contains a dicts of any instance of a model under nue.models with the rest of the parameters in the dict corresponding as arguments to train the model within it's respective model.train function. 
    
        The first key:value pair of the dict can be any name for the model with the model instance as the value. No need to include X_train or Y_train params. 
    
    :type models: list
    :param verbose_test: Determines the verbosity of the Ensemble while testing. If true, prints the accuracy and prediction(s) of the model. Does not determine verbosity of the models in the ensemble.
    :type verbose_test: bool 
    ''' 
   
    def __init__(self, models = [], verbose_test = False):
        self.models = models
        self.verbose_test = verbose_test
       
    def train(self, X_train, Y_train):
       
        '''
        Train each model on a training set, X_train  
        
        :param X_train: The training data, of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The labels corresponding to the training data, of shape (samples, 1) or (samples, )
        :type Y_train: numpy.ndarray 
        ''' 
       
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.trained_models = [self._train_models(model) for model in self.models] 
       
    def test(self, X_test, Y_test, voting = 'hard', weights = None):
        
        '''
        Test the model on the testing set, X_test 
        
        :param X_train: The training data, of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The labels corresponding to the training data, of shape (samples, 1) or (samples, )
        :type Y_train: numpy.ndarray  
        :param voting: Choose between 'hard' or 'soft' voting, where 'hard' is based on pure class predictions while 'soft' is based on raw probabilities.
        :type voting: str
        :param weights: A set of weights, used when calculating the highest average probability or most predicted class. Default is simple 1 / n where n is the total number of predictions
        :type weights: numpy.ndarray
        '''
        
        self.X_test = X_test 
        self.Y_test = Y_test
        self.voting = voting
        self.weights = weights
        self.tested_models = [self._test_models(model) for model in self.trained_models]   

        if self.voting == 'hard':
            self.prediction = self._hard_prediction()
        elif self.voting == 'soft':
            self.prediction = self._soft_predictions()
        self.accuracy = accuracy(self.Y_test.flatten(), self.prediction)

        print(colored('\nENSEMBLE RESULTS\n', 'green', attrs = ['bold', 'underline']))
        print(f"Accuracy: {self.accuracy}%")
        
        return self.prediction
        
    def train_test(self):
        '''
        Train and test the models on training and testing sets, X_train and X_test 
        ''' 
           
    def _train_models(self, model):
        model_meta = list(model.items())
        model_id, model_instance = model_meta[0] 
        model_name = type(model_instance).__name__        
        if model_name == 'SVM':
            trained_model = self._train_svm(model)
        elif model_name == 'KNN':
            trained_model = self._train_knn(model)
        elif model_name == 'DecisionTree':
            trained_model = self._train_decision_tree(model)
        elif model_name == 'LogisticRegression':
            trained_model = self._train_logistic_regression(model) 
        return trained_model
      
    def _test_models(self, model): 
        model_meta = list(model.items())
        _, model_instance = model_meta[0] 
        model_name = type(model_instance).__name__ 
       
        if model_name == 'SVM':
            model_args = self._test_svm(model)
        elif model_name == 'KNN':
            model_args = self._test_knn(model)
        elif model_name == 'DecisionTree':
            model_args = self._test_decision_tree(model)
        elif model_name == 'LogisticRegression':
            model_args = self._test_logistic_regression(model) 
        
        return model_args 
        
    def _train_svm(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        train_hyparams = {k:v for k, v in dict(model).items() if k in ['modality', 'C', 'alpha', 'epochs', 'metric_freq']}
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        train_loss, train_acc, _ = model_instance.train(self.X_train, self.Y_train, **train_hyparams)        
        trained_model = {model_id: model_instance, **dict(model), 'Training Loss':train_loss, 'Train Accuracy': train_acc, 'Weights': model_instance.weights, 'Bias': model_instance.bias}
        return trained_model

#   could do the following
#   print(f"Beginning in {colored('3 seconds', 'blue', attrs = ['bold', 'underline'])}")  # add functionality for a user to pause via a 'pause' param.
#   time.sleep(3)
    
    def _train_knn(self, model): # note that for the KNN to propertly work with majority voting, it must output predictions for an equal amount of samples to the other models.
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        train_hyparams = {k:v for k, v in dict(model_args).items() if k in ['K', 'modality', 'distance_metric']}
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        model_instance.train(self.X_train, self.Y_train, **train_hyparams)        
        trained_model = {model_id: model_instance, **dict(model_args)}
        return trained_model
    
    def _train_decision_tree(self, model):
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        train_hyparams = {k:v for k, v in dict(model_args).items() if k in ['max_depth', 'min_sample_split', 'modality', 'alpha']}
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        train_acc, train_uncertainty = model_instance.train(self.X_train, self.Y_train, **train_hyparams)        
        trained_model = {model_id: model_instance, **dict(model_args), 'Training Uncertainty': train_uncertainty, 'Training Accuracy': train_acc}
        return trained_model
   
    def _train_logistic_regression(self, model):
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        train_hyparams = {k:v for k,v in dict(model_args).items() if k in ['alpha', 'epochs', 'metric_freq']}
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        train_loss, train_acc, _ = model_instance.train(self.X_train, self.Y_train, **train_hyparams)        
        trained_model = {model_id: model_instance, **dict(model_args), 'Training Loss': train_loss, 'Training Accuracy': train_acc, 'Weights': model_instance.weights, 'Bias': model_instance.bias}
        return trained_model
   
  # -------- TEST MODELS --------  
    
    def _test_svm(self, model):
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        model_type = type(model_instance).__name__
        test_hyparams = {k:v for k,v in dict(model_args).items() if k in ['platt_kwargs', 'zero_one']}  
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n")
        if self.voting == 'hard':
            test_loss, test_acc, preds = model_instance.test(self.X_test, self.Y_test, return_probs = False, **test_hyparams)
            tested_model = {model_id: model_instance, **dict(model_args), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, 'Predictions': preds}
        elif self.voting == 'soft':
            test_loss, test_acc, preds, probs = model_instance.test(self.X_test, self.Y_test, return_probs = True, **test_hyparams)
            tested_model = {model_id: model_instance, **dict(model_args), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, 'Predictions': preds, 'Probabilities': probs}
        return tested_model
        
    def _test_knn(self, model):
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        model_type = type(model_instance).__name__
        test_hyparams = {k:v for k,v in dict(model_args).items() if k in ['platt_kwargs', 'testing_size']}
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n")
        if self.voting == 'hard':
            test_acc, preds = model_instance.test(self.X_test, self.Y_test, return_probs = False, **test_hyparams) 
            tested_model = {model_id: model_instance, **dict(model_args), 'Testing Accuracy': test_acc, 'Predictions': preds} 
        elif self.voting == 'soft':
            test_acc, preds, probs = model_instance.test(self.X_test, self.Y_test, return_probs = True, **test_hyparams) 
            tested_model = {model_id: model_instance, **dict(model_args), 'Testing Accuracy': test_acc, 'Predictions': preds, 'Probabilities': probs}
        return tested_model 
        
    def _test_decision_tree(self, model):
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n")
        if self.voting == 'hard':
            uncertainty, test_acc, preds  = model_instance.test(self.X_test, self.Y_test, return_probs = False)
            tested_model = {model_id: model_instance, **dict(model_args), 'Uncertainty': uncertainty, 'Testing Accuracy': test_acc, 'Predictions': preds} 
        elif self.voting == 'soft':
            uncertainty, test_acc, preds, probs = model_instance.test(self.X_test, self.Y_test, return_probs = True)
            tested_model = {model_id: model_instance, **dict(model_args), 'Uncertainty': uncertainty, 'Testing Accuracy': test_acc, 'Predictions': preds, 'Probabilities':probs}  
        return tested_model

    def _test_logistic_regression(self, model):
        model_args = list(model.items())
        model_id, model_instance = model_args.pop(0)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"{model_id}", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        if self.voting == 'hard':
            test_loss, test_acc, preds = model_instance.test(self.X_test, self.Y_test)
            tested_model = {model_id: model_instance, **dict(model_args), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, "Predictions": preds} 
        elif self.voting == 'soft':
            test_loss, test_acc, preds, probs = model_instance.test(self.X_test, self.Y_test, return_probs = True)
            tested_model = {model_id: model_instance, **dict(model_args), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, "Predictions": preds, 'Probabilities': probs}    
        return tested_model
  
    def _hard_prediction(self):
        self.model_preds = []
        for model in self.tested_models:
            predictions = model['Predictions'].flatten() 
            self.model_preds.append(predictions) 
        self.model_preds = np.array(self.model_preds)
        self.majority_pred = np.max(self.model_preds, axis = 0).flatten()
        return self.majority_pred
  
    def _soft_predictions(self):
        self.model_probs = []
        for model in self.tested_models:
            probabilities = model['Probabilities'].flatten()
            self.model_probs.append(probabilities)
        np_probs = np.array(self.model_probs)
        mean_probs = np.mean(np_probs, axis = 0)
        self.majority_preds = np.where(mean_probs > .5, 1, 0)
        return self.majority_preds        

