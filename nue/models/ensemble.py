import numpy as np
import pprint
import time
from nue.metrics import accuracy
from termcolor import colored

'''
TODO 
- [X] Ensemble Train Method
    - [X] Refactor train methods of each model
    - [X] Build the train function

- [ ] Ensemble Predict / Test Method
    - [X] Refactor test methods of each model.
    - [ ] Build the testing for soft / hard:
        - [X] Ensure all models have the ability to return probabilities.
            - [X] SVMs
                - [X] Ensure that probabilities are returned with respect to it being 1
            - [X] Log Reg
            - [X] KNN
                - [X] Ensure that probabilities are returned with respect to it being 1
            - [X] Decision Tree
                - [X] Ensure thast probabilitie asre returned with respect to it being 1
        - [X] Need the KNN to output the same # of predictions as the other models. Otherwise, won't work.  
        - [X] Insert Hard Majority Voting to compute predictions
        - [ ] Insert Soft Majority Voting to compute predictions.
            - [X] The KNN needs to propertly output probabilities. Calculate the probabilities as looking at the nearest k labels and getting their ratio. 
            - [ ] Feed that raw output to the logistic regression to platt scale.
            
            
            
----- notes -------

> Just need to continue now            

'''

class EnsembleClassifier:
    
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
        self.prediction = self._hard_prediction()
        
        self.accuracy = accuracy(self.Y_test.flatten(), self.prediction)

        if self.verbose_test:
            print(colored('\nENSEMBLE RESULTS\n', 'green', attrs = ['bold', 'underline']))
            print(f"Accuracy: {self.accuracy}%")
            print(f"Predictions:\n\n{self.prediction}")
        
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
            print(tested_model)
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
            print(tested_model) 
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
            print(tested_model)
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
            print(tested_model)
        return tested_model
  
    def _hard_prediction(self):
      
        self.model_preds = []
       
        for model in self.tested_models:
            # get a new dict of modelname:modelinstance, 'predictions': predictions
            model_id, model_instance = list(model.items())[0] # gets the model id and instance
            predictions = model['Predictions'].flatten() 
            self.model_preds.append(predictions) 
        self.model_preds = np.array(self.model_preds)
        self.majority_pred = np.max(self.model_preds, axis = 0).flatten()

        return self.majority_pred
    
    @property
    def models(self):
        return self._models
    
    @models.setter
    def models(self, models):
        assert isinstance(models, list), "models must be in type dict, in form of 'model-name': model instance"
        self._models = models 
        
    @property
    def voting(self):
        return self._voting
    
    @voting.setter
    def voting(self, voting):
        assert isinstance(voting, str), "voting must be type str."
        self._voting = voting
        
    @property 
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        assert isinstance(weights, (type(None), np.ndarray)), "weights must be type numpy.ndarray."
        self._weights = weights 
        
    @property
    def X_train(self):
        return self._X_train
    
    @X_train.setter
    def X_train(self, X_train):
        assert isinstance(X_train, np.ndarray), "X_train must be type numpy.ndarray"
        self._X_train = X_train 
    
    @property
    def Y_train(self):
        return self._Y_train
    
    @Y_train.setter 
    def Y_train(self, Y_train):
        assert isinstance(Y_train, np.ndarray), 'Y_train must be type numpy.ndarray'
        self._Y_train = Y_train
       
    @property 
    def verbose_train(self):
        return self._verbose_train
    
    @verbose_train.setter
    def verbose_train(self, verbose):
        assert isinstance(verbose, bool), 'verbose_train musat be type bool'
        self._verbose_train = verbose 
        
        
         