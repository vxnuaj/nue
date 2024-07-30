import numpy as np
import pprint
import time
from termcolor import colored

'''
TODO 
- [X] Ensemble Train Method
    - [X] Refactor train methods of each model
    - [X] Build the train function
- [ ] Ensemble Predict / Test Method
    - [X] Refactor test methods of each model.
    - [ ] Build the testing function
        - [ ] Ensure all models have the ability to return probabilities.
            - [X] SVMs
            - [X] Log Reg
            - [ ] KNN - create a class for PlattsMethod and use it here as + replace the svm code.
                I left off here, trying to create a class and implement it in the SVM. Need to test if it works now. Then move onto KNN.
            - [X] Decision Tree
'''

class EnsembleClassifier:
    
    '''
    Initalize a set of Ensemble models.
    ''' 
    
    def __init__(self):
        pass
       
    def train(self, X_train, Y_train, models= []):
       
        '''
        Train each model on a training set, X_train  
        
        :param X_train: The training data, of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The labels corresponding to the training data, of shape (samples, 1) or (samples, )
        :type Y_train: numpy.ndarray 
        :param models: The set of models used to run the Voting Classifier. Contains a dicts of any instance of a model under nue.models with the rest of the parameters in the dict corresponding as arguments to train the model within it's respective model.train function. 
    
        The first key:value pair of the dict can be any name for the model with the model instance as the value. No need to include X_train or Y_train params.
        
        :type models: list
        ''' 
       
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.models = models
        self.trained_models = [] # Trained models are appended to this list in the `self._train_models` func.
       
        for model in self.models:
            self._train_models(model) 
       
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
      
        for model in self.trained_models:
            self._test_models(model)
           
    def train_test(self):
        '''
        Train and test the models on training and testing sets, X_train and X_test 
        ''' 
            
    def _train_models(self, model):
        model_meta = list(model.items())
        model_id, model_instance = model_meta[0] 
        model_name = type(model_instance).__name__        

        if model_name == 'SVM':
            model = self._train_svm(model)
        elif model_name == 'KNN':
            self._train_knn(model)
        elif model_name == 'DecisionTree':
            model = self._train_decision_tree(model)
        elif model_name == 'LogisticRegression':
            model = self._train_logistic_regression(model) 
            
        self.trained_models.append(model)
      
    def _test_models(self, model): 
        model_meta = list(model.items())
        model_id, model_instance = model_meta[0]
        model_name = type(model_instance.__name__) 
       
        if model_name == 'SVM':
            self._test_svm(model)
        elif model_name == 'KNN':
            self._test_knn(model)
        elif model_name == 'DecisionTree':
            self._test_decision_tree(model)
        elif model_name == 'LogisticRegression':
            self._test_logistic_regression(model) 
        
    def _train_svm(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_hyparams = dict(model)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'{model_id}'", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        train_loss, train_acc, params = model_instance.train(self.X_train, self.Y_train, **model_hyparams)        
        model = {model_id: model_instance, **dict(model), 'Training Loss':train_loss, 'Train Accuracy': train_acc, 'Weights': model_instance.weights, 'Bias': model_instance.bias}
        return model

#   could do the following
#   print(f"Beginning in {colored('3 seconds', 'blue', attrs = ['bold', 'underline'])}")  # add functionality for a user to pause via a 'pause' param.
#   time.sleep(3)
    
    def _train_knn(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_hyparams = dict(model)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'{model_id}'", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        model_instance.train(self.X_train, self.Y_train, **model_hyparams)        
        model = {model_id: model_instance, **dict(model)}
        print(model)
        return model      
    
    def _train_decision_tree(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_hyparams = dict(model)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'{model_id}'", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        train_acc, train_uncertainty = model_instance.train(self.X_train, self.Y_train, **model_hyparams)        
        model = {model_id: model_instance, **dict(model), 'Training Uncertainty': train_uncertainty, 'Training Accuracy': train_acc}
        return model
   
    def _train_logistic_regression(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_hyparams = dict(model)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TRAINING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'{model_id}'", 'blue')} | {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        train_loss, train_acc, _ = model_instance.train(self.X_train, self.Y_train, **model_hyparams)        
        model = {model_id: model_instance, **dict(model), 'Training Loss': train_loss, 'Training Accuracy': train_acc, 'Weights': model_instance.weights, 'Bias': model_instance.bias}
        return model
   
  # -------- TEST MODELS --------  
    
    def _test_svm(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_type = type(model_instance).__name__  
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'model_id", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n")
        if self.voting == 'hard':
            test_loss, test_acc, preds = model_instance.test(self.X_test, self.Y_test)   
            model = {model_id: model_instance, **dict(model), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, 'Predictions': preds}
        elif self.voting == 'soft':
            test_loss, test_acc, preds, probs = model_instance.test(self.X_test, self.Y_test, return_probs = False)
            model = {model_id: model_instance, **dict(model), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, 'Predictions': probs}
        print(model)
        return model
        
    def _test_knn(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'model_id", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n")
        test_acc, preds = model_instance.test(self.X_test, self.Y_test)
        model = {model_id: model_instance, **dict(model), 'Testing Accuracy': test_acc, 'Predictions': preds}
        print(model)
        return model
    
    def _test_decision_tree(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'model_id", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n")
        if self.voting == 'hard':
            preds, test_acc, uncertainty = model_instance.test(self.X_test, self.Y_test, return_probs = True)
            model = {model_id: model_instance, **dict(model), 'Uncertainty': uncertainty, 'Testing Accuracy': test_acc, 'Predictions': preds} 
        elif self.voting == 'soft':
            preds, probs, test_acc, uncertainty = model_instance.test(self.X_test, self.Y_test, return_probs = True)
            model = {model_id: model_instance, **dict(model), 'Uncertainty': uncertainty, 'Testing Accuracy': test_acc, 'Predictions': preds, 'Probabilities': probs}
        print(model)
        return model

    def _test_logistic_regression(self, model):
        model = list(model.items())
        model_id, model_instance = model.pop(0)
        model_type = type(model_instance).__name__
        print(f"\n{colored('NOW TESTING:', 'blue', attrs = ['underline', 'bold'])} {colored(f"'model_id", 'blue')} |  {colored('MODEL TYPE', 'blue', attrs = ['underline', 'bold'])}: {colored(model_type, 'blue')}\n") 
        if self.voting == 'hard':
            test_loss, test_acc, preds = model_instance.test(self.X_test, self.Y_test)
            model = {model_id: model_instance, **dict(model), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, "Predictions": preds} 
        elif self.voting == 'soft':
            test_loss, test_acc, preds, probs = model_instance.test(self.X_test, self.Y_test)
            model = {model_id: model_instance, **dict(model), 'Testing Loss': test_loss, 'Testing Accuracy': test_acc, "Predictions": preds, "Probabilities": probs} 
        print(model) 
        return model
    
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
        
        
         