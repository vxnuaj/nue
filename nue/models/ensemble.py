import numpy as np

'''
TODO 
'''

class EnsembleClassifier:
    
    '''
    Initalize a set of Ensemble models.
    ''' 
    
    def __init__(self):
        pass
       
    def train(self, X_train, Y_train, models= [], weights = None, voting = 'hard', verbose = False):
       
        '''
        Train each model on a training set, X_train  
        
        :param X_train: The training data, of shape (samples, features)
        :type X_train: numpy.ndarray
        :param Y_train: The labels corresponding to the training data, of shape (samples, 1) or (samples, )
        :type Y_train: numpy.ndarray 
        :param models: The set of models used to run the Voting Classifier. Contains a dicts of any instance of a model under nue.models with the rest of the parameters in the dict corresponding as arguments to train the model within it's respective model.train function. 
    
        The first key:value pair of the dict can be any name for the model with the model instance as the value. No need to include X_train or Y_train params.
        
        :type models: list
        :param voting: Choose between 'hard' or 'soft' voting, where 'hard' is based on pure class predictions while 'soft' is based on raw probabilities.
        :type voting: str
        :param weights: A set of weights, used when calculating the highest average probability or most predicted class. Default is simple 1 / n where n is the total number of predictions
        :type weights: numpy.ndarray
        ''' 
        
       
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.models = models
        self.voting = voting
        self.weights = weights
        self.verbose_train = verbose
       
        for model in models:
           
            self._train_sub_models(model) 
        
            
    def _train_sub_models(self, model):
        
        model_meta = list(model.items())
        model_id, model_instance = model_meta[0] 
        model_name = type(model_instance).__name__        

        print(f"Training {model_id} | Model Type: {model_name}")
              
        if model_name == 'SVM':
            self._train_svm(model)
        elif model_name == 'KNN':
            self._train_knn(model)
        elif model_name == 'DecisionTree':
            self._train_decision_tree(model)
      
      
        return
       
    def _train_svm(self, model):
       
        '''
        TODO ; left off here. 
        
        ''' 
       
        print(model) 
        
        print('train svm\n')
    
    def _train_knn(self, model):
        print('train knn\n')
   
    def _train_decision_tree(self, model):
        print('train destree\n')
    
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
        
        
         