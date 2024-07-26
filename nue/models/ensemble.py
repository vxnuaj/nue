import numpy as np

'''
TODO
'''

class VotingClassifier:
    
    '''
    Run a set of classifiers and return a generalized prediction using `soft` or `hard` voting.
    The set of models can be a variety of types.
    
    :param models: The set of models used to run the Voting Classifier. Any model instance under nue.models, with the key belonging to the assigned name of the model.
    :type models: dict
    :param voting: Choose between 'hard' or 'soft' voting, where 'hard' is based on pure class predictions while 'soft' is based on raw probabilities.
    :type voting: str
    :param weights: A set of weights, used when calculating the highest average probability or most predicted class. Default is simple 1 / n where n is the total number of predictions
    :type weights: numpy.ndarray 
    ''' 
    
    def __init__(self, models = {}, voting = 'hard', weights = None):
        self.models = models
        self.voting = voting
        self.weights = weights
       
    def fit(self, X_train, Y_train):
       
        '''
        Fit each model on a training set, X_train  
        
        :param X_train: The training data, of shape samples, features
        :type X_train: numpy.ndarray
        :param Y_train: The labels corresponding to the training data.
        :type Y_train: numpy.ndarray 
        ''' 
       
        self.X_train = X_train
        self.Y_trian = Y_train 
        
        return