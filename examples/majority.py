import numpy as np
from nue.models import SVM, KNN, DecisionTree, MajorityClassifier, LinearRegression, LogisticRegression
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

''' Pre-processing data '''

train = csv_to_numpy('data/EnsembleTrain.csv')
test = csv_to_numpy('data/EnsembleTest.csv')

X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

''' Initializing Models '''

verbose_train = False
verbose_test = False
seed = 1
voting = 'soft'
weights = None
  
lin_reg = LinearRegression(seed = 1, verbose_train = verbose_train, verbose_test=verbose_test)
log_reg = LogisticRegression(seed = 1, verbose_train = verbose_train, verbose_test=verbose_test)
svm = SVM(seed = 1, verbose_train = verbose_train, verbose_test=verbose_test)
knn = KNN(verbose_test=verbose_test)
dtree = DecisionTree(verbose_train = verbose_train, verbose_test=verbose_test)
platt_kwargs = {
    'seed':1,
    'verbose_train': False,
    'verbose_test': False,
    'alpha': .01,
    'epochs': 1500,
    'metric_freq': 750,
    'Y_test': Y_test
}


''' Initializing model hyperparams '''

logreg = {
  
    'logreg!': log_reg,
    'epochs': 10000,
    'alpha': .1,
    'metric_freq': 1000,
    'platt_kwargs': platt_kwargs
}

svm = {
    'svm!':svm,
    'modality': 'soft',
    'C': .01,
    'alpha': .0001,
    'epochs': 1000,
    'metric_freq': 1000, 
    'platt_kwargs':platt_kwargs,
    'zero_one': True # Must be true for the Ensemble to work if the labels are binary 0s and 1s. Typically will be set as True.
}

knn = {
    
    'knn!': knn,
    'K': 50,
    'modality': 'brute',
    'distance_metric': 2,
    'platt_kwargs':platt_kwargs,
    'testing_size': 'all'
}

decision_tree = {
    
    'dtree!': dtree,
    'max_depth': 100,
    'min_sample_split': 2,
    'modality': 'entropy',
    'alpha': None,
    'platt_kwargs': platt_kwargs
}

models = [logreg, svm, knn, decision_tree]

''' Initializing Majority Voting Classifier '''

model = MajorityClassifier(models, verbose_test=verbose_test)

''' Training the Ensembled models'''

model.train(X_train, Y_train)
model.test(X_test, Y_test, voting = voting, weights = weights)
