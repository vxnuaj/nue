import numpy as np
from nue.models import SVM, KNN, DecisionTree, EnsembleClassifier, LinearRegression, LogisticRegression
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

''' Pre-processing data '''

train = csv_to_numpy('data/ensembleTrain.csv')
test = csv_to_numpy('data/ensembleTest.csv')

X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

''' Initializing Models '''

verbose_train = False
verbose_test = True
seed = 1

lin_reg = LinearRegression(seed = 1, verbose_train = verbose_train, verbose_test=verbose_test)
log_reg = LogisticRegression(seed = 1, verbose_train = verbose_train, verbose_test=verbose_test)
svm = SVM(seed = 1, verbose_train = verbose_train, verbose_test=verbose_test)
knn = KNN(verbose_test=verbose_test)
dtree = DecisionTree(verbose_train = verbose_train, verbose_test=verbose_test)


''' Initializing model hyperparams '''

linreg_train = {
   
    'linreg!': lin_reg,
    'modality': 'ols', 
    'alpha': .0001,
    'epochs': 5000,
    'metric_freq': 1000
}

logreg_train = {
  
    'logreg!': log_reg,
    'alpha': .0001,
    'epochs': 5000,
    'metric_freq': 1000 
    
}

svm_train = {
    
    'svm!':svm,
    'modality': 'soft',
    'C': .01,
    'alpha': .0001,
    'epochs': 1000,
    'metric_freq': 1000  

}

knn_train = {
    
    'knn!': knn,
    'K': 10,
    'modality': 'brute',
    'distance_metric': 2
     
}

dtree_train = {
    
    'dtree!': dtree,
    'max_depth': 100,
    'min_sample_split': 2,
    'modality': 'entropy',
    'alpha': None,
}

models = [linreg_train, logreg_train, svm_train, knn_train, dtree_train]

''' Initializing Ensemble '''

model = EnsembleClassifier()

''' Training the Ensemble models'''

model.train(X_train, Y_train, models = models)
