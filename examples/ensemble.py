import numpy as np
from nue.models import SVM, KNN, DecisionTree, EnsembleClassifier
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

''' Pre-processing data '''

train = csv_to_numpy('data/ensembleTrain.csv')
test = csv_to_numpy('data/ensembleTest.csv')

X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

''' Initializing Models '''

svm = SVM(seed = 1)
knn = KNN()
dtree = DecisionTree()

''' Initializing model hyperparams '''

svm_train = {
    
    'svm!':svm,
    'modality': 'soft',
    'C': .01,
    'alpha': .0001,
    'epochs': 250,
    'verbose': True,
    'metric_freq': 1 

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
    'verbose': False
    
}

models = [svm_train, knn_train, dtree_train]

''' Initializing Ensemble '''

model = EnsembleClassifier()

''' Training the Ensemble models'''

model.train(X_train, Y_train, models = models)




