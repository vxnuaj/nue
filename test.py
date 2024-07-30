import numpy as np
import pprint
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy
from nue.models import SVM

train = csv_to_numpy('data/ensembleTrain.csv')
test = csv_to_numpy('data/ensembleTest.csv')

X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

svm = SVM(seed = 1)


svm_train = {
    
    'svm!':svm,
    'modality': 'soft',
    'C': .01,
    'alpha': .0001,
    'epochs': 250,
    'verbose': True,
    'metric_freq': 1 

}

print(type(svm).__name__)

svm_dict = {key:val for key, val in svm_train.items() if type(val).__name__!="SVM"}

svm.train(X_train, Y_train, **svm_dict)

#svm.train(X_train, Y_train, )
