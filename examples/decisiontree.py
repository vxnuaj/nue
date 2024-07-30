from nue.models import DecisionTree
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split, label_encoding

''' Pre-processing data '''

data = csv_to_numpy('data/DesTreeData.csv')

train_data, test_data = train_test_split(data, train_split = .8)

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(test_data, y_col = 'last')

X_train, Y_train = X_train, Y_train.astype(int)
X_test, Y_test = X_test, Y_test.astype(int)

''' Setting hyperparameters '''

max_depth = 1000
min_sample_split = 2
modality = 'entropy'

alpha = .1
verbose_train = True
verbose_test = True
return_prob = True

''' Instantiating model '''

model = DecisionTree(verbose_train = verbose_train, verbose_test = verbose_test)

''' Training and testing the Decision Tree'''

model.train(X_train, Y_train, max_depth = max_depth, min_sample_split = min_sample_split, modality = modality, alpha = alpha)
pred, probs = model.test(X_test, Y_test, return_probs = True)

