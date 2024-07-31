from nue.models import KNN
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split


''' Pre-processing data '''

train_data = csv_to_numpy('data/DecTreeTrain.csv')
test_data = csv_to_numpy('data/DecTreeTest.csv')

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_train, Y_train = X_train, Y_train

X_test, Y_test = x_y_split(test_data, y_col = 'last')
X_test, Y_test = X_test, Y_test.flatten()

Y_train, Y_test = Y_train.flatten(), Y_test.flatten()

''' Setting hyperparameters '''

testing_size = 160
verbose_test = True
K = 25
modality = 'brute'
distance_metric = 2
return_probs = True

''' Instantiating model '''

model = KNN(verbose_test=verbose_test)

''' Training and testing the KNN '''

model.train(X_train, Y_train, K = K, modality = modality, distance_metric=distance_metric)
acc, pred, prob = model.test(X_test, Y_test, testing_size = testing_size, return_probs = return_probs)

print('pred', pred)
print('prob',  prob)