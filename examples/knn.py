from nue.models import KNN
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split


''' Pre-processing data '''

train_data = csv_to_numpy('data/fashion-mnist_train.csv')
test_data = csv_to_numpy('data/fashion-mnist_test.csv')


X_train, Y_train = x_y_split(train_data, y_col = 'first')
X_train, Y_train = X_train / 255, Y_train.T

X_test, Y_test = x_y_split(test_data, y_col = 'first')
X_test, Y_test = X_test / 255, Y_test.flatten()

X_train, X_test = X_train.T, X_test.T

Y_train, Y_test = Y_train.flatten(), Y_test.flatten()

''' Setting hyperparameters '''

testing_size = 20
verbose = True
K = 50


''' Instantiating model '''

model = KNN(K)


''' Training and testing the KNN '''

model.fit(X_train, Y_train)
model.predict(X_test, Y_test, testing_size = testing_size, verbose=verbose)