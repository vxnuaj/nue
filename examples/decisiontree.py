from nue.models import DecisionTree
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split

''' Pre-processing data '''

train_data = csv_to_numpy('data/DecTreeTrain.csv')
test_data = csv_to_numpy('data/DecTreeTrain.csv')

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(test_data, y_col = 'last')

X_train, Y_train = X_train.T, Y_train.T.astype(int)
X_test, Y_test = X_test.T, Y_test.T.astype(int)

''' Setting hyperparameters '''

max_depth = 200
min_node_samples = 1

''' Instantiating model '''

model = DecisionTree(max_depth = max_depth, min_node_samples=min_node_samples)

''' Training and testing the Decision Tree'''

model.fit(X_train, Y_train)
model.predict(X_test, Y_test, verbose = True)