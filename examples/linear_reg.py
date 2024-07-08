from nue.models import LinearRegression 
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split 


''' Pre-processing data '''

data = csv_to_numpy('data/linear_regression_dataset.csv')
train, test = train_test_split(data, .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')


''' Setting hyperparameters '''

alpha = .001
epochs = 100000


''' Instantiating the model '''

model = LinearRegression(seed = 1)


''' Training and Testing the model '''

model.train(X_train, Y_train, alpha, epochs, verbose = False)
model.test(X_test, Y_test, verbose = False)


''' Checking Final Metrics '''

model.metrics(mode = 'both')
