from nue.models import LinearRegression 
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split, ZNorm

''' Pre-processing data '''

data = csv_to_numpy('data/linear_regression_dataset.csv')
train, test = train_test_split(data, .8)

X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

z_norm1 = ZNorm()
z_norm2 = ZNorm()

X_train = z_norm1.fit_normalize(X_train)
X_test = z_norm1.normalize(X_test)

Y_train = z_norm2.fit_normalize(Y_train)
Y_test = z_norm2.fit_normalize(Y_test)

''' Setting hyperparameters '''

alpha = .01
epochs = 10000
verbose_1 = False
verbose_2 = True

''' Instantiating the model '''

model = LinearRegression(modality = 'ols',  seed = 1)

''' Training and Testing the model '''

model.train(X_train, Y_train, alpha = alpha, epochs = epochs, verbose = verbose_1)
model.test(X_test, Y_test, verbose = verbose_2)


''' Checking Final Metrics '''

model.metrics(mode = 'both')
