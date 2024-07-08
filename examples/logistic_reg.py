from nue.models import LogisticRegression
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy 

''' Pre-processing data '''

data = csv_to_numpy('data/binaryclass.csv')
train, test = train_test_split(data, .75)
X_train, Y_train = x_y_split(train, y_col= 'last')
X_test, Y_test = x_y_split(test, y_col='last')


''' Setting hyperparameters '''

alpha = .1
epochs = 100000


''' Instantiating model '''

model = LogisticRegression(seed = 1)


''' Training and Testing the model '''

model.train(X_train, Y_train, alpha, epochs, verbose = False)
model.test(X_test, Y_test, verbose = False)


''' Checking final metrics '''

model.metrics(mode = 'both')