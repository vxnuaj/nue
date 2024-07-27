from nue.models import LogisticRegression
from nue.preprocessing import ZNorm, x_y_split, train_test_split, csv_to_numpy 

''' Pre-processing data '''

data = csv_to_numpy('data/heart.csv')
train, test = train_test_split(data, .75)
X_train, Y_train = x_y_split(train, y_col= 'last')
X_test, Y_test = x_y_split(test, y_col='last')

znorm_1 = ZNorm()

X_train = znorm_1.fit_normalize(X_train)
X_test = znorm_1.normalize(X_test)


''' Setting hyperparameters '''

alpha = .01
epochs = 1000

''' Instantiating model '''

model = LogisticRegression(seed = 1)


''' Training and Testing the model '''

model.train(X_train, Y_train, alpha, epochs, verbose = False)
model.test(X_test, Y_test, verbose = True)


''' Checking final metrics '''

model.metrics(mode = 'both')