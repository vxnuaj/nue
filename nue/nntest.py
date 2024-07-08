from nue.models import NN 
from nue.preprocessing import x_y_split, csv_to_numpy 

''' Pre-processing data '''

train_data = csv_to_numpy('data/fashion-mnist_train.csv')
test_data = csv_to_numpy('data/fashion-mnist_test.csv')

X_train, Y_train = x_y_split(train_data)
X_test, Y_test = x_y_split(test_data)

X_train = X_train / 255
X_test = X_test / 255

''' Setting hyperparameters '''

alpha = .1
epochs = 50
hidden_size = 32
num_classes = 10

save_model_filepath = 'files/model'
model_filepath = 'files/model.npz'

''' Instantiating model '''

model = NN(seed = 1)

''' Training and Testing the model '''

params = model.train(X_train, Y_train, hidden_size, num_classes, alpha, epochs, verbose = True, metric_freq=1, save_model_filepath = save_model_filepath) 

model.test(X_test, Y_test, model_filepath, verbose = True) 

''' Checking final metrics '''

model.metrics(mode = 'both')