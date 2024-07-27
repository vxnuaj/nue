from nue.models import SVM
from nue.preprocessing import x_y_split, csv_to_numpy

''' Pre-processing data '''

train_data = csv_to_numpy('data/SVM.csv')
test_data = csv_to_numpy('data/testSVM.csv')

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(train_data, y_col = 'last')


''' Setting parameters '''

alpha = .001
epochs = 10000
verbose = True
seed = 0
modality = 'soft'
C = .1

''' Instantiating SVM '''

model = SVM(seed = seed, modality = modality, C = C)

''' Training and Testing the SVM '''

model.train(X_train, Y_train, verbose = verbose)
model.test(X_test, Y_test, verbose = verbose)

''' Support Vectors '''

support_vector, geometric_margin = model.support_vector()
print("\nSupport Vectors\n")
print("\n".join([f"{i}" for i in support_vector]))
print("\nGeometric Margins\n")
print("\n".join([f"{i}" for i in geometric_margin]))