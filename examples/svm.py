from nue.models import SVM
from nue.preprocessing import x_y_split, csv_to_numpy

''' Pre-processing data '''

train_data = csv_to_numpy('data/ensembleTrain.csv')
test_data = csv_to_numpy('data/ensembleTest.csv')

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(train_data, y_col = 'last')

''' Setting parameters '''

alpha = .0
epochs = 1000
seed = 0
modality = 'soft'
verbose_train = False
verbose_test = True
platt_train_verbose = True
platt_test_verbose = True
return_probs = True
platt_kwargs = {'seed': 1, 'verbose_train': False, 'verbose_test': False, 'Y_train': Y_train, 'alpha': .1, 'epochs': 10000, 'verbose_test':verbose_test}


''' Instantiating SVM '''

model = SVM(seed = seed, verbose_train = verbose_train, verbose_test = verbose_test)

''' Training and Testing the SVM, and returning raw Probabilities via Platt's Method '''

model.train(X_train, Y_train, modality = modality, alpha = alpha, epochs = epochs)
loss, acc, preds, probs = model.test(X_test, Y_test, return_probs)


''' 
Support Vectors

support_vector, geometric_margin = model.support_vector()

print("\nSupport Vectors:\n")
print("\n".join([f"{i}" for i in support_vector]))
print("\nGeometric Margins:\n")
print("\n".join([f"{i}" for i in geometric_margin]))
'''