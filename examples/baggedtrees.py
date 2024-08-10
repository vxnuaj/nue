from nue.models import BaggedTrees
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

''' Pre processing data'''
data = csv_to_numpy('data/DecTreeData.csv')
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

''' Setting parameters / hyperparamters '''
verbose_test = True
n_bootstrap = 10
alpha_range = (.1, 1)
dtree_dict = {
    'verbose_test': True,
    'max_depth': 100,
    'min_sample_split': 2,
    'modality': 'gini'
}

''' Initializing, Training, and Testing the model'''

model = BaggedTrees(verbose_test = verbose_test)
model.train(X_train, Y_train, n_bootstrap = n_bootstrap, dtree_dict = dtree_dict, alpha_range = alpha_range)
model.test(X_test, Y_test)

