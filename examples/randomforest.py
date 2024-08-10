from nue.models import RandomForest
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

''' Preprocessing data '''

data = csv_to_numpy('data/DecTreeData.csv')
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

''' Setting parameters / hyperparameters '''

verbose_test = True
max_features = 7
n_bootstraps = 10
n_extremely_randomized_feats = 20
rtree_dict = {

    'verbose_train': False,
    'min_node_samples': 2,
    'max_depth': 100,
    'criterion': 'gini',

}

''' Initializing, Training, and Testing the Random Forest '''

model = RandomForest(verbose_test = verbose_test, n_extremely_randomized_feats = n_extremely_randomized_feats)
model.train(X_train, Y_train, max_features = max_features, n_bootstraps = n_bootstraps, rtree_dict = rtree_dict)
model.test(X_test, Y_test)

