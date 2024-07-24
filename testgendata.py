import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# of size samples, feature for X or samples, for Y
X, Y = make_classification(n_samples = 100, n_features = 10, n_informative=8, n_redundant = 2, n_classes = 3, random_state=1)
Y = Y.reshape(-1, 1)

train_data = np.concatenate((X, Y), axis = 1)

train, test = train_test_split(train_data, random_state = 1)

train_data = pd.DataFrame(train)
test_data = pd.DataFrame(test)

train_data.to_csv('nue/data/DecTreeTrain.csv', index = None)
test_data.to_csv('nue/data/DecTreeTest.csv', index = None)