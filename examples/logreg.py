import numpy as np
import pandas as pd

from nue import logreg as lr

''' Pre processing data'''

data = pd.read_csv('examples/data/randomtrain.csv')
data = np.array(data)

X_train = data[:, :2].T
Y_train = data[:, 2].T.reshape(-1, 200)

print(X_train.shape)
print(Y_train.shape)

''' Running model '''

model = lr.LogisticRegression(X_train, Y_train, 2, .1,  50000)

model.model()

