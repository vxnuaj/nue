import numpy as np
import pandas as pd

from neu.models import nn

''' Pre-processing data '''

data = pd.read_csv('examples/data/mnist_train.csv')
data = np.array(data) # 60000, 785

Y_train = data[:, 0].T.reshape(1, -1)# 1, 60000
X_train = data[:, 1:786].T / 255 # 784, 60000

''' Running model '''

model = nn.NN(X_train, Y_train, 784, 10, 32, .1, 1000)

model.model()