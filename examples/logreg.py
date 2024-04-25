import numpy as np
import pandas as pd
import nue.models.logreg as lr

data = pd.read_csv('examples/data/randomtrain.csv')
data = np.array(data)

X_train = data[:, :2].T
Y_train = data[:, 2].T.reshape(-1, 200)

print(X_train.shape)
print(Y_train.shape)

model = lr.LogisticRegression(X_train, Y_train, 2, .1,  50000)

model.model()

#(1, 13) @ (13, 303)