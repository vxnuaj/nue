"""
Implement pre-built machine learning models with the call of a single function!

**Pre-Built Models**:

:func:`linreg` - Contains the implementation of linear regression.

:func:`logreg` - Contains the implementation of logistic regression.

:func:`nn` - Contains the implementation of a neural network, for classification tasks.

**Sample Usage:**

.. code-block:: python
    
    from nue import nn
    
    # Pre-processing data
    data = pd.read_csv('examples/data/mnist_train.csv')
    data = np.array(data) # 60000, 785
    Y_train = data[:, 0].T.reshape(1, -1)# 1, 60000
    X_train = data[:, 1:786].T / 255 # 784, 60000
    
    # Instantiating class
    model = nn.NN(X_train, Y_train, 784, 10, 32, .1, 1000)
    
    #Single function call
    model.model()

See more at the :doc:`examples`.
"""

from models.linreg import LinearRegression
from models.logreg import LogisticRegression
from models.nn import NN
