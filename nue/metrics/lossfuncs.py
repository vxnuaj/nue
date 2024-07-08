import numpy as np

def mse(y, pred):

    '''
    Compute the Mean-Squared-Error for the model.

    :return: Float defining the MSE metric.
    :rtype: float  
    '''

    loss = np.sum(np.square(y - pred)) / y.size
    return loss

def log_loss(y, pred, eps = 1e-8):
    loss = - np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))  
    return loss

def cce(one_hot_y, a, eps = 1e-8):
    loss = - np.sum(one_hot_y * np.log(a + eps)) / one_hot_y.shape[1]
    return loss