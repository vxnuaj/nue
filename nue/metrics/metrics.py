import numpy as np

def r2_score(y, pred):
    total_sum_squares = np.sum(np.square(y - np.mean(y)))
    sum_squared_residual = np.sum(np.square(y - pred))
    r2 =  1 - (sum_squared_residual / total_sum_squares)
    return r2 

def logistic_accuracy(y, a):
    pred = np.round(a, decimals = 0) 
    acc = np.sum(pred == y) / y.size * 100
    return acc

def nn_accuracy(y, a):
    pred = np.argmax(a, axis = 0)
    acc = np.sum(pred == y) / y.size * 100
    return acc

