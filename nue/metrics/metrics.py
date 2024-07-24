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

def svm_accuracy(y, z):
    pred = np.sign(z) 
    acc = np.sum(y == pred) / y.size * 100
    return acc

def entropy(Y):
    eps = 1e-10
        
    _, freqs = np.unique(Y.flatten(), return_counts = True)
    probs = freqs / Y.size
    ent = - np.sum(probs * np.log(probs + eps))        

    return ent

def gini(Y):
    _, freqs = np.unique(Y.flatten(), return_counts = True)
    
    probs = freqs / Y.size
    gini_impurity = 1 - np.sum(np.square(probs))
    return gini_impurity
    
def dt_accuracy(y, pred):
    acc = np.sum(y == pred) / y.size * 100
    return acc