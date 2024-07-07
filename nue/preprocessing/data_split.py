import numpy as np
import pandas as pd

def train_test_split(data, train_split):
    '''
    Function to split data into train and testing sets.
 
    :param data: The data in numpy.ndarray format, assumed to be in (features, samples)
    :type data: numpy.ndarray
    :param train_split: The percentage dataset which will serve as the training set.
    :type train_split: float
  
    :return train: The training set, delimited by `train_split` of shape (features, samples) 
    :rtype train: numpy.ndarray
    :return test: The testing set, delimited by 1 - `train_split` of shape (features, samples) 
    :rtype test: numpy.ndarray
    ''' 
    
    if not isinstance(data, np.ndarray):
        raise ValueError('Data must be type ndarray!')
    if not isinstance(train_split, float):
        raise ValueError('train_split must be type float!')
    shape = data.shape
    split_val = round(train_split * shape[1])
    train = data[:, :split_val]
    test = data[:, split_val:] 
    return train, test

def x_y_split(data, y_col = 'first'):
    ''' 
    Function to split an array into labels and features 
    :param data: The data in numpy.ndarray format, assumed to be in (features, samples)
    :type data: numpy.ndarray
    :param y_col: The column index of the labels, y. Must be either the first or the last column
    :type y_col: str
    
    :return X: The features, of shape (features, samples)
    :rtype X: numpy.ndarray
    :return Y: The labels, of shape (1, samples)
    :rtype Y: numpy.ndarray 
    '''

    if not isinstance(data, np.ndarray):
        raise ValueError('data must be type float!')
    if y_col.lower() not in ['first', 'last']:
        raise ValueError("y_col must be type str of 'first' or 'last'")
      
    if y_col.lower() == 'first':
        X = data[1:, :]
        Y = data[0, :].reshape(1, -1)
    elif y_col.lower() == 'last':
        X = data[:-1, :]
        Y = data[-1, :].reshape(1, -1)

    return X, Y 
      