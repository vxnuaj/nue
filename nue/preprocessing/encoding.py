import numpy as np

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

def label_encoding(y):
    '''
    Encode your categorical labels into numerical labels. 
   
    :param y: The original categorical labels for a dataset, of shape (samples, )
    :type y: numpy.ndarray
    '''
   
    word_to_int = {word: i for i, word in enumerate(np.unique(y))} # Makes a dictionary mapping each label to a unique index, given the use of np.unique()
    vectorize = np.vectorize(lambda word: word_to_int[word]) # The lambda function takes in 'word' and uses the dictionary mapping to return it's corresponding integer. np.vectorize is ran to enable the function to run on an numpy.ndarray
    y_encoded = vectorize(y) # We use the vectorized function on the original set of labels to get integer encoded labels.
    return y_encoded    
    