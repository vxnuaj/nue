import numpy as np

class ZNorm():

    '''
    Class, to normalize your dataset via Z-Score Normalization.
    '''

    def __init__(self):

        self._mean = None
        self._std = None

    def fit(self, data):
    
        '''
        Fit your dataset to the ZNorm Instance. Compute the mean, standard deviation, and variance. 
        
        :param data: Your input data to normalize, of shape (samples, features) if features or shape (samples, ) or (samples, 1) if training data.
        :type data: numpy.ndarray 
        
        :return self._mean: The mean of your dataset
        :rtype self._mean: float or int
        :return self._std: The standard deviation of your dataset
        :rtype self._std: float or int
        :return self._var: The variance of your dataset
        :rtype self._var: float or int
        '''
      
        self.data_dim = len(data.shape) 
       
        if len(data.shape) == 1:
            self._mean = np.mean(data)
            self._std = np.std(data)
            self._var = np.square(self._std)
      
        elif len(data.shape) == 2: 
            self._mean = np.mean(data, axis = 0)
            self._std = np.std(data, axis = 0)
            self._var = np.square(self._std)
  
        else:
            raise ValueError('data must be 1d or 2d. higher dimensionality is not supported yet!')
    
        return self._mean, self._std, self._var
    
    def fit_normalize(self, data, transpose = False):
       
        '''
        Fit and transform your dataset to the ZNorm Instance 
       
        :param data: Your input data to normalize, of shape (samples, features) if features or shape (samples, ) or (samples, 1) if training data.
        :type data: numpy.ndarray
        
        :return: Your normalized data
        :rtype: numpy.ndarray 
        '''
      
        self.data_dim = len(data.shape)
       
        if self.data_dim == 1:
            self._mean = np.mean(data)
            self._std = np.std(data)
            self._var = np.square(self._std) 
        
        elif self.data_dim == 2: 
            self._mean = np.mean(data, axis = 0)
            self._std = np.std(data, axis = 0)
            self._var = np.square(self._std)
  
        else:
            raise ValueError('data must be 1d or 2d. higher dimensionality is not supported yet!')
    
        data_norm = (data - self._mean) / self._std  
        
        return data_norm
    
    def normalize(self, data):
      
        '''
        Normalize your data based on the computed standard deviation and mean, done prior. Must have used `ZNorm.fit()` or `ZNorm.fit_transform()` prior 
        
        :param data: Your input data to normalize, of shape (samples, features) if features or shape (samples, ) or (samples, 1) if training data.
        :type data: numpy.ndarray
        
        :return: Your normalized data
        :rtype: numpy.ndarray  
        '''
      
       
        if self._mean is None or self._std is None:
            raise ValueError('Need to run ZNorm.fit() or ZNorm.fit_transform() on a set of data prior to running ZNorm.transform().')
   
        if len(data.shape) != self.data_dim:
            raise ValueError('the shape of your input data does not match the dimensions of data used to fit ZNorm.') 
    
        data_norm = (data - self._mean) / self._std
        
        return data_norm
    
    