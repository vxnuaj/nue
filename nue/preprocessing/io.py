import numpy as np
import pandas as pd

def csv_to_numpy(file_path:str, delimiter:str = None, delim_whitespace:bool = False):
    '''
    Converts CSV files to ndarray object
    
    :param file_path: The file path of the CSV file, assumed to be in shape (samples, features)
    :type file_path: str
    :param delimeter: The spacing delimeter used in the CSV file.
    :type delimieter: str
    :param delim_whitespace: True if the delimeter is whitespace or '\t' 
    :type delim_whitespace: str
   
    :return: CSV file as an ndarray object
    :rtype: numpy.ndarray in shape (features, samples) 
    
    
    '''

    if delim_whitespace == False: 
        data = pd.read_csv(file_path, delimiter = delimiter)
        data = np.array(data)
        data=data.T
    elif delim_whitespace == True:
        data = pd.read_csv(file_path, delim_whitespace=True)
        data = np.array(data)
        data = data.T
    return data