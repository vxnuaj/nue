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

def dt_entropy(p:np.ndarray = None, p1:float = None, p2:float = None, class_num:int = 2):
 
    eps = 1e-8
    
    if class_num > 2:
        if not isinstance(p, np.ndarray):
            raise ValueError("p must be type numpy.ndarray!")

        ent = - np.sum(p * np.log(p + eps))

    elif class_num == 2:
        if not isinstance(p1, float) or not isinstance(p2, float):
            raise ValueError('p1 and p2 must be type float!')
        elif not 0 < p1 < 1 or not 0 < p2 < 1:
            raise ValueError('p1 and p2 must be a probability between range 0 and 1!')

        ent = - p1 * np.log(p1 + eps) + ( 1 - p2 ) * np.log(1 - p2 + eps) 

    elif class_num < 2:
        raise ValueError('class_num, cannot be less than 2! must be binary or multiclass!') 
    
    return ent

def dt_weighted_entropy(dataset_size:int, subset_sizes:np.ndarray, subset_entropy:np.ndarray):
  
    #Elements of subset_size and subset_entropy must match with each other
   
    if not isinstance(dataset_size, int) or dataset_size < 1:
        raise ValueError("dataset_size must be type int and postive!")
    elif not isinstance(subset_sizes, np.ndarray):
        raise ValueError("subset_size must be type numpy.ndarray!")
    elif not isinstance(subset_entropy, np.ndarray):
        raise ValueError("subset_entropy must be type numpy.ndarray!")
  
    weighted_ent = np.sum((subset_sizes / dataset_size) * subset_entropy)
    
    return weighted_ent 

def dt_information_gain(dataset_ent:float, weighted_ent:float):
    if not isinstance(dataset_ent, float):
        raise ValueError('dataset_ent must be type float!')
    elif not isinstance(weighted_ent, float):
        raise ValueError("weighted_ent must be type float!")
        
    information_gain = dataset_ent - weighted_ent
    return information_gain
    
  
    
def dt_gini_impurity(p:np.ndarray = None, p1:float = None, p2:float = None, class_num:int = 2):
 
    if class_num > 2:
        if p is None:
            raise ValueError("p is None, must assign number of classes to p!")
        impurity = 1 - np.sum(np.square(p)) # AVERAGE OR NOT?
    elif class_num == 2:
        if p1 is None or p2 is None:
            raise ValueError("p1 and p2 must both be assigned to a value of type float!")
        impurity = p1 * ( 1 - p1 ) + p2 * ( 1 - p2 )

    return impurity
    