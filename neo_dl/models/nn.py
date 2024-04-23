import numpy as np

class NN:
    def __init__(self):
        return

    def layer(self, neurons, input_size):
        w = np.random.randn(neurons, input_size)
        b = np.random.randn(neurons, 1)
        return w, b
    
    


    # create a function that defines layers individually. #then iterate that over 