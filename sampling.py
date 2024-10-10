import numpy as np 

def sample(x, n=10, replacement=True):
    return np.random.choice(x, n, replace=replacement)