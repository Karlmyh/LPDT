import numpy as np
from numba import njit

@njit
def insample_ssq(y):
    """In sample sum of squares.
    """
    if len(y)>0:
        return np.var(y) * len(y)
    else:
        return 0


@njit
def mse(X, dt_Y, d, split):
    """Compute MSE decrease for one pair of dimension and split point.

    Parameters
    ----------
    
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
 
    d : int in 0, ..., dim - 1
        The splitting dimension.
    
    split : float
        The splitting point.
    
    Returns
    -------
    
    mse_reduction : float
        The decrease of mse. 

    """
    
    before_mse = insample_ssq( dt_Y )
    
   
    after_mse = insample_ssq( dt_Y[ X[:,d] < split ] ) + insample_ssq( dt_Y[ X[:,d] >= split ] )

   
    (before_mse - after_mse) / len(dt_Y)
    
    return (before_mse - after_mse) / len(dt_Y)

