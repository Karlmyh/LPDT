import numpy as np
from scipy.stats import laplace
import math


class RegressionLaplaceEstimator(object):
    """ 
    Regression estimator that record the privatised labels based on Laplacian mechanism. 
    
    Parameters
    ----------
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    epsilon : float
        The privacy budget.
        
    Y_range : float
        The range of label.
        
    public_Y_mean : float
        The mean of public data, recorded for leveraging public data.

    proportion_budget : float in [0,1]
        The propotion of privacy budgets spent on the numerator.
    
    min_ratio_pri_est : float
        If the privately estimated sample in the grid is less that n * min_ratio_pri_est, 
        use public estimation.
        
    Attributes
    ----------
    y_hat : float
        The final estimation.
    
    """
    def __init__(self, 
                 X_range,
                 epsilon,
                 Y_range,
                 public_Y_mean,
                 proportion_budget,
                 min_ratio_pri_est,
                 ):
        
        
        self.X_range = X_range
        self.noise_level_Z = 2 * Y_range / epsilon
        self.noise_level_W = 4           / epsilon
        self.public_Y_mean = public_Y_mean
        self.min_ratio_pri_est = min_ratio_pri_est
        
        
    def fit(self):
        self.y_hat = self.public_Y_mean
        
    def get_data(self, Y, in_idx):
        noise_Z = laplace.rvs(size = in_idx.shape[0], scale = self.noise_level_Z)
        noise_W = laplace.rvs(size = in_idx.shape[0], scale = self.noise_level_W)
        Z = (Y.ravel() * in_idx.ravel() + noise_Z).tolist()
        W = (in_idx.ravel() + noise_W).tolist()

        if len(Z) > 0:
            if np.sum(W)  < ( Y.shape[0] * self.min_ratio_pri_est ):
                self.y_hat = self.public_Y_mean
            else:
                self.y_hat =  np.array(Z).sum() / np.array(W).sum()
                
    def predict(self, test_X):
        y_predict = np.full(test_X.shape[0], self.y_hat)
        return y_predict
    
    
class RegressionRandomPermutationEstimator(object):
    """ 
    Regression estimator that record the privatised labels based on random
    response mechanism.
    
    Parameters
    ----------
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    epsilon : float
        The privacy budget.
        
    Y_range : float
        The range of label.
        
    public_Y_mean : float
        The mean of public data, recorded for leveraging public data.
        
    proportion_budget : float in [0,1]
        The propotion of privacy budgets spent on the numerator.
    
    min_ratio_pri_est : float
        If the privately estimated sample in the grid is less that n * min_ratio_pri_est, 
        use public estimation.
        
    Attributes
    ----------
    y_hat : float
        The final estimation.
    
    """   
    def __init__(self, 
                 X_range,
                 epsilon,
                 Y_range,
                 public_Y_mean,
                 proportion_budget,
                 min_ratio_pri_est,
                 ):
        
        self.X_range = X_range
        self.rev_prob = 1 / ( 1 + math.exp( - epsilon * proportion_budget / 2 ))
        self.noise_level_Y =   Y_range / epsilon / (1 - proportion_budget)
        self.public_Y_mean = public_Y_mean
        self.count_for_debug = 0
        self.y_sum_for_debug = 0
        self.min_ratio_pri_est = min_ratio_pri_est
       
        
    def fit(self):
        self.y_hat = self.public_Y_mean
        
    def get_data(self, Y, in_idx):
        privatized_idx = in_idx.copy()
        rev_idx = np.random.choice(2, in_idx.shape[0], p = [self.rev_prob, 1- self.rev_prob], replace = True)
        privatized_idx[rev_idx.astype("bool")] = 1 - privatized_idx[rev_idx.astype("bool")]
        noise_Y = laplace.rvs(size = Y.shape[0], scale = self.noise_level_Y)
        V = (Y.ravel()  + noise_Y)  * (privatized_idx.ravel() + self.rev_prob - 1 ) 
        U = privatized_idx + self.rev_prob - 1 
        self.U_sum = U.sum() / ( 2 * self.rev_prob - 1)
        self.V_sum = V.sum() / ( 2 * self.rev_prob - 1)

        if len(U) > 0:
            if np.sum(U) / (2 * self.rev_prob - 1) < ( Y.shape[0] * self.min_ratio_pri_est ):
                self.y_hat = self.public_Y_mean
            else:
                self.y_hat =  V.sum() / U.sum() 
        
    def predict(self, test_X):
        y_predict = np.full(test_X.shape[0], self.y_hat)
        return y_predict
    
    
    