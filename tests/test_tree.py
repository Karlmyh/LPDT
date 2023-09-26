import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from LPDT import LPDTreeRegressor


def sample(d, n):
    X = np.random.rand(n, d)
    y = np.sin(16 * X[:,0]) + np.random.normal(0,1,n)
    return X, y


# test for LPDTreeRegressor class with fixed parameters
def test_tree():
    dim = 2
    n = 2000
    nq = 1000
    n_test = 1000
    
    np.random.seed(6)
    X_train, y_train = sample(dim, n)
    X_pub, y_pub = sample(dim, nq)
    X_test, y_test = sample(dim, n_test)
    
    model = LPDTreeRegressor(
                 splitter = "msemaxedge", 
                 epsilon = 1,
                 public_X = X_pub,
                 public_Y = y_pub,
                 min_samples_split = 5, 
                 min_samples_leaf = 2,
                 max_depth = 5, 
                 random_state = 666,
                 search_number = 10,
                 threshold = 0,
                 estimator = "random_permutation",
                 proportion_budget = 0.5,
                 range_x = "unit",
                 min_ratio_pri_est = 0,).fit(X_train, y_train)
    
    print(MSE(model.predict(X_test), y_test))


   




