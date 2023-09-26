import numpy as np
from sklearn.model_selection import GridSearchCV
from LPDT import LPDTreeRegressor


def sample(d, n):
    X = np.random.rand(n, d)
    y = np.sin(16 * X[:,0]) + np.random.normal(0,1,n)
    return X, y


# test for LPDTreeRegressor class with cross validation
def test_tree():
    dim = 2
    n = 2000
    nq = 1000
    n_test = 1000
    
    np.random.seed(6)
    X_train, y_train = sample(dim, n)
    X_pub, y_pub = sample(dim, nq)
    X_test, y_test = sample(dim, n_test)
    

    
    parameters={"min_samples_split":[2],
                "min_samples_leaf":[10,20,40,60,80,100],
                "max_depth":[1, 2, 3, 4, 5, 6],
                "public_X":[X_pub],
                "public_Y": [y_pub],
                "epsilon": [1],
                "splitter": ['msemaxedge'],
                "estimator":["random_permutation"],
                "proportion_budget" :[ 0.3, 0.5, 0.7]
                }
    cv_model = GridSearchCV(estimator = LPDTreeRegressor(), param_grid = parameters,n_jobs = 10, cv = 5)
    cv_model.fit(X_train, y_train)
    print( -cv_model.score(X_test, y_test))
   




