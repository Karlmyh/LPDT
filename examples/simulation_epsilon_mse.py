import numpy as np

from sklearn.model_selection import GridSearchCV
from LPDT import LPDTreeRegressor


def sample(d, n):
    X = np.random.rand(n, d)
    y = np.sin(16 * X[:,0]) + np.random.normal(0,1,n)
    return X, y


repeat_time = 50
log_file_path = "./results/epsilon_mse.csv"
n_train, n_pub, n_test = 6000, 2000, 2000
dim = 1
epsilon_vec = [2,3,4,5,6,7,8,9,10]


for epsilon in epsilon_vec:      
    for iterate in range(repeat_time):
    
        X_train, y_train = sample(dim, n_train)
        X_pub, y_pub = sample(dim, n_pub)
        X_test, y_test = sample(dim, n_test)

        method = "LPDT"
        parameters={"min_samples_split":[2],
                    "min_samples_leaf":[10, 20, 40, 60, 80, 100],
                    "max_depth":[1, 2, 3, 4, 5, 6],
                    "public_X":[X_pub],
                    "public_Y": [y_pub],
                    "epsilon": [epsilon],
                    "splitter": ['msemaxedge'],
                    "estimator":["random_permutation"],
                    "proportion_budget" :[ 0.3, 0.5, 0.7]
                    }
        cv_model = GridSearchCV(estimator = LPDTreeRegressor(), param_grid = parameters, n_jobs = 10,cv = 5).fit(X_train, y_train)
        score = - cv_model.score(X_test, y_test)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{}\n".format(method,iterate,epsilon,score,
                                                             n_train)
            f.writelines(logs)




