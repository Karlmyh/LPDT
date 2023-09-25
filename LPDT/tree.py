import numpy as np
from sklearn.metrics import mean_squared_error as MSE

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter, MidPointRandomSplitter, MaxEdgeRandomSplitter, MSEReductionSplitter, MSEReductionMidpointSplitter, MSEReductionMaxEdgeSplitter
from ._estimator import RegressionRandomPermutationEstimator, RegressionLaplaceEstimator



SPLITTERS = {"purely": PurelyRandomSplitter,
             "midpoint": MidPointRandomSplitter, 
             "maxedge": MaxEdgeRandomSplitter, 
             "msereduction": MSEReductionSplitter,
             "msemaxedge": MSEReductionMaxEdgeSplitter,
             "msemidpoint":MSEReductionMidpointSplitter,
             }

ESTIMATORS = {"laplace": RegressionLaplaceEstimator,
              "random_permutation": RegressionRandomPermutationEstimator,
                }




class BaseRecursiveTree(object):
    """ Abstact Recursive Tree Structure.
    
    
    Parameters
    ----------
    splitter : splitter keyword in SPLITTERS
        Splitting scheme

    estimator : estimator keyword in ESTIMATORS
        Estimation scheme
        
    epsilon : float
        Privacy budget.

    min_samples_split : int
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int
        The minimum number of samples required in the subnodes to split an internal node.
    
    max_depth : int
        Maximum depth of the individual regression estimators.
        
    random_state : int
        Random state for building the tree.
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.

    proportion_budget : float in [0,1]
        The propotion of privacy budgets spent on the numerator.
    
    min_ratio_pri_est : float
        If the privately estimated sample in the grid is less that n * min_ratio_pri_est, 
        use public estimation.
        
    Attributes
    ----------
    n_samples : int
        Number of samples.
    
    dim : int
        Dimension of covariant.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the support, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    tree_ : binary tree object defined in _tree.py
    
    """
    def __init__(self, 
                 splitter = None, 
                 epsilon = None,
                 min_samples_split = None,
                 min_samples_leaf = None,
                 max_depth = None, 
                 random_state = None,
                 search_number = None,
                 threshold = None,
                 estimator = None,
                 proportion_budget = None,
                 min_ratio_pri_est = None,
                ):
        
        self.splitter = splitter
        self.epsilon = epsilon
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.search_number = search_number
        self.threshold = threshold
        self.estimator = estimator
        self.proportion_budget = proportion_budget
        self.min_ratio_pri_est = min_ratio_pri_est

        
             
    def _fit(self, X, Y, range_x = "unit"):
        
        self.n_samples, self.dim = X.shape
        # check the boundary
        if range_x in ["unit"]:
            X_range = np.array([np.zeros(self.dim),np.ones(self.dim)])
        if range_x in ['auto']:
            X_range = np.zeros(shape = (2, self.dim))
            X_range[0] = X.min(axis = 0) - 0.01 * (X.max(axis = 0) - X.min(axis = 0))
            X_range[1] = X.max(axis = 0) + 0.01 * (X.max(axis = 0) - X.min(axis = 0))
        self.X_range = X_range

        # begin
        SPLITTERS[self.splitter]
        splitter = SPLITTERS[self.splitter](self.random_state, self.search_number, self.threshold)
        Estimator = ESTIMATORS[self.estimator]
        
        # initiate a tree structure
        self.tree_ = TreeStruct(self.dim)
  
        # recursively build the tree
        builder = RecursiveTreeBuilder(splitter, 
                                       Estimator, 
                                       self.min_samples_split, 
                                       self.min_samples_leaf,
                                       self.max_depth, 
                                       self.epsilon,
                                       self.Y_range,
                                       self.proportion_budget,
                                      self.min_ratio_pri_est,
                                       )
      
        builder.build(self.tree_, X, Y, X_range)
        return self
        
   
        
        
    def apply(self, X):
        """Reture the belonging cell ids. 
        """
        return self.tree_.apply(X)
    
    
    def get_node_idx(self,X):
        """Reture the belonging cell ids. 
        """
        return self.apply(X)
    
    def get_node(self,X):
        """Reture the belonging node. 
        """
        return [self.tree_.leafnode_fun[i] for i in self.get_node_idx(X)]
    
    def get_all_node(self):
        """Reture all nodes. 
        """
        return list(self.tree_.leafnode_fun.values())
    
    def get_all_node_idx(self):
        """Reture all node indexes. 
        """
        return list(self.tree_.leafnode_fun.keys())
        
    
    def predict(self, X):
        
        y_hat = self.tree_.predict(X)
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis = 1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis = 1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat[np.logical_not(is_inboundary)] = 0
        return y_hat
    





class LPDTreeRegressor(BaseRecursiveTree):
    """Locally private regression tree.
    
    Parameters
    ----------
    splitter : splitter keyword in SPLITTERS
        Splitting scheme

    estimator : estimator keyword in ESTIMATORS
        Estimation scheme
        
    epsilon : float
        Privacy budget.
        
    public_X : (nq, d) numpy array
        The public X
        
    public_Y : (nq, ) numpy array
        The public Y

    min_samples_split : int
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int
        The minimum number of samples required in the subnodes to split an internal node.
    
    max_depth : int
        Maximum depth of the individual regression estimators.
        
    random_state : int
        Random state for building the tree.
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.

    proportion_budget : float in [0,1]
        The propotion of privacy budgets spent on the numerator.
    
    min_ratio_pri_est : float
        If the privately estimated sample in the grid is less that n * min_ratio_pri_est, 
        use public estimation.
        
    range_x : {"unit" or "auto"}
        If unit, use [0,1]^d as domain. 
        If auto, use the range learnt by public data.
    """
    def __init__(self, splitter = "msemaxedge", 
                 epsilon = 1,
                 public_X = None,
                 public_Y = None,
                 min_samples_split = 5, 
                 min_samples_leaf = 2,
                 max_depth = 2, 
                 random_state = 666,
                 search_number = 10,
                 threshold = 0,
                 estimator = "random_permutation",
                 proportion_budget = 0.5,
                 range_x = "unit",
                 min_ratio_pri_est = 0,
                 ):
        super(LPDTreeRegressor, self).__init__(splitter = splitter,
                                               epsilon = epsilon, 
                                               min_samples_split = min_samples_split,
                                               min_samples_leaf = min_samples_leaf,
                                               max_depth = max_depth, 
                                               random_state = random_state,
                                               search_number = search_number,
                                               threshold = threshold,
                                               estimator = estimator,
                                               proportion_budget = proportion_budget,
                                               min_ratio_pri_est = min_ratio_pri_est,
                                              )
        
        
        self.public_X = public_X
        self.public_Y = public_Y
        self.range_x = range_x
        
        
    def get_partition(self, X, Y, range_x = "unit"):
        """Fit the tree with public data.
        """
        return self._fit( X, Y, range_x = range_x)
        
    
    
    def attribute_data(self, X, Y):
        """Fit the estimator with private data.
        """
        test_idx = self.apply(X)
        for node_idx in self.get_all_node_idx():
            in_idx  = test_idx == node_idx
            self.tree_.leafnode_fun[node_idx].get_data(Y, in_idx)
            
    
    
    def fit(self, X, Y):
        
        if self.public_X is None:
            raise ValueError
        else:
            self.Y_range = Y.max() - Y.min()
            self.get_partition(self.public_X, self.public_Y, self.range_x)
        self.attribute_data(X, Y)
        return self
    
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['min_samples_split',"min_samples_leaf", "max_depth",
                    "splitter", "epsilon", "public_X", "public_Y",
                    "search_number", "threshold", "estimator",
                   "proportion_budget", "range_x","min_ratio_pri_est"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self

    def score(self, X, y):
        """Reture the regression score, i.e. MSE.
        """
        return - MSE(self.predict(X),y)
    
    

