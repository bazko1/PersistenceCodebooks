import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation

def _sample(X, max_points=None, weight_function=None, random_state=None):
    rnd = validation.check_random_state(random_state)
    rows = X.shape[0]

    if not max_points or rows <= max_points:
        return X

    p = None
    if weight_function:
        p = np.zeros(rows)
        for row in range(rows):
            p[row] = weight_function(X[row])
        p /= np.sum(p)

    return X[rnd.choice(rows, max_points, p=p, replace=False)]

class RandomPDSampler(BaseEstimator):
    def __init__(self, max_points=None, weight_function=None, random_state=None):
        self.max_points = max_points
        self.weight_function = weight_function
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [_sample(
            np.concatenate(X), 
            self.max_points, 
            self.weight_function, 
            self.random_state
        )]

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
class GridPDSampler(BaseEstimator):
    def __init__(self, grid_shape, max_points, weight_function=None, random_state=None):
        self.grid_shape = grid_shape
        self.max_points = max_points
        self.weight_function = weight_function
        self.random_state = random_state
       
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        out = []
        
        X = np.concatenate(X)
        y_points = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), self.grid_shape[0] + 1)
        x_points = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), self.grid_shape[1] + 1)
        
        #Create and sample grids
        for y in range(1, len(y_points)):
            if y == 1:
                indices = y_points[y-1] <= X[:, 1]
            else:
                indices = y_points[y-1] <  X[:, 1]
            indices &= X[:, 1] < y_points[y]
            y_split = X[indices]

            for x in range(1, len(x_points)):            
                if x == 1:
                    indices = x_points[x-1] <= y_split[:, 0]
                else:
                    indices = x_points[x-1] <  y_split[:, 0]
                indices &= y_split[:, 0] < x_points[x]
                grid = y_split[indices]

                out.append(
                    _sample(
                        grid, 
                        self.max_points, 
                        self.weight_function, 
                        self.random_state
                    )
                )
        
        return [np.concatenate(out)]
                
    def fit_transform(self, X, y=None):
        return self.transform(X)



class DiagramConsolidator(BaseEstimator):
    '''
    Simple class taking list of persistence diagrams and returning single consolidated diagram
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.concatenate(X)
