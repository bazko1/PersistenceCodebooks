import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation

def _sample(X, max_points=None, weight_function=None, random_state=None):
    rnd = validation.check_random_state(random_state)
    rows = X.shape[0]

    if max_points is None or rows <= max_points:
        return X

    p = None
    if weight_function:
        p = np.zeros(rows)
        for row in range(rows):
            p[row] = weight_function(X[row])
        p /= np.sum(p)

    return X[rnd.choice(rows, max_points, p=p, replace=False)]

def _grid_generator(X, y_points, x_points):
    for y in range(1, len(y_points)):
        if y == 1:
            mask = y_points[y-1] <= X[:, 1]
        else:
            mask = y_points[y-1] <  X[:, 1]
        mask &= X[:, 1] <= y_points[y]
        y_split = X[mask]

        for x in range(1, len(x_points)):
            if x == 1:
                mask = x_points[x-1] <= y_split[:, 0]
            else:
                mask = x_points[x-1] <  y_split[:, 0]
            mask &= y_split[:, 0] <= x_points[x]

            yield y_split[mask]


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

        cells_populations, _, _ = np.histogram2d(x=X[:,0], y=X[:,1], bins=(x_points, y_points))
        cells_populations = cells_populations.T
        samples_to_take = np.zeros(cells_populations.shape, dtype=np.int32)
        points_to_distribute = self.max_points

        sorting_indices = np.unravel_index(
            cells_populations.argsort(axis=None),
            cells_populations.shape)
        cells_left = cells_populations.size

        #Distribute samples to cells, moving leftover samples uniformly to rest of cells
        for cell_indices in np.column_stack(sorting_indices):
            y_i, x_i = cell_indices
            population = cells_populations[y_i, x_i]
            samples = points_to_distribute // cells_left
            if population < samples:
                points_to_distribute -= population
                samples_to_take[y_i, x_i] = population
            else:
                points_to_distribute -= samples
                samples_to_take[y_i, x_i] = samples
            cells_left -= 1

        #Sample each grid cell
        for grid_cell, samples in zip(
                _grid_generator(X, y_points, x_points),
                samples_to_take.flat):
            out.append(
                _sample(
                    grid_cell,
                    samples,
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
