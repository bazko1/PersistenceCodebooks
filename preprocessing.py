import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation


class RandomPDSampler(BaseEstimator):
    def __init__(self, max_points=None, weight_function=None, random_state=None):
        self.max_points = max_points
        self.weight_function = weight_function
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rnd = validation.check_random_state(self.random_state)

        out_diagram = np.concatenate(X)
        if not self.max_points or len(out_diagram) <= self.max_points:
            return [out_diagram]

        p = None
        if self.weight_function:
            rows = out_diagram.shape[0]
            p = np.zeros(rows)
            for row in range(rows):
                p[row] = self.weight_function(out_diagram[row])
            p /= np.sum(p)

        choice = rnd.choice(len(out_diagram), self.max_points, p=p, replace=False)

        return [out_diagram[choice]]

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
