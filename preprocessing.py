import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
		
class RandomPDSampler(BaseEstimator, TransformerMixin):
    def __init__(self, max_points=None, weight_function=None):
        self.max_points = max_points
        self.weight_function = weight_function
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        out_diagram = np.concatenate(X)
        if not self.max_points or len(out_diagram) <= self.max_points:
            return [out_diagram]
        
        p = None
        if self.weight_function:
            persistencies = out_diagram[:,1]
            p = np.vectorize(self.weight_function)(persistencies)
            p /= np.sum(p)

        choice = np.random.choice(len(out_diagram), self.max_points, p=p, replace=False)
        
        return [out_diagram[choice]]

class DiagramConsolidator(BaseEstimator, TransformerMixin):
    '''
    Simple class taking list of persistence diagrams and returning single consolidated diagram
    '''
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return np.concatenate(X)