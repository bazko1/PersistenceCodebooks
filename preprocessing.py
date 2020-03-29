import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# class DiagramScaler(BaseEstimator, TransformerMixin):
#     """
#     This is a class for diagram transformation to be in range [0, 1] x [0, 1] with the respect 
#     to maximum value of all (x, y) column of all given diagrams.
#     """
#     def __init__(self):
#         return None
    
#     def fit(self, X, y=None):
#         diagram_column_maxes = [np.max(diagram, axis=0) for diagram in X]
#         self.all_diag_max = np.max(diagram_column_maxes, axis=0)
#         return self
    
#     def transform(self, X):
#         return X / self.all_diag_max
		
		
class RandomPDSampler(BaseEstimator, TransformerMixin):
    def __init__(self, max_points=None, weight_function=None):
        self.max_points = max_points
        self.weight_function = weight_function
        
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        out_diagram = np.concatenate(X)
        if not self.max_points or len(out_diagram[0]) <= self.max_points:
            return [out_diagram]
        
        p = None
        if self.weight_function:
            persistencies = out_diagram[:,1]
            p = np.vectorize(self.weight_function)(persistencies)
            p /= np.sum(p)

        choice = np.random.choice(len(out_diagram), self.max_points, p=p, replace=False)
        
        return [out_diagram[choice]]
