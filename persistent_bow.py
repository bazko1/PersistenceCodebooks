import numpy as np
from sklearn.cluster import KMeans

class PersistentBow(KMeans):
    def fit(self, X, y=None, sample_weight=None):
        X = np.concatenate(X)
        return super().fit(X, y, sample_weight)
    
    def predict(self, X, sample_weight=None):
        out = []
        for diagram in X:
            out.append(super().predict(diagram, sample_weight))
        return out
        