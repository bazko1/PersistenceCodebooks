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
        return np.array(out)

    def transform(self, X):
        '''
        Returns list of bags-of-words
        '''
        out = []
        for diagram in X:
            pred = super().predict(diagram)
            histogram = np.bincount(pred, minlength=self.n_clusters)
            out.append(histogram)

        return np.array(out)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X, sample_weight)
