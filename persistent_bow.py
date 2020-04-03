import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin

class PersistentBow(BaseEstimator, TransformerMixin, ClusterMixin):
    def __init__(self, cluster, sampler=None):
        self.cluster = cluster
        self.sampler = sampler

    @property
    def n_clusters(self):
        return self.cluster.n_clusters

    def fit(self, X, y=None, sample_weight=None):
        if self.sampler:
            X = self.sampler.fit_transform(X, y)
        X = np.concatenate(X)
        self.cluster.fit(X, y, sample_weight)

        return self

    def predict(self, X, sample_weight=None):
        out = []
        for diagram in X:
            out.append(self.cluster.predict(diagram, sample_weight))
        return np.array(out)

    def transform(self, X):
        '''
        Returns list of bags-of-words
        '''
        out = []
        for diagram in X:
            pred = self.cluster.predict(diagram)
            histogram = np.bincount(pred, minlength=self.n_clusters)
            out.append(histogram)

        return np.array(out)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X, sample_weight)
