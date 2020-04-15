import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.preprocessing import MaxAbsScaler


class PersistentBow(BaseEstimator, TransformerMixin, ClusterMixin):

    def __init__(self,
                 cluster,
                 *,
                 transformator = BirthPersistenceTransform(),
                 scaler = DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler = None
                 ):
        self.cluster = cluster
        self.transformator = transformator
        self.scaler = scaler
        self.sampler = sampler

    @property
    def n_clusters(self):
        return self.cluster.n_clusters

    def fit(self, X, y=None, sample_weight=None):
        # TODO: To be discussed, is it enough repeated operations that we should prepare Pipeline[transformator, scaler?
        if self.transformator:
            self.transformator.fit(X, y)
        if self.scaler:
            self.scaler.fit(X, y)
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
        if self.transformator:
            X = self.transformator.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)

        for diagram in X:
            pred = self.cluster.predict(diagram)
            histogram = np.bincount(pred, minlength=self.n_clusters)
            out.append(histogram)

        return np.array(out)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X, sample_weight)
