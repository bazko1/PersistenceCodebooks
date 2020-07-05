import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.preprocessing import MaxAbsScaler


class PersistenceBow(BaseEstimator, TransformerMixin, ClusterMixin):
    def __init__(self,
                 cluster,
                 *,
                 transformator=BirthPersistenceTransform(),
                 scaler=DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler=None,
                 normalize=True,
                 cluster_weighting=None
                 ):
        self.cluster = cluster
        self.transformator = transformator
        self.scaler = scaler
        self.sampler = sampler
        self.normalize = normalize
        self.cluster_weighting = cluster_weighting

    @property
    def n_clusters(self):
        return self.cluster.n_clusters

    def fit(self, X, y=None, sample_weight=None):
        if self.transformator:
            X = self.transformator.fit_transform(X, y)
        if self.scaler:
            X = self.scaler.fit_transform(X, y)
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
            weights_ = None

            if self.cluster_weighting:
                weights_ = tuple(map(self.cluster_weighting, diagram))
            histogram = np.bincount(pred, weights=weights_, minlength=self.n_clusters)

            if self.normalize:
                norm = np.linalg.norm(histogram)
                if not np.isclose(norm, 0):
                    histogram = np.array([np.sign(el) * np.sqrt(np.abs(el)) for el in histogram]) \
                                / norm

            out.append(histogram)

        return np.array(out)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X, sample_weight)


class StablePersistenceBow(BaseEstimator, TransformerMixin, ClusterMixin):
    def __init__(self,
                 mixture,
                 *,
                 transformator=BirthPersistenceTransform(),
                 scaler=DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler=None,
                 normalize=True,
                 cluster_weighting=None):
        self.mixture = mixture
        self.transformator = transformator
        self.scaler = scaler
        self.sampler = sampler
        self.normalize = normalize
        self.cluster_weighting = cluster_weighting

    def fit(self, X, y=None):
        if self.transformator:
            X = self.transformator.fit_transform(X, y)
        if self.scaler:
            X = self.scaler.fit_transform(X, y)
        if self.sampler:
            X = self.sampler.fit_transform(X, y)

        X = np.concatenate(X)
        self.mixture.fit(X, y)

        return self

    def predict(self, X):
        out = []
        for diagram in X:
            out.append(self.mixture.predict(diagram))

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
            probabilities = self.mixture.predict_proba(diagram)
            if self.cluster_weighting:
                probabilities *= np.array(list(map(self.cluster_weighting, diagram))).reshape(-1, 1)
            histogram = np.sum(probabilities, axis=0) * self.mixture.weights_

            if self.normalize:
                norm = np.linalg.norm(histogram)
                if not np.isclose(norm, 0):
                    histogram = np.array([np.sign(el) * np.sqrt(np.abs(el)) for el in histogram]) \
                                / norm

            out.append(histogram)

        return np.array(out)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
