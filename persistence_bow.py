import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.preprocessing import MaxAbsScaler


class PersistenceBow(BaseEstimator, TransformerMixin, ClusterMixin):
    """
    Class used for vectorization of persistence diagrams.

    Implements algorithm described in section 3.1 of `Persistence Codebooks for Topological Data Analysis <https://arxiv.org/pdf/1802.04852.pdf#subsection.3.1>`.
    Original uses KNN for clustering, but this class should be able to use any hard-clustering class compatible with scikit api.
    """

    def __init__(self,
                 cluster,
                 *,
                 transformator=BirthPersistenceTransform(),
                 scaler=DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler=None,
                 normalize=True,
                 cluster_weighting=None
                 ):
        """
        PersistenceBow constructor.

        Parameters:
            cluster: Clustering object (sklearn API consistent) should contain n_clusters attribute.
                    Eg. sklearn.cluster.KMeans.
            transformator: PD flow initial transformator.
            scaler: PD flow initial scaler.
            sampler: Data sampler to be used during train.
            normalize: If normalize PBow by taking the square root of each component
                    and dividing it by the norm of the whole vector.
            cluster_weighting: Weighting function to be applied on diagrams (R^2 -> R).
                               If None, all observations are assigned equal weight.
        """
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
        """
        Fit the PersitenceBow class on a list of persistence diagrams.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels.
        """
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
        """
        Cluster predict on a list of persistence diagrams.
        """
        out = []
        for diagram in X:
            out.append(self.cluster.predict(diagram, sample_weight))
        return np.array(out)

    def transform(self, X):
        """
        Compute persistence-bags-of-words for each diagram.
        
        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
        Returns:
            numpy array with n (number of diagrams) n_clusters shaped numpy arrays containing 
            pbow calculation for each diagram.
        """
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
    """
    Class used for stable vectorization of persistence diagrams.

    Implements algorithm described in section 3.4 of `Persistence Codebooks for Topological Data Analysis <https://arxiv.org/pdf/1802.04852.pdf#subsection.3.4>`.
    Uses gaussian mixture model in order to be stable with respect to 1-Wasserstein distance between the diagrams.
    """

    def __init__(self,
                 mixture,
                 *,
                 transformator=BirthPersistenceTransform(),
                 scaler=DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler=None,
                 normalize=True,
                 cluster_weighting=None):
        """
        StablePersistenceBow constructor.
        
        Parameters:
            mixture: Gaussian mixture model implementation compatible with sklern API.
                    Should contain n_components, weights_ attribute.
            transformator: PD flow initial transformator.
            scaler: PD flow initial scaler.
            sampler: Data sampler to be used during train.
            normalize: If normalize PBow by taking the square root of each component
                    and dividing it by the norm of the whole vector.
            cluster_weighting: Weighting function to be applied on diagrams (R^2 -> R).
                               If None, all observations are assigned equal weight.
        """
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
        """
        Gaussian mixture predict on each diagram.
        """
        out = []
        for diagram in X:
            out.append(self.mixture.predict(diagram))

        return np.array(out)

    def transform(self, X):
        """
        Compute stable persistence-bags-of-words for each diagram.
        
        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
        Returns:
            numpy array with n (number of diagrams) n_components shaped numpy arrays containing 
            spbow calculation for each diagram.
        """
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
