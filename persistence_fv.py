import numpy as np
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.preprocessing import MaxAbsScaler
from preprocessing import RandomPDSampler

def fixarray(x):
    d = x.shape
    x = x.T.reshape([d[0]*d[1],])
    x = x.reshape([d[0],d[1]])
    return x

class PersistenceFV(BaseEstimator, TransformerMixin):
    """Fit GMM and compute Fisher vectors"""

    def __init__(self,
                 gmm_clusters_number=10,
                 init_mode='kmeans',
                 transformator=BirthPersistenceTransform(),
                 scaler=DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler=None):
        self.gmm_clusters_number = gmm_clusters_number
        self.init_mode = init_mode
        self.transformator = transformator
        self.scaler = scaler
        self.sampler = sampler
        self.gmm_ = None

    def fit(self, X, y=None):
        if self.transformator:
            X = self.transformator.fit_transform(X, y)
        if self.scaler:
            X = self.scaler.fit_transform(X, y)
        if self.sampler:
            X = self.sampler.fit_transform(X, y)
        X = np.float32(np.concatenate(X))

        means, covars, priors, ll, posteriors = gmm(
            X,
            n_clusters=self.gmm_clusters_number,
            init_mode=self.init_mode,
        )
        means = means.transpose()
        covars = covars.transpose()
        self.gmm_ = means, covars, priors

        return self
    
    def set_model(self, means, covars, priors):
        self.gmm_ = means, covars, priors
        return self
    
    def transform(self, X, y=None):
        if self.transformator:
             X = self.transformator.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)
        return np.array(list(map(lambda x: self.__fisher_vector(x), X)))

    def __fisher_vector(self, x):
        """Compute Fisher vector from feature vector x."""
        means, covars, priors = self.gmm_
        x = np.float32(x.T)
        # fixing arrays due to bug in vlfeat
        x = fixarray(x)
        means = fixarray(means)
        covars = fixarray(covars)
        return fisher(x, means, covars, priors, improved=True)

