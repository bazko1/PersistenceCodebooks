import numpy as np
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.preprocessing import MaxAbsScaler
from preprocessing import RandomPDSampler


class FisherVectorTransformer(BaseEstimator, TransformerMixin):
    """Fit GMM and compute Fisher vectors"""

    def __init__(self,
                 gmm_clusters_number=10,
                 sampler=RandomPDSampler(1000),
                 init_mode='kmeans'):

        self.gmm_clusters_number = gmm_clusters_number
        self.init_mode = init_mode
        self.transformator = BirthPersistenceTransform()
        self.scaler = DiagramScaler(
            use=True,
            scalers=[((0,),
                      MaxAbsScaler(copy=False)),
                      ((1,),
                      MaxAbsScaler(copy=False))])

        self.sampler = sampler
        self.gmm_ = None

    def fit(self, X, y=None):

        X = self.transformator.fit_transform(X, y)
        X = self.scaler.fit_transform(X, y)

        if self.sampler:
            X = np.float32(self.sampler.fit_transform(X, y))[0]
        else:
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

    def transform(self, X, y=None):
        X = self.transformator.transform(X)
        X = self.scaler.transform(X)
        return np.array(list(map(lambda x: self.__fisher_vector(x), X)))


    def __fisher_vector(self, x):
        """Compute Fisher vector from feature vector x."""
        means, covars, priors = self.gmm_
        x = np.float32(x.transpose())
        return fisher(x, means, covars, priors, improved=True)
