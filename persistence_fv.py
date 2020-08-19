import numpy as np
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.preprocessing import MaxAbsScaler
from preprocessing import RandomPDSampler

class PersistenceFV(BaseEstimator, TransformerMixin):
    """Fisher vector wrap for persistence diagrams.

    Implements fisher vector algorithms for comparison
    experiments purposes as described in section 3.7.
    https://arxiv.org/pdf/1802.04852.pdf?fbclid=IwAR0Or4JbGpvQPr7Il9bLZ7vVZetyOCjRPNF1MuOJ1H9bEwNl7inp4VgUhmo#subsection.3.7
    """
    def __init__(self,
                 fisher_vector=None,
                 transformator=BirthPersistenceTransform(),
                 scaler=DiagramScaler(use=True, scalers=[((0,), MaxAbsScaler(copy=False)), ((1,), MaxAbsScaler(copy=False))]),
                 sampler=None):
        """
        Parameters:
            fisher_vector: Fisher vector encoder object (sklearn API consistent).
                    Eg. FisherVectorEncoder.
            transformator: PD flow initial transformator.
            scaler: PD flow initial scaler.
            sampler: Data sampler to be used during train.
        """
        self.fisher_vector = fisher_vector
        self.transformator = transformator
        self.scaler = scaler
        self.sampler = sampler

    def fit(self, X, y=None):
        """
        Fits underlying fisher vector, transformator, and scaler.
        
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
        
        self.fisher_vector.fit(X, y)

        return self

    def transform(self, X, y=None):
        """Computes the fisher vector for each diagram from `X` (after transforming and scaling it first)."""
        if self.transformator:
             X = self.transformator.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)
            
        return np.array([self.fisher_vector.transform(diagram) for diagram in X])

    def __fisher_vector(self, x):
        """Compute Fisher vector from feature vector x."""
        means, covars, priors = self.gmm_
        x = np.float32(x)
        return fisher(x, means, covars, priors, improved=True, normalized=True)

