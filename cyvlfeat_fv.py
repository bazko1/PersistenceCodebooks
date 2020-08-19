from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn.base import BaseEstimator, TransformerMixin

class FisherVectorEncoder(BaseEstimator, TransformerMixin):
    '''
    Sklearn API compatible wrapper around cyvlfeat Fisher vector encoder and GMM
    '''
    def __init__(cluster_number, init_mode="kmeans", fast=False):
        '''
        Parameters:
            cluster_number: Number of gaussian mixtures to fit 
            init_mode: {'rand', 'kmeans', 'custom'}
                The initialization mode:
                  - rand: Initial mean positions are randomly  chosen among
                          data samples
                  - kmeans: The K-Means algorithm is used to initialize the cluster
                            means
                  - custom: The intial parameters are provided by the user, through
                            the use of ``init_priors``, ``init_means`` and
                            ``init_covars``. Note that if those arguments are given
                            then the ``init_mode`` value is always considered as
                            ``custom``            scaler: PD flow initial scaler.
            fast: If ``True``, uses slightly less accurate computations but significantly
                increase the speed in some cases (particularly with a large number of
                Gaussian modes).
        '''
        self.cluster_number = cluster_number
        self.init_mode = init_mode
        self.improved = improved
        self.fast = fast
        self._gmm = None
    
    def fit(X, y=None):
        means, covars, priors, ll, posteriors = gmm(
            X,
            n_clusters=self.cluster_number,
            init_mode=self.init_mode,
        )
        self._gmm = means, covars, priors
        
        return self
    
    def transform(X):
        means, covars, priors = self.gmm_
        X = np.float32(X)
        return fisher(X, means, covars, priors, improved=True, fast=self.fast)
    
    