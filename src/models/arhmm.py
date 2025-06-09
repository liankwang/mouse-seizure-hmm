from typing import List, Dict, Tuple, Any
from jaxtyping import Float

import torch
from torch.distributions import MultivariateNormal

class LinearRegressionObservations(object):
    """
    Wrapper for a collection of Gaussian observation parameters.
    """
    num_states: int      # K
    data_dim: int        # D (dimension of observed data y_t)
    covariate_dim: int   # M (dimension of covariates phi_t)

    weights: Float[torch.Tensor, "num_states data_dim covariate_dim"]       # W_k (K, D, M)
    covs: Float[torch.Tensor, "num_states data_dim data_dim"]               # Q_k (K, D, D)

    def __init__(self, num_states: int, data_dim: int, covariate_dim: int) -> None:
        """
        Initialize a collection of observation parameters for an HMM whose
        observation distributions are linear regressions. The HMM has
        `num_states` (i.e. K) discrete states, `data_dim` (i.e. D)
        dimensional observations, and `covariate_dim` covariates.
        In an ARHMM, the covariates will be functions of the past data.
        """
        self.num_states = num_states
        self.data_dim = data_dim
        self.covariate_dim = covariate_dim

        # Initialize the model parameters
        self.weights = torch.zeros((num_states, data_dim, covariate_dim))
        self.covs = torch.tile(torch.eye(data_dim), (num_states, 1, 1))

    @staticmethod
    def precompute_suff_stats(dataset: List[Dict[str, Any]]) -> None:
        """
        Compute the sufficient statistics of the linear regression for each
        data dictionary in the dataset. This modifies the dataset in place.

        Parameters
        ----------
        dataset: a list of data dictionaries.

        Returns
        -------
        Nothing, but the dataset is updated in place to have a new `suff_stats`
            key, which contains a tuple of sufficient statistics.
        """
        ###
        # YOUR CODE BELOW
        #
        for data in dataset:
            x = data['data']
            phi = data['covariates']

            data['suff_stats'] = (torch.ones(x.shape[0]),       # 1 (T,)
                                  torch.einsum('ti,tj->tij', x, x), # x_t x_t^T (T, D, D)
                                  torch.einsum('ti,tj->tij', x,phi), # x_t phi_t^T (T, D, M)
                                  torch.einsum('ti,tj->tij', phi,phi) # phi_t phi_t^T (T, M, M)
                                  )
        #
        ###

    def log_likelihoods(
            self,
            data: Dict[str, Any]
            ) -> Float[torch.Tensor, "num_timesteps num_states"]:
        """
        Compute the matrix of log likelihoods of data for each state.
        (I like to use torch.distributions for this, though it requires
         converting back and forth between numpy arrays and pytorch tensors.)

        Parameters
        ----------
        data: a dictionary with multiple keys, including "data", the TxD array
            of observations for this mouse.

        Returns
        -------
        log_likes: a TxK array of log likelihoods for each datapoint and
            discrete state.
        """
        ###
        # YOUR CODE BELOW
        #
        x = data['data'] # (T, D)
        phi = data['covariates'] # (T, M)

        log_likes = torch.zeros((x.shape[0], self.num_states))
        for k in range(self.num_states):
          mean = phi @ self.weights[k].T # (T, D)
          #eigvals = torch.linalg.eigh(self.covs[k]) # (D, D)
          #print(f"State {k} eigvals: {eigvals}")
          mvn = MultivariateNormal(mean, self.covs[k])
          log_likes[:, k] = mvn.log_prob(x)

        #
        ###
        return log_likes

    def M_step(
        self,
        stats: Tuple[
            Float[torch.Tensor, "num_states"],
            Float[torch.Tensor, "num_states data_dim data_dim"],
            Float[torch.Tensor, "num_states data_dim covariate_dim"],
            Float[torch.Tensor, "num_states covariate_dim covariate_dim"]
        ]
    ) -> None:
        """
        Compute the linear regression parameters given the expected
        sufficient statistics.

        Note: add a little bit (1e-4 * I) to the diagonal of each covariance
            matrix to ensure that the result is positive definite.


        Parameters
        ----------
        stats: a tuple of expected sufficient statistics

        Returns
        -------
        Nothing, but self.weights and self.covs are updated in place.
        """
        Ns, t1s, t2s, t3s = stats
        ###
        # YOUR CODE BELOW
        eps = torch.eye(self.data_dim) * 1e-4

        for k in range(self.num_states):
          W_k = t2s[k] @ t3s[k].inverse() # (D, M) @ (M, M)
          self.weights[k] = W_k
          tmp = t1s[k] - 2 * t2s[k] @ W_k.T + W_k @ t3s[k] @ W_k.T

          cov_k = tmp / Ns[k] + eps
          cov_k = 0.5 * (cov_k + cov_k.T)  # enforce symmetry
          self.covs[k] = cov_k
        #
        #
        #
        #
        ###


def precompute_ar_covariates(
    dataset: List[Dict[str, Any]],
    num_lags: int = 2,
    fit_intercept: bool = True
    ) -> None:
    """Precomputes autoregressive (AR) covariates for time series data.

    For each data item in `dataset` (which must contain a 'data' key mapping
    to a TxD_in tensor), this function generates AR features. These include
    `num_lags` lagged versions of the 'data' tensor (zero-padded at the start)
    and optionally an intercept term. The resulting covariate matrix is added
    to each data item's dictionary under the key 'covariates' (in-place).

    Args:
        dataset: A list of dictionaries. Each dictionary is expected to have
                 a 'data' key with a 2D PyTorch Tensor (time_steps x features).
                 These dictionaries are modified in-place.
        num_lags: Number of past time steps to include as covariates.
                  Defaults to 2.
        fit_intercept: If True, adds a constant intercept term (a column of
                       ones) to the covariates. Defaults to True.

    Returns:
        None. The `dataset` dictionaries are modified directly.
    """
    for data in dataset:
        x = data["data"]
        data_dim = x.shape[1]

        ###
        # YOUR CODE BELOW
        G = num_lags
        T = x.shape[0]

        padding = torch.zeros((num_lags, data_dim))
        x = torch.cat((padding, x), dim=0)

        phi = []
        for g in range(1, G+1):
          phi.append(x[G - g:T + G - g])

        phi = torch.cat(phi, dim=1)

        if fit_intercept:
          phi = torch.cat([phi, torch.ones((T,1))], dim=1)

        data["covariates"] = phi
        #
        ###