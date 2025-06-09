from typing import List, Dict, Tuple, Any
from jaxtyping import Float

import torch
from torch.distributions import MultivariateNormal


class GaussianObservations(object):
    """
    Wrapper for a collection of Gaussian observation parameters.
    """
    # Instance variable type hints
    num_states: int
    data_dim: int
    means: Float[torch.Tensor, "num_states data_dim"]
    covs: Float[torch.Tensor, "num_states data_dim data_dim"]

    def __init__(self, num_states: int, data_dim: int) -> None:
        """
        Initialize a collection of observation parameters for a Gaussian HMM
        with `num_states` (i.e. K) discrete states and `data_dim` (i.e. D)
        dimensional observations.
        """
        self.num_states = num_states
        self.data_dim = data_dim
        self.means = torch.zeros((num_states, data_dim), dtype=torch.float32)
        self.covs = torch.tile(torch.eye(data_dim), (num_states, 1, 1))

    @staticmethod
    def precompute_suff_stats(dataset: List[Dict[str, Any]]) -> None:
        """
        Compute the sufficient statistics of the Gaussian distribution for each
        data dictionary in the dataset. This modifies the dataset in place.

        Parameters
        ----------
        dataset: a list of data dictionaries.

        Returns
        -------
        Nothing, but the dataset is updated in place to have a new `suff_stats`
            key, which contains a tuple of sufficient statistics.
        """
        for data in dataset:
            x = data['data']
            data['suff_stats'] = (torch.ones(len(x)),                  # 1
                                  x,                                   # x_t
                                  torch.einsum('ti,tj->tij', x, x))    # x_t x_t^T

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
        x = data["data"]

        ###
        # YOUR CODE BELOW
        #
        log_likes = torch.zeros((x.shape[0], self.num_states), dtype=torch.float32)
        chunk_size = 100000
        for k in range(self.num_states):
            #print("Computing log_likelihoods for state", k)
            eigvals = torch.linalg.eigvalsh(self.covs[k]) # (D, D)
            #print(f"State {k} eigvals: {eigvals}")
            is_pd = torch.all(eigvals > 0)
            #print(f"State {k}: min eigenvalue = {eigvals.min().item()}, PD = {is_pd}")

            mvn = MultivariateNormal(self.means[k], self.covs[k])
            
            for start in range(0, x.shape[0], chunk_size):
                end = start + chunk_size
                log_likes[start:end, k] = mvn.log_prob(x[start:end])
        #
        ###
        return log_likes

    def M_step(
        self,
        stats: Tuple[ # This tuple structure comes from compute_expected_suff_stats
            Float[torch.Tensor, "num_states"],               # Ns (expected counts per state)
            Float[torch.Tensor, "num_states data_dim"],      # t1s (expected sum_of_x per state)
            Float[torch.Tensor, "num_states data_dim data_dim"] # t2s (expected sum_of_xxT per state)
        ]
        ) -> None:
        """
        Compute the Gaussian parameters give the expected sufficient statistics.

        Note: add a little bit (1e-4 * I) to the diagonal of each covariance
            matrix to ensure that the result is positive definite.

        Parameters
        ----------
        stats: a tuple of expected sufficient statistics

        Returns
        -------
        Nothing, but self.means and self.covs are updated in place.
        """
        Ns, t1s, t2s = stats

        ###
        # YOUR CODE BELOW
        eps = torch.eye(self.data_dim, device=self.covs.device, dtype=torch.float32) * 1e-4

        for k in range(self.num_states):
            #print("Computing cov in M_step for state", k)
            self.means[k] = t1s[k] / Ns[k] # (D, 1)
            # tmp = t2s[k] - torch.outer(t1s[k], t1s[k]) / Ns[k] # (D, D)
            # self.covs[k] = tmp / Ns[k] + eps # (D, D)
            cov = t2s[k] / Ns[k] - torch.outer(self.means[k], self.means[k])
            
            # Ensure symmetry
            cov = 0.5 * (cov + cov.T)

            # Add small value to diagonal for numerical stability
            min_eigenval = torch.linalg.eigvalsh(cov).min()
            if min_eigenval < 1e-8:
                print(f"Fixing non-PD matrix in state {k}")
                correction = (1e-4 - min_eigenval).clamp(min=1e-4)
                cov += torch.eye(self.data_dim, device=cov.device) * correction
            else:
                cov += eps

            self.covs[k] = cov

            # Check if covs[k] is positive definite
            # eigvals = torch.linalg.eigvalsh(self.covs[k])
            # is_pd = torch.all(eigvals > 0)
            # print(f"State {k}: min eigenvalue = {eigvals.min().item()}, PD = {is_pd}")

        #
        ###