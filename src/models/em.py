from typing import Dict, Tuple, Union
from jaxtyping import Float

import torch

def E_step(
    initial_dist: Float[torch.Tensor, "num_states"],
    transition_matrix: Float[torch.Tensor, "num_states num_states"],
    log_likes: Float[torch.Tensor, "num_timesteps num_states"]
    ) -> Dict[str, Union[
    Float[torch.Tensor, "num_timesteps num_states"], # For "expected_states"
    Float[torch.Tensor, ""]                         # For "marginal_ll" (scalar tensor)
    ]]:
    """
    Fun the forward and backward passes and then combine to compute the
    posterior probabilities q(z_t=k).

    Parameters
    ----------
    initial_dist: $\pi$, the initial state distribution. Length K, sums to 1.
    transition_matrix: $P$, a KxK transition matrix. Rows sum to 1.
    log_likes: $\log \ell_{t,k}$, a TxK matrix of _log_ likelihoods.

    Returns
    -------
    posterior: a dictionary with the following key-value pairs:
        expected_states: a TxK matrix containing $q(z_t=k)$
        marginal_ll: the marginal log likelihood from the forward pass.
    """
    ###
    # YOUR CODE BELOW
    alphas, marginal_ll = forward_pass(initial_dist, transition_matrix, log_likes)

    betas = backward_pass(transition_matrix, log_likes)
    m = log_likes.max(dim=1, keepdim=True).values
    l = torch.exp(log_likes - m)

    tmp = alphas * betas * l
    expected_states = tmp / tmp.sum(axis=1, keepdims=True)

    #
    ###

    # Package the results into a dictionary summarizing the posterior
    posterior = dict(expected_states=expected_states,
                     marginal_ll=marginal_ll)
    return posterior


def forward_pass(
    initial_dist: Float[torch.Tensor, "num_states"],
    transition_matrix: Float[torch.Tensor, "num_states num_states"],
    log_likes: Float[torch.Tensor, "num_timesteps num_states"]
    ) -> Tuple[
    Float[torch.Tensor, "num_timesteps num_states"], # alphas
    Float[torch.Tensor, ""]                         # marginal_ll (scalar tensor)
    ]:
    """
    Perform the (normalized) forward pass of the HMM.

    Parameters
    ----------
    initial_dist: $\pi$, the initial state distribution. Length K, sums to 1.
    transition_matrix: $P$, a KxK transition matrix. Rows sum to 1.
    log_likes: $\log \ell_{t,k}$, a TxK matrix of _log_ likelihoods.

    Returns
    -------
    alphas: TxK matrix with _normalized_ forward messages $\tilde{\alpha}_{t,k}$
    marginal_ll: Scalar marginal log likelihood $\log p(x | \Theta)$
    """
    alphas = torch.zeros_like(log_likes)
    marginal_ll = 0

    ###
    # YOUR CODE BELOW
    #
    num_timesteps = log_likes.shape[0]
    prev = None
    for t in range(num_timesteps):
      if t == 0:
          alpha_t = initial_dist
      else:
          alpha_t = torch.matmul(prev, transition_matrix)

      # Save alphas
      alphas[t] = alpha_t

      # Subtract max for numerical stability
      max_loglike_t = log_likes[t].max()
      stable_loglike = log_likes[t] - max_loglike_t

      alpha_unnorm = alpha_t * torch.exp(stable_loglike)
      norm = alpha_unnorm.sum()
      prev = alpha_unnorm / norm

      marginal_ll += torch.log(norm) + max_loglike_t

    #
    ###

    return alphas, marginal_ll

def backward_pass(
    transition_matrix: Float[torch.Tensor, "num_states num_states"],
    log_likes: Float[torch.Tensor, "num_timesteps num_states"]
    ) -> Float[torch.Tensor, "num_timesteps num_states"]:
    """
    Perform the (normalized) backward pass of the HMM.

    Parameters
    ----------
    transition_matrix: $P$, a KxK transition matrix. Rows sum to 1.
    log_likes: $\log \ell_{t,k}$, a TxK matrix of _log_ likelihoods.

    Returns
    -------
    betas: TxK matrix with _normalized_ backward messages $\tilde{\beta}_{t,k}$
    """
    betas = torch.zeros_like(log_likes)
    ###
    # YOUR CODE BELOW
    T, K = log_likes.shape
    betas[T-1, :] = torch.ones(K)

    # Range from T-2
    for t in range(T-2, -1, -1):
      prev_beta = betas[t+1]
      prev_log_l = log_likes[t+1, :]
      prev_l = torch.exp(prev_log_l - prev_log_l.max())

      next_beta_unnorm = transition_matrix @ (prev_beta * prev_l)

      betas[t] = next_beta_unnorm / next_beta_unnorm.sum()

    #
    ###

    return betas


