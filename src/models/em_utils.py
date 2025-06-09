from typing import Tuple, Any, Dict, List, Union
from jaxtyping import Float

import torch
from torch.distributions import Normal

from src.utils import Number

def sticky_transitions(
    num_states: int,
    stickiness: float = 0.95
    ) -> Float[torch.Tensor, "num_states num_states"]:
    """Creates a 'sticky' transition probability matrix.

    Diagonal elements (self-transitions) are set to `stickiness`. The
    remaining probability (1 - `stickiness`) is distributed equally among
    the other (`num_states` - 1) transitions for each row.

    Args:
        num_states: The number of states (K), determining matrix size (KxK).
        stickiness: Probability of self-transition. Defaults to 0.95.

    Returns:
        A PyTorch Tensor (shape: `num_states` x `num_states`) representing
        the sticky transition matrix, with float32 data type.
    """
    P = stickiness * torch.eye(num_states)
    P += (1 - stickiness) / (num_states - 1) * (1 - torch.eye(num_states))
    return P


def soft_band_transitions(num_states, stickiness=0.95, bandwidth=1, epsilon=1e-4):
    P = torch.full((num_states, num_states), epsilon)
    for i in range(num_states):
        start = max(0, i - bandwidth)
        end = min(num_states, i + bandwidth + 1)
        band_size = end - start - 1
        P[i, i] = stickiness
        for j in range(start, end):
            if i != j:
                P[i, j] += (1 - stickiness - band_size * epsilon) / band_size
        P[i] /= P[i].sum()  # Normalize row
    return P


def random_args(
    num_timesteps: int,
    num_states: int,
    seed: int = 0,
    offset: Number = 0,
    scale: Number = 1
    ) -> Tuple[
    Float[torch.Tensor, "num_states"],
    Float[torch.Tensor, "num_states num_states"],
    Float[torch.Tensor, "num_timesteps num_states"]
    ]:
    """Generates random HMM parameters: initial distribution, transition matrix, and log likelihoods.

    Sets the PyTorch random seed for reproducibility. It creates a uniform
    initial state distribution (`pi`), a 'sticky' transition matrix (`P`) via
    `sticky_transitions`, and log likelihoods sampled from a Normal(0,1)
    distribution, adjusted by `offset` and `scale`.

    Args:
        num_timesteps: Number of time steps (T) for log likelihoods.
        num_states: Number of hidden states (K).
        seed: Seed for PyTorch's random number generator. Defaults to 0.
        offset: Offset added to the log likelihoods. Defaults to 0.
        scale: Scale factor for the random component of log likelihoods. Defaults to 1.

    Returns:
        A tuple `(pi, P, log_likes)` containing:
        - `pi` (torch.Tensor): Uniform initial state distribution (shape: K).
        - `P` (torch.Tensor): Sticky transition probability matrix (shape: KxK).
        - `log_likes` (torch.Tensor): Simulated log likelihoods (shape: TxK).
    """
    torch.manual_seed(seed)
    pi = torch.ones(num_states) / num_states
    P = sticky_transitions(num_states)
    log_likes = offset + scale * Normal(0,1).sample((num_timesteps, num_states))
    return pi, P, log_likes

