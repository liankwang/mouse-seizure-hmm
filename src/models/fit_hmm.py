from typing import List, Dict, Tuple, Any, Union
from jaxtyping import Float
from tqdm.auto import trange

import torch
from src.data_processing.utils import combine
from models.em import E_step
from models.em_utils import sticky_transitions, soft_band_transitions
from models.ghmm import GaussianObservations
from models.arhmm import LinearRegressionObservations, precompute_ar_covariates

def fit_hmm_wrapper(train_dataset,
                    test_dataset,
                    model_type,
                    num_states=50,
                    transition_matrix_method="sticky",
                    stickiness=0.95,
                    bandwidth=2,
                    num_lags=2,
                    num_iters=50):
    
    if model_type not in ["ghmm", "arhmm"]:
        raise ValueError(f"Unknown HMM type: {model_type}")

    # Initialize distribution over latent states
    initial_dist = torch.ones(num_states) / num_states

    # Create the fixed transition matrix        
    if transition_matrix_method == "soft_band":
        print("Creating soft band transition matrix...")
        transition_matrix = soft_band_transitions(num_states, stickiness, bandwidth)
    else:
        print("Creating sticky transition matrix...")
        transition_matrix = sticky_transitions(num_states, stickiness)


    # Precompute sufficient statistics and initialize Observations object
    data_dim = train_dataset[0]["data"].shape[1]
    if model_type == "ghmm":
        print("Precomputing sufficient statistics for Gaussian HMM...")
        GaussianObservations.precompute_suff_stats(train_dataset)
        GaussianObservations.precompute_suff_stats(test_dataset)
        observations = GaussianObservations(num_states, data_dim)
    elif model_type == "arhmm":
        print("Precomputing AR covariates and sufficient statistics for AR HMM...")
        precompute_ar_covariates(train_dataset, num_lags)
        precompute_ar_covariates(test_dataset, num_lags)
        LinearRegressionObservations.precompute_suff_stats(train_dataset)
        LinearRegressionObservations.precompute_suff_stats(test_dataset)
        observations = LinearRegressionObservations(num_states, data_dim,
                                            num_lags * data_dim + 1)
    
    # Fit the HMM with EM
    train_lls, test_lls, posteriors, test_posteriors = fit_hmm(
        train_dataset[:1],
        test_dataset[:1],
        initial_dist,
        transition_matrix,
        observations,
        num_iters
    )

    return train_lls, test_lls, posteriors, test_posteriors


def fit_hmm(
    train_dataset: List[Dict[str, Any]],
    test_dataset: List[Dict[str, Any]],
    initial_dist: Float[torch.Tensor, "num_states"],
    transition_matrix: Float[torch.Tensor, "num_states num_states"],
    observations: Any,
    num_iters: int = 50,
    seed: int = 0
    ) -> Tuple[
    Float[torch.Tensor, "num_iters"],
    Float[torch.Tensor, "num_iters"],
    List[Dict[
    str,
    Union[
        Float[torch.Tensor, "num_timesteps num_states"],
        Float[torch.Tensor, ""]
        ]]],
    List[Dict[
    str,
    Union[
        Float[torch.Tensor, "num_timesteps num_states"],
        Float[torch.Tensor, ""]
        ]]]
    ]:
    """
    Fit a Hidden Markov Model (HMM) with expectation maximization (EM).

    Note: This is only a partial fit, as this method will treat the initial
    state distribution and the transition matrix as fixed!

    Parameters
    ----------
    train_dataset: a list of dictionary with multiple keys, including "data",
        the TxD array of observations for this mouse, and "suff_stats", the
        tuple of sufficient statistics.

    test_dataset: as above but only used for tracking the test log likelihood
        during training.

    initial_dist: a length-K vector giving the initial state distribution.

    transition_matrix: a K x K matrix whose rows sum to 1.

    observations: an Observations object with `log_likelihoods` and `M_step`
        functions.

    seed: random seed for initializing the algorithm.

    num_iters: number of EM iterations.

    Returns
    -------
    train_lls: array of likelihoods of training data over EM iterations
    test_lls: array of likelihoods of testing data over EM iterations
    posteriors: final list of posterior distributions for the training data
    test_posteriors: final list of posterior distributions for the test data
    """
    # Get some constants
    num_states = observations.num_states
    num_train = sum([len(data["data"]) for data in train_dataset])
    num_test = sum([len(data["data"]) for data in test_dataset])

    # Check the initial distribution and transition matrix
    assert initial_dist.shape  == (num_states,) and \
        torch.all(initial_dist >= 0) and \
        torch.isclose(initial_dist.sum(), torch.tensor(1.0))
    assert transition_matrix.shape  == (num_states, num_states) and \
        torch.all(transition_matrix >= 0) and \
        torch.allclose(transition_matrix.sum(axis=1), torch.tensor(1.0))

    # Initialize with a random posterior
    posteriors = initialize_posteriors(train_dataset, num_states, seed=seed)
    #print("Now computing suff stats...")
    stats = compute_expected_suff_stats(train_dataset, posteriors)

    # Track the marginal log likelihood of the train and test data
    train_lls = []
    test_lls = []

    # Main loop
    for itr in trange(num_iters, desc="Fitting HMM with EM"):
        ###
        # YOUR CODE BELOW
        #
        # M step: update the parameters of the observations using the
        #         expected sufficient stats.
        observations.M_step(stats)

        # E step: computhe the posterior for each data dictionary in the dataset
        posteriors = []
        for data in train_dataset:
          lls = observations.log_likelihoods(data)
          
          posteriors.append(E_step(initial_dist, transition_matrix, lls))

        # Compute the expected sufficient statistics under the new posteriors
        stats = compute_expected_suff_stats(train_dataset, posteriors)

        # Store the average train likelihood
        avg_train_ll = sum([p["marginal_ll"] for p in posteriors]) / num_train
        train_lls.append(avg_train_ll)

        # Compute the posteriors for the test dataset too
        test_posteriors = []
        for data in test_dataset:
          lls = observations.log_likelihoods(data)
          test_posteriors.append(E_step(initial_dist, transition_matrix, lls))

        # Store the average test likelihood
        avg_test_ll = sum([p["marginal_ll"] for p in test_posteriors]) / num_test
        test_lls.append(avg_test_ll)

        #
        ###

    # convert lls to arrays
    train_lls = torch.stack(train_lls)
    test_lls = torch.stack(test_lls)
    return train_lls, test_lls, posteriors, test_posteriors



def compute_expected_suff_stats(
    dataset: List[Dict[str, Any]],
    posteriors: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, ...]:
    """
    Compute a tuple of normalized sufficient statistics, taking a weighted sum
    of the posterior expected states and the sufficient statistics, then
    normalizing by the length of the sequence. The statistics are combined
    across all mice (i.e. all the data dictionaries and posterior dictionaries).

    Parameters
    ----------
    dataset: a list of dictionary with multiple keys, including "data", the TxD
        array of observations for this mouse, and "suff_stats", the tuple of
        sufficient statistics.

    Returns
    -------
    stats: a tuple of normalized sufficient statistics. E.g. if the
        "suff_stats" key has four arrays, the stats tuple should have four
        entires as well. Each entry should be a K x (size of statistic) array
        with the expected sufficient statistics for each of the K discrete
        states.
    """
    assert isinstance(dataset, list)
    assert isinstance(posteriors, list)

    # Helper function to compute expected counts and sufficient statistics
    # for a single time series and corresponding posterior.
    def _compute_expected_suff_stats(data, posterior):
        ###
        # YOUR CODE BELOW
        # Hint: einsum might be useful
        q = posterior['expected_states']
        x = data['data']
        suff_stats = data['suff_stats']

        if len(suff_stats) == 3:
          # Gaussian HMM
          Ns, t1s, t2s = suff_stats

          Ns = q.sum(dim=0) # (K x 1)
          t1s = torch.einsum('tk,td->kd', q, t1s) # (K x D)
          t2s = torch.einsum('tk,tij->kij', q, t2s) # (K x D x D)
          stats = (Ns, t1s, t2s)

        elif len(suff_stats) == 4:
          # AR HHM
          Ns, t1s, t2s, t3s = suff_stats

          Ns = q.sum(dim=0) # (K, 1)
          t1s = torch.einsum('tk,tij->kij', q, t1s) # (K, D, D)
          t2s = torch.einsum('tk,tij->kij', q, t2s) # (K, D, M)
          t3s = torch.einsum('tk,tij->kij', q, t3s) # (K, M, M)
          stats = (Ns, t1s, t2s, t3s)

        return (x.shape[0], stats)
        #
        ###

    # Sum the expected stats over the whole dataset
    combined_T = 0
    combined_stats = None
    for data, posterior in zip(dataset, posteriors):
        this_T, these_stats = _compute_expected_suff_stats(data, posterior)
        combined_T, combined_stats = combine(
            combined_T, combined_stats, this_T, these_stats)
    return combined_stats


def initialize_posteriors(
    dataset: List[Dict[str, Any]],
    num_states: int,
    seed: int = 0
    ) -> List[Dict[str, Union[Float[torch.Tensor, "num_timesteps num_states"], float]]]:
    """Initializes a list of random posterior distributions for an HMM.

    For each item in the input `dataset` (where each item is a dictionary
    expected to have a 'data' key providing time series length), this function
    creates a corresponding dictionary. This dictionary contains randomly
    initialized 'expected_states' (posterior probabilities, normalized per
    time step) and a 'marginal_ll' (marginal log likelihood) set to -infinity.
    Sets the PyTorch random seed for reproducibility.

    Args:
        dataset: A list of dictionaries. Each dictionary should contain a 'data'
                 key, where `len(data['data'])` gives the number of time steps
                 for that data sequence.
        num_states: The number of hidden states (K) for the HMM.
        seed: Seed for PyTorch's random number generator. Defaults to 0.

    Returns:
        A list of dictionaries. Each dictionary has two keys:
        - 'expected_states': A 2D PyTorch Tensor (shape: T x K) of randomly
          initialized posterior probabilities (rows sum to 1). T varies per item.
        - 'marginal_ll': A float, initialized to `-torch.inf`.
    """
    torch.manual_seed(seed)
    posteriors = []
    for data in dataset:
        expected_states = torch.rand(len(data["data"]), num_states)
        expected_states /= expected_states.sum(axis=1, keepdims=True)
        posteriors.append(dict(expected_states=expected_states,
                               marginal_ll=-torch.inf))
    return posteriors


