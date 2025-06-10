from typing import Dict, Union, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from .utils import palette

def plot_ll_vs_iter(train_lls, test_lls):
    fig, ax = plt.subplots()
    ax.plot(train_lls, label="train")
    ax.plot(test_lls, '-r', label="test")
    ax.set_xlabel("iteration")
    ax.set_ylabel("avg marginal log likelihood")
    ax.grid(True)
    ax.legend()
    return fig


def plot_data_and_states(
    data: Dict[str, Union[torch.Tensor, np.ndarray, Any]],
    states: Union[torch.Tensor, np.ndarray],
    spc: int = 4,
    slc: slice = slice(100, 10000),
    title: Optional[str] = None,
    ) -> None:
    """
    Plots principal component data along with discrete states over time.

    The function visualizes time-series data (e.g., principal components)
    as line plots. The discrete states, passed via the `states` argument,
    are shown as a background image. The `data` dictionary is also expected
    to contain a "labels" key, which is sliced and assigned to an internal
    variable, though its direct use in subsequent plotting commands within
    this snippet is not apparent; the `states` argument is used for the imshow background.

    Args:
        data (Dict[str, Union[torch.Tensor, np.ndarray, Any]]): A dictionary
            containing the data to plot. It must include:
            - "data": A 2D array-like (num_timesteps, num_features) of the
                      main data trajectories (e.g., principal components).
            - "times": A 1D array-like of timestamps corresponding to the data.
            - "labels": A 1D array-like of discrete labels. This is accessed
                        internally, though the primary state visualization uses
                        the `states` argument.
        states (Union[torch.Tensor, np.ndarray]): A 1D array-like of discrete
            states corresponding to each time step, used for the background
            visualization.
        spc (int, optional): Spacing factor for plotting multiple principal
            components vertically. Defaults to 4.
        slc (slice, optional): A slice object to select a portion of the
            data and states to plot. Defaults to slice(0, 900).
        title (Optional[str], optional): The title for the plot. If None,
            a default title "data and discrete states" is used.
            Defaults to None.

    Returns:
        None: The function generates and displays a plot using matplotlib.
    """
    times = data["times"][slc]
    x = data["data"][slc]
    seizures = data['seizures'][slc]
    num_timesteps, data_dim = x.shape
    print("Seizures: ")
    print(seizures)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot latent states in background
    ax.imshow(states[None, slc],
                cmap="cubehelix", aspect="auto",
                extent=(0, times[-1] - times[0], -data_dim * spc, spc))

    # Plot time series line plot
    ax.plot(times - times[0],
            x - spc * np.arange(data_dim),
            ls='-', lw=3, color='w')
    ax.plot(times - times[0],
            x - spc * np.arange(data_dim),
            ls='-', lw=2, color=palette[0])

    # Add seizure statues at the bottom
    from matplotlib.colors import ListedColormap
    cmap_seizure = ListedColormap(["#66c2a5", "#fc8d62", "#8da0cb"])

    ax.imshow(seizures[None, :],
            cmap = cmap_seizure,
            aspect = "auto",
            extent=(0, times[-1] - times[0], -data_dim * spc - spc, -data_dim * spc))

    from matplotlib.patches import Patch
    mapping = {0: 'normal', 1: 'interictal', 2: 'true seizure'}
    legend_patches = [
        Patch(color=cmap_seizure(i), label=mapping[i])
        for i in sorted(mapping)
    ]
    ax.legend(handles=legend_patches, loc='upper right', title='seizure status')


    # Set ticks and labels
    ax.set_yticks(-spc * np.arange(data_dim + 1))
    ax.set_yticklabels(list(np.arange(data_dim)) + ['seizure'])
    ax.set_ylabel("feature")
    ax.set_xlim(0, times[-1] - times[0])
    ax.set_xlabel("time [ms]")

    if title is None:
        ax.set_title("data and discrete states")
    else:
        ax.set_title(title)
    
    return fig


def plot_state_usage_hist(hmm_states, num_states):
    # Sort states by usage
    hmm_usage = torch.bincount(hmm_states, minlength=num_states)
    hmm_order = torch.argsort(hmm_usage, descending=True)

    fig, ax = plt.subplots()
    ax.bar(torch.arange(num_states).int().numpy(), hmm_usage[hmm_order].numpy())
    ax.set_xlabel("state index [ordered]")
    ax.set_ylabel("num frames")
    ax.set_title("histogram of inferred state usage")

    return fig


def plot_posterior_heatmap(hmm_state_probs, train_dataset):
    T, K = hmm_state_probs.shape
    timestamps = train_dataset[0]['times'] / 1000
    seizures = train_dataset[0]['seizures']

    # Set your desired x-axis limits (time step range)
    x_start = 0
    x_end = min(50000, T)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(hmm_state_probs.T, 
                   aspect='auto', interpolation='none', cmap='viridis',
                   extent=[timestamps[0], timestamps[-1], K - 1, 0])  # extent sets correct axis ranges
    
    # Add seizure statuses at the bottom
    from matplotlib.colors import ListedColormap
    cmap_seizure = ListedColormap(["#66c2a5", "#fc8d62", "#8da0cb"])
    ax.imshow(seizures[None, :],
              cmap=cmap_seizure,
              aspect="auto",
              extent=[timestamps[0], timestamps[-1], -1, 0])
    from matplotlib.patches import Patch
    mapping = {0: 'normal', 1: 'interictal', 2: 'true seizure'}
    legend_patches = [
        Patch(color=cmap_seizure(i), label=mapping[i])
        for i in sorted(mapping)
    ]
    ax.legend(handles=legend_patches, loc='upper right', title='seizure status')
    
    # Set ticks and titles
    step = max(1, K // 10)  # Display at most 10 labels, evenly spaced
    ax.set_yticks(np.arange(0, K, step))  # Set y-ticks with a step
    ax.set_yticklabels(np.arange(0, K, step).astype(int))  # Set y-tick labels as integers
    ax.set_xlabel('timestamp')
    ax.set_ylabel('hidden state')
    ax.set_title('heatmap of posterior state probabilities')
    ax.set_xlim(timestamps[x_start], timestamps[x_end - 1])

    fig.colorbar(im, ax=ax, label='posterior probability')

    return fig


def plot_state_duration_hist(hmm_states, num_states):
    # Constants
    min_duration_frames = 10  # 300 ms / 30 ms per frame

    # Compute durations per state
    durations_by_state = {k: [] for k in range(num_states)}
    prev_state = hmm_states[0]
    start = 0

    for t in range(1, len(hmm_states)):
        if hmm_states[t] != prev_state:
            duration = t - start
            if duration >= min_duration_frames:
                durations_by_state[prev_state].append(duration)
            prev_state = hmm_states[t]
            start = t

    # Add last segment
    duration = len(hmm_states) - start
    if duration >= min_duration_frames:
        durations_by_state[prev_state].append(duration)

    # Compute statistics (mean ± IQR, filtering out short bouts)
    means = []
    iqr_lows = []
    iqr_highs = []

    for k in range(num_states):
        durations = np.array(durations_by_state[k])
        if len(durations) == 0:
            means.append(0)
            iqr_lows.append(0)
            iqr_highs.append(0)
        else:
            mean = durations.mean()
            q25, q75 = np.percentile(durations, [25, 75])
            iqr_low = np.clip(mean - q25, 0, None)
            iqr_high = np.clip(q75 - mean, 0, None)
            means.append(mean)
            iqr_lows.append(iqr_low)
            iqr_highs.append(iqr_high)

    # Plot
    x = np.arange(num_states)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, means, yerr=[iqr_lows, iqr_highs], capsize=5)
    ax.set_xlabel("state index")
    ax.set_ylabel("bout duration (frames, ≥300ms)")
    ax.set_title("Mean ± IQR duration of continuous states (≥300ms)")
    step = max(1, len(x) // 10)  # Display at most 10 labels, evenly spaced
    ax.set_xticks(x[::step])
    ax.set_xticklabels(x[::step].astype(int))

    return fig
