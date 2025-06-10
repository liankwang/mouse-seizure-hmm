import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from .visualize_states import plot_ll_vs_iter, plot_data_and_states, plot_state_usage_hist, plot_posterior_heatmap, plot_state_duration_hist
from .make_videos import generate_videos

def visualize(train_dataset, train_lls, test_lls, train_posteriors, 
              save_path, 
              time_match_path, video_path,
              show=False, dpi=300):
    
    time_match = pd.read_table(time_match_path, sep=',')
    
    hmm_state_probs = train_posteriors[0]['expected_states']
    hmm_states = hmm_state_probs.argmax(axis=1)
    num_states = train_posteriors[0]['expected_states'].shape[1]

    print("Generating plots...")
    figs = {"ll_vs_iter": plot_ll_vs_iter(train_lls, test_lls),
            "data_and_states": plot_data_and_states(train_dataset[0], hmm_states),
            "state_usage_hist": plot_state_usage_hist(hmm_states, num_states),
            "posterior_heatmap": plot_posterior_heatmap(hmm_state_probs, train_dataset),
            "state_duration_hist": plot_state_duration_hist(hmm_states.tolist(), num_states)
            }
    
    # Save all figures
    for name, fig in figs.items():
        #Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_path}/{name}.png", dpi=dpi)
        if show:
            plt.show()
        plt.close()

    # Generate grid videos
    print("Generating videos...")
    generate_videos(train_dataset, train_posteriors, hmm_states, num_states, time_match, video_path, save_path)


