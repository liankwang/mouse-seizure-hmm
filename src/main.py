import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Adds project root to path

import argparse
import yaml
from pathlib import Path
from datetime import datetime

from src.data_processing.load_data import load_data, process_data
from src.visualization.visualize_data import plot_feature_distributions, get_feature_names
from src.models.fit_hmm import fit_hmm_wrapper
from src.visualization.visualize import visualize


def main():
    # Parse arguments for config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--show', action='store_true', help='Show plots')
    args = parser.parse_args()

    # Load config file and store defaults for unspecified parameters
    config = load_config(args.config)
    data_path = Path(config['data']['data_path']) if 'data_path' in config['data'] else None
    start_time = config['data']['start_time'] if 'start_time' in config['data'] else None
    end_time = config['data']['end_time'] if 'end_time' in config['data'] else None
    pitch_roll = config['data']['pitch_roll'] if 'pitch_roll' in config['data'] else True
    original_sps = config['data']['original_sps'] if 'original_sps' in config['data'] else None
    target_sps = config['data']['target_sps'] if 'target_sps' in config['data'] else None
    downsampling_method = config['data']['downsampling_method'] if 'downsampling_method' in config['data'] else None
    n_pca_components = config['data']['n_pca_components'] if 'n_pca_components' in config['data'] else None
    stickiness = config['model']['stickiness'] if 'stickiness' in config['model'] else 0.95
    model_type = config['model']['type'] if 'type' in config['model'] else None
    num_states = config['model']['num_states'] if 'num_states' in config['model'] else 50
    transition_matrix_method = config['model']['transition_matrix_method'] if 'transition_matrix_method' in config['model'] else None
    num_iters = config['model']['num_iters'] if 'num_iters' in config['model'] else 50
    output_path = Path(config['output']['output_path']) if 'output_path' in config['output'] else 'output'
    dpi = config['output']['dpi'] if 'dpi' in config['output'] else 300
    time_match_path = Path(config['data']['time_match_path']) if 'time_match_path' in config['data'] else None

    # Load dataset
    data = load_data(
        data_path,
        start_time=start_time,
        end_time=end_time,
        pitch_roll=pitch_roll,
        sampling_frequency=original_sps
    )

    # Load and process dataset
    train_dataset, test_dataset = process_data(
        data,
        downsampling_method=downsampling_method,
        target_sps=target_sps,
        original_sps=original_sps,
        n_pca_components=n_pca_components
    )
    
    # Visualize data distributions
    feature_names = get_feature_names(data, downsampling_method, n_pca_components)
    plot_feature_distributions(
        train_data=train_dataset[0]['data'],
        test_data=test_dataset[0]['data'],
        feature_names=feature_names,
        downsampling_method=downsampling_method
    )
    
    # Fit HMM model
    train_lls, test_lls, train_posteriors, test_posteriors = fit_hmm_wrapper(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_type=model_type,
        num_states=num_states,
        transition_matrix_method=transition_matrix_method,
        stickiness=stickiness,
        num_iters=num_iters
    )

    # Get save name for outputs
    current_date = datetime.now().strftime("%Y%m%d")
    if 'save_name' not in config['output']:
        save_name = f"{current_date}_{model_type}_{start_time.replace(':', '')}to{end_time.replace(':', '')}_{downsampling_method}_{num_states}"
    else:
        save_name = f"{current_date}_{config['save_name']}"
    
    # Create visualizations
    visualize(
        train_dataset=train_dataset,
        train_lls=train_lls,
        test_lls=test_lls,
        train_posteriors=train_posteriors,
        save_path=f"{output_path}/{save_name}",
        time_match_path=time_match_path,
        show=args.show,
        dpi=dpi
    )


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    main()