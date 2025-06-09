import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Adds project root to path

from data_processing.load_data import load_data, process_data
from models.fit_hmm import fit_hmm_wrapper

import matplotlib.pyplot as plt

data_path = Path('data/250509_seizure_I7_1_interpolated.parquet')

data = load_data(data_path,
                 start_time="22:00",
                 end_time="23:59",
                 pitch_roll=True,
                 sampling_frequency=500)

train_dataset, test_dataset = process_data(data,
                                           downsampling_method='pca',
                                           target_sps=50,
                                           original_sps=500)

train_lls, test_lls, train_posteriors, test_posteriors = \
    fit_hmm_wrapper(train_dataset,
                    test_dataset,
                    model_type='ghmm')

plt.plot(train_lls, label="train")
plt.plot(test_lls, '-r', label="test")
plt.xlabel("iteration")
plt.ylabel("avg marginal log lkhd")
plt.grid(True)
plt.legend()
plt.show()