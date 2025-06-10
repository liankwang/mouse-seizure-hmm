From the parent directory, run ```src/main.py --config configs/insert_name.yaml``` where ```insert_name.yaml``` is the name of the desired YAML file.

```
configs/                    # Create config files here to define parameters for a given run
|── sample.yaml             # Sample config file for reference

data/                       # Store all raw data here. The script expects the following files:
|── 250509_seizure_I7_1_interpolated.parquet
|── 250509_seizure_I7_1_timestamp.txt
|── 250509_seizure_I7_1.avi

src/                        # This is the main directory with all code scripts
|—— main.py                 # Main script that runs entire pipeline
|── data_processing/
    |── load_data.py        # Implements load_data() and process_data(), which do heavy lifting. Called directly in main script.
    |── downsample.py       # Helper script that defines various downsampling methods
    |── kalman_imu.py       # Helper script that defines KalmanIMU class
    |── utils.py            # Helper functions used in data processing
|── models/
    |—— fit_hmm.py          # Implements algorithm to fit an HMM (both Gaussian and AR). Called directly in main script.
    |── arhmm.py            # Defines the LinearRegressionObservations class for the ARHMM model
    |── ghmm.py             # Defines the GaussianObservations class for the Gaussian HMM model
    |—— em.py               # Implements forward-backward algorithm for EM
    |—— em_utils.py         # Helper functions used in em.py
|—— visualization/
    |–– visualize.py        # Implements pipeline to visualize results. Called directly in main script.
    |—— visualize_data.py   # Defines function to plot feature distributions
    |–– visualize_states.py # Defines various functions to visualize inferred latent states
    |–– make_videos.py      # Defines function to make grid videos
    |–– make_videos_utils.py # Helper functions used to make videos
|–– utils.py                # Helper functions used throughout src/
     
output/                     # All script outputs will be saved here

env.yaml                    # Required packages and dependencies (incomplete) 
```

The implementation of the Gaussian HMM and autoregressive HMM models, EM algorithm, and algorithm to fit the HMM models were all adapted from Lab 7 of Scott Linderman's Machine Learning Methods for Neural Data Analysis class at Stanford. The class materials are available at \texttt{slinderman.github.io/stats320/}.