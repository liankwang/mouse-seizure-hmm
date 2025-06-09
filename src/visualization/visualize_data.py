import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_feature_distributions(train_data, test_data, feature_names, downsampling_method):
    train_np = train_data.cpu().numpy()
    test_np = test_data.cpu().numpy()

    num_features = train_np.shape[1]
    fig, ax = plt.subplots(figsize=(15, 3*num_features), nrows=num_features, ncols=2)

    for i in range(num_features):
        sns.histplot(train_np[:, i], kde=True, color='blue', stat="density", bins=50, ax=ax[i, 0])
        ax[i, 0].set_title(f'Train - {feature_names[i]}')
        ax[i, 0].set_xlabel(feature_names[i])

        sns.histplot(test_np[:, i], kde=True, color='orange', stat="density", bins=50, ax=ax[i, 1])
        ax[i, 1].set_title(f'Test - {feature_names[i]}')
        ax[i, 1].set_xlabel(feature_names[i])
    
    plt.suptitle(f'Feature distributions for {downsampling_method} downsampling', fontsize=16)
    plt.tight_layout()

    return fig    

def get_feature_names(data, downsampling_method, n_pca_components):
    feature_names = data.columns[2:].tolist()  # Assuming first two columns are time and EEG
    if downsampling_method == 'pca' or downsampling_method == 'pca_overlap':
        feature_names = [f'PC{i+1}' for i in range(n_pca_components)]
    return feature_names