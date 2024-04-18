import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score

# Function to read data from files
def read_data(file_name):
    path = f"C:/Users/akmik/Onedrive/Desktop/CSCI 347/Mini Project 3/{file_name}"
    with open(path, 'r') as f:
        data = np.loadtxt(f)
    return data[:2000]

# Concatenating data from different files
file_names = [
    'mfeat-fou',
    'mfeat-fac',
    'mfeat-kar',
    'mfeat-pix',
    'mfeat-zer',
    'mfeat-mor',
]
data_concatenated = np.concatenate([read_data(file_name) for file_name in file_names], axis=1)

# Removing rows with NaN values
data_concatenated = data_concatenated[~np.all(np.isnan(data_concatenated), axis=1)]

# Applying PCA
pca = PCA()
pca_transformed = pca.fit_transform(data_concatenated)

# Define range of values for minpts and epsilon for question 11
minpts_values = [5, 10, 15, 20, 25]
epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5]

# ---------------------- Question 11 ----------------------

# Experiment with DBSCAN using different minpts and epsilon values
for epsilon in epsilon_values:
    for minpts in minpts_values:
        dbscan = DBSCAN(eps=epsilon, min_samples=minpts)
        labels = dbscan.fit_predict(data_concatenated)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 label indicates noise points
        print(f"For (minpts={minpts}, epsilon={epsilon}), Number of clusters: {num_clusters}")

# ---------------------- Question 12 ----------------------

# Lists to store precision values
precision_original = []
precision_reduced = []

# Define range of values for k
k_values = [5, 10, 15, 20, 25]

# Experiment with DBSCAN using different minpts, epsilon, and k values
for minpts in minpts_values:
    for epsilon in epsilon_values:
        for k in k_values:
            # Original data
            dbscan = DBSCAN(eps=epsilon, min_samples=minpts)
            labels_original = dbscan.fit_predict(data_concatenated)
            precision_original.append(precision_score(labels_original, k*np.ones_like(labels_original), average='micro'))
            
            # Reduced-dimensionality data
            dbscan = DBSCAN(eps=epsilon, min_samples=minpts)
            labels_reduced = dbscan.fit_predict(pca_transformed)
            precision_reduced.append(precision_score(labels_reduced, k*np.ones_like(labels_reduced), average='micro'))

# Reshape precision lists for plotting
precision_original = np.array(precision_original).reshape(len(minpts_values), len(epsilon_values), len(k_values))
precision_reduced = np.array(precision_reduced).reshape(len(minpts_values), len(epsilon_values), len(k_values))

# Plotting using heatmaps
fig, axs = plt.subplots(2, len(epsilon_values), figsize=(15, 8), sharex='col', sharey='row')

for i, eps_val in enumerate(epsilon_values):
    for j in range(2):
        if j == 0:
            im = axs[j, i].imshow(precision_original[:, i, :], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axs[j, i].set_title(f'Original Data (Eps={eps_val})')
        else:
            im = axs[j, i].imshow(precision_reduced[:, i, :], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axs[j, i].set_title(f'Reduced Data (Eps={eps_val})')
        axs[j, i].set_ylabel('MinPts')
        axs[j, i].set_xlabel('K')
        fig.colorbar(im, ax=axs[j, i], label='Precision')

plt.tight_layout()
plt.show()