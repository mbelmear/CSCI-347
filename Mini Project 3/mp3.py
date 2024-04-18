import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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

# Function to calculate covariance
def calculate_covariance(vec1, vec2=None):
    if vec2 is None:
        vec2 = vec1
    vec1_mean = vec1.mean()
    vec2_mean = vec2.mean()
    covar = 0
    for i in range(vec1.shape[0]):
        covar += (vec1[i] - vec1_mean) * (vec2[i] - vec2_mean)
    return (covar / (vec1.shape[0] - 1))

# Function to get explained variance
def get_variance_explained(eigen_values):
    cum_sum = np.cumsum(eigen_values)
    variance_explained = cum_sum / np.sum(eigen_values)
    return variance_explained

# Plotting explained variance ratio
principal_component_values = np.arange(pca.n_components_) + 1
plt.plot(principal_component_values, pca.explained_variance_ratio_, 'o-')
plt.xlim(0, 10)
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.show()

# Define range of values for minpts and epsilon
minpts_values = [5, 10, 15, 20, 25]
epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5]

# Experiment with DBSCAN using different minpts and epsilon values
for minpts in minpts_values:
    for epsilon in epsilon_values:
        dbscan = DBSCAN(eps=epsilon, min_samples=minpts)
        labels = dbscan.fit_predict(pca_transformed)
        # Count the number of clusters found
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 label indicates noise points
        print(f"For (minpts={minpts}, epsilon={epsilon}), Number of clusters: {num_clusters}")
