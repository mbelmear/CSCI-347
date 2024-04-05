import math  # Import math module for mathematical operations
import pandas as pd  # Import pandas library for data manipulation
import numpy as np  # Import numpy library for numerical operations
import numpy.linalg as LA  # Import numpy's linear algebra module
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.decomposition import PCA  # Import PCA from scikit-learn for principal component analysis
from sklearn.cluster import KMeans, DBSCAN  # Import KMeans and DBSCAN clustering algorithms

# Define the path to the Boston data CSV file
boston_data_path = "C:\\Users\\akmik\\OneDrive\\Desktop\\CSCI 347\\HW\\HW5\\Boston.csv"

# Generate random data
mu = np.array([0, 0])  # Mean of random data
Sigma = np.array([[1, 0], [0, 1]])  # Covariance matrix of random data
X1, X2 = np.random.multivariate_normal(mu, Sigma, 1000).T  # Generate random multivariate normal distribution
D = np.array([X1, X2]).T  # Combine X1 and X2 to create random dataset D

# Scatter plot of original data
plt.scatter(x=D[:,0], y=D[:,1], s=1)  # Scatter plot x and y coordinates of D
plt.xlabel('First Attribute')  # Set x-axis label
plt.ylabel('Second Attribute')  # Set y-axis label
plt.title('Original Data')  # Set plot title
plt.show()  # Display plot

# Define rotation and scaling matrices
R = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4)],  # Rotation matrix
              [math.sin(math.pi / 4), math.cos(math.pi / 4)]])  # Rotation matrix
S = np.array([[5, 0], [0, 2]])  # Scaling matrix

# Transform data with rotation and scaling
D_RS = np.dot(np.dot(D, R), S)  # Perform rotation and scaling transformation on data D

# Scatter plot of original and transformed data
plt.scatter(x=D[:,0], y=D[:,1], color='blue', marker='o', s=1, label='Original')  # Scatter plot of original data
plt.scatter(x=D_RS[:,0], y=D_RS[:,1], color='red', marker='s', s=1, label='Transformed')  # Scatter plot of transformed data
plt.xlabel('First Attribute')  # Set x-axis label
plt.ylabel('Second Attribute')  # Set y-axis label
plt.title('Original vs Transformed Data')  # Set plot title
plt.legend()  # Display legend
plt.show()  # Display plot

# Function to calculate covariance between vectors
def covariance(v1, v2=None):
    if v2 is None:
        v2 = v1
    v1_mean = np.mean(v1)  # Mean of vector v1
    v2_mean = np.mean(v2)  # Mean of vector v2
    co_var = np.sum((v1 - v1_mean) * (v2 - v2_mean))  # Covariance calculation
    return (co_var / (v1.shape[0] - 1))  # Return covariance

# Function to calculate covariance matrix
def covarianceMatrix(m):
    covar_m = np.ndarray((m.shape[1], m.shape[1]))  # Initialize covariance matrix
    for i in range(m.shape[1]):
        for j in range(m.shape[1]):
            covar_m[i, j] = covariance(m[:,i], m[:,j])  # Calculate covariance between columns
    return covar_m  # Return covariance matrix

# Calculate covariance matrix of transformed data
print(covarianceMatrix(D_RS))

total_var = 0
for col_index in range(D_RS.shape[1]):
    total_var += covariance(D_RS[:,col_index])
print(total_var)

# Perform PCA on transformed data
pca = PCA(n_components=2)  # Initialize PCA with 2 components
pca_transformed_D = pca.fit_transform(D_RS)  # Fit and transform data with PCA

# Scatter plot of PCA-transformed data
plt.scatter(x=pca_transformed_D[:,0], y=pca_transformed_D[:,1], color='blue', marker='o', s=1)  # Scatter plot of PCA-transformed data
plt.xlabel('First Principal Component')  # Set x-axis label
plt.ylabel('Second Principal Component')  # Set y-axis label
plt.title('PCA Transformed Data')  # Set plot title
plt.show()  # Display plot

# Calculate covariance matrix of PCA-transformed data
print(covarianceMatrix(pca_transformed_D))

# Explained variance ratio of first and second principal components
print(pca.explained_variance_ratio_[0])  # Explained variance ratio of first principal component
print(pca.explained_variance_ratio_[1])  # Explained variance ratio of second principal component

# Load the Boston housing dataset from CSV file
boston_data = pd.read_csv(boston_data_path)  # Read Boston housing data from CSV file

# Perform PCA analysis on Boston housing data
pca = PCA(n_components=2)  # Initialize PCA with 2 components
pca_transformed_D_boston = pca.fit_transform(boston_data)  # Fit and transform Boston housing data with PCA

# Function to normalize data using Z-score normalization
def zScoreNormalize(m):
    z_score = np.ndarray(m.shape)  # Initialize array for normalized data
    for row_index in range(m.shape[0]):
        for col_index in range(m.shape[1]):
            col_arr = m[:,col_index]  # Get column array
            col_std_div = (covariance(col_arr)) ** (1/2)  # Calculate standard deviation
            col_mean = col_arr.mean()  # Calculate column mean
            x_ij = m[row_index, col_index]  # Get data point
            x_ij_zscore = (x_ij - col_mean) / col_std_div  # Calculate Z-score
            z_score[row_index, col_index] = x_ij_zscore  # Store normalized value
    return z_score  # Return normalized data

# Normalize PCA-transformed Boston data and plot
pca_transformed_D_boston_normalized = zScoreNormalize(pca_transformed_D_boston)  # Normalize PCA-transformed Boston data

plt.scatter(x=pca_transformed_D_boston_normalized[:,0], y=pca_transformed_D_boston_normalized[:,1], color='blue', marker='o', s=1)  # Scatter plot of normalized PCA-transformed Boston data
plt.xlabel('First Principal Component (Normalized)')  # Set x-axis label
plt.ylabel('Second Principal Component (Normalized)')  # Set y-axis label
plt.title('Normalized PCA Transformed Boston Data')  # Set plot title
plt.show()  # Display plot

# Calculate covariance matrix of normalized PCA-transformed Boston data
Sigma = np.cov(pca_transformed_D_boston_normalized, ddof=1)  # Calculate covariance matrix with degrees of freedom as 1
evalues, evectors = LA.eig(Sigma)  # Calculate eigenvalues and eigenvectors
idx = evalues.argsort()[::-1]  # Sort eigenvalues in descending order
evalues = evalues[idx]  # Sort eigenvalues
evectors = evectors[:, idx]  # Sort eigenvectors
total_vars = np.ndarray(shape=(13,2), dtype=np.complex128)  # Initialize array for total variance
for i in range(13):
    var_i = (sum(evalues[:i+1])) / sum(evalues)  # Calculate total variance explained by principal components
    total_vars[i] = [i, var_i]  # Store total variance

# Scatter plot of total variance explained by principal components
plt.scatter(total_vars[:,0], total_vars[:,1])  # Scatter plot of total variance explained
plt.ylim(0.9999999999999999, 1)  # Set y-axis limits
plt.xlabel('Number of Principal Components')  # Set x-axis label
plt.ylabel('Total Variance Explained')  # Set y-axis label
plt.title('Total Variance Explained by Principal Components')  # Set plot title
plt.show()  # Display plot

print(evalues[0]/total_var)

# Total variance explained by first two principal components
total_var = sum(np.diag(Sigma))
print((evalues[0] + evalues[1]) / total_var)

kmeans = KMeans(n_clusters=3, init='random', max_iter=300, random_state=0)  # Initialize KMeans clustering
pred_labels = kmeans.fit_predict(pca_transformed_D_boston_normalized)  # Fit and predict cluster labels
centers = kmeans.cluster_centers_  # Get cluster centers

plt.scatter(pca_transformed_D_boston_normalized[:,0],
            pca_transformed_D_boston_normalized[:,1],
            c=pred_labels)  # Scatter plot of clustered data
plt.scatter(centers[:,0], centers[:,1], s=50, c='red', label='Centroids')  # Scatter plot of centroids
plt.xlabel('First Principal Component (Normalized)')  # Set x-axis label
plt.ylabel('Second Principal Component (Normalized)')  # Set y-axis label
plt.title('KMeans Clustering of Normalized PCA Transformed Boston Data')  # Set plot title
plt.legend()  # Display legend
plt.show()  # Show plot

dbs = DBSCAN(eps=0.5, min_samples=3)  # Initialize DBSCAN clustering
pred_labels = dbs.fit_predict(pca_transformed_D_boston_normalized)  # Fit and predict cluster labels

plt.scatter(pca_transformed_D_boston_normalized[:,0],
            pca_transformed_D_boston_normalized[:,1],
            c=pred_labels)  # Scatter plot of clustered data
plt.xlabel('First Principal Component (Normalized)')  # Set x-axis label
plt.ylabel('Second Principal Component (Normalized)')  # Set y-axis label
plt.title('DBSCAN Clustering of Normalized PCA Transformed Boston Data')  # Set plot title
plt.show()  # Show plot