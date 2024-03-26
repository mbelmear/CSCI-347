import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Question 1: Dot Product Calculation
# -----------------------------

# Build matrix 'A' and vector 'V'
A = np.array([[2, 1], [1, 3]], dtype=int)
V = np.array([-1, 1], dtype=int)

# Initialize an array to store dot products (pre-allocate for efficiency)
dot_product = np.zeros(A.shape[1])

# Loop through rows in matrix A and calculate dot product with vector 'V'
for i in range(A.shape[0]):
    a_row = A[i, :]  # Get the current row
    dot_product[i] = np.dot(a_row, V)  # Efficient dot product calculation

# Print the dot products
print(dot_product)

# -----------------------------
# Question 2: Linear Transformation and Data Analysis
# -----------------------------

# Part 1: Define Matrix A, Data Set D, and Scatter Plot

# Define matrix 'A'
A = np.matrix([[((3**(1/2))/2), -(1/2)], [(1/2), ((3**(1/2))/2)]])
print(A)

# Build data set 'D'
D = np.array([[1, 1.5], [1, 2], [3, 4], [-1,-1], [-1, 1], [1, -2], [2, 2], [2, 3]])
print(D)

# Extract X1 and X2 values from data set 'D'
X1 = D[:, 0]
X2 = D[:, 1]

# Create a scatter plot of X1 and X2
plt.scatter(x=X1, y=X2, color='blue', marker='o', s=100, edgecolors='black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of X1 and X2')
plt.show()

# Part 2: Linear Transformation
# Initialize an array to store linearly transformed data (pre-allocate)
linearTransformationData = np.zeros_like(D)

# Loop through each row in data set 'D' and perform transformation
for row_index, row in enumerate(D):
    dot = np.dot(A, row)  # Perform matrix multiplication
    linearTransformationData[row_index] = dot.tolist()[0]  # Convert to list for assignment

# Print out the transformed data
for index, row in enumerate(linearTransformationData):
    print(f"x{index+1}: ", row)

# Part 3: Visualization of Original vs. Transformed Data
# Extract transformed X1 and X2 values
X1_transformed = linearTransformationData[:, 0]
X2_transformed = linearTransformationData[:, 1]

# Create scatter plots of original and transformed X1 and X2 values
plt.scatter(x=X1, y=X2, color='blue', marker='o', s=100, edgecolors='black')
plt.scatter(x=X1_transformed, y=X2_transformed, color='red', marker='s', s=100, edgecolors='black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of original v.s. transformed X1 & X2')
plt.legend(['Original', 'Transformed'])
plt.show()

# Part 4: Multivariate Mean Calculation
def multivariate_mean(m):
  """
  Calculates the multivariate mean of a matrix.

  Args:
      m: A NumPy array representing the data matrix.

  Returns:
      A NumPy array representing the multivariate mean vector.
  """
  # Output array (i.e. mean array)
  mean = [0] * m.shape[1]
  
  # Iterate over columns to calculate column means
  for col_index in range(m.shape[1]):
    col_arr = m[:, col_index]
    col_mean = col_arr.mean()
    mean[col_index] = col_mean
  
  # Return the multivariate mean
  return mean

multiDimMean = multivariate_mean(D)
print("Multivariate Mean:", multiDimMean)

# Part 5: Mean Centering Data
meanCenteredData = np.zeros_like(D)
for row_index, row in enumerate(D):
    for row_col_index, value in enumerate(row):
        meanCenteredData[row_index][row_col_index] = value - multiDimMean[row_col_index]

print(meanCenteredData)

# Part 6: Visualization of Original vs. Mean-Centered Data
X1_mean_centered = meanCenteredData[:, 0]
X2_mean_centered = meanCenteredData[:, 1]
plt.scatter(x=X1, y=X2, color='blue', marker='o', s=100, edgecolors='black')
plt.scatter(x=X1_mean_centered, y=X2_mean_centered, color='red', marker='s', s=100, edgecolors='black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of original v.s. mean-centered X1 & X2')
plt.legend(['Original', 'Mean-Centered'])
plt.show()

# Part 7: Covariance Calculation
def covariance(v1, v2=None):
  """
  Calculates the covariance between two vectors.

  Args:
      v1: A NumPy array representing the first vector.
      v2 (optional): A NumPy array representing the second vector (defaults to None, which is same as v1).

  Returns:
      A float representing the covariance between v1 and v2.
  """
  if v2 is None: v2 = v1
  v1_mean = v1.mean()
  v2_mean = v2.mean()
  co_var = 0
  for i in range(v1.shape[0]):
    co_var += (v1[i] - v1_mean) * (v2[i] - v2_mean)
  return (co_var / (v1.shape[0] - 1))

def covarianceMatrix(m):
  """
  Calculates the covariance matrix of a data matrix.

  Args:
      m: A NumPy array representing the data matrix.

  Returns:
      A NumPy array representing the covariance matrix.
  """
  covar_m = np.zeros((m.shape[1], m.shape[1]))
  for i in range(m.shape[1]):
    for j in range(m.shape[1]):
      covar_m[i, j] = covariance(m[:, i], m[:, j])
  return covar_m

print(covarianceMatrix(D))

# Part 8: Covariance Matrix of Mean-Centered Data
print(covarianceMatrix(meanCenteredData))

# Part 9: Z-Score Normalization
def zScoreNormalize(m):
  """
  Normalizes a data matrix using z-score normalization.

  Args:
      m: A NumPy array representing the data matrix.

  Returns:
      A NumPy array representing the normalized data matrix.
  """
  z_score = np.zeros_like(m)
  for row_index in range(m.shape[0]):
    for col_index in range(m.shape[1]):
      col_arr = m[:, col_index]
      col_std_div = (covariance(col_arr)) ** (1/2)
      col_mean = col_arr.mean()
      x_ij = m[row_index, col_index]
      x_ij_zscore = (x_ij - col_mean) / col_std_div
      z_score[row_index, col_index] = x_ij_zscore
  return z_score

print(covarianceMatrix(zScoreNormalize(D)))
