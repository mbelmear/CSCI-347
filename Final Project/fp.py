import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/akmik/OneDrive/Desktop/CSCI 347/Final Project/OnlineNewsPopularity.csv", header=0)

# Assuming you have loaded your data into a DataFrame named 'data'
cols1 = data.iloc[:, [7, 3]]  # Selecting 'num_hrefs' and 'n_tokens_content'

# Compute the correlation matrix
correlation_matrix1 = cols1.corr()

# Plot the heatmap for the first pair of features
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap - num_hrefs vs n_tokens_content')
plt.show()


# Now let's choose another pair of features, for example, 'avg_positive_polarity' and 'avg_negative_polarity'
cols2 = data.iloc[:, [50, 53]]  # Selecting 'avg_positive_polarity' and 'avg_negative_polarity'

# Compute the correlation matrix for the second pair of features
correlation_matrix2 = cols2.corr()

# Plot the heatmap for the second pair of features
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap - avg_positive_polarity vs avg_negative_polarity')
plt.show()