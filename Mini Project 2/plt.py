import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_degree_distribution(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create a graph from the DataFrame
    graph = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')

    # Get the degree of each node in the graph
    degrees = dict(graph.degree())

    # Plot the degree distribution on a log-log scale
    plt.figure(figsize=(10, 6))
    plt.title("Degree Distribution (log-log scale)")
    plt.xlabel("Degree (log)")
    plt.ylabel("Frequency (log)")

    # Compute degree distribution
    degree_values = list(degrees.values())
    degree_counts = pd.Series(degree_values).value_counts().sort_index()
    degree_counts.plot(marker='o', linestyle='None')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)

    plt.show()

# Path to the CSV file
file_path = r"C:\Users\akmik\OneDrive\Desktop\CSCI 347\Mini Project 2\large_twitch_edges.csv"

# Plot the degree distribution
plot_degree_distribution(file_path)