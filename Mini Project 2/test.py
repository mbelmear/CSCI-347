import pandas as pd
import networkx as nx

def preprocess_graph(file_path, sample_size=None):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Optionally, take a sample of the graph
    if sample_size is not None:
        df = df.sample(n=sample_size)

    # Create a graph from the DataFrame
    graph = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')

    # Extract the largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    graph_lcc = graph.subgraph(largest_cc).copy()

    return graph_lcc

def average_shortest_path_length_from_file(file_path, sample_size=None):
    # Preprocess the graph
    graph = preprocess_graph(file_path, sample_size)

    # Calculate the average shortest path length using unweighted Dijkstra's algorithm
    avg_shortest_path_length = nx.average_shortest_path_length(graph, method='unweighted')

    return avg_shortest_path_length

# Path to the CSV file
file_path = r"C:\Users\akmik\OneDrive\Desktop\CSCI 347\Mini Project 2\large_twitch_edges.csv"

# Optionally, specify the sample size
sample_size = 10000

# Calculate and print the average shortest path length
avg_length = average_shortest_path_length_from_file(file_path, sample_size)
print("Average Shortest Path Length (largest connected component):", avg_length)