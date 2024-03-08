import pandas as pd
import networkx as nx

def get_largest_connected_component(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create a graph from the DataFrame
    graph = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')

    # Find all connected components
    connected_components = nx.connected_components(graph)

    # Get the largest connected component
    largest_cc = max(connected_components, key=len)
    
    # Extract the largest connected component
    graph_lcc = graph.subgraph(largest_cc).copy()

    return graph_lcc

def average_shortest_path_length_lcc(file_path):
    # Get the largest connected component
    graph_lcc = get_largest_connected_component(file_path)

    # Calculate the average shortest path length using unweighted Dijkstra's algorithm
    avg_shortest_path_length = nx.average_shortest_path_length(graph_lcc, method='unweighted')

    return avg_shortest_path_length

# Path to the CSV file
file_path = r"C:\Users\akmik\OneDrive\Desktop\CSCI 347\Mini Project 2\large_twitch_edges.csv"

# Calculate and print the average shortest path length of the largest connected component
avg_length_lcc = average_shortest_path_length_lcc(file_path)
print("Average Shortest Path Length (largest connected component):", avg_length_lcc)