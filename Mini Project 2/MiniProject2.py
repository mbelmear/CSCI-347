import csv
from collections import defaultdict

def read_csv(file_path):
    """
    Read the CSV file containing edges and return a list of edges.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of edges, where each edge is represented as a tuple (vertex1, vertex2).
    """
    edges = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if row[0] != 'numeric_id_1' and row[1] != 'numeric_id_2':
                edges.append((int(row[0]), int(row[1])))
    return edges

def num_vertices(edges):
    """
    Calculate the number of vertices in the graph.

    Parameters:
        edges (list): A list of edges, where each edge is represented as a tuple (vertex1, vertex2).

    Returns:
        int: The number of vertices in the graph.
    """
    vertices = set()
    for edge in edges:
        vertices.update(edge)
    return len(vertices)

def degree_of_vertex(edges, vertex):
    """
    Calculate the degree of a given vertex in the graph.

    Parameters:
        edges (list): A list of edges, where each edge is represented as a tuple (vertex1, vertex2).
        vertex (int): The vertex for which degree needs to be calculated.

    Returns:
        int: The degree of the given vertex.
    """
    degree = 0
    for edge in edges:
        if vertex in edge:
            degree += 1
    return degree

def clustering_coefficient(edges, vertex):
    """
    Calculate the clustering coefficient of a given vertex in the graph.

    Parameters:
        edges (list): A list of edges, where each edge is represented as a tuple (vertex1, vertex2).
        vertex (int): The vertex for which clustering coefficient needs to be calculated.

    Returns:
        float: The clustering coefficient of the given vertex.
    """
    neighbors = set()
    for edge in edges:
        if vertex in edge:
            neighbors.add(edge[0] if edge[0] != vertex else edge[1])
    num_neighbors = len(neighbors)
    if num_neighbors < 2:
        return 0.0
    
    total_possible_edges = num_neighbors * (num_neighbors - 1) / 2
    actual_edges = 0
    for u in neighbors:
        for v in neighbors:
            if u < v and (u, v) in edges:
                actual_edges += 1
    return actual_edges / total_possible_edges

def betweenness_centrality(edges, vertex):
    """
    Calculate the betweenness centrality of a given vertex in the graph.

    Parameters:
        edges (list): A list of edges, where each edge is represented as a tuple (vertex1, vertex2).
        vertex (int): The vertex for which betweenness centrality needs to be calculated.

    Returns:
        float: The betweenness centrality of the given vertex.
    """
    def get_shortest_paths(edges, vertex):
        paths = defaultdict(int)
        queue = [vertex]
        visited = set(queue)
        distance = {vertex: 0}
        while queue:
            current_vertex = queue.pop(0)
            for edge in edges:
                if current_vertex in edge:
                    neighbor = edge[0] if edge[0] != current_vertex else edge[1]
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
                        distance[neighbor] = distance[current_vertex] + 1
                        paths[neighbor] += 1
                    elif distance[neighbor] == distance[current_vertex] + 1:
                        paths[neighbor] += 1
        return paths

    total_paths = defaultdict(int)
    for vertex in range(num_vertices(edges)):
        paths = get_shortest_paths(edges, vertex)
        for key, value in paths.items():
            total_paths[key] += value

    vertex_paths = total_paths[vertex]
    total_possible_paths = (num_vertices(edges) - 1) * (num_vertices(edges) - 2) / 2
    return vertex_paths / total_possible_paths

def average_shortest_path_length(edges):
    """
    Calculate the average shortest path length in the graph.

    Parameters:
        edges (list): A list of edges, where each edge is represented as a tuple (vertex1, vertex2).

    Returns:
        float: The average shortest path length in the graph.
    """
    def bfs_shortest_path(edges, start):
        visited = {start}
        queue = [(start, 0)]
        while queue:
            node, depth = queue.pop(0)
            for edge in edges:
                if node in edge:
                    neighbor = edge[0] if edge[0] != node else edge[1]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return sum(depth for _, depth in queue) / len(queue)

    total_lengths = 0
    for vertex in range(num_vertices(edges)):
        total_lengths += bfs_shortest_path(edges, vertex)
    return total_lengths / num_vertices(edges)

def adjacency_matrix(edges):
    """
    Generate the adjacency matrix of the graph.

    Parameters:
        edges (list): A list of edges, where each edge is represented as a tuple (vertex1, vertex2).

    Returns:
        list: The dense adjacency matrix of the graph.
    """
    num_verts = num_vertices(edges)
    matrix = [[0] * num_verts for _ in range(num_verts)]
    for edge in edges:
        matrix[edge[0]][edge[1]] = 1
        matrix[edge[1]][edge[0]] = 1
    return matrix

def eigenvector_centrality(adjacency_matrix):
    """
    Calculate the eigenvector centrality using power iteration.

    Parameters:
        adjacency_matrix (list): The dense adjacency matrix of the graph.

    Returns:
        list: The eigenvector corresponding to the dominant eigenvector of the adjacency matrix.
    """
    def normalize(vector):
        norm = sum(vector)
        return [x / norm for x in vector]

    n = len(adjacency_matrix)
    x = [1] * n
    prev_x = [0] * n
    epsilon = 1e-8
    while sum((x[i] - prev_x[i]) ** 2 for i in range(n)) > epsilon:
        prev_x = x.copy()
        x = [0] * n
        for i in range(n):
            for j in range(n):
                x[i] += adjacency_matrix[i][j] * prev_x[j]
        x = normalize(x)
    return x

file_path = r"C:\Users\akmik\OneDrive\Desktop\CSCI 347\Mini Project 2\large_twitch_edges.csv"
edges = read_csv(file_path)

# Number of vertices
print("Number of vertices:", num_vertices(edges))

# Degree of vertex 1
print("Degree of vertex 1:", degree_of_vertex(edges, 1))

# Clustering coefficient of vertex 1
print("Clustering coefficient of vertex 1:", clustering_coefficient(edges, 1))

# Betweenness centrality of vertex 1
print("Betweenness centrality of vertex 1:", betweenness_centrality(edges, 1))

# Average shortest path length
print("Average shortest path length:", average_shortest_path_length(edges))

# Adjacency matrix
print("Adjacency matrix:")
adj_matrix = adjacency_matrix(edges)
for row in adj_matrix:
    print(row)

# Eigenvector centrality
print("Eigenvector centrality:", eigenvector_centrality(adj_matrix))