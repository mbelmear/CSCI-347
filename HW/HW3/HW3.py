import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#defining the graph for use in the relevant problems
graph = nx.Graph()

graph.add_edge(1, 2)
graph.add_edge(1, 3)
graph.add_edge(2, 3)
graph.add_edge(3, 4)
graph.add_edge(3, 5)
graph.add_edge(3, 12)
graph.add_edge(4, 5)
graph.add_edge(5, 11)
graph.add_edge(6, 7)
graph.add_edge(6, 12)
graph.add_edge(7, 12)
graph.add_edge(8, 12)
graph.add_edge(9, 12)
graph.add_edge(10, 12)
graph.add_edge(11, 12)

#answer to question 3
print("Betweeness centrality: ", nx.betweenness_centrality(graph, normalized=False), "\n")

#answer to question 4
print("Prestige centrality: ", nx.eigenvector_centrality(graph), "\n")

#answer to question 5
print("Average shortest path length: ", nx.average_shortest_path_length(graph), "\n")

#answer to question 6
degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
degree_count = nx.degree_histogram(graph)

plt.bar(range(len(degree_count)), degree_count, width=0.8, color='b')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()

#answer to question 9 
n = 200
p = 0.1
random_graph = nx.erdos_renyi_graph(n, p)

betweenness_centrality = nx.betweenness_centrality(random_graph)

node_degrees = dict(random_graph.degree())

betweenness_values = [betweenness_centrality[node] for node in random_graph.nodes()]

node_degrees_values = [node_degrees[node] for node in random_graph.nodes()]

plt.figure(figsize=(10, 8))

nx.draw(random_graph, 
        pos=nx.spring_layout(random_graph, seed=42),  
        node_size=[3000 * bc for bc in betweenness_values],  
        node_color=node_degrees_values,  
        cmap=plt.cm.plasma,  
        alpha=0.7,  
        with_labels=False)  

norm = plt.Normalize(vmin=min(node_degrees_values), vmax=max(node_degrees_values))
sm = cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
sm.set_array([])
cbar = plt.colorbar(sm, label='Node Degree', cax=plt.gca().inset_axes([0.95, 0.1, 0.05, 0.8]))

plt.title('Erdos-Renyi Random Graph Visualization')
plt.show()