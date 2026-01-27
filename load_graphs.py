# import pickle
# import numpy as np
# import networkx as nx

# def load_graphs_from_file(graphs_file, names_file, colors_file):
#     """
#     Loads graphs, their names, and their colors from pickle files and converts graphs
#     to NumPy adjacency list representation.
    
#     Parameters:
#         graphs_file (str): Path to the pickle file containing the graphs.
#         names_file (str): Path to the pickle file containing the graph names.
#         colors_file (str): Path to the pickle file containing the graph colors.
#     Returns:
#         tuple: A list of adjacency lists, a list of graph names, and a list of graph colors.
#     """
#     with open(graphs_file, 'rb') as file:
#         graphs_dict = pickle.load(file)
    
#     with open(names_file, 'rb') as file:
#         names_dict = pickle.load(file)
        
#     with open(colors_file, 'rb') as file:
#         colors_dict = pickle.load(file)
    
#     # Extract graphs, names, and colors in corresponding order
#     adj_lists = []
#     names = []
#     colors = []
    
#     print(f"Processing {len(graphs_dict)} graphs...")
    
#     for i, key in enumerate(graphs_dict.keys()):
#         nx_graph = graphs_dict[key]
#         graph_name = names_dict.get(key, key)
#         graph_color = colors_dict.get(key, "?")
        
#         # Print information about the current graph
#         print(f"Graph {i+1}/{len(graphs_dict)}: {graph_name}")
#         print(f"  - Nodes: {nx_graph.number_of_nodes()}")
#         print(f"  - Edges: {nx_graph.number_of_edges()}")
#         print(f"  - Known colors: {graph_color}")
        
#         # Convert NetworkX graph to adjacency list
#         adj_list = networkx_to_adj_list(nx_graph)
        
#         adj_lists.append(adj_list)
#         names.append(graph_name)
#         colors.append(graph_color)
#     print("Over")
#     return adj_lists, names, colors

# def networkx_to_adj_list(nx_graph):
#     """
#     Convert a NetworkX graph to an adjacency list representation.
    
#     Args:
#         nx_graph (networkx.Graph): NetworkX graph
        
#     Returns:
#         list: Adjacency list where adj_list[i] contains neighbors of node i
#     """
#     # Get number of nodes
#     num_nodes = nx_graph.number_of_nodes()
    
#     # Ensure nodes are sequential integers from 0 to n-1
#     # If not, create a mapping
#     node_map = {}
#     for i, node in enumerate(sorted(nx_graph.nodes())):
#         node_map[node] = i
    
#     # Create adjacency list
#     adj_list = [[] for _ in range(num_nodes)]
    
#     # Add edges to adjacency list
#     for src, dst in nx_graph.edges():
#         src_idx = node_map.get(src, src)
#         dst_idx = node_map.get(dst, dst)
        
#         adj_list[src_idx].append(dst_idx)
#         adj_list[dst_idx].append(src_idx)  # For undirected graphs
    
#     return adj_list

import pickle
import numpy as np
import networkx as nx

def load_graphs_from_file(graphs_file, names_file, colors_file, return_nx=False):
    """
    Loads graphs, their names, and their colors from pickle files and converts graphs
    to NumPy adjacency list representation or keeps them as NetworkX graphs.
    
    Parameters:
        graphs_file (str): Path to the pickle file containing the graphs.
        names_file (str): Path to the pickle file containing the graph names.
        colors_file (str): Path to the pickle file containing the graph colors.
        return_nx (bool): If True, return original NetworkX graphs instead of adjacency lists
    Returns:
        tuple: A list of graphs (as adj lists or NetworkX graphs), a list of graph names, and a list of graph colors.
    """
    with open(graphs_file, 'rb') as file:
        graphs_dict = pickle.load(file)
    
    with open(names_file, 'rb') as file:
        names_dict = pickle.load(file)
        
    with open(colors_file, 'rb') as file:
        colors_dict = pickle.load(file)
    
    # Extract graphs, names, and colors in corresponding order
    graph_objects = []
    names = []
    colors = []
    
    print(f"Processing {len(graphs_dict)} graphs...")
    
    for i, key in enumerate(graphs_dict.keys()):
        nx_graph = graphs_dict[key]
        graph_name = names_dict.get(key, key)
        graph_color = colors_dict.get(key, "?")
        
        # Print information about the current graph
        print(f"Graph {i+1}/{len(graphs_dict)}: {graph_name}")
        print(f"  - Nodes: {nx_graph.number_of_nodes()}")
        print(f"  - Edges: {nx_graph.number_of_edges()}")
        print(f"  - Known colors: {graph_color}")
        
        if return_nx:
            # Keep the NetworkX graph
            graph_objects.append(nx_graph)
        else:
            # Convert NetworkX graph to adjacency list
            adj_list = networkx_to_adj_list(nx_graph)
            graph_objects.append(adj_list)
        
        names.append(graph_name)
        colors.append(graph_color)
    print("Over")
    return graph_objects, names, colors

def networkx_to_adj_list(nx_graph):
    """
    Convert a NetworkX graph to an adjacency list representation.
    
    Args:
        nx_graph (networkx.Graph): NetworkX graph
        
    Returns:
        list: Adjacency list where adj_list[i] contains neighbors of node i
    """
    # Get number of nodes
    num_nodes = nx_graph.number_of_nodes()
    
    # Ensure nodes are sequential integers from 0 to n-1
    # If not, create a mapping
    node_map = {}
    for i, node in enumerate(sorted(nx_graph.nodes())):
        node_map[node] = i
    
    # Create adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    
    # Add edges to adjacency list
    for src, dst in nx_graph.edges():
        src_idx = node_map.get(src, src)
        dst_idx = node_map.get(dst, dst)
        
        adj_list[src_idx].append(dst_idx)
        adj_list[dst_idx].append(src_idx)  # For undirected graphs
    
    return adj_list


def load_adj_lists_from_citation(pickle_path):
    """
    Load a pickle file containing NetworkX graphs and
    return adjacency lists for each graph.

    Args:
        pickle_path (str): Path to pickle file

    Returns:
        dict: {graph_name: {node: [neighbors]}}
    """
    with open(pickle_path, "rb") as f:
        graphs_dict = pickle.load(f)

    graph_objects = []

    for name, G in graphs_dict.items():
        if not isinstance(G, nx.Graph):
            raise TypeError(f"{name} is not a NetworkX graph")

        adj_list = networkx_to_adj_list(G)
        graph_objects.append(adj_list)


    return graph_objects