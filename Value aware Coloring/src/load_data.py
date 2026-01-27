import pickle
import numpy as np


def load_data(file_path):
    """
    Load data from a pickle file and convert to NumPy adjacency list representation.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        list: List of adjacency lists, each represented as a list of lists
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    adj_lists = []

    if isinstance(data, list):
        for graph_data in data:
            edge_index = graph_data['edge_index']
            adj_list = edge_index_to_adj_list(edge_index)
            adj_lists.append(adj_list)
    else:
        edge_index = data['edge_index']
        adj_list = edge_index_to_adj_list(edge_index)
        adj_lists.append(adj_list)

    return adj_lists


def edge_index_to_adj_list(edge_index):
    """
    Convert an edge index tensor to an adjacency list.

    Args:
        edge_index: Tensor of shape [2, num_edges] containing edge indices

    Returns:
        list: Adjacency list where adj_list[i] contains neighbors of node i
    """
    edges = edge_index.t().tolist()

    # Find the maximum node index to determine the size of the adjacency list
    max_node = max(max(src, dst) for src, dst in edges)

    # Initialize the adjacency list
    adj_list = [[] for _ in range(max_node + 1)]

    # Add edges to the adjacency list
    for src, dst in edges:
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # For undirected graphs

    return adj_list