# src/load_data.py
import pickle
import networkx as nx

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    graphs = []
    if isinstance(data, list):
        for graph in data:
            if not isinstance(graph, nx.Graph):
                raise TypeError(f"Expected a NetworkX Graph, but got {type(graph)}")
            graphs.append(graph)
    elif isinstance(data, nx.Graph):
        graphs.append(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected a NetworkX Graph or list of Graphs.")
    
    return graphs

# def load_data(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     graphs = []
#     if isinstance(data, list):
#         for graph_data in data:
#             graph = nx.Graph()
#             edge_index = graph_data['edge_index']
#             edges = edge_index.t().tolist()
#             graph.add_edges_from(edges)
#             graphs.append(graph)
#     else:
#         graph = nx.Graph()
#         edge_index = data['edge_index']
#         edges = edge_index.t().tolist()
#         graph.add_edges_from(edges)
#         graphs.append(graph)
#     return graphs

# def load_data(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
    
#     graphs = []
#     if isinstance(data, list):
#         for graph in data:
#             if not isinstance(graph, nx.Graph):
#                 raise TypeError(f"Expected a NetworkX Graph, but got {type(graph)}")
#             graphs.append(graph)
#     elif isinstance(data, nx.Graph):
#         graphs.append(data)
#     else:
#         raise TypeError(f"Unsupported data type: {type(data)}. Expected a NetworkX Graph or list of Graphs.")
    
#     return graphs
