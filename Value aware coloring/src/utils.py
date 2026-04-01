import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def load_model(model_path, model_class, *model_args, **model_kwargs):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        model_class (torch.nn.Module): The model class to load.
        *model_args: Arguments to pass to model_class constructor.
        **model_kwargs: Keyword arguments to pass to model_class constructor.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Create model instance
    model = model_class(*model_args, **model_kwargs)

    # Load state dictionary
    try:
        # Try loading with specified map_location
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        # Fallback to default loading
        model.load_state_dict(torch.load(model_path))

    # Set to evaluation mode
    model.eval()

    return model

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.theta0 = nn.Linear(in_dim, out_dim, bias=False)
        self.theta1 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, A_hat, H):
        """
        Forward pass for a single GCN layer.
        Args:
            A_hat (torch.Tensor): Normalized adjacency matrix (N x N).
            H (torch.Tensor): Feature matrix from the previous layer (N x Cl).
        Returns:
            torch.Tensor: Updated feature matrix for the next layer (N x Cl+1).
        """
        propagated_H = torch.matmul(A_hat, H)
        H_next = torch.matmul(H, self.theta0.weight) + torch.matmul(propagated_H, self.theta1.weight)
        return H_next

class DeepGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(DeepGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, adjacency_matrix, degree_matrix, features):
        """
        Forward pass for the GCN.
        Args:
            adjacency_matrix (torch.Tensor): Adjacency matrix (N x N).
            degree_matrix (torch.Tensor): Degree matrix (N x N).
            features (torch.Tensor): Input feature matrix (N x hidden_dim).
        Returns:
            torch.Tensor: Predicted output (N x hidden_dim).
        """
        # Normalize adjacency matrix: A_hat = D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag_embed(torch.pow(degree_matrix.diag(), -0.5))
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        A_hat = torch.matmul(torch.matmul(D_inv_sqrt, adjacency_matrix), D_inv_sqrt)

        H = features

        # Hidden layers with ReLU
        for layer in self.hidden_layers:
            H = F.relu(layer(A_hat, H))

        H = self.output_layer(H)

        return torch.sigmoid(H)  # Final output

    def from_adj_list(self, adj_list, device=None):
        """
        Helper method to create adjacency and degree matrices from adjacency list.

        Args:
            adj_list (list): Adjacency list representation of graph
            device: PyTorch device to place tensors on

        Returns:
            tuple: (adjacency_matrix, degree_matrix, features) for model input
        """
        num_nodes = len(adj_list)

        # Create adjacency matrix from adjacency list
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                adj_matrix[i, j] = 1

        # Calculate degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        degree_matrix = np.diag(degrees).astype(np.float32)

        # Create feature matrix (all ones)
        features = np.ones((num_nodes, self.hidden_dim), dtype=np.float32)

        # Convert to PyTorch tensors
        if device is None:
            device = next(self.parameters()).device

        adj_tensor = torch.tensor(adj_matrix, device=device)
        degree_tensor = torch.tensor(degree_matrix, device=device)
        features_tensor = torch.tensor(features, device=device)

        return adj_tensor, degree_tensor, features_tensor

