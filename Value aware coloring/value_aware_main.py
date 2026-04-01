import pickle
import networkx as nx
import os
import copy
import sys
import random
import torch
from torch.utils.data import Dataset
from contextlib import contextmanager
import torch.nn as nn
import torch.nn.functional as F
import time
from queue import Queue
import numpy as np
from collections import defaultdict

class Degree12Reducer:
    """
    Performs degree-1 (pendant) and degree-2 (vertex folding) reductions
    on a NetworkX graph and returns the reduced graph.
    """

    def __init__(self, graph: nx.Graph):
        # Work on a copy so original graph is untouched
        self.graph = copy.deepcopy(graph)

        # Stats
        self.initial_nodes = self.graph.number_of_nodes()
        self.initial_edges = self.graph.number_of_edges()
        self.pendant_reductions = 0
        self.vertex_reductions = 0

    def pendant_reduction(self):
        """
        Remove a degree-1 vertex.
        """
        deg1_nodes = {v for v, d in self.graph.degree() if d == 1}
        if not deg1_nodes:
            return False
        v = random.choice(tuple(deg1_nodes))
        self.graph.remove_node(v)
        self.pendant_reductions += 1
        return True

    def vertex_folding(self):
        """
        Degree-2 vertex folding using v as representative:
        If v has neighbors u and w, and u and w are NOT adjacent,
        remove u and w, and connect v to all neighbors of u and w.
        """

        deg2_nodes = {v for v, d in self.graph.degree() if d == 2}
        if not deg2_nodes:
            return False
        for v in list(deg2_nodes):
            if len(list(self.graph.neighbors(v))) < 2:
                deg2_nodes.remove(v)
                continue
            u, w = list(self.graph.neighbors(v))
            if self.graph.has_edge(u, w):
                deg2_nodes.remove(v)

        if not deg2_nodes:
            return False
        v = random.choice(tuple(deg2_nodes))
        u, w = list(self.graph.neighbors(v))
        # Collect external neighbors of u and w
        external_neighbors = set()
        for x in self.graph.neighbors(u):
            if x not in {u, v, w}:
                external_neighbors.add(x)

        for x in self.graph.neighbors(w):
            if x not in {u, v, w}:
                external_neighbors.add(x)

        # Remove u and w
        self.graph.remove_node(u)
        self.graph.remove_node(w)

        # Remove old edges of v (to avoid stale connections)
        self.graph.remove_edges_from(list(self.graph.edges(v)))

        # Connect v to all external neighbors
        for x in external_neighbors:
            self.graph.add_edge(v, x)

        self.vertex_reductions += 1
        return True

    def reduce_graph(self):
        """
        Apply reductions until no more are possible.
        Returns:
            reduced_graph (nx.Graph)
            stats (dict)
        """
        changed = True
        while changed:
            changed = False
            if self.pendant_reduction():
                changed = True
                continue

            if self.vertex_folding():
                changed = True
                continue

        stats = {
            "initial_nodes": self.initial_nodes,
            "final_nodes": self.graph.number_of_nodes(),
            "initial_edges": self.initial_edges,
            "final_edges": self.graph.number_of_edges(),
            "pendant_reductions": self.pendant_reductions,
            "vertex_reductions": self.vertex_reductions,
            "node_reduction_ratio":
                1 - self.graph.number_of_nodes() / self.initial_nodes
                if self.initial_nodes > 0 else 0.0
        }

        number_reduced = self.initial_nodes - self.graph.number_of_nodes()
        if number_reduced > 0:
            checker = True
        else:
            checker = False

        return self.graph, stats, checker


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


def load_model(model_path, model_class, device, *model_args, **model_kwargs):
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
    # 1. Decide device safely
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = model_class(*model_args, **model_kwargs)

    # 3. Load weights safely
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Move model to device
    model.to(device)

    """ Load state dictionary
    try:
        # Try loading with specified map_location
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    except:
        # Fallback to default loading
        model.load_state_dict(torch.load(model_path))"""

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

def predict_colors(model, adj_list, device, time_budget=1000, max_queue_size=4):
    """
    Implements tree search algorithm to find maximum independent set using
    multiple probability maps from trained GCN model. Uses NumPy adjacency list.

    Args:
        model: Trained DeepGCN model
        adj_list (numpy.ndarray): Adjacency list representation of the graph
        time_budget (int): Time limit in seconds
        max_queue_size (int): Maximum number of states to keep in queue

    Returns:
        int: Minimum number of colors required
    """
    num_nodes = len(adj_list)
    max_degree = max(len(neighbors) for neighbors in adj_list)

    class GraphState:
        def __init__(self, adj_list, labels=None, colors=0):
            self.adj_list = adj_list
            self.labels = labels if labels is not None else {}
            self.colors = colors
            self.num_nodes = len(adj_list)
            self._edge_density = None  # Cache for edge density

        def is_completely_labeled(self):
            return len(self.labels) == self.num_nodes

        def get_unlabeled_nodes(self):
            """Returns list of unlabeled node indices"""
            return [n for n in range(self.num_nodes) if n not in self.labels]

        def get_unlabeled_subgraph(self):
            """Returns subgraph adjacency list of unlabeled vertices"""
            unlabeled = self.get_unlabeled_nodes()
            if not unlabeled:
                return [], [], {}

            node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unlabeled)}

            # Create new adjacency list for the subgraph
            subgraph_size = len(unlabeled)
            subgraph_adj_list = [[] for _ in range(subgraph_size)]

            # Optimized approach for building subgraph
            for i, node in enumerate(unlabeled):
                for neighbor in self.adj_list[node]:
                    if neighbor in node_map:
                        subgraph_adj_list[i].append(node_map[neighbor])

            return subgraph_adj_list, unlabeled, node_map

        def get_edge_density(self):
            """Returns edge density of unlabeled subgraph with caching"""
            if self._edge_density is not None:
                return self._edge_density

            subgraph_adj_list, _, _ = self.get_unlabeled_subgraph()
            num_nodes = len(subgraph_adj_list)
            if num_nodes <= 1:  # Handle cases with 0 or 1 node
                self._edge_density = 0
                return 0

            num_edges = sum(len(neighbors) for neighbors in subgraph_adj_list) // 2

            if num_nodes == 0:
                self._edge_density = 0
            else:
                self._edge_density = num_edges / num_nodes

            return self._edge_density

        def get_edge_density_with_colors_used(self):
            """Returns edge density of unlabeled subgraph with multiplying with colors used/(max-degree+1)"""
            edge_density = self.get_edge_density()
            if edge_density == 0:
                return 0
            else:
                M = edge_density * (self.colors / max_degree + 1)
                return M

    def update_queue(Q, new_state, max_size):
        """
        Updates queue to maintain only max_size states with lowest edge density
        """
        # Get all states including the new one
        states = list(Q.queue) + [new_state]

        # Sort states by edge density (lower is better)
        states.sort(key=lambda x: x.get_edge_density())

        # Keep only the first max_size states (lowest edge density)
        states = states[:max_size]

        # Clear the queue and add back the selected states
        Q.queue.clear()
        for state in states:
            Q.put(state)

    # Initialize
    Q = Queue()
    initial_state = GraphState(adj_list)
    Q.put(initial_state)
    min_colours = float('inf')
    state_counter=0

    # Track processed states to avoid duplicates
    processed_signatures = set()

    # Main loop
    # while not Q.empty() and time.time() - start_time < time_budget:
    while not Q.empty():
        # Select state with minimum edge density from queue
        states = list(Q.queue)

        # Skip iteration if queue is empty
        if not states:
            break

        # Find the state with minimum edge density
        current_state = min(states, key=lambda x: x.get_edge_density_with_colors_used())
        Q.queue.remove(current_state)
        state_counter += 1

        # Check for duplicate states
        current_sig = tuple(sorted(current_state.labels.items()))
        if current_sig in processed_signatures:
            continue
        processed_signatures.add(current_sig)

        # Get subgraph of unlabeled vertices
        subgraph_adj_list, original_nodes, node_map = current_state.get_unlabeled_subgraph()
        if len(subgraph_adj_list) == 0:
            # This state is completely labeled
            if current_state.colors < min_colours:
                min_colours = current_state.colors
            continue

        # Get all MIS of the remaining unlabelled subgraph
        model_mis_list = predict_mis(model, subgraph_adj_list, device)

        # If no MIS found, continue with next state
        if not model_mis_list:
            continue

        for mis in model_mis_list:
            # Create a new state by copying current state
            new_state = GraphState(adj_list, copy.deepcopy(current_state.labels), current_state.colors)

            # Labels those vertices which have label 1 in the mis as new_state.colors+1.
            col = new_state.colors + 1

            # Map MIS nodes from subgraph indices back to original graph indices
            for node_idx, val in mis.items():
                if isinstance(node_idx, str):
                    node_idx = int(node_idx)
                if val == 1:
                    # Convert subgraph index to original graph index
                    if node_idx < len(original_nodes):  # Safety check
                        orig_idx = original_nodes[node_idx]
                        new_state.labels[orig_idx] = col

            new_state.colors = col

            # After processing the MIS:
            if new_state.is_completely_labeled():
                # Update best solution if current is better
                if new_state.colors < min_colours:
                    min_colours = new_state.colors
            else:
                # Check for duplicate state
                new_sig = tuple(sorted(new_state.labels.items()))
                if new_sig not in processed_signatures:
                    # Update queue with new state while maintaining size limit
                    update_queue(Q, new_state, max_queue_size)

    # If no solution was found, return a conservative upper bound (number of nodes)
    if min_colours == float('inf'):
        return num_nodes

    print("state_counter:", state_counter)
    return min_colours


def predict_mis(model, adj_list, device, time_budget=60, num_maps=32, max_solutions=16):
    """
    Optimized tree search algorithm to find multiple maximum independent sets using
    multiple probability maps from trained GCN model. Uses NumPy adjacency list representation.

    Args:
        model: Trained DeepGCN model
        adj_list (numpy.ndarray): Adjacency list representation of the graph
                                 [node_idx][neighbor_indices]
        time_budget (int): Time limit in seconds
        num_maps (int): Number of probability maps to generate
        max_solutions (int): Maximum number of solutions to return

    Returns:
        list: List of MIS solutions (each a dict of node indices) with the best size
    """
    device = next(model.parameters()).device
    model.eval()

    class GraphState:
        def __init__(self, adj_list, labels=None):
            self.adj_list = adj_list
            self.labels = labels if labels is not None else {}
            self.num_nodes = len(adj_list)

        def is_completely_labeled(self):
            return len(self.labels) == self.num_nodes

        def get_unlabeled_nodes(self):
            """Returns list of unlabeled node indices"""
            return [n for n in range(self.num_nodes) if n not in self.labels]

        def get_unlabeled_subgraph(self):
            """Returns subgraph adjacency list of unlabeled vertices"""
            unlabeled = self.get_unlabeled_nodes()
            if not unlabeled:
                return [], [], {}

            node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unlabeled)}

            # Create new adjacency list for the subgraph
            subgraph_size = len(unlabeled)
            subgraph_adj_list = [[] for _ in range(subgraph_size)]

            # Optimized approach for building subgraph
            for i, node in enumerate(unlabeled):
                for neighbor in self.adj_list[node]:
                    if neighbor in node_map:
                        subgraph_adj_list[i].append(node_map[neighbor])

            return subgraph_adj_list, unlabeled, node_map

        def get_mis_signature(self):
            """Returns a tuple of sorted MIS nodes (for detecting unique solutions)"""
            return tuple(sorted(n for n, label in self.labels.items() if label == 1))

    def get_probability_maps(subgraph_adj_list, node_mapping):
        """Get M probability maps for subgraph using the model"""
        # Create adjacency matrix from adjacency list
        sub_num_nodes = len(subgraph_adj_list)
        if sub_num_nodes == 0:
            return torch.tensor([])

        adj_matrix = np.zeros((sub_num_nodes, sub_num_nodes), dtype=np.float32)

        # Optimized approach for building adjacency matrix
        for i, neighbors in enumerate(subgraph_adj_list):
            if neighbors:  # Only process if there are neighbors
                for j in neighbors:
                    adj_matrix[i, j] = 1

        # Convert to tensors
        adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

        # Calculate degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        degree_matrix_tensor = torch.diag(torch.tensor(degrees, dtype=torch.float32)).to(device)

        # Create feature tensor (all ones)
        features = torch.ones(sub_num_nodes, model.hidden_dim).to(device)

        with torch.no_grad():
            prob_maps = model(adj_matrix_tensor, degree_matrix_tensor, features)  # Shape: [num_nodes, num_maps]

        return prob_maps

    # Initialize
    start_time = time.time()
    Q = Queue()
    initial_state = GraphState(adj_list)
    Q.put(initial_state)

    # Track solutions by size
    solutions_by_size = defaultdict(set)  # {size: set of solution signatures}
    best_solutions = {}  # {signature: solution_dict}
    best_size = 0

    # Track processed states to avoid duplicates
    processed_signatures = set()

    # Main loop
    while time.time() - start_time < time_budget and not Q.empty():
        # Randomly select a state from queue - maintain diversity in search
        states = list(Q.queue)
        current_state = random.choice(states)
        Q.queue.remove(current_state)

        # Check if we've already processed a similar state
        current_sig = tuple(sorted(current_state.labels.items()))
        if current_sig in processed_signatures:
            continue
        processed_signatures.add(current_sig)

        # Get subgraph of unlabeled vertices
        subgraph_adj_list, original_nodes, node_map = current_state.get_unlabeled_subgraph()
        if len(subgraph_adj_list) == 0:
            continue

        # Get M probability maps
        prob_maps = get_probability_maps(subgraph_adj_list, node_map)  # [num_nodes, num_maps]
        if prob_maps.nelement() == 0:
            continue

        # Process each probability map
        for m in range(num_maps):
            # Create a new state by copying current state
            new_state = GraphState(adj_list, copy.deepcopy(current_state.labels))

            # Get probabilities for map m and sort vertices
            probs = prob_maps[:, m]
            vertices_indices = torch.argsort(probs, descending=True).cpu().numpy()
            vertices = [original_nodes[i] for i in vertices_indices]

            # Label vertices greedily until we hit a labeled vertex
            for v in vertices:
                if v in new_state.labels:  # If we hit a labeled vertex, break
                    break

                # Label current vertex as 1 (in MIS)
                new_state.labels[v] = 1

                # Label neighbors as 0 (not in MIS)
                for neighbor in adj_list[v]:
                    if neighbor not in new_state.labels:
                        new_state.labels[neighbor] = 0

            # Check if the state is completely labeled
            if new_state.is_completely_labeled():
                # Get solution size and signature
                mis_size = sum(1 for v in new_state.labels.values() if v == 1)
                solution_signature = new_state.get_mis_signature()

                if mis_size >= best_size:
                    # If we found a better size, clear previous solutions
                    if mis_size > best_size:
                        solutions_by_size.clear()
                        best_solutions.clear()
                        best_size = mis_size

                    # Add new solution if unique
                    if solution_signature not in solutions_by_size[mis_size]:
                        solutions_by_size[mis_size].add(solution_signature)
                        best_solutions[solution_signature] = dict(new_state.labels)

                        # If we have enough solutions, we can start being more selective
                        if len(best_solutions) >= max_solutions:
                            time_budget = min(time_budget, time.time() - start_time + 5)  # Give 5 more seconds
            else:
                # Only add to queue if it has a unique signature
                new_sig = tuple(sorted(new_state.labels.items()))
                if new_sig not in processed_signatures:
                    Q.put(new_state)

    # Return list of best solutions, up to max_solutions
    if not best_solutions:
        return []

    # Convert to list of solutions
    if len(best_solutions) <= max_solutions:
        return list(best_solutions.values())
    else:
        # Random sampling to maintain diversity as in original algorithm
        return random.sample(list(best_solutions.values()), max_solutions)


def find_colors(adj_list, model_path, device, output_stream):  # GBS Method 
    start_time = time.time()
    hidden_dim = 32
    num_layers = 20
    model = load_model(model_path, DeepGCN, device, hidden_dim, num_layers)
    num_nodes = len(adj_list)
    print(num_nodes)
    # num_edges = sum(len(neighbors) for neighbors in adj_list) // 2
    # print(num_edges)
    start_time = time.time()
    num_colors = predict_colors(model, adj_list, device)
    processing_time = time.time() - start_time
    print(f"  - Execution time: {processing_time:.2f} seconds", file=output_stream)
    print(f"  - Colors required: {num_colors}", file=output_stream)


class ValueAwareGraphDataset(Dataset):
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        data = self.graph_data[idx]
        graph = data['reduced_graph']
        mis_solutions = data['mis_solutions']

        # Create adjacency matrix
        adjacency_matrix = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float32)

        # Create degree matrix
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

        # Extract select and nonselect values from nodes
        nodes = sorted(graph.nodes())
        select_values = torch.tensor([graph.nodes[n]['select_value'] for n in nodes], dtype=torch.float32)
        nonselect_values = torch.tensor([graph.nodes[n]['nonselect_value'] for n in nodes], dtype=torch.float32)

        # Stack them for convenience in training
        node_features = torch.stack([select_values, nonselect_values], dim=1)

        # Create solution labels
        labels = torch.stack([
            torch.tensor([mis.get(n, 0) for n in nodes], dtype=torch.float32)
            for mis in mis_solutions
        ])  # Shape: [num_mis_solutions, num_nodes]

        return adjacency_matrix, degree_matrix, node_features, labels


def load_value_aware_model(model_path, device, hidden_dim=32, num_layers=20, num_maps=32):
    """
    Load a trained ValueAwareDeepGCN model.

    Args:
        model_path (str): Path to saved model state dict
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GCN layers
        num_maps (int): Number of probability maps
        device (str): Device to load model to ('cpu' or 'cuda')

    Returns:
        nn.Module: Loaded ValueAwareDeepGCN model
    """
    model = ValueAwareDeepGCN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_maps=num_maps
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


class ValueAwareDeepGCN(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=20, num_maps=32):
        """
        Initialize a Value-Aware GCN that takes features directly in hidden_dim.

        Args:
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of GCN layers
            num_maps (int): Number of probability maps to output
        """
        super(ValueAwareDeepGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_maps = num_maps

        # Hidden GCN layers
        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])

        # Output layer: transforms hidden_dim to num_maps (multiple probability maps)
        self.output_layer = nn.Linear(hidden_dim, num_maps)

    def forward(self, adjacency_matrix, degree_matrix, features):
        """
        Forward pass for the value-aware GCN.

        Args:
            adjacency_matrix (torch.Tensor): Adjacency matrix (N x N)
            degree_matrix (torch.Tensor): Degree matrix (N x N)
            features (torch.Tensor): Node features [N x hidden_dim] already in hidden dimension

        Returns:
            torch.Tensor: Multiple probability maps (N x num_maps)
        """
        # Normalize adjacency matrix: A_hat = D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag_embed(
            torch.pow(degree_matrix.diag() + 1e-8, -0.5))  # Add epsilon to avoid division by zero
        A_hat = torch.matmul(torch.matmul(D_inv_sqrt, adjacency_matrix), D_inv_sqrt)

        # Input features are already in hidden dimension
        H = features

        # Apply GCN layers with residual connections
        for layer in self.hidden_layers:
            H_new = F.relu(layer(A_hat, H))
            H = H_new + H  # Add residual connection for better gradient flow

        # Output layer produces multiple probability maps
        output = self.output_layer(H)

        # Apply sigmoid to get probabilities
        return torch.sigmoid(output)  # Shape: [N x num_maps]



def predict_value_aware_mis(model,adj_list,select_values,nonselect_values,device,time_budget=60,num_maps=32,max_solutions=16):

    model.eval()

    class GraphState:
        def __init__(self, adj_list,labels=None):
            self.adj_list=adj_list
            self.labels = labels if labels is not None else {}
            self.num_nodes = len(adj_list)

        def is_completely_labeled(self):
            return len(self.labels) == self.num_nodes

        def get_unlabeled_nodes(self):
            return [n for n in range(self.num_nodes) if n not in self.labels]

        def signature(self):
            return tuple(sorted(n for n, v in self.labels.items() if v == 1))

        def get_unlabelled_subgraph(self):
            unlabeled = self.get_unlabeled_nodes()
            if not unlabeled:
                return [],[],{}
            node_map={old_idx:new_idx for new_idx,old_idx in enumerate(unlabeled)}

            subgraph_size=len(unlabeled)
            subgraph_adj_list=[[] for _ in range (subgraph_size)]
            sub_select = [0] * subgraph_size
            sub_nonselect = [0] * subgraph_size

            for new_idx,old_idx in enumerate(unlabeled):
                sub_select[new_idx] = select_values[old_idx]
                sub_nonselect[new_idx] = nonselect_values[old_idx]

            for i,node in enumerate(unlabeled):
                for neighbor in self.adj_list[node]:
                    if neighbor in node_map:
                        subgraph_adj_list[i].append(node_map[neighbor])

            return subgraph_adj_list, sub_select,sub_nonselect,unlabeled,node_map

    def get_probability_maps(subgraph_adj_list,select_values,nonselect_values,node_mapping):

        sub_num_nodes=len(subgraph_adj_list)
        #node_map = {old: i for i, old in enumerate(nodes)}
        if sub_num_nodes==0:
            return torch.tensor([])

        # Build adjacency matrix
        adj_matrix = torch.zeros((sub_num_nodes, sub_num_nodes), dtype=torch.float32, device=device)

        for i,neighbors in enumerate(subgraph_adj_list):
            if neighbors:
                for j in neighbors:
                    adj_matrix[i,j]=1

        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1)).to(device)

        # Value arrays
        sel = torch.tensor([select_values[n] for n in range(sub_num_nodes)],dtype=torch.float32,device=device)
        nonsel = torch.tensor([nonselect_values[n] for n in range(sub_num_nodes)],dtype=torch.float32,device=device)

        features = torch.ones((sub_num_nodes, model.hidden_dim), device=device)
        features[:, 0] = sel
        features[:, 1] = nonsel

        with torch.no_grad():
            prob_maps = model(adj_matrix, degree_matrix, features)

        return prob_maps

    start_time = time.time()

    Q = Queue()
    initial_state=GraphState(adj_list)
    Q.put(initial_state)

    # Track solutions by size
    solutions_by_size=defaultdict(set)
    best_solutions={}
    best_value=0
    solutions = []

    # Track processed states to avoid duplicates
    seen_signatures = set()

    while time.time() - start_time < time_budget and not Q.empty():

        current_state = random.choice(list(Q.queue))
        Q.queue.remove(current_state)

        current_sig=current_state.signature()
        if current_sig in seen_signatures:
            continue
        seen_signatures.add(current_sig)

        subgraph_adj_list,sub_select,sub_nonselect, original_nodes,node_map = current_state.get_unlabelled_subgraph()
        if len(subgraph_adj_list) == 0:
            continue

        prob_maps = get_probability_maps(subgraph_adj_list,sub_select,sub_nonselect,node_map)
        if prob_maps.nelement() == 0:
            continue

        for m in range(min(num_maps, prob_maps.size(1))):

            new_state = GraphState(adj_list,copy.deepcopy(current_state.labels))

            probs = prob_maps[:, m]
            vertices_indices=torch.argsort(probs,descending=True).cpu().numpy()
            vertices=[original_nodes[v] for v in vertices_indices]

            for v in vertices:

                if v in new_state.labels:
                    break

                new_state.labels[v] = 1

                for nbr in adj_list[v]:
                    if nbr not in new_state.labels:
                        new_state.labels[nbr] = 0

            if new_state.is_completely_labeled():

                total_value = sum(select_values[n] if l == 1 else nonselect_values[n] for n, l in new_state.labels.items())

                sig = new_state.signature()
                if total_value >= best_value: # if we found a better value, clear previous solutions
                    if total_value>best_value:
                        solutions_by_size.clear()
                        best_solutions.clear()
                        best_value=total_value

                    if sig not in solutions_by_size[total_value]:
                        solutions_by_size[total_value].add(sig)
                        best_solutions[sig]=dict(new_state.labels)

                        if len(best_solutions)>=max_solutions:
                            time_budget=min(time_budget,time.time()-start_time+5) # Give 5 more seconds

            else:
                sig = new_state.signature()
                if sig not in seen_signatures:
                    Q.put(new_state)


    if not best_solutions:
        return[]

    if len(best_solutions)<=max_solutions:
        return list(best_solutions.values())
    else:
        return random.sample(list(best_solutions.values()),max_solutions)


def adjlist_to_nx(adj_list):
    G = nx.Graph()
    n = len(adj_list)
    G.add_nodes_from(range(n))
    for u, nbrs in enumerate(adj_list):
        for v in nbrs:
            if u < v:  # avoid duplicate edges
                G.add_edge(u, v)
    return G


def predict_value_colors(model, adj_list, device, time_budget=1000, max_queue_size=4):

    # ----------------------------------------
    # Reduce graph
    # ----------------------------------------

    original_graph = adjlist_to_nx(adj_list)

    reducer = SimplifiedValueAwareGraph(original_graph)
    reduced_graph, merged_history = reducer.reduce_graph()

    reduced_nodes = list(reduced_graph.nodes())
    node_map = {old: new for new, old in enumerate(reduced_nodes)}

    n = len(reduced_nodes)

    red_adj_list = [[] for _ in range(n)]

    for old_u in reduced_nodes:
        for old_v in reduced_graph.neighbors(old_u):
            red_adj_list[node_map[old_u]].append(node_map[old_v])

    values = {
        node_map[old]: {
            "select_value": reduced_graph.nodes[old]["select_value"],
            "nonselect_value": reduced_graph.nodes[old]["nonselect_value"]
        }
        for old in reduced_nodes
    }

    max_degree = max(len(neigh) for neigh in red_adj_list) if red_adj_list else 0

    # ----------------------------------------
    # GraphState
    # ----------------------------------------

    class GraphState:

        def __init__(self, adj_list, values, labels=None, active=None, colors=0):

            self.adj_list = adj_list
            self.values = {k: v.copy() for k, v in values.items()}

            if labels is None:
                self.labels = {i: [] for i in values}
            else:
                self.labels = {k: v.copy() for k, v in labels.items()}

            if active is None:
                self.active = set(values.keys())
            else:
                self.active = set(active)

            self.colors = colors

        # -------------------------------

        def is_resolved(self):
            return len(self.active) == 0

        # -------------------------------

        def remove_fully_resolved_nodes(self):

            for n in list(self.active):

                if (self.values[n]["select_value"] == 0 and
                        self.values[n]["nonselect_value"] == 0):

                    self.active.remove(n)

        # -------------------------------

        def build_active_adj_list(self):
            """
            Build adjacency list for active nodes only.

            Returns:
                active_adj_list : adjacency list of active subgraph
                active_nodes    : list mapping new_idx -> old_idx
                node_map        : dict mapping old_idx -> new_idx
            """

            active_nodes = sorted(self.active)

            node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_nodes)}

            size = len(active_nodes)

            active_adj_list = [[] for _ in range(size)]

            for old_u in active_nodes:

                new_u = node_map[old_u]

                for old_v in self.adj_list[old_u]:

                    if old_v in node_map:
                        active_adj_list[new_u].append(node_map[old_v])

            return active_adj_list, active_nodes, node_map

        def isolate_zero_select_nodes(self):

            for node in list(self.active):

                if (self.values[node]["select_value"] == 0 and
                        self.values[node]["nonselect_value"] > 0):

                    for nbr in list(self.adj_list[node]):
                        if nbr in self.active and node in self.adj_list[nbr]:
                            self.adj_list[nbr].remove(node)

                    self.adj_list[node].clear()

        # -------------------------------

        def get_edge_density(self):

            active_nodes = list(self.active)

            if len(active_nodes) <= 1:
                return 0

            edges = 0

            for node in active_nodes:
                for nbr in self.adj_list[node]:
                    if nbr in self.active:
                        edges += 1

            edges //= 2

            return edges / len(active_nodes)

        # -------------------------------

        def get_edge_density_with_colors_used(self):

            edge_density = self.get_edge_density()

            if edge_density == 0:
                return 0

            return edge_density * (self.colors / (max_degree + 1) + 1)

        # -------------------------------
        
        def get_value_density(self):
          if not self.active:
            return 0

          total_value = 0

          for n in self.active:
            total_value += self.values[n]["select_value"]
            total_value += self.values[n]["nonselect_value"]

          return total_value / len(self.active)


        def get_priority_score(self):
          edge_density = self.get_edge_density()
          value_density = self.get_value_density()
          return (edge_density *  (self.colors / (max_degree + 1) + 1)*0.5 + 0.5 * (value_density/len(reduced_nodes)))

        def signature(self):

            active_sig = tuple(sorted(self.active))

            value_sig = tuple(
                (n,
                 self.values[n]["select_value"],
                 self.values[n]["nonselect_value"])
                for n in sorted(self.values)
            )

            return (active_sig, value_sig)

        # -------------------------------

        def get_value_arrays(self):

            select_vals = [self.values[i]["select_value"] for i in range(len(self.adj_list))]
            nonselect_vals = [self.values[i]["nonselect_value"] for i in range(len(self.adj_list))]

            return select_vals, nonselect_vals

    # ----------------------------------------
    # Queue pruning
    # ----------------------------------------

    def update_queue(Q, new_state, max_size):

        states = list(Q.queue) + [new_state]

        states.sort(
            key=lambda x: x.get_priority_score()
        )

        states = states[:max_size]

        Q.queue.clear()

        for s in states:
            Q.put(s)

    # ----------------------------------------
    # Initialize search
    # ----------------------------------------

    Q = Queue()

    initial_state = GraphState(red_adj_list, values)

    Q.put(initial_state)

    processed_signatures = set()

    min_colors = float('inf')

    start_time = time.time()

    # ----------------------------------------
    # Main search loop
    # ----------------------------------------

    while not Q.empty() and time.time() - start_time < time_budget:

        states = list(Q.queue)

        if not states:
            break

        current_state = min(
            states,
            key=lambda x: x.get_priority_score()
        )

        Q.queue.remove(current_state)

        current_sig = current_state.signature()

        if current_sig in processed_signatures:
            continue

        processed_signatures.add(current_sig)

        if current_state.is_resolved():

            if current_state.colors < min_colors:
                min_colors = current_state.colors

            continue

        active_adj_list, active_nodes, node_map = current_state.build_active_adj_list()

        select_vals = [current_state.values[n]["select_value"] for n in active_nodes]
        nonselect_vals = [current_state.values[n]["nonselect_value"] for n in active_nodes]

        #select_vals, nonselect_vals = current_state.get_value_arrays()

        model_mis_list = predict_value_aware_mis(
            model,
            active_adj_list,
            select_vals,
            nonselect_vals,
            device
        )

        if not model_mis_list:
            continue

        for reduced_mis in model_mis_list:

            new_state = GraphState(
                current_state.adj_list,
                current_state.values,
                current_state.labels,
                current_state.active,
                current_state.colors
            )

            new_color = new_state.colors + 1

            for sub_idx, decision in reduced_mis.items():

                if isinstance(sub_idx, str):
                    sub_idx = int(sub_idx)

                red_node=active_nodes[sub_idx]

                if decision == 1 and new_state.values[red_node]["select_value"] > 0:

                    new_state.values[red_node]["select_value"] = 0
                    new_state.labels[red_node].append(new_color)

                elif decision == 1 and new_state.values[red_node]["select_value"] == 0:
                    new_state.values[red_node]["nonselect_value"] = 0
                    new_state.labels[red_node].append(new_color)
                else:

                    new_state.values[red_node]["nonselect_value"] = 0

            new_state.colors = new_color

            new_state.isolate_zero_select_nodes()

            new_state.remove_fully_resolved_nodes()

            if new_state.is_resolved():

                if new_state.colors < min_colors:
                    min_colors = new_state.colors
                    print("resolved",min_colors)

            else:

                new_sig = new_state.signature()

                if new_sig not in processed_signatures:
                    update_queue(Q, new_state, max_queue_size)

    if min_colors == float('inf'):
        return n

    return min_colors,time.time() - start_time


class SimplifiedValueAwareGraph:
    def __init__(self, graph):
        """
        Initialize a simplified value-aware graph for reduction tracking.
        This version doesn't track inclusion status and only uses standard folding rules.

        Parameters:
            graph (nx.Graph): Original NetworkX graph
            logger: Optional logger object for logging
        """
        self.graph = copy.deepcopy(graph)

        # Initialize node attributes
        for node in self.graph.nodes():
            self.graph.nodes[node]['select_value'] = 1
            self.graph.nodes[node]['nonselect_value'] = 0

        # Keep track of merged nodes
        self.merged_history = {}
        for node in self.graph.nodes():
            self.merged_history[node] = {
                "select": [node],  # include this node if selected
                "nonselect": []  # include nothing if not selected
            }

        """for node in self.graph.nodes():
            self.merged_history[node] = [node]  # Initially, each node only contains itself """

    def pendant_fold(self):
        """
        Perform one pendant vertex folding reduction on the graph.
        Updates the neighbor node with new values and removes the leaf vertex.
        Uses equations (8) and (9) from the specification.

        Returns:
            bool: True if a fold was performed; False otherwise.
        """
        for v in list(self.graph.nodes()):
            if self.graph.degree(v) == 1:
                # Get the neighbor
                nbrs = list(self.graph.neighbors(v))
                u = nbrs[0]

                # Get values
                select_u = self.graph.nodes[u]['select_value']
                nonselect_u = self.graph.nodes[u]['nonselect_value']
                select_v = self.graph.nodes[v]['select_value']
                nonselect_v = self.graph.nodes[v]['nonselect_value']

                # Apply standard pendant folding rule (equations 8 and 9)
                new_select = select_u + nonselect_v
                new_nonselect = nonselect_u + select_v

                # update merged history
                mh_u = self.merged_history.get(u, {"select": [u], "nonselect": []})
                mh_v = self.merged_history.get(v, {"select": [v], "nonselect": []})

                self.merged_history[u] = {"select": mh_u["select"] + mh_v["nonselect"],
                                          "nonselect": mh_v["select"] + mh_u["nonselect"]}
                # Update merged history
                # self.merged_history[u] = self.merged_history.get(u, [u]) + self.merged_history.get(v, [v])

                # Update the neighbor node with new values
                self.graph.nodes[u]['select_value'] = new_select
                self.graph.nodes[u]['nonselect_value'] = new_nonselect

                # Remove the leaf vertex
                self.graph.remove_node(v)
                if v in self.merged_history:
                    del self.merged_history[v]

                return True
        return False

    def vertex_fold(self):
        """
        Perform one vertex folding reduction on the graph for degree-2 vertices.
        Creates a representative node with values calculated according to
        equations (6) and (7) from the specification.

        Returns:
            bool: True if a fold was performed; False otherwise.
        """
        for v in list(self.graph.nodes()):
            if self.graph.degree(v) == 2:
                nbrs = list(self.graph.neighbors(v))
                if len(nbrs) != 2:
                    continue  # Safety check
                u, w = nbrs

                # Only fold if the two neighbors are not directly connected
                if self.graph.has_edge(u, w):
                    continue

                # Use the minimum node ID as the representative
                # rep_node = min(u, v, w)
                rep_node = v

                # Get values
                select_u = self.graph.nodes[u]['select_value']
                nonselect_u = self.graph.nodes[u]['nonselect_value']
                select_v = self.graph.nodes[v]['select_value']
                nonselect_v = self.graph.nodes[v]['nonselect_value']
                select_w = self.graph.nodes[w]['select_value']
                nonselect_w = self.graph.nodes[w]['nonselect_value']

                # Apply standard vertex folding rule (equations 6 and 7)
                new_select = select_u + select_w + nonselect_v
                new_nonselect = select_v + nonselect_u + nonselect_w

                # Collect all external neighbors of u, v, and w
                external_nbrs = set()
                for node in set(self.graph.neighbors(u)) | set(self.graph.neighbors(w)):
                    if node not in {u, v, w}:
                        external_nbrs.add(node)

                # Log the values if logger is available
                mh_u = self.merged_history.get(u, {"select": [u], "nonselect": []})
                mh_v = self.merged_history.get(v, {"select": [v], "nonselect": []})
                mh_w = self.merged_history.get(w, {"select": [w], "nonselect": []})

                self.merged_history[rep_node] = {"select": mh_v["nonselect"] + mh_u["select"] + mh_w["select"],
                                                 "nonselect": mh_v["select"] + mh_u["nonselect"] + mh_w["nonselect"]}
                """# Update merged history
                merged_nodes = []
                for node in [u, v, w]:
                    merged_nodes.extend(self.merged_history.get(node, [node]))
                self.merged_history[rep_node] = merged_nodes"""

                # Remove all original nodes
                self.graph.remove_node(u)
                self.graph.remove_node(v)
                if w in self.graph:  # Check if w still exists
                    self.graph.remove_node(w)
                # Remove merged history of u and w
                if u in self.merged_history:
                    del self.merged_history[u]
                if w in self.merged_history:
                    del self.merged_history[w]

                # Add the representative node with new values
                self.graph.add_node(rep_node,
                                    select_value=new_select,
                                    nonselect_value=new_nonselect)

                # Connect representative to external neighbors
                for nbr in external_nbrs:
                    self.graph.add_edge(rep_node, nbr)

                return True
        return False

    def reduce_graph(self):
        """
        Reduce the graph by repeatedly applying folding operations until
        no more reductions are possible.

        Returns:
            tuple: (reduced graph, merged history)
        """
        reduction_performed = True

        while reduction_performed:
            reduction_performed = False

            # Try pendant folding
            if self.pendant_fold():
                reduction_performed = True
                continue

            # Try vertex folding
            if self.vertex_fold():
                reduction_performed = True
                continue

        return self.graph, self.merged_history

    def get_node_values(self):
        """
        Get the select and nonselect values for all nodes in the graph.

        Returns:
            dict: Dictionary mapping node IDs to (select_value, nonselect_value) tuples
        """
        result = {}
        for node in self.graph.nodes():
            result[node] = (
                self.graph.nodes[node]['select_value'],
                self.graph.nodes[node]['nonselect_value']
            )
        return result

    def calculate_max_value(self):
        """
        Calculate the maximum possible value achievable from the current graph state.

        Returns:
            int: The maximum value possible from the current graph
        """
        max_value = 0
        for node in self.graph.nodes():
            # Add the maximum between select and nonselect value for each node
            max_value += max(
                self.graph.nodes[node]['select_value'],
                self.graph.nodes[node]['nonselect_value']
            )
        return max_value

    def reconstruct_solution(self):
        """
        Reconstruct a solution from the reduced graph by choosing the maximum
        value option (select or nonselect) for each node.

        Returns:
            dict: Dictionary mapping node IDs to boolean (True if selected)
        """
        solution = {}

        # For each node in the reduced graph, choose the option with higher value
        for node in self.graph.nodes():
            select_val = self.graph.nodes[node]['select_value']
            nonselect_val = self.graph.nodes[node]['nonselect_value']

            # Choose to select the node if its select_value is higher
            selected = select_val > nonselect_val

            # Map the node and all its merged components to this decision
            for orig_node in self.merged_history[node]:
                solution[orig_node] = selected

        return solution

    def lift_solution(self, reduced_solution):
        """
        Lift an MIS solution from the reduced graph back to the original graph.

        Parameters:
            reduced_solution (dict):
                Mapping reduced_node -> 0 or 1

        Returns:
            dict:
                Mapping original_node -> 0 or 1
        """
        original_solution = {}

        for red_node, decision in reduced_solution.items():
            if red_node not in self.merged_history:
                if decision == 1:
                    original_solution[red_node] = 1
                else:
                    original_solution[red_node] = 0
                continue

            info = self.merged_history[red_node]

            # Choose correct branch based on decision
            chosen_nodes = info["select"] if decision == 1 else info["nonselect"]
            non_chosen_nodes = info["nonselect"] if decision == 1 else info["select"]

            for v in chosen_nodes:
                original_solution[v] = 1
            for v in non_chosen_nodes:
                original_solution[v] = 0

        return original_solution

    def remove_labelled_nodes(self, labelled_nodes):
        graph_1 = self.graph.copy()
        for v in labelled_nodes:
            if v in graph_1.nodes():
                graph_1.remove_node(v)
            else:
                continue
        return graph_1


def color_graph(adj_list, model_path, device, output_stream):  # value-aware approach
    #start_time = time.time()
    model = load_value_aware_model(model_path, device, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_maps=NUM_MAPS)
    # Predict colors with original stdout
    #start_time = time.time()
    num_colors,processing_time = predict_value_colors(model, adj_list, device)
    #processing_time = time.time() - start_time

    # Write both execution time and colors to file
    print(f"  - Execution time: {processing_time:.2f} seconds", file=output_stream)
    print(f"  - Colors required: {num_colors}", file=output_stream)


def build_undirected_graph(adj_list):
    G = nx.Graph()
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if u < v:  # Ensure each edge is added only once for undirected graph
                G.add_edge(u, v)
    return G


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
        print(f"Graph {i + 1}/{len(graphs_dict)}: {graph_name}")
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


if __name__ == "__main__":
    HIDDEN_DIM = 32  # Hidden dimension
    NUM_LAYERS = 20  # Deeper model
    NUM_MAPS = 32  # Number of probability maps
    EPOCHS = 200  # Training epochs
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 1
    TIME_BUDGET = 60  # Time budget for prediction (seconds)
    SEED = 42

    # Store the original stdout
    original_stdout = sys.stdout


    @contextmanager
    def redirect_stdout(new_stdout):
        """Context manager to temporarily redirect stdout"""
        sys.stdout = new_stdout
        try:
            yield
        finally:
            sys.stdout = original_stdout
            
            
    graphs_file = "networkx_graphs.pkl"
    output_file = "output_value.txt"
    names_file = "graph_names.pkl"
    colors_file = "graph_colors.pkl"
    model_path1 = "gcn_model.pth"
    model_path2 = "value_aware_gcn.pth"
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    start_time = time.time()
    with redirect_stdout(original_stdout):
        adj_lists, names, colors = load_graphs_from_file(graphs_file, names_file, colors_file)
    print(f"Loaded {len(adj_lists)} graphs in {time.time() - start_time:.2f} seconds")

    # Open output file for writing
    output_stream = open(output_file, 'a', buffering=1)
    print("Starting graph processing...", file=output_stream)

    c = [20,45,47,50,58,60,64]

    # Process each graph
    for i, adj_list in enumerate(adj_lists):
        if i not in c:
            continue
        # Predict colors with original stdout
        print(names[i], file=output_stream)
        num_nodes = len(adj_list)
        print(num_nodes)
        num_edges = sum(len(neighbors) for neighbors in adj_list) // 2
        print(num_edges)

        G = build_undirected_graph(adj_list)
        reducer = Degree12Reducer(G)
        reduced_G, stats, checker = reducer.reduce_graph()
        print("Original nodes:", G.number_of_nodes(), file=output_stream)
        print("Reduced nodes:", reduced_G.number_of_nodes(), file=output_stream)
        print("Stats:", stats, file=output_stream)
        adj_list2 = networkx_to_adj_list(reduced_G)

        start_time = time.time()
        with redirect_stdout(original_stdout):

            find_colors(adj_list, model_path1, device, output_stream)
            find_colors(adj_list2, model_path1, device, output_stream)
            if G.number_of_nodes() == reduced_G.number_of_nodes():
                print("No reduction")
            else:
                color_graph(adj_list, model_path2, device, output_stream)

    output_stream.close()




