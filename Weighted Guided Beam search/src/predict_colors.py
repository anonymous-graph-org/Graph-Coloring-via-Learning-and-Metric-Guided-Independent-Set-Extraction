import time
import random
from queue import Queue
import copy
import numpy as np
import networkx as nx
from src.predict_mis import predict_mis


def predict_colors(model, adj_list, time_budget=1000, max_queue_size=4, weights=None):
    """
    Implements tree search algorithm to find maximum independent set using
    multiple probability maps from trained GCN model. Uses NumPy adjacency list.

    Args:
        model: Trained DeepGCN model
        adj_list (numpy.ndarray): Adjacency list representation of the graph
        time_budget (int): Time limit in seconds
        max_queue_size (int): Maximum number of states to keep in queue
        weights (list): Weights for different metrics [w_edge_density, w_avg_degree, w_progress, w_efficiency]

    Returns:
        int: Minimum number of colors required
    """
    # Default weights if none provided: edge density (0.4), avg degree (0.3), progress (-0.2), efficiency (-0.1)
    if weights is None:
        weights = [0.4, 0.3, -0.2, -0.1]
    num_nodes = len(adj_list)
    max_degree = max(len(neighbors) for neighbors in adj_list)

    class GraphState:
        def __init__(self, adj_list, labels=None, colors=0):
            self.adj_list = adj_list
            self.labels = labels if labels is not None else {}
            self.colors = colors
            self.num_nodes = len(adj_list)

            # Cache for metrics
            self._edge_density = None
            self._avg_degree = None
            self._combined_metric = None

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
            max_possible_edges = (num_nodes * (num_nodes - 1)) // 2

            if max_possible_edges == 0:
                self._edge_density = 0
            else:
                self._edge_density = num_edges / max_possible_edges

            return self._edge_density

        def get_avg_degree(self):
            """Returns average degree of unlabeled subgraph with caching"""
            if self._avg_degree is not None:
                return self._avg_degree

            subgraph_adj_list, _, _ = self.get_unlabeled_subgraph()
            num_nodes = len(subgraph_adj_list)

            if num_nodes == 0:
                self._avg_degree = 0
                return 0

            total_degree = sum(len(neighbors) for neighbors in subgraph_adj_list)
            self._avg_degree = total_degree / num_nodes

            return self._avg_degree

        def get_progress_rate(self):
            """Returns the proportion of vertices that have been colored"""
            labeled_count = len(self.labels)
            return labeled_count / self.num_nodes

        def get_color_efficiency(self):
            """Returns the ratio of vertices colored to colors used"""
            labeled_count = len(self.labels)
            if self.colors == 0:
                return 0
            return labeled_count / (self.colors * self.num_nodes)

        def get_average_closeness_centrality(self):
            """ Returns average closeness centrality of a graph """
            subgraph_adj_list, _, _ = self.get_unlabeled_subgraph()
            G = nx.Graph()
            for node, neighbors in subgraph_adj_list.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)

            # Compute closeness centrality for all nodes
            closeness_values = nx.closeness_centrality(G)

            # Compute and return the average
            avg_closeness = sum(closeness_values.values()) / len(closeness_values)
            return avg_closeness

        def get_edge_density_with_colors_used(self):
            """Returns edge density of unlabeled subgraph with multiplying with colors used/(max-degree+1)"""
            edge_density=self.get_edge_density()
            if edge_density == 0:
                return 0
            else:
                M=edge_density*(self.colors/max_degree+1)
                return M

        def get_combined_metric(self, weights):
            """Calculate combined metric using weighted metrics"""
            if self._combined_metric is not None:
                return self._combined_metric

            edge_density = self.get_edge_density()
            avg_degree = self.get_avg_degree() / max_degree  # Normalize
            progress_rate = self.get_progress_rate()
            color_efficiency = self.get_color_efficiency()

            # Combine metrics using weights
            self._combined_metric = (weights[0] * edge_density +
                                     weights[1] * avg_degree +
                                     weights[2] * progress_rate +
                                     weights[3] * color_efficiency)

            return self._combined_metric

    def update_queue(Q, new_state, max_size, weights):
        """
        Updates queue to maintain only max_size states with lowest combined metric
        """
        # Get all states including the new one
        states = list(Q.queue) + [new_state]
        states.sort(key=lambda x: x.get_combined_metric(weights))


        # Keep only the first max_size states (lowest combined metric)
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

    # Track processed states to avoid duplicates
    processed_signatures = set()

    # Main loop
    while not Q.empty():
        # Get all states in the queue
        states = list(Q.queue)

        # Skip iteration if queue is empty
        if not states:
            break

        # Sort states by combined metric
        states.sort(key=lambda x: x.get_combined_metric(weights))

        # Select the best state (lowest combined metric)
        current_state = states[0]
        Q.queue.remove(current_state)

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
        model_mis_list = predict_mis(model, subgraph_adj_list)

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
                    update_queue(Q, new_state, max_queue_size, weights)

    # If no solution was found, return a conservative upper bound (number of nodes)
    if min_colours == float('inf'):
        return num_nodes


    return min_colours
