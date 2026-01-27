import time
from queue import Queue
import networkx as nx
import copy
import torch
from src import config
import numpy as np
from src.predict import predict_value_aware_mis
from src.load_model import load_value_aware_model
from src.value_aware_reduction import ValueAwareGraph
from src. simple_reduction import SimplifiedValueAwareGraph

def adjlist_to_nx(adj_list):
    G = nx.Graph()
    n = len(adj_list)
    G.add_nodes_from(range(n))
    for u, nbrs in enumerate(adj_list):
        for v in nbrs:
            if u < v:   # avoid duplicate edges
                G.add_edge(u, v)
    return G

def predict_colors(model, adj_list, logger,time_budget=1000, max_queue_size=4):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        current_state = min(states, key=lambda x: x.get_edge_density())
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

        # Create a value-aware graph for prediction (without providing MIS solution) - pass the logger
        original_graph= adjlist_to_nx(subgraph_adj_list)
        value_graph_pred = SimplifiedValueAwareGraph(original_graph, logger=logger)
        # Reduce the graph
        red_graph, merged_history = value_graph_pred.reduce_graph()

        model_red_mis_list= predict_value_aware_mis(model, red_graph, time_budget=config.TIME_BUDGET,num_maps=config.NUM_MAPS, device=device)

        if not model_red_mis_list:
            continue

        # Returns a single MIS solution
        #reduced_mis = predict_value_aware_mis(model, red_graph, time_budget=config.TIME_BUDGET,num_maps=config.NUM_MAPS, device=device)




        for reduced_mis in model_red_mis_list:
            original_mis = value_graph_pred.lift_solution(reduced_mis)
            # If no MIS found, continue with next state
            if not original_mis:
                continue
            # Create a new state by copying current state
            new_state = GraphState(adj_list, copy.deepcopy(current_state.labels), current_state.colors)

            # Labels those vertices which have label 1 in the mis as new_state.colors+1.
            col = new_state.colors + 1

            # Map MIS nodes from subgraph indices back to original graph indices
            for node_idx, val in original_mis.items():
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

    return min_colours