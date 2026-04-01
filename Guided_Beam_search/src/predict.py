import time
from queue import Queue
import copy
import numpy as np
from src.predict_mis import predict_mis


def predict_colors(model, adj_list, time_budget=1000, max_queue_size=4):
    """
    Implements tree search algorithm to find maximum independent set using
    multiple probability maps from trained GCN model. Uses NumPy adjacency list."""
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
                    update_queue(Q, new_state, max_queue_size)

    # If no solution was found, return a conservative upper bound (number of nodes)
    if min_colours == float('inf'):
        return num_nodes


    return min_colours
