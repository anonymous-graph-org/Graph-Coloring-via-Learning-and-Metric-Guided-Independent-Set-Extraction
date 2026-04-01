import torch
import numpy as np
import time
from queue import Queue
import random
import copy
from collections import defaultdict

def predict_mis(model, adj_list, time_budget=60, num_maps=32, max_solutions=16):
    """
    Optimized tree search algorithm to find multiple maximum independent sets using
    multiple probability maps from trained GCN model. Uses NumPy adjacency list representation."""
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
            for v in vertices[:3]:
                if v in new_state.labels:  # If we hit a labeled vertex, break
                    continue
                    
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
