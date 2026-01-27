import torch
import networkx as nx
import time
from queue import Queue
import random
import copy
from collections import defaultdict

def predict_mis(model, graph, time_budget=200, num_maps=32, max_solutions=4):
    device = next(model.parameters()).device
    model.eval()
    
    class GraphState:
        def __init__(self, graph, labels=None):
            self.graph = graph
            self.labels = labels if labels is not None else {}
            
        def is_completely_labeled(self):
            return len(self.labels) == len(self.graph)
            
        def get_unlabeled_subgraph(self):
            """Returns subgraph of unlabeled vertices"""
            unlabeled = [n for n in self.graph.nodes() if n not in self.labels]
            return self.graph.subgraph(unlabeled)

        def get_mis_signature(self):
            """Returns a tuple of sorted MIS nodes (for detecting unique solutions)"""
            return tuple(sorted(n for n, label in self.labels.items() if label == 1))
            
    def get_probability_maps(g):
        """Get M probability maps for graph g using the model"""
        adj_matrix = torch.tensor(nx.adjacency_matrix(g).todense(), dtype=torch.float32).to(device)
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        num_nodes = adj_matrix.size(0)
        features = torch.ones(num_nodes, model.hidden_dim).to(device)
        
        with torch.no_grad():
            prob_maps = model(adj_matrix, degree_matrix, features)  # Shape: [num_nodes, num_maps]
        return prob_maps
    
    # Initialize
    start_time = time.time()
    Q = Queue()
    initial_state = GraphState(graph)
    Q.put(initial_state)
    
    # Track solutions by size
    solutions_by_size = defaultdict(set)  # {size: set of solution signatures}
    best_solutions = {}  # {signature: solution_dict}
    best_size = 0
    
    # Main loop
    while time.time() - start_time < time_budget and not Q.empty():
        # Randomly select a state from queue
        states = list(Q.queue)
        current_state = random.choice(states)
        Q.queue.remove(current_state)
        
        # Get subgraph of unlabeled vertices
        subgraph = current_state.get_unlabeled_subgraph()
        if len(subgraph) == 0:
            continue
            
        # Get M probability maps
        prob_maps = get_probability_maps(subgraph)  # [num_nodes, num_maps]
        
        # Process each probability map
        for m in range(num_maps):
            # Create a new state by copying current state
            new_state = GraphState(graph, copy.deepcopy(current_state.labels))
            
            # Get probabilities for map m and sort vertices
            probs = prob_maps[:, m]
            vertices = sorted(list(subgraph.nodes()), 
                           key=lambda x: probs[list(subgraph.nodes()).index(x)],
                           reverse=True)
            
            # Label vertices greedily until we hit a labeled vertex
            for v in vertices:
                if v in new_state.labels:  # If we hit a labeled vertex, break
                    break
                    
                # Label current vertex as 1 (in MIS)
                new_state.labels[v] = 1
                
                # Label neighbors as 0 (not in MIS)
                for neighbor in graph.neighbors(v):
                    if neighbor not in new_state.labels:
                        new_state.labels[neighbor] = 0
            
            # After breaking from the vertex loop:
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
                # Add to queue for further exploration
                Q.put(new_state)
    
    # Return list of best solutions, up to max_solutions
    if not best_solutions:
        return []
    random_solutions = random.sample(list(best_solutions.values()), min(max_solutions, len(best_solutions)))

    # Return the randomly chosen solutions
    return random_solutions