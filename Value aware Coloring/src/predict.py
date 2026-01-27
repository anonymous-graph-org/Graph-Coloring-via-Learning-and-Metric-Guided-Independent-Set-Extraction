import torch
import networkx as nx
import time
from queue import Queue
import random
import copy

"""def predict_value_aware_mis(model, graph, time_budget=30, num_maps=32, device='cpu'):
    
    Implements tree search algorithm to find maximum independent set using
    multiple probability maps from a value-aware GCN model, prioritizing
    nodes based on probability*select_value/nonselect_value ratio.

    Args:
        model: Trained ValueAwareDeepGCN model
        graph (networkx.Graph): Input graph with 'select_value' and 'nonselect_value' node attributes
        time_budget (int): Time limit in seconds
        num_maps (int): Number of probability maps to use
        device (str): Device to use for inference ('cpu' or 'cuda')

    Returns:
        dict: Best MIS solution found (node labels as 0 or 1)
    
    model.eval()
    
    class GraphState:
        def __init__(self, graph, labels=None):
            self.graph = graph
            self.labels = labels if labels is not None else {}
            
        def is_completely_labeled(self):
            return len(self.labels) == len(self.graph)
            
        def get_unlabeled_subgraph(self):
            #Returns subgraph of unlabeled vertices with their attributes
            unlabeled = [n for n in self.graph.nodes() if n not in self.labels]
            return self.graph.subgraph(unlabeled)
            
    def get_probability_maps(g):
        #Get probability maps for graph g using the value-aware model
        # Create adjacency matrix
        adj_matrix = torch.tensor(nx.adjacency_matrix(g).todense(), dtype=torch.float32).to(device)
        
        # Create degree matrix
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1)).to(device)
        
        # Extract select and nonselect values
        nodes = sorted(g.nodes())
        select_values = torch.tensor([g.nodes[n]['select_value'] for n in nodes], dtype=torch.float32).to(device)
        nonselect_values = torch.tensor([g.nodes[n]['nonselect_value'] for n in nodes], dtype=torch.float32).to(device)
        
        # Create features tensor with select/nonselect values as first two dimensions and ones for the rest
        num_nodes = len(nodes)
        features = torch.ones(num_nodes, model.hidden_dim, device=device)
        features[:, 0] = select_values
        features[:, 1] = nonselect_values
        
        # Get predictions
        with torch.no_grad():
            prob_maps = model(adj_matrix, degree_matrix, features)  # Shape: [num_nodes, num_maps]
        
        return prob_maps, nodes, select_values, nonselect_values
    
    # Initialize
    start_time = time.time()
    Q = Queue()
    initial_state = GraphState(graph)
    Q.put(initial_state)
    best_solution = None
    best_value = -1
    
    # Main loop
    while time.time() - start_time < time_budget and not Q.empty():
        # Randomly select a state from queue
        states = list(Q.queue)
        if not states:
            break
        current_state = random.choice(states)
        Q.queue.remove(current_state)
        
        # Get subgraph of unlabeled vertices
        subgraph = current_state.get_unlabeled_subgraph()
        if len(subgraph) == 0:
            continue
            
        # Get probability maps and values
        prob_maps, nodes, select_values, nonselect_values = get_probability_maps(subgraph)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1
        
        # Process each probability map
        for m in range(min(num_maps, prob_maps.size(1))):
            # Create a new state by copying current state
            new_state = GraphState(graph, copy.deepcopy(current_state.labels))
            
            # Get probabilities for map m 
            probs = prob_maps[:, m]
            
            # Calculate value-weighted score for each node
            value_ratio = 1
            
            # Weight the probability by the value ratio (higher is better)
            weighted_scores = probs * value_ratio
            
            # Sort vertices by the weighted score
            nodes_list = list(subgraph.nodes())
            vertices = sorted(nodes_list, 
                           key=lambda x: weighted_scores[nodes_list.index(x)].item(),
                           reverse=True)
            
            # Label vertices greedily until we hit a labeled vertex
            for v in vertices:
                if v in new_state.labels:
                    break
                    
                # Label current vertex as 1 (in MIS)
                new_state.labels[v] = 1
                
                # Label neighbors as 0 (not in MIS)
                for neighbor in graph.neighbors(v):
                    if neighbor not in new_state.labels:
                        new_state.labels[neighbor] = 0
            
            # After breaking from the vertex loop:
            if new_state.is_completely_labeled():
                # Calculate total value of the MIS
                total_value = 0
                for node, label in new_state.labels.items():
                    if label == 1:  # Node is in MIS
                        total_value += graph.nodes[node]['select_value']
                    else:  # Node is not in MIS
                        total_value += graph.nodes[node]['nonselect_value']
                
                # Update best solution if current is better (by value)
                if total_value > best_value:
                    best_value = total_value
                    best_solution = new_state.labels
            else:
                # Add to queue for further exploration
                Q.put(new_state)
                
    return best_solution if best_solution is not None else {}"""

def predict_value_aware_mis(model, graph, time_budget=30, num_maps=32,
                                  max_solutions=16, device='cpu'):
    """
    Generate multiple value-aware MIS solutions using multiple probability maps.

    Returns:
        list of dicts: Each dict maps node -> {0,1}
    """
    model.eval()

    class GraphState:
        def __init__(self, graph, labels=None):
            self.graph = graph
            self.labels = labels if labels is not None else {}

        def is_completely_labeled(self):
            return len(self.labels) == len(self.graph)

        def get_unlabeled_subgraph(self):
            unlabeled = [n for n in self.graph.nodes() if n not in self.labels]
            return self.graph.subgraph(unlabeled)

        def get_unlabeled_nodes(self):
            """Returns list of unlabeled node indices"""
            return [n for n in range(self.num_nodes) if n not in self.labels]

        def signature(self):
            return tuple(sorted(n for n, v in self.labels.items() if v == 1))

    def get_probability_maps(g):
        adj_matrix = torch.tensor(
            nx.adjacency_matrix(g).todense(),
            dtype=torch.float32,
            device=device
        )
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

        nodes = sorted(g.nodes())
        select_values = torch.tensor(
            [g.nodes[n]['select_value'] for n in nodes],
            dtype=torch.float32,
            device=device
        )
        nonselect_values = torch.tensor(
            [g.nodes[n]['nonselect_value'] for n in nodes],
            dtype=torch.float32,
            device=device
        )

        features = torch.ones(len(nodes), model.hidden_dim, device=device)
        features[:, 0] = select_values
        features[:, 1] = nonselect_values

        with torch.no_grad():
            prob_maps = model(adj_matrix, degree_matrix, features)

        return prob_maps, nodes

    # Initialize
    start_time = time.time()
    Q = Queue()
    Q.put(GraphState(graph))

    # Track processed states to avoid duplicates
    #seen_signatures = set()

    solutions = []
    seen_signatures = set()

    # Main loop
    while time.time() - start_time < time_budget and not Q.empty():
        current_state = random.choice(list(Q.queue))
        Q.queue.remove(current_state)

        subgraph = current_state.get_unlabeled_subgraph()
        if len(subgraph) == 0:
            continue

        prob_maps, nodes = get_probability_maps(subgraph)

        for m in range(min(num_maps, prob_maps.size(1))):
            new_state = GraphState(graph, copy.deepcopy(current_state.labels))
            probs = prob_maps[:, m]

            vertices = sorted(
                list(subgraph.nodes()),
                key=lambda x: probs[nodes.index(x)].item(),
                reverse=True
            )

            for v in vertices:
                if v in new_state.labels:
                    break
                new_state.labels[v] = 1
                for nbr in graph.neighbors(v):
                    if nbr not in new_state.labels:
                        new_state.labels[nbr] = 0

            if new_state.is_completely_labeled():
                total_value = sum(
                    graph.nodes[n]['select_value'] if l == 1
                    else graph.nodes[n]['nonselect_value']
                    for n, l in new_state.labels.items()
                )

                sig = new_state.signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    solutions.append((copy.deepcopy(new_state.labels), total_value))
            else:
                Q.put(new_state)

    # Sort by value (descending)
    solutions.sort(key=lambda x: x[1], reverse=True)

    # Return only label dicts
    return [s[0] for s in solutions[:max_solutions]]
