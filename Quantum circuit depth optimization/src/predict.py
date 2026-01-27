from src.predict_mis import predict_mis
import heapq
from copy import deepcopy

def predict_colors(graph, model, queue_capacity=4):
    """
    Predict the number of colors needed to color a graph using beam search approach.
    Handles mixed graphs with both directed and undirected edges.
    Only directed edges contribute to indegree calculations.
    
    Args:
        graph: The dependency graph representation of a quantum circuit
        model: The model used for predicting MIS
        queue_capacity: Maximum size of the queue (beam width)
        time_budget: Time budget for the algorithm in seconds
        
    Returns:
        min_colors: Minimum number of colors needed
        coloring: Dictionary mapping nodes to their color assignments
    """
    
    # Helper function to extract processable nodes (in-degree 0)
    # Only considers directed edges for indegree calculation
    def extract_processable(g):
        processable_nodes = []
        for node in g.nodes():
            # Count only directed edges for indegree
            directed_in_degree = sum(1 for _, _, data in g.in_edges(node, data=True) 
                                    if data.get('directed', True))
            if directed_in_degree == 0:
                processable_nodes.append(node)
        return g.subgraph(processable_nodes)
    
    # Helper function to calculate edge density of a graph
    def calculate_edge_density(g):
        if len(g.nodes()) <= 1:
            return 0.0
        return len(g.edges()) / len(g.nodes()) if len(g.nodes()) > 0 else 0.0
    
    # Create a unique identifier for each state
    # This avoids direct comparison of graph objects
    state_counter = 0
    
    # Initialize queue, coloring, and tracking variables
    queue = []
    
    # Use state_counter to ensure uniqueness in comparisons
    edge_density = calculate_edge_density(graph)
    heapq.heappush(queue, (edge_density, 0, state_counter, graph, {}))
    state_counter += 1
    
    best_coloring = None
    min_colors = float('inf')
    
    while queue:
        # Dequeue the graph state with minimum edge density
        _, num_colors, _, current_graph, current_coloring = heapq.heappop(queue)
        
        # If the graph is empty, we found a solution
        if len(current_graph.nodes()) == 0:
            if num_colors < min_colors:
                min_colors = num_colors
                best_coloring = current_coloring
            continue
        
        # Extract nodes with no incoming directed edges
        executable_graph = extract_processable(current_graph)
        
        if len(executable_graph.nodes()) == 0:
            print(f"Warning: No processable nodes found with {len(current_graph.nodes())} nodes remaining")
            # Try to recover by selecting nodes with minimum indegree
            min_indegree = float('inf')
            min_indegree_nodes = []
            
            for node in current_graph.nodes():
                directed_in_degree = sum(1 for _, _, data in current_graph.in_edges(node, data=True) 
                                       if data.get('directed', True))
                if directed_in_degree < min_indegree:
                    min_indegree = directed_in_degree
                    min_indegree_nodes = [node]
                elif directed_in_degree == min_indegree:
                    min_indegree_nodes.append(node)
            
            executable_graph = current_graph.subgraph(min_indegree_nodes)
        
        # Find multiple MIS for the executable subgraph using your provided function
        try:
            mis_candidates = predict_mis(model,executable_graph)
        except Exception as e:
            print(f"Error in predict_mis: {e}")
            # Fallback to a simple greedy approach if MIS prediction fails
            mis_nodes = list(executable_graph.nodes())
            if mis_nodes:
                fallback_mis = {}
                first_node = mis_nodes[0]
                for node in executable_graph.nodes():
                    fallback_mis[node] = 1 if node == first_node else 0
                mis_candidates = [fallback_mis]
            else:
                continue
        
        # Process each MIS candidate
        new_states = []
        
        for mis in mis_candidates:
            # mis is a dictionary mapping nodes to 0 or 1
            # Get nodes that are in the MIS (labeled as 1)
            mis_nodes = [node for node, label in mis.items() if label == 1]
            
            if not mis_nodes:
                print("Warning: Empty MIS, skipping")
                continue
                
            # Create new graph by removing MIS nodes
            new_graph = deepcopy(current_graph)
            new_graph.remove_nodes_from(mis_nodes)
            
            # Update coloring
            new_coloring = current_coloring.copy()
            for node in mis_nodes:
                new_coloring[node] = num_colors
            
            # Calculate edge density for priority
            edge_density = calculate_edge_density(new_graph)
            
            # Add to new states with a unique counter to avoid direct graph comparisons
            new_states.append((edge_density, new_graph, new_coloring))
        
        # Sort by edge density
        new_states.sort(key=lambda x: x[0])
        new_states = new_states[:queue_capacity]
        
        # Enqueue new states with incremented color count and unique ID
        for edge_density, new_graph, new_coloring in new_states:
            if len(new_graph.nodes()) < len(current_graph.nodes()):
                heapq.heappush(queue, (edge_density, num_colors + 1, state_counter, new_graph, new_coloring))
                state_counter += 1
            else:
                print("Warning: No progress made in reducing graph size")
    
    # If we found a solution, return the number of colors and coloring
    if min_colors < float('inf'):
        print(f"Found solution with {min_colors} colors")
        return min_colors, best_coloring
    else:
        print("No solution found within time budget")
        return None, None