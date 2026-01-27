import networkx as nx

def validate_mis_solution(graph, solution):
    """
    Validates if a given solution is a valid Maximum Independent Set (MIS) for a graph.
    
    Args:
        graph (networkx.Graph): The graph to check against
        solution (dict): Dictionary mapping nodes to 0 or 1, where 1 means included in MIS
        
    Returns:
        dict: Dictionary containing validation results:
            - 'is_valid' (bool): Whether the solution is a valid independent set
            - 'is_maximal' (bool): Whether the solution is maximal (cannot add more nodes)
            - 'violations' (list): List of node pairs that violate independence
            - 'total_value' (float): Total value based on select/nonselect values
            - 'mis_size' (int): Number of nodes in the independent set
    """
    # Get the set of nodes in the MIS
    mis_nodes = set(node for node, value in solution.items() if value == 1)
    
    # Check if it's a valid independent set
    is_valid = True
    violations = []
    
    for node in mis_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor in mis_nodes:
                is_valid = False
                violations.append((node, neighbor))
    
    # Check if it's maximal
    is_maximal = True
    for node in graph.nodes():
        if node not in mis_nodes:
            # Check if this node can be added to the MIS
            can_add = True
            for neighbor in graph.neighbors(node):
                if neighbor in mis_nodes:
                    can_add = False
                    break
            if can_add:
                is_maximal = False
                break
    
    # Calculate total value
    total_value = 0
    for node in graph.nodes():
        if solution.get(node, 0) == 1:
            total_value += max(1,graph.nodes[node]['select_value'])
        else:
            total_value += graph.nodes[node]['nonselect_value']
            
    return {
        'is_valid': is_valid,
        'is_maximal': is_maximal,
        'violations': violations,
        'total_value': total_value,
        'mis_size': len(mis_nodes)
    }