import pickle

def load_graphs(pkl_file_path):
    """
    Load a dictionary of quantum circuit dependency graphs from a pickle file
    
    Parameters:
        pkl_file_path (str): Path to the pickle file containing the graph dictionary
        
    Returns:
        dict: Dictionary where keys are circuit names and values are NetworkX graph objects
    
    Example:
        >>> graphs = load_graphs("qasm_graphs_small.pkl")
        >>> print(list(graphs.keys()))
        ['bell_state', 'grover', 'qft']
        >>> G = graphs['grover']
        >>> print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Successfully loaded {len(graphs)} graphs from {pkl_file_path}")
        return graphs
    except FileNotFoundError:
        print(f"Error: File {pkl_file_path} not found")
        return {}
    except Exception as e:
        print(f"Error loading graphs: {e}")
        return {}
