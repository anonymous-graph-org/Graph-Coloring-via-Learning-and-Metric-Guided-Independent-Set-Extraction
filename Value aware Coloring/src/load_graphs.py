import pickle
import networkx as nx

def load_graphs(filename):
    """
    Load graphs from a pickle file.
    
    Parameters:
        filename (str): Path to the pickle file containing graphs.
        
    Returns:
        list: A list of NetworkX graphs.
    """
    with open(filename, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

# Example usage
if __name__ == "__main__":
    # Adjust the path as needed
    graphs = load_graphs("graphs.pickle")
    print(f"Loaded {len(graphs)} graphs")
    # Print some statistics about the first graph
    if graphs:
        g = graphs[0]
        print(f"First graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")