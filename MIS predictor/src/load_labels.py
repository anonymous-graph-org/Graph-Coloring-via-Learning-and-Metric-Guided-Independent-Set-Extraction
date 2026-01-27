import pickle
def load_labels(label_file, graphs):
    """
    Loads labels from a pickle file and converts them to a dictionary where each key
    is a graph index and value is a list of all possible MIS solutions for that graph.

    Args:
        label_file (str): Path to the pickle file containing multiple MIS solutions per graph.

    Returns:
        dict: Dictionary where key is graph index and value is list of binary label vectors,
              each vector representing one possible MIS solution for that graph.
    """
    with open(label_file, 'rb') as f:
        graph_mis_dict = pickle.load(f)
    
    labels = []
    for i in sorted(graph_mis_dict.keys()):
        # graph_mis_dict[i] is now a list of MIS solutions
        all_solutions = graph_mis_dict[i]
        # Convert each solution dictionary to a binary vector
        binary_solutions = []
        for solution in all_solutions:
            binary_vector = [solution[node] for node in sorted(solution.keys())]
            binary_solutions.append(binary_vector)
        labels.append(binary_solutions)
    
    return labels