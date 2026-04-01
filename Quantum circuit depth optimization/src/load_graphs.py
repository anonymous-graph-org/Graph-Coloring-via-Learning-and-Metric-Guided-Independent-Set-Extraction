import pickle

def load_graphs(pkl_file_path):
    # Load a dictionary of quantum circuit dependency graphs from a pickle file
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
