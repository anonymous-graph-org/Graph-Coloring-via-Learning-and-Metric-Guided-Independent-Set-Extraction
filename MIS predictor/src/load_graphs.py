import pickle

def load_graphs_from_file(graphs_file, names_file, colors_file):
    """
    Loads graphs, their names, and their colors from pickle files and returns them as lists.
    
    Parameters:
        graphs_file (str): Path to the pickle file containing the graphs.
        names_file (str): Path to the pickle file containing the graph names.
        colors_file (str): Path to the pickle file containing the graph colors.
    Returns:
        tuple: A list of NetworkX graphs, a list of their names, and a list of their colors.
    """
    with open(graphs_file, 'rb') as file:
        graphs_dict = pickle.load(file)
    
    with open(names_file, 'rb') as file:
        names_dict = pickle.load(file)
        
    with open(colors_file, 'rb') as file:
        colors_dict = pickle.load(file)
    
    # Extract graphs, names, and colors in corresponding order
    graphs = []
    names = []
    colors = []
    for key in graphs_dict.keys():
        graphs.append(graphs_dict[key])
        names.append(names_dict.get(key, key))
        colors.append(colors_dict.get(key, "?"))
    
    return graphs, names, colors
