import numpy as np
import time
import pickle
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import networkx as nx
from contextlib import contextmanager

# Set base directory to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Make sure all modules can be imported
sys.path.append(BASE_DIR)

import os
import datetime
from src.predict_colors import predict_colors
from src.load_model import load_model
from src.model import DeepGCN

# Store the original stdout
original_stdout = sys.stdout

# Open output file immediately with flushing enabled
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"output_{timestamp}.txt"
output_file = open(os.path.join(BASE_DIR, filename), "w", buffering=1)


# output_file = open(os.path.join(BASE_DIR, "output.txt"), "w", buffering=1)  # buffering=1 enables line buffering for flushing

@contextmanager
def redirect_stdout(new_stdout):
    """Context manager to temporarily redirect stdout"""
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout


def log(message):
    """Log message to both console and output file with immediate flushing"""
    print(message)
    print(message, file=output_file)
    output_file.flush()  # Explicitly flush after each write


def load_data(file_path):
    """
    Load NetworkX graphs from a pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        list: List of NetworkX graphs
    """
    log(f"Attempting to load data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    log(f"Data loaded from pickle file. Type: {type(data)}")

    graphs = []
    if isinstance(data, list):
        log(f"Data is a list with {len(data)} items")
        for i, graph in enumerate(data):
            if isinstance(graph, nx.Graph):
                graphs.append(graph)
                if i < 3:  # Log details for first few graphs
                    log(f"  Graph {i}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            else:
                log(f"  Item {i} is not a NetworkX Graph, but a {type(graph)}. Skipping.")
    elif isinstance(data, nx.Graph):
        log("Data is a single NetworkX Graph")
        graphs.append(data)
        log(f"  Graph has {data.number_of_nodes()} nodes and {data.number_of_edges()} edges")
    elif isinstance(data, dict):
        log(f"Data is a dictionary with {len(data)} entries")
        # Check if this is a dictionary of NetworkX graphs
        graph_count = 0
        for key, value in data.items():
            if isinstance(value, nx.Graph):
                graph_count += 1
                graphs.append(value)
                if graph_count <= 3:  # Log details for first few graphs
                    log(f"  Graph for key '{key}': {value.number_of_nodes()} nodes, {value.number_of_edges()} edges")

        if graph_count > 0:
            log(f"Successfully extracted {graph_count} NetworkX graphs from dictionary")
        else:
            # Try a different approach - maybe the dictionary contains graph data
            log("No NetworkX graphs found in dictionary values. Trying to create graphs from data...")
            for key, value in data.items():
                try:
                    # Try to create a graph from the dictionary data
                    new_graph = nx.Graph()
                    # Add code to construct the graph based on your specific dictionary structure
                    # This is a placeholder and will depend on your data format
                    log(f"Created graph for key '{key}'")
                    graphs.append(new_graph)
                except Exception as e:
                    log(f"Could not create graph for key '{key}': {e}")
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected a NetworkX Graph, list of Graphs, or dictionary.")

    if not graphs:
        raise ValueError("No valid NetworkX graphs could be loaded from the file")

    log(f"Successfully loaded {len(graphs)} NetworkX graphs")
    return graphs


def networkx_to_adj_list(nx_graph):
    """
    Convert a NetworkX graph to an adjacency list representation.

    Args:
        nx_graph (networkx.Graph): NetworkX graph

    Returns:
        list: Adjacency list where adj_list[i] contains neighbors of node i
    """
    # Get number of nodes
    num_nodes = nx_graph.number_of_nodes()

    # Ensure nodes are sequential integers from 0 to n-1
    # If not, create a mapping
    node_map = {}
    for i, node in enumerate(sorted(nx_graph.nodes())):
        node_map[node] = i

    # Create adjacency list
    adj_list = [[] for _ in range(num_nodes)]

    # Add edges to adjacency list
    for src, dst in nx_graph.edges():
        src_idx = node_map.get(src, src)
        dst_idx = node_map.get(dst, dst)

        adj_list[src_idx].append(dst_idx)
        adj_list[dst_idx].append(src_idx)  # For undirected graphs

    return adj_list


class WeightPredictor(nn.Module):
    """
    Simple neural network to predict optimal weights for graph coloring metrics
    based on graph features.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super(WeightPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4 weights
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # Split into positive and negative weights
        w_pos = self.softmax(x[:, :2])  # First two weights are positive
        w_neg = -1 * self.softmax(x[:, 2:])  # Last two weights are negative

        # Ensure weights sum to 0 (since we're subtracting negatives)
        w_pos_sum = torch.sum(w_pos, dim=1, keepdim=True)
        w_neg_sum = torch.sum(w_neg, dim=1, keepdim=True)

        w_pos = w_pos / (w_pos_sum + w_neg_sum.abs()) * 2
        w_neg = w_neg / (w_pos_sum + w_neg_sum.abs()) * 2

        # Combine weights
        weights = torch.cat((w_pos, w_neg), dim=1)

        return weights

def load_weight_predictor(trained_weights_dir, input_dim, device="cpu"):
    """
    Load a trained WeightPredictor model and normalization parameters.
    """
    # Load normalization parameters
    with open(os.path.join(trained_weights_dir, "normalization_params.pkl"), "rb") as f:
        normalization_params = pickle.load(f)

    # Recreate model architecture
    model = WeightPredictor(input_dim=input_dim)
    model.load_state_dict(
        torch.load(
            os.path.join(trained_weights_dir, "weight_predictor.pth"),
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, normalization_params

def extract_graph_features(adj_list):
    """
    Extract features from a graph represented as an adjacency list.

    Args:
        adj_list: Adjacency list representation

    Returns:
        numpy.ndarray: Feature vector
    """
    num_nodes = len(adj_list)

    # Convert to NetworkX for easier feature extraction
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            G.add_edge(i, j)

    # Basic graph features
    density = nx.density(G)

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    std_degree = np.std(degrees) if degrees else 0

    # Connectivity measures
    try:
        diameter = nx.diameter(G)
    except (nx.NetworkXError, nx.NetworkXNotImplemented):
        diameter = -1  # Use -1 for disconnected graphs

    try:
        avg_shortest_path = nx.average_shortest_path_length(G)
    except (nx.NetworkXError, nx.NetworkXNotImplemented):
        avg_shortest_path = -1

    # Clustering and community features
    clustering_coef = nx.average_clustering(G)

    # Combine all features
    features = np.array([
        num_nodes,
        G.number_of_edges(),
        density,
        avg_degree,
        max_degree,
        min_degree,
        std_degree,
        diameter,
        avg_shortest_path,
        clustering_coef
    ])

    return features


def find_best_weights(model, adj_list, num_trials=20):
    """
    Find the best weights for a given graph using random search.

    Args:
        model: Trained DeepGCN model
        adj_list: Adjacency list representation of the graph
        num_trials: Number of random weight combinations to try

    Returns:
        tuple: (best_weights, best_colors) - Best weights found and corresponding colors
    """
    best_colors = float('inf')
    best_weights = None

    log(f"Starting random search with {num_trials} trials")
    for trial in range(num_trials):
        # Generate random weights that sum to 1
        w1, w2, w3, w4 = np.random.dirichlet(np.ones(4), size=1)[0]

        # Make w3 and w4 negative (for progress and efficiency metrics)
        weights = [w1, w2, -w3, -w4]

        # Run beam search with these weights
        start_time = time.time()
        colors = predict_colors(model, adj_list, weights=weights)
        end_time = time.time()

        log(f"  Trial {trial + 1}/{num_trials}: Weights={weights}, Colors={colors}, Time={end_time - start_time:.2f}s")

        # Update best weights if current is better
        if colors < best_colors:
            best_colors = colors
            best_weights = weights
            log(f"New best result! Colors={colors}, Weights={weights}")

    log(f"Finished random search. Best result: Colors={best_colors}, Weights={best_weights}")
    return best_weights, best_colors


def train_weight_predictor(features, best_weights, epochs=100, batch_size=8):
    """
    Train a neural network to predict optimal weights based on graph features.

    Args:
        features: Array of graph features
        best_weights: Array of corresponding best weights
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        tuple: (model, normalization_params) - Trained model and normalization parameters
    """
    log(f"Starting weight predictor training with {epochs} epochs")
    log(f"Input features shape: {features.shape}, Target weights shape: {best_weights.shape}")

    # Normalize features
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    features_normalized = (features - feature_mean) / (feature_std + 1e-8)

    # Show feature statistics
    log("Feature statistics:")
    for i in range(features.shape[1]):
        log(f"  Feature {i}: mean={feature_mean[i]:.4f}, std={feature_std[i]:.4f}")

    # Convert to PyTorch tensors
    X = torch.tensor(features_normalized, dtype=torch.float32)
    y = torch.tensor(best_weights, dtype=torch.float32)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = WeightPredictor(input_dim=features.shape[1])
    log(f"Model initialized with input dimension {features.shape[1]}")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    log("Starting training...")
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            # Print progress for each batch
            if batch_count % 5 == 0 or batch_count == len(train_loader):
                log(f"  Epoch {epoch + 1}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.6f}")

        train_loss /= len(train_loader)
        all_train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        all_val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            log(f"New best model! Val Loss: {val_loss:.6f}")

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print progress
        log(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")

        # Print weight examples
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_indices = np.random.choice(len(X_val), min(3, len(X_val)), replace=False)
                log("Sample predictions:")
                for idx in sample_indices:
                    input_features = X_val[idx:idx + 1]
                    true_weights = y_val[idx].cpu().numpy()
                    pred_weights = model(input_features).cpu().numpy()[0]
                    log(f"  Sample {idx}: True={true_weights}, Pred={pred_weights}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log(f"Restored best model with validation loss: {best_val_loss:.6f}")

    # Normalization parameters
    normalization_params = {
        'mean': feature_mean,
        'std': feature_std
    }

    log("Training completed!")

    return model, normalization_params


def predict_optimal_weights(adj_list, predictor_model, normalization_params):
    """
    Predict optimal weights for a given graph.

    Args:
        adj_list: Adjacency list representation of the graph
        predictor_model: Trained WeightPredictor model
        normalization_params: Dictionary with feature normalization parameters

    Returns:
        numpy.ndarray: Predicted optimal weights
    """
    # Extract features
    features = extract_graph_features(adj_list)

    # Normalize features
    features_normalized = (features - normalization_params['mean']) / (normalization_params['std'] + 1e-8)

    # Convert to PyTorch tensor
    X = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)

    # Predict weights
    predictor_model.eval()
    with torch.no_grad():
        weights = predictor_model(X).squeeze().numpy()

    return weights


def main():
    # Create output directory for saving trained models and results
    trained_weights_dir = os.path.join(BASE_DIR, "trained_weights")
    #os.makedirs(trained_weights_dir, exist_ok=True)

    log("*" * 80)
    log("STARTING TESTING PROCESS")
    log("*" * 80)
    log(f"Current working directory: {os.getcwd()}")
    log(f"Base directory: {BASE_DIR}")
    log(f"Output directory: {trained_weights_dir}")
    log("*" * 80)

    # Configuration paths - Using relative paths
    training_data_file = os.path.join(BASE_DIR, 'graphs.pickle')  # Training graphs
    test_graphs_file = os.path.join(BASE_DIR, 'networkx_graphs.pkl')  # Test graphs
    test_names_file = os.path.join(BASE_DIR, 'graph_names.pkl')
    test_colors_file = os.path.join(BASE_DIR, 'graph_colors.pkl')
    model_dir = os.path.join(BASE_DIR, 'model_parameters')

    # Check for model directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        log(f"Created model directory: {model_dir}")

    model_path = os.path.join(model_dir, 'gcn_model.pth')  # GCN model path

    # Load the GCN model
    log("Loading GCN model...")
    try:
        log(f"Loading model from: {model_path}")
        hidden_dim = 32
        num_layers = 20
        gcn_model = load_model(model_path, DeepGCN, hidden_dim, num_layers)
        log("GCN model loaded successfully")
    except Exception as e:
        log(f"Error loading GCN model: {e}")
        log("Exiting.")
        return
    """
    # Load training data using custom loader for NetworkX graphs
    log("\n" + "=" * 50)
    log("LOADING TRAINING DATA")
    log("=" * 50)
    try:
        log(f"Reading training data from: {training_data_file}")
        train_nx_graphs = load_data(training_data_file)
        log(f"Loaded {len(train_nx_graphs)} training graphs")

        # Convert NetworkX graphs to adjacency lists
        log("Converting NetworkX graphs to adjacency lists...")
        train_adj_lists = []
        for i, nx_graph in enumerate(train_nx_graphs):
            if i < 5 or i % 10 == 0:  # Log details for some graphs
                log(f"  Converting graph {i + 1}/{len(train_nx_graphs)}: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
            adj_list = networkx_to_adj_list(nx_graph)
            train_adj_lists.append(adj_list)
        log(f"Converted {len(train_adj_lists)} NetworkX graphs to adjacency lists")
    except Exception as e:
        log(f"Error loading training data: {e}")
        log("Exiting.")
        return

    # Select a subset of training graphs to find optimal weights
    log("\n" + "=" * 50)
    log("SELECTING TRAINING GRAPHS")
    log("=" * 50)
    # Use at most 50 graphs for training to keep computation manageable
    num_train_graphs = min(50, len(train_adj_lists))
    log(f"Selecting {num_train_graphs} graphs from {len(train_adj_lists)} total graphs")

    train_indices = np.random.choice(len(train_adj_lists), num_train_graphs, replace=False)
    log(f"Selected graph indices: {train_indices}")

    selected_train_graphs = [train_adj_lists[i] for i in train_indices]
    log(f"Selected {len(selected_train_graphs)} graphs for weight optimization")

    # Find best weights for each training graph
    log("\n" + "=" * 50)
    log("FINDING OPTIMAL WEIGHTS FOR TRAINING GRAPHS")
    log("=" * 50)
    features = []
    best_weights_list = []

    for i, graph in enumerate(selected_train_graphs):
        log(f"\nProcessing training graph {i + 1}/{num_train_graphs}")
        log(f"Graph size: {len(graph)} nodes, {sum(len(neighbors) for neighbors in graph) // 2} edges")

        # Extract features
        log("Extracting graph features...")
        graph_features = extract_graph_features(graph)
        log(f"Extracted features: {graph_features}")

        # Find best weights using random search
        log("Finding best weights using random search...")
        best_weights, best_colors = find_best_weights(gcn_model, graph, num_trials=10)
        log(f"Best weights: {best_weights}, Colors: {best_colors}")

        features.append(graph_features)
        best_weights_list.append(best_weights)

    # Convert to numpy arrays
    features = np.array(features)
    best_weights_list = np.array(best_weights_list)

    # Save the training data
    log("\nSaving training data...")
    with open(os.path.join(trained_weights_dir, "training_data.pkl"), 'wb') as f:
        pickle.dump({
            'features': features,
            'best_weights': best_weights_list
        }, f)
    log(f"Saved training data to {os.path.join(trained_weights_dir, 'training_data.pkl')}")

    with open(os.path.join(trained_weights_dir, "training_data.pkl"), 'rb') as f:
       data = pickle.load(f)

    features = data['features']
    best_weights_list = data['best_weights']
    # Train the weight predictor model
    log("\n" + "=" * 50)
    log("TRAINING WEIGHT PREDICTOR MODEL")
    log("=" * 50)
    predictor_model, normalization_params = train_weight_predictor(
        features, best_weights_list, epochs=100, batch_size=8
    )

    # Save the predictor model and normalization parameters
    log("\nSaving predictor model and normalization parameters...")
    torch.save(predictor_model.state_dict(), os.path.join(trained_weights_dir, "weight_predictor.pth"))
    with open(os.path.join(trained_weights_dir, "normalization_params.pkl"), 'wb') as f:
        pickle.dump(normalization_params, f)
    log(f"Saved predictor model to {os.path.join(trained_weights_dir, 'weight_predictor.pth')}")
    log(f"Saved normalization parameters to {os.path.join(trained_weights_dir, 'normalization_params.pkl')}")
    """

    # Load test graphs
    log("\n" + "=" * 50)
    log("LOADING PRE-TRAINED WEIGHT PREDICTOR")
    log("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 10  # MUST match training
    predictor_model, normalization_params = load_weight_predictor(trained_weights_dir,input_dim=input_dim,device=device)

    log("Weight predictor and normalization parameters loaded successfully")
    # Configuration paths - Using relative paths


    log("\n" + "=" * 50)
    log("LOADING TEST GRAPHS")
    log("=" * 50)
    try:
        # Check if test files exist
        if os.path.exists(test_graphs_file):
            log(f"Loading test graphs from: {test_graphs_file}")
            from src.load_graphs import load_graphs_from_file
            with redirect_stdout(original_stdout):
                test_adj_lists, graph_names, known_colors = load_graphs_from_file(
                    test_graphs_file, test_names_file, test_colors_file
                )
            log(f"Loaded {len(test_adj_lists)} test graphs")
            log(f"Sample graph names: {graph_names[:5]}")
            log(f"Sample known colors: {known_colors[:5]}")
        else:
            # If test files don't exist, use a subset of training graphs as test set
            log("Test graph files not found. Using a subset of training graphs for testing.")
            # Select different graphs than those used for training
            """all_indices = set(range(len(train_adj_lists)))
            train_indices_set = set(train_indices)
            available_indices = list(all_indices - train_indices_set)

            # Use at most 20 graphs for testing
            num_test_graphs = min(20, len(available_indices))
            test_indices = np.random.choice(available_indices, num_test_graphs, replace=False)
            log(f"Selected test graph indices: {test_indices}")

            test_adj_lists = [train_adj_lists[i] for i in test_indices]
            # Create placeholder names and colors
            graph_names = [f"Graph_{i}" for i in test_indices]
            known_colors = ["?" for _ in test_indices]

            log(f"Selected {num_test_graphs} graphs for testing")"""
    except Exception as e:
        log(f"Error loading test graphs: {e}")
        log("Exiting.")
        return



    # Test the trained predictor on test graphs
    log("\n" + "=" * 50)
    log("TESTING WEIGHT PREDICTOR ON TEST GRAPHS")
    log("=" * 50)

    # Create a CSV file for detailed results
    csv_file = open(os.path.join(trained_weights_dir, "detailed_results.csv"), 'a',
                    buffering=1)  # Enable line buffering
    csv_file.write(
        "graph_name,known_colors,default_colors,predicted_colors,default_weights,predicted_weights,improvement\n")
    csv_file.flush()  # Flush after writing header

    default_weights = [0.4, 0.3, -0.2, -0.1]  # Default weights for comparison
    total_default = 0
    total_predicted = 0
    better_count = 0
    worse_count = 0
    same_count = 0

    for i, adj_list in enumerate(test_adj_lists):
        abc=len(adj_list)
        if abc > 200:
            continue
        graph_name = graph_names[i] if i < len(graph_names) else f"Graph_{i}"
        log(f"\nTesting graph {i + 1}/{len(test_adj_lists)}: {graph_name}")
        log(f"Graph size: {len(adj_list)} nodes, {sum(len(neighbors) for neighbors in adj_list) // 2} edges")

        # Get known optimal colors if available
        known_color = "?"
        if i < len(known_colors) and known_colors[i] != "?":
            try:
                known_color = int(known_colors[i])
                log(f"Known colors: {known_color}")
            except:
                known_color = "?"

        # Test with default weights
        """log("\nTesting with default weights...")
        start_time = time.time()
        default_result = predict_colors(gcn_model, adj_list, weights=default_weights)
        default_time = time.time() - start_time
        log(f"Default weights: {default_weights}")
        log(f"Result: {default_result} colors in {default_time:.2f}s")"""

        # Predict optimal weights for this graph
        log("\nPredicting optimal weights...")
        weight_prediction_start = time.time()
        predicted_weights = predict_optimal_weights(adj_list, predictor_model, normalization_params)
        weight_prediction_time = time.time() - weight_prediction_start
        log(f"Predicted weights: {predicted_weights}")
        log(f"Weight prediction time: {weight_prediction_time:.2f}s")

        # Test with predicted weights
        log("\nTesting with predicted weights...")
        start_time = time.time()
        predicted_result = predict_colors(gcn_model, adj_list, weights=predicted_weights)
        predicted_time = time.time() - start_time
        log(f"Predicted weights: {predicted_weights}")
        log(f"Result: {predicted_result} colors in {predicted_time:.2f}s")

        # Compare results
        improvement = default_result - predicted_result
        if improvement > 0:
            comparison = "BETTER"
            better_count += 1
        elif improvement < 0:
            comparison = "WORSE"
            worse_count += 1
        else:
            comparison = "SAME"
            same_count += 1

        log(f"\nComparison: {comparison}")
        log(f"Default weights: {default_result} colors")
        log(f"Predicted weights: {predicted_result} colors")
        log(f"Improvement: {improvement} colors")

        # Write to CSV
        csv_file.write(
            f"{graph_name},{known_color},{default_result},{predicted_result},\"{default_weights}\",\"{predicted_weights.tolist()}\",{improvement}\n")
        csv_file.flush()  # Explicitly flush after each write

        # Update totals
        total_default += default_result
        total_predicted += predicted_result

    # Write summary
    log("\n" + "=" * 50)
    log("SUMMARY")
    log("=" * 50)

    if len(test_adj_lists) > 0:
        avg_default = total_default / len(test_adj_lists)
        avg_predicted = total_predicted / len(test_adj_lists)

        log(f"Average colors with default weights: {avg_default:.2f}")
        log(f"Average colors with predicted weights: {avg_predicted:.2f}")
        log(f"Average improvement: {avg_default - avg_predicted:.2f} colors")
        log(f"Better: {better_count} graphs ({better_count / len(test_adj_lists) * 100:.1f}%)")
        log(f"Same: {same_count} graphs ({same_count / len(test_adj_lists) * 100:.1f}%)")
        log(f"Worse: {worse_count} graphs ({worse_count / len(test_adj_lists) * 100:.1f}%)")
    else:
        log("No test graphs were processed.")

    # Close files
    csv_file.close()
    log("Training and testing complete!")
    log("Results saved in the 'trained_weights' directory")

    # Close files
    output_file.close()

    print("\nTraining and testing complete!")
    print(f"Results saved in the 'trained_weights' directory")


if __name__ == "__main__":
    main()
