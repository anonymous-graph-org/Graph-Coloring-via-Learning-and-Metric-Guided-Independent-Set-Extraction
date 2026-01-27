from src.load_data import load_data
from src.load_graphs_from_file import load_graphs_from_file
from src.color_with_value_aware import predict_colors
import sys
import os
import torch
import random
import time
from contextlib import contextmanager
from src.load_model import load_value_aware_model
from src.model import  ValueAwareDeepGCN
from src import config
from src.logger import Logger

# Store the original stdout
original_stdout = sys.stdout


@contextmanager
def redirect_stdout(new_stdout):
    """Context manager to temporarily redirect stdout"""
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout


# Add debug print function that prints to both console and file
def debug_print(message, file=None):
    print(message)
    if file:
        print(message, file=file)
        file.flush()  # Ensure it's written immediately


# Configuration
graphs_file = "networkx_graphs.pkl"
output_file = "output_red_colors.txt"
model_path = "value_aware_gcn.pth"
names_file = "graph_names.pkl"
colors_file = "graph_colors.pkl"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(config.TEST_DATA_PATH), exist_ok=True)

logger = Logger(config.OUTPUT_FILE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log(f"Device: {device}")
logger.log(f"Config: Hidden Dim={config.HIDDEN_DIM}, Layers={config.NUM_LAYERS}, Maps={config.NUM_MAPS}")

# Set random seeds for reproducibility
random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

start_time = time.time()
hidden_dim = 32
num_layers = 20
model = load_value_aware_model(config.MODEL_SAVE_PATH, hidden_dim=config.HIDDEN_DIM,num_layers=config.NUM_LAYERS,num_maps=config.NUM_MAPS,device="cpu")


start_time = time.time()
with redirect_stdout(original_stdout):
    adj_lists, names, colors = load_graphs_from_file(graphs_file, names_file, colors_file)
print(f"Loaded {len(adj_lists)} graphs in {time.time() - start_time:.2f} seconds")

# Open output file for writing
output_stream = open(output_file, 'a', buffering=1)
print("Starting graph processing...", file=output_stream)

# Process each graph
for i, adj_list in enumerate(adj_lists):
    if i==20:
      # Predict colors with original stdout
      start_time = time.time()
      with redirect_stdout(original_stdout):
          num_colors = predict_colors(model, adj_list,logger)
      processing_time = time.time() - start_time

      # Write both execution time and colors to file
      print(f"  - Execution time: {processing_time:.2f} seconds", file=output_stream)
      print(f"  - Colors required: {num_colors}", file=output_stream)


output_stream.close()
